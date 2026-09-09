"""Open-vocabulary retrieval baseline on one Workshop W1--W10 variant.

No language model is used.  The task structure is fixed and every role is
filled by CLIP image-text similarity between the role's function phrase and a
crop from the raw camera frames, exactly as in the Living Room runner.

Workshop differs from the Living Room in one way that matters: the components
start inside closed storage, so there is nothing to score until a region has
been opened.  Retrieval has no mechanism for deciding where to look -- that
decision is what a language model or a functional search would contribute --
so it follows the scene's own fixed inspection order to exhaustion and scores
whatever that reveals.  Consuming the whole order is the honest choice: a
retrieval policy that stopped early would be making a discovery decision it
has no basis for.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.models import Action
from baseline_common.physical_benchmark import (
    GOAL_COMPLETE_STATUS,
    write_execution_result,
)
from mujoco_scenes.baseline_workshop_runtime import (
    MirroredWorkshopExecutor,
    WorkshopPhysicalExecutor,
    workshop_goal_reached,
)
from vlm_tamp_baseline.workshop_runtime import (
    DEFAULT_EXPECTED_ROOT,
    STORAGE_BACKENDS,
    WorkshopPlanningRuntime,
    WorkshopSymbolicExecutor,
    canonical_workshop_actions,
    compare_workshop_actions,
)

from .retrieval import (
    CROP_CONTEXT_FRACTION,
    CLIPRetriever,
    assign_distinct,
    read_annotations,
)
from .roles import WORKSHOP_ROLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="W1-W10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the retrieved sequence through the physical Workshop "
            "scene.  Retrieval issues no model requests, so this isolates how "
            "far a purely similarity-grounded assignment gets on the robot."
        ),
    )
    # Accepted and ignored: the shared batch runner sends the same inference
    # flags to every method and retrieval grounds with CLIP, not a model.
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--decoding", help=argparse.SUPPRESS)
    parser.add_argument("--max-total-actions", type=int, help=argparse.SUPPRESS)
    return parser


def _role(key: str):
    return next(role for role in WORKSHOP_ROLES if role.key == key)


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        output = prepare_run_directory(arguments.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    started_at = time.monotonic()
    runtime = WorkshopPlanningRuntime(
        arguments.variant,
        output,
        expected_root=arguments.expected_root.resolve(),
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        camera_count=arguments.camera_count,
        physical_execution=bool(arguments.execute),
    )
    goal = arguments.goal or runtime.goal
    # Inspection has to advance the *planning* runtime as well, because that
    # is what renders the frames and owns `visible_object_ids`.  Executing
    # only against the physical scene opened the real drawers while the
    # runtime still believed the cell was closed, so CLIP was handed an empty
    # candidate set and every role came back unfilled.
    physical = WorkshopPhysicalExecutor(runtime) if arguments.execute else None
    executor = (
        MirroredWorkshopExecutor(physical, WorkshopSymbolicExecutor(runtime))
        if physical is not None
        else WorkshopSymbolicExecutor(runtime)
    )
    try:
        predicted: list[dict[str, Any]] = []
        action_history: list[dict[str, Any]] = []
        executed_actions = 0

        def run(row: dict[str, Any]) -> bool:
            """Record one planned action and advance the world by it.

            In planning-only mode this advances the symbolic rollout, which is
            what reveals storage contents for scoring.  Under --execute the
            physical scene decides and the rollout mirrors it.
            """
            nonlocal executed_actions
            predicted.append(row)
            outcome = executor.execute(
                Action(row["skill"], dict(row["arguments"]))
            )
            if physical is not None:
                executed_actions += 1
            action_history.append({
                "action": row,
                "success": outcome.success,
                "failure_code": outcome.failure_code,
                "message": outcome.message,
            })
            return outcome.success

        # Fixed inspection order, to exhaustion: see the module docstring.
        # The order is the scene's canonical storage sequence; retrieval has no
        # basis for choosing a different one.
        inspection_order = [
            runtime.backend_by_region_id[name] for name in STORAGE_BACKENDS
        ]
        aborted = False
        for region_id in inspection_order:
            if not run({"skill": "INSPECT", "arguments": {"region_id": region_id}}):
                aborted = True
                break

        runtime.observe()
        stage = "initial" if runtime.revision == 0 else f"revision_{runtime.revision:03d}"
        observation_dir = output / "observations" / stage
        annotations = read_annotations(observation_dir)

        retriever = CLIPRetriever(device=arguments.clip_device)
        # The frame joint is the fixed repair target, not a selectable
        # candidate: electing it as a tool or a part would let the target
        # masquerade as its own solution.
        candidate_ids = sorted(
            item for item in runtime.visible_object_ids
            if item != runtime.target_object_id
        )
        object_scores = retriever.score(
            annotations,
            observation_dir,
            [_role(key).phrase for key in ("turning_tool", "threaded_part")],
            "object",
            candidate_ids=candidate_ids,
        )
        tools = assign_distinct(
            object_scores, _role("turning_tool").phrase, 1, taken=()
        )
        parts = assign_distinct(
            object_scores, _role("threaded_part").phrase, 1, taken=tools
        )
        selection = {"turning_tool": tools, "threaded_part": parts}
        complete = len(tools) == 1 and len(parts) == 1 and not aborted

        if complete:
            target = runtime.target_object_id
            surface = runtime.backend_by_region_id[
                "MAIN_WORKBENCH_ZONE"
            ]
            # Same order as the ground-truth plan: seat the part, then bring
            # the tool, drive, and leave the tool on the work surface.
            for row in (
                {"skill": "PICK", "arguments": {"object_id": parts[0]}},
                {"skill": "INSERT", "arguments": {
                    "fastener_id": parts[0], "target_id": target}},
                {"skill": "PICK", "arguments": {"object_id": tools[0]}},
                {"skill": "FASTEN", "arguments": {
                    "tool_id": tools[0], "fastener_id": parts[0],
                    "target_id": target}},
                {"skill": "PLACE", "arguments": {
                    "object_id": tools[0], "region_id": surface}},
            ):
                if not run(row):
                    break

        predicted_outcome = "FEASIBLE" if complete else "INFEASIBLE"
        if not complete:
            predicted.append({
                "operator": "TERMINATE_INFEASIBLE",
                "arguments": ["NO_RETRIEVED_ROLE_FILLER"],
            })

        backend_by_id = {**runtime.object_by_backend, **runtime.region_by_backend}
        comparison = compare_workshop_actions(
            canonical_workshop_actions(action_history, backend_by_id)
            if action_history else [],
            runtime.expected.actions,
        )
        comparison.update({
            "variant": runtime.variant,
            "predicted_outcome": predicted_outcome,
            "expected_outcome": runtime.expected.intended_outcome,
            "outcome_match": predicted_outcome == runtime.expected.intended_outcome,
            "gt_was_model_input": False,
        })

        physical_goal_satisfied = workshop_goal_reached(physical, runtime)
        write_json(output / "retrieval_trace.json", {
            "schema_version": 1,
            "roles": [
                {"key": r.key, "phrase": r.phrase, "count": r.count, "kind": r.kind}
                for r in WORKSHOP_ROLES
            ],
            "selection": selection,
            "object_scores": object_scores.to_dict(),
            "candidate_object_ids": candidate_ids,
            "inspection_order": inspection_order,
            "inspection_aborted": aborted,
            "crops_taken_from": "raw_<camera>.png (unannotated)",
            "crop_context_fraction": CROP_CONTEXT_FRACTION,
            "rejection_criterion": (
                "cardinality: a role is unfilled when the observable objects, "
                "after the fixed inspection order is exhausted, cannot supply "
                "a distinct candidate"
            ),
        })
        write_json(output / "method_manifest.json", {
            "method": "Open-vocabulary retrieval (CLIP), no language model",
            "environment": "workshop",
            "evaluation_mode": (
                "PHYSICAL_EXECUTION_PLUS_GT_SEQUENCE_COMPARISON"
                if arguments.execute else "PLANNING_ONLY_GT_SEQUENCE_COMPARISON"
            ),
            "physical_execution": bool(arguments.execute),
            "model_requests": 0,
            "camera_count": arguments.camera_count,
            "seed": arguments.seed,
            "gt_visible_to_model": False,
            "hidden_storage_contents_visible_to_model": False,
        })

        payload = {
            "baseline": "retrieval",
            "environment": "workshop",
            "variant": runtime.variant,
            "goal": goal,
            "camera_count": arguments.camera_count,
            "seed": arguments.seed,
            "physical_execution": bool(arguments.execute),
            "executed_actions": executed_actions,
            "direct_payload_pose_writes": (
                physical.direct_payload_pose_writes if physical else 0
            ),
            "physical_goal_satisfied": physical_goal_satisfied,
            "result": {
                "status": "RETRIEVED" if complete else "NO_RETRIEVED_ROLE_FILLER",
                "actions": predicted,
                "selection": selection,
                "action_history": action_history,
            },
            "gt_comparison": comparison,
        }
        write_json(output / "episode_result.json", payload)
        write_json(output / "gt_sequence_comparison.json", comparison)

        if arguments.execute:
            write_execution_result(
                output,
                scene="workshop",
                method="retrieval",
                protocol=arguments.protocol,
                variant=runtime.variant,
                camera_count=arguments.camera_count,
                seed=arguments.seed,
                success=physical_goal_satisfied,
                executed_actions=executed_actions,
                # CLIP retrieval issues no language-model requests at all.
                model_calls=0,
                raw_vlm_requests=0,
                replans=0,
                planning_latency_s=0.0,
                elapsed_seconds=time.monotonic() - started_at,
                terminal_status=(
                    GOAL_COMPLETE_STATUS if physical_goal_satisfied
                    else "NO_RETRIEVED_ROLE_FILLER" if not complete
                    else "EXECUTION_FAILED"
                ),
                expected_outcome=runtime.expected.intended_outcome,
                predicted_outcome=predicted_outcome,
            )
        print(f"[workshop retrieval] {runtime.variant}: {payload['result']['status']}, "
              f"{executed_actions} actions, goal={physical_goal_satisfied}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
