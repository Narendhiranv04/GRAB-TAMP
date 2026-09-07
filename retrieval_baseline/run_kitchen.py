"""Open-vocabulary retrieval baseline on one Kitchen K1--K12 variant.

No language model is used.  The task skeleton is fixed and every role is
filled by CLIP image-text similarity between the role's function phrase and a
crop from the raw camera frames, exactly as in the Living Room and Workshop
runners.

Two things make Kitchen harder than the other two domains for this baseline,
and both are properties of the task rather than of the implementation:

- The plan is a 24-action sequence with ordering constraints, not a set of
  placements.  Each source must be picked, applied to every drink vessel and
  returned before the next is picked, because the robot has one gripper; each
  food vessel must be served before its utensil is seated in it.  The skeleton
  in `roles.KITCHEN_TASK_SKELETON` encodes that order.
- Several roles are near-identical in appearance.  The stirring implement and
  the eating utensils are all slender implements; the drink and food vessels
  are all open vessels.  Similarity cannot separate them by what they are for,
  which is the capability under test.

Required items may be inside closed storage, and retrieval has no basis for
choosing where to look, so it inspects the scene's regions in their fixed
order to exhaustion before scoring -- the same policy as the Workshop runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.execution import MuJoCoActionExecutor
from baseline_common.models import Action
from baseline_common.physical_benchmark import (
    GOAL_COMPLETE_STATUS,
    write_execution_result,
)
from mujoco_scenes.baseline_kitchen_runtime import (
    ARTICULATION_SPECS,
    BaselineKitchenRuntime,
)

from .retrieval import (
    CROP_CONTEXT_FRACTION,
    CLIPRetriever,
    assign_distinct,
    read_annotations,
)
from .roles import KITCHEN_ROLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="K1-K12 or internal ID")
    parser.add_argument("--physical-variant", help="Alias accepted by the batch runner")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--close-on-complete", action="store_true")
    # Accepted and ignored: the shared batch runner sends the same inference
    # flags to every method, and retrieval grounds with CLIP, not a model.
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--decoding", help=argparse.SUPPRESS)
    parser.add_argument("--max-model-calls", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-total-actions", type=int, help=argparse.SUPPRESS)
    return parser


def _expected_outcome(variant: str) -> str | None:
    """Feasible/infeasible label for a Kitchen variant.

    Resolved from the scene configuration rather than assumed from the label's
    number, so a catalogue change cannot silently mislabel a trial.
    """
    from mujoco_scenes.final_paper_variant_labels import resolve_variant_name
    from mujoco_scenes.scene_loader import KITCHEN_FEASIBILITY_VARIANTS
    import yaml

    document = yaml.safe_load(
        KITCHEN_FEASIBILITY_VARIANTS.read_text(encoding="utf-8")
    )
    internal = resolve_variant_name("kitchen", variant)
    entry = (document.get("variants") or {}).get(internal) or {}
    outcome = entry.get("intended_outcome")
    return str(outcome) if outcome else None


def _role(key: str):
    return next(role for role in KITCHEN_ROLES if role.key == key)


def main() -> None:
    arguments = build_parser().parse_args()
    variant = arguments.physical_variant or arguments.variant
    if not variant:
        raise SystemExit("--variant or --physical-variant is required")
    try:
        output = prepare_run_directory(arguments.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    started_at = time.monotonic()
    runtime = BaselineKitchenRuntime.from_variant(
        variant,
        output,
        camera_count=arguments.camera_count,
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        # --headless was parsed and then ignored, so this runner inherited
        # BaselineKitchenRuntime's show_viewer=True default and launched an
        # interactive passive viewer per episode -- even under the batch
        # runner, which passes --headless for exactly this reason.  The viewer
        # throttles stepping to render rate: four concurrent episodes sat at
        # 43% CPU on an idle 24-core host and wrote nothing for 54 minutes.
        # The other two Kitchen runners already pass this.
        show_viewer=not arguments.headless,
    )
    goal = arguments.goal or getattr(runtime, "goal", "")
    executor = MuJoCoActionExecutor(
        runtime.dispatcher,
        effect_sink=runtime.accept_effects,
    )
    try:
        runtime.open()
        planned: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        executed = 0

        def run(skill: str, **args: str) -> bool:
            nonlocal executed
            row = {"skill": skill, "arguments": args}
            planned.append(row)
            outcome = executor.execute(Action(skill, args))
            executed += 1
            history.append({
                "action": row,
                "success": outcome.success,
                "failure_code": outcome.failure_code,
                "message": outcome.message,
            })
            return outcome.success

        # Inspect every region in its fixed order before scoring: required
        # items may be in closed storage and retrieval cannot choose where to
        # look.  A closed region simply yields no candidates.
        # The articulated regions are the closed storage; `countertop` and
        # `serving_area` are open supports and are not inspectable.
        inspected_regions = []
        for region_id in sorted(ARTICULATION_SPECS):
            if run("INSPECT", region_id=region_id):
                inspected_regions.append(region_id)

        observation, _ = runtime.observe()
        observation_dir = output / "observations" / "initial"
        annotations = read_annotations(observation_dir)

        retriever = CLIPRetriever(device=arguments.clip_device)
        role_keys = (
            "water_source", "grounds_source", "stir_tool",
            "drink_vessel", "food_vessel", "eating_utensil",
        )
        scores = retriever.score(
            annotations,
            observation_dir,
            [_role(key).phrase for key in role_keys],
            "object",
        )

        # Filled in a fixed order, each role taking only candidates no earlier
        # role claimed.  The order is not tuned: it follows the task skeleton,
        # so a wrong early pick propagates, which is the behaviour being
        # measured rather than something to be smoothed over.
        selection: dict[str, list[str]] = {}
        taken: list[str] = []
        for key in role_keys:
            role = _role(key)
            chosen = assign_distinct(scores, role.phrase, role.count, taken=taken)
            selection[key] = chosen
            taken.extend(chosen)

        complete = all(len(selection[k]) == _role(k).count for k in role_keys)

        if complete:
            water = selection["water_source"][0]
            grounds = selection["grounds_source"][0]
            stirrer = selection["stir_tool"][0]
            drinks = selection["drink_vessel"]
            foods = selection["food_vessel"]
            utensils = selection["eating_utensil"]
            counter = "countertop"
            serving = "serving_area"

            ok = True
            # Brew: one source at a time, applied to every drink vessel, then
            # returned to the countertop before the next source is picked.
            for source in (water, grounds):
                ok = run("PICK", object_id=source)
                if not ok:
                    break
                for target in drinks:
                    if not run("POUR", source_id=source, target_id=target):
                        ok = False
                        break
                if not ok or not run("PLACE", object_id=source, region_id=counter):
                    ok = False
                    break
            # Stir every drink vessel with the one retrieved implement.
            if ok and run("PICK", object_id=stirrer):
                for target in drinks:
                    if not run("STIR", tool_id=stirrer, target_id=target):
                        ok = False
                        break
                ok = ok and run("PLACE", object_id=stirrer, region_id=counter)
            # Serve the drink vessels, then each food vessel followed
            # immediately by its own utensil.
            if ok:
                for vessel in drinks:
                    if not (run("PICK", object_id=vessel)
                            and run("PLACE", object_id=vessel, region_id=serving)):
                        ok = False
                        break
            if ok:
                for vessel, utensil in zip(foods, utensils):
                    if not (run("PICK", object_id=vessel)
                            and run("PLACE", object_id=vessel, region_id=serving)
                            and run("PICK", object_id=utensil)
                            and run("PLACE_SERVING_UTENSIL",
                                    object_id=utensil, region_id=vessel)):
                        ok = False
                        break

        predicted_outcome = "FEASIBLE" if complete else "INFEASIBLE"
        if not complete:
            planned.append({
                "operator": "TERMINATE_INFEASIBLE",
                "arguments": ["NO_RETRIEVED_ROLE_FILLER"],
            })
        goal_satisfied = bool(runtime.goal_verifier(runtime.observe_state()))

        write_json(output / "retrieval_trace.json", {
            "schema_version": 1,
            "roles": [
                {"key": r.key, "phrase": r.phrase, "count": r.count, "kind": r.kind}
                for r in KITCHEN_ROLES
            ],
            "selection": selection,
            "object_scores": scores.to_dict(),
            "inspected_regions": inspected_regions,
            "crops_taken_from": "raw_<camera>.png (unannotated)",
            "crop_context_fraction": CROP_CONTEXT_FRACTION,
            "rejection_criterion": (
                "cardinality: a role is unfilled when the observable objects, "
                "after the fixed inspection order is exhausted, cannot supply "
                "enough distinct candidates"
            ),
        })
        write_json(output / "method_manifest.json", {
            "method": "Open-vocabulary retrieval (CLIP), no language model",
            "environment": "kitchen",
            "evaluation_mode": "PHYSICAL_EXECUTION",
            "physical_execution": True,
            "model_requests": 0,
            "camera_count": arguments.camera_count,
            "seed": arguments.seed,
            "gt_visible_to_model": False,
            "hidden_storage_contents_visible_to_model": False,
        })
        payload = {
            "baseline": "retrieval",
            "environment": "kitchen",
            "variant": variant,
            "goal": goal,
            "camera_count": arguments.camera_count,
            "seed": arguments.seed,
            "physical_execution": True,
            "executed_actions": executed,
            "physical_goal_satisfied": goal_satisfied,
            "result": {
                "status": "RETRIEVED" if complete else "NO_RETRIEVED_ROLE_FILLER",
                "actions": planned,
                "selection": selection,
                "action_history": history,
            },
        }
        write_json(output / "episode_result.json", payload)

        write_execution_result(
            output,
            scene="kitchen",
            method="retrieval",
            protocol=arguments.protocol,
            variant=str(variant),
            camera_count=arguments.camera_count,
            seed=arguments.seed,
            success=goal_satisfied,
            executed_actions=executed,
            model_calls=0,
            raw_vlm_requests=0,
            replans=0,
            planning_latency_s=0.0,
            elapsed_seconds=time.monotonic() - started_at,
            terminal_status=(
                GOAL_COMPLETE_STATUS if goal_satisfied
                else "NO_RETRIEVED_ROLE_FILLER" if not complete
                else "EXECUTION_FAILED"
            ),
            # Feasibility comes from the variant label: K1-K6 are the
            # feasible Kitchen variants and K7-K12 the infeasible ones, per the
            # published variant catalogue.
            expected_outcome=_expected_outcome(variant),
            predicted_outcome=predicted_outcome,
        )
        print(f"[kitchen retrieval] {variant}: "
              f"{payload['result']['status']}, {executed} actions, "
              f"goal={goal_satisfied}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
