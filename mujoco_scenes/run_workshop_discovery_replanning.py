"""Run ROBUST-TAMP on one Workshop W1--W10 variant with physical execution.

The Kitchen and Living Room counterparts already existed; Workshop did not,
which is why that column was empty.  The planning loop, planner, and prompt are
shared -- only the runtime, the executor, and the feasibility verdict are
scene-specific.

Two Workshop-specific points. Its executors are synchronous, so
`WorkshopSkillDispatcher` presents them through the stepped protocol the
executive drives.  And inspection must advance the *planning* runtime as well
as the physical scene, because the planning runtime is what renders frames and
owns `visible_object_ids` -- which is precisely the signal discovery-triggered
replanning fires on.  `MirroredWorkshopExecutor` keeps the two in step; without
it the real drawer opens while the runtime still believes the cell is closed,
and no discovery event is ever raised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from baseline_common.artifacts import prepare_run_directory
from mujoco_scenes.baseline_workshop_runtime import (
    MirroredWorkshopExecutor,
    WorkshopPhysicalExecutor,
)
from vlm_tamp_baseline.workshop_runtime import (
    DEFAULT_EXPECTED_ROOT,
    WorkshopPlanningRuntime,
    WorkshopSymbolicExecutor,
)

from .tamp.baseline_observation_bridge import (
    BaselineRuntimeSnapshotObserver,
    observed_skill_precheck,
)
from .tamp.discovery_planner import OpenAIDiscoveryPlanner, OpenAIPlannerConfig
from .tamp.discovery_replanning import DiscoveryReplanningExecutive
from .tamp.events import EventLog
from .tamp.robust_tamp_reporting import write_robust_tamp_artifacts
from .tamp.workshop_skill_dispatcher import WorkshopSkillDispatcher


def run_episode(
    *,
    variant: str,
    output_dir: str | Path,
    goal: str | None = None,
    base_url: str,
    model: str,
    api_key: str = "",
    max_replans: int = 5,
    max_model_calls: int | None = None,
    protocol: str = "native",
    seed: int = 0,
    max_actions: int = 80,
    camera_count: int = 5,
    max_tokens: int = 4096,
    timeout_seconds: float = 600.0,
    decoding: str = "paper",
    expected_root: Path = DEFAULT_EXPECTED_ROOT,
    image_width: int = 960,
    image_height: int = 540,
) -> dict[str, object]:
    if protocol not in {"native", "single_call"}:
        raise ValueError("protocol must be 'native' or 'single_call'")
    if protocol == "single_call":
        if max_model_calls not in {None, 1}:
            raise ValueError("single_call protocol requires max_model_calls=1")
        max_model_calls = 1
    output = prepare_run_directory(output_dir)

    runtime = WorkshopPlanningRuntime(
        variant,
        output,
        expected_root=Path(expected_root).resolve(),
        image_width=image_width,
        image_height=image_height,
        camera_count=camera_count,
        physical_execution=True,
    )
    # The instruction comes from the scene config, never from this runner: one
    # source of truth across methods is what makes the comparison fair.
    task_goal = goal or runtime.goal
    physical = WorkshopPhysicalExecutor(runtime)
    executor = MirroredWorkshopExecutor(
        physical, WorkshopSymbolicExecutor(runtime)
    )
    dispatcher = WorkshopSkillDispatcher(executor)
    observer = BaselineRuntimeSnapshotObserver(runtime)
    planner = OpenAIDiscoveryPlanner(
        OpenAIPlannerConfig(
            base_url=base_url,
            model=model,
            scene="workshop",
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            seed=seed,
            trace_dir=output / "model_calls",
            decoding=decoding,
        )
    )
    events = EventLog(output / "discovery_replanning_events.jsonl")
    executive = DiscoveryReplanningExecutive(
        scene="workshop",
        goal=task_goal,
        observer=observer,
        planner=planner,
        dispatcher=dispatcher,
        goal_verifier=lambda _goal, _state, _history: runtime.goal_verifier(),
        pre_action_check=observed_skill_precheck,
        max_replans=max_replans,
        max_model_calls=max_model_calls,
        max_actions=max_actions,
        event_log=events,
    )
    try:
        started = time.monotonic()
        executive.start()
        while executive.busy:
            executive.update()
        success = executive.mode == "complete"
        result = {
            "scene": "workshop",
            "method": "discovery_replanning",
            "protocol": protocol,
            "variant": runtime.variant,
            "goal": task_goal,
            "success": success,
            "status": executive.mode.upper(),
            "executed_actions": dispatcher.executed_actions,
            "replans": executive.replans,
            "model_calls": executive.model_calls,
            "raw_vlm_requests": planner.call_count,
            "seed": seed,
            "camera_count": camera_count,
            "physical_execution": True,
            "planning_latency_s": round(executive.planning_latency_s, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            # The Kitchen and Living Room runners both record these; this one
            # did not, so every Workshop ROBUST-TAMP episode wrote an empty
            # `history` while its event log held the skills that actually ran
            # -- 109 of 338 episodes across the grid.  Any consumer reading
            # the result file rather than the JSONL undercounted the run.
            "history": list(executive.history),
            "last_event": (
                executive.last_event.as_dict() if executive.last_event else None
            ),
            "terminal_failure": executive.terminal_failure,
            "mirror_divergences": list(executor.mirror_divergences),
            "events_path": str(output / "discovery_replanning_events.jsonl"),
            "model_calls_path": str(output / "model_calls"),
        }
        # `infeasibility_proven` is the authority on Workshop: a rejection is
        # only earned once every storage region has actually been inspected, so
        # exhausting the budget cannot masquerade as one.
        verdict = write_robust_tamp_artifacts(
            output,
            scene="workshop",
            protocol=protocol,
            variant=runtime.variant,
            camera_count=camera_count,
            seed=seed,
            success=success,
            executed_actions=dispatcher.executed_actions,
            model_calls=executive.model_calls,
            raw_vlm_requests=planner.call_count,
            replans=executive.replans,
            planning_latency_s=round(executive.planning_latency_s, 6),
            elapsed_seconds=round(time.monotonic() - started, 6),
            status=executive.mode,
            terminal_failure=executive.terminal_failure,
            expected_outcome=runtime.expected.intended_outcome,
            infeasibility_proven=runtime.infeasibility_proven(),
        )
        result.update(verdict)
        (output / "discovery_replanning_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    finally:
        events.close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-replans", type=int, default=5)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="paper",
        help=(
            "Decoding condition. 'paper' is this method's published "
            "temperature-0 / 4096-token setting; 'model-native' draws from "
            "baseline_common.inference, the same source the other baselines "
            "read, and is what the comparison table runs."
        ),
    )
    # Workshop always executes physically, so there is no --execute flag and no
    # viewer to disable: its runtime renders offscreen.
    arguments = parser.parse_args()
    result = run_episode(
        variant=arguments.variant,
        output_dir=arguments.output_dir,
        goal=arguments.goal,
        base_url=arguments.base_url,
        model=arguments.model,
        api_key=arguments.api_key,
        max_replans=arguments.max_replans,
        max_model_calls=arguments.max_model_calls,
        protocol=arguments.protocol,
        seed=arguments.seed,
        max_actions=arguments.max_actions,
        camera_count=arguments.camera_count,
        max_tokens=arguments.max_tokens,
        timeout_seconds=arguments.timeout_seconds,
        decoding=arguments.decoding,
        expected_root=arguments.expected_root,
        image_width=arguments.image_width,
        image_height=arguments.image_height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
