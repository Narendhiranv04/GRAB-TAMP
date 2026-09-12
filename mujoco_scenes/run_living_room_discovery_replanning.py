"""Run image-conditioned discovery-based replanning in the MuJoCo living room."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .living_room_discovery_runtime import LivingRoomDiscoveryRuntime
from .tamp.baseline_observation_bridge import (
    BaselineRuntimeSnapshotObserver,
    observed_skill_precheck,
)
from .tamp.discovery_planner import OpenAIDiscoveryPlanner, OpenAIPlannerConfig
from .tamp.robust_tamp_reporting import (
    expected_outcome_for,
    write_robust_tamp_artifacts,
)
from .tamp.discovery_replanning import DiscoveryReplanningExecutive
from .tamp.events import EventLog


ACTION_CATALOG = Path(__file__).parent / "configs" / "discovery_action_catalog.json"


def _check_model_server(base_url: str, model: str, api_key: str = "") -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = Request(base_url.rstrip("/") + "/models", headers=headers, method="GET")
    try:
        with urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Model server preflight failed at {request.full_url}: {error}") from error
    models = {
        str(row.get("id"))
        for row in payload.get("data", ())
        if isinstance(row, dict) and row.get("id")
    } if isinstance(payload, dict) else set()
    if models and model not in models:
        raise RuntimeError(f"Model {model!r} is unavailable; server exposes {sorted(models)}")


def _arguments(arguments: object, aliases: dict[str, str]) -> str:
    if not isinstance(arguments, dict):
        return "{}"
    display = {
        key: f"{value} ({aliases[value].replace('_', ' ')})"
        if isinstance(value, str) and value in aliases
        else value
        for key, value in arguments.items()
    }
    return json.dumps(display, sort_keys=True)


def _print_event(record: object, aliases: dict[str, str]) -> None:
    if not isinstance(record, dict):
        return
    event = record.get("event")
    if event == "plan_accepted":
        print(f"[living-discovery] {'REPLAN' if record.get('is_replan') else 'PLAN'}:", flush=True)
        for index, action in enumerate(record.get("actions", ()), start=1):
            print(f"  {index}. {action.get('name')} {_arguments(action.get('arguments', {}), aliases)}", flush=True)
    elif event == "discovery_skill_started":
        print(f"[living-discovery] START {record.get('action')} {_arguments(record.get('arguments', {}), aliases)}", flush=True)
    elif event == "discovery_skill_finished":
        action = record.get("action", {})
        name = action.get("name") if isinstance(action, dict) else action
        outcome = "OK" if record.get("success") else "FAILED"
        print(f"[living-discovery] {outcome} {name}: {record.get('message', '')}", flush=True)
    elif event in {"failure_replan_requested", "goal_replan_requested", "discovery_episode_complete", "discovery_episode_failed"}:
        print(f"[living-discovery] {event}: {record.get('message', '')}", flush=True)


def run_episode(
    *,
    variant: str,
    output_dir: str | Path,
    goal: str,
    base_url: str,
    model: str,
    api_key: str = "",
    max_replans: int = 5,
    max_model_calls: int | None = None,
    protocol: str = "native",
    seed: int = 0,
    max_actions: int = 40,
    camera_count: int = 5,
    max_tokens: int = 8192,
    timeout_seconds: float = 600.0,
    decoding: str = "paper",
    show_viewer: bool = True,
    viewer_camera: str = "free",
    enable_thinking: bool | None = None,
    preflight_model_server: bool = True,
) -> dict[str, object]:
    if protocol not in {"native", "single_call"}:
        raise ValueError("protocol must be 'native' or 'single_call'")
    if protocol == "single_call":
        if max_model_calls not in {None, 1}:
            raise ValueError("single_call protocol requires max_model_calls=1")
        max_model_calls = 1
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if preflight_model_server:
        _check_model_server(base_url, model, api_key)
    runtime = LivingRoomDiscoveryRuntime(
        variant,
        output,
        camera_count=camera_count,
        show_viewer=show_viewer,
        viewer_camera=viewer_camera,
    )
    events = EventLog(
        output / "discovery_replanning_events.jsonl",
        sink=lambda record: _print_event(record, runtime.object_annotation_aliases),
    )
    planner = OpenAIDiscoveryPlanner(
        OpenAIPlannerConfig(
            base_url=base_url,
            model=model,
            scene="living_room",
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            enable_thinking=enable_thinking,
            seed=seed,
            trace_dir=output / "model_calls",
            # The table's condition, not this method's own: see
            # OpenAIPlannerConfig.  `paper` stays the default so an existing
            # caller is unchanged.
            decoding=decoding,
        ),
        action_catalog=ACTION_CATALOG,
    )
    executive = DiscoveryReplanningExecutive(
        scene="living_room",
        goal=goal,
        observer=BaselineRuntimeSnapshotObserver(runtime),
        planner=planner,
        dispatcher=runtime.dispatcher,
        goal_verifier=lambda _goal, _state, _history: runtime.goal_verifier(),
        effect_sink=runtime.accept_effects,
        pre_action_check=observed_skill_precheck,
        max_replans=max_replans,
        max_model_calls=max_model_calls,
        max_actions=max_actions,
        event_log=events,
    )
    try:
        started = time.monotonic()
        runtime.open()
        executive.start()
        while executive.busy:
            runtime.sync(executive.status)
            executive.update()
        runtime.sync(executive.status)
        result = {
            "scene": "living_room",
            "method": "robust_tamp",
            "protocol": protocol,
            "variant": variant,
            "goal": goal,
            "success": executive.mode == "complete",
            "status": executive.mode.upper(),
            "executed_actions": executive.executed_actions,
            "replans": executive.replans,
            "model_calls": executive.model_calls,
            "raw_vlm_requests": planner.call_count,
            "seed": seed,
            "camera_count": camera_count,
            "planning_latency_s": round(executive.planning_latency_s, 6),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "history": list(executive.history),
            "last_event": executive.last_event.as_dict() if executive.last_event else None,
            "events_path": str(output / "discovery_replanning_events.jsonl"),
            "model_calls_path": str(output / "model_calls"),
            "planner_input_boundary": {
                "five_rgb_views": camera_count == 5,
                "hidden_inventory_exposed": False,
                "ground_truth_plan_exposed": False,
                "private_goal_evaluator_exposed": False,
                "replans_from_current_observation": True,
                "model_call_condition": (
                    "UNBOUNDED_BY_CALL_COUNT"
                    if max_model_calls is None
                    else f"MAX_{max_model_calls}_MODEL_CALLS"
                ),
            },
        }
        (output / "discovery_replanning_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The shared artifact the metrics actually read.  Without it this
        # method produced no scored episode at all, and without the verdict
        # inside it every GT-infeasible variant would score as a plain
        # failure -- the defect that cost VLM-TAMP and OWL-TAMP their
        # infeasible-rejection numbers until it was fixed.
        verdict = write_robust_tamp_artifacts(
            output,
            scene="living_room",
            protocol=protocol,
            variant=variant,
            camera_count=camera_count,
            seed=seed,
            success=bool(executive.mode == "complete"),
            executed_actions=executive.executed_actions,
            model_calls=executive.model_calls,
            raw_vlm_requests=planner.call_count,
            replans=executive.replans,
            planning_latency_s=round(executive.planning_latency_s, 6),
            elapsed_seconds=round(time.monotonic() - started, 6),
            status=executive.mode,
            terminal_failure=executive.terminal_failure,
            expected_outcome=expected_outcome_for("living_room", variant),
        )
        result.update(verdict)
        return result
    finally:
        events.close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(f"L{index}" for index in range(1, 11)), default="L1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-replans", type=int, default=5)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-model-calls",
        type=int,
        help="Maximum VLM requests for the episode; use 1 for the single-call condition.",
    )
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--timeout-seconds", type=float, default=1800.0,
        help=(
            "Model-call deadline.  Raised from 600s on 2026-09-10: under the "
            "re-run's concurrency the served median climbed from 174s to 265s "
            "with a p90 of exactly 600s, so 6.2%% of ROBUST-TAMP calls were "
            "being discarded at the deadline and charged to the replan budget. "
            "A deadline that binds makes the result depend on how many "
            "episodes happened to be scheduled alongside it rather than on "
            "the method."
        ),
    )
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="paper",
        help=(
            "Decoding condition. 'paper' is this method's published "
            "temperature-0 / 4096-token setting and stays the default; "
            "'model-native' draws from baseline_common.inference, the same "
            "source VLM-TAMP, OWL-TAMP and ViLaIn-TAMP read, and is what the "
            "comparison table runs."
        ),
    )
    parser.add_argument("--camera-count", choices=(1, 3, 5), type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera", default="free")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", action="store_true")
    thinking.add_argument("--no-thinking", action="store_true")
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
        show_viewer=not arguments.headless,
        viewer_camera=arguments.camera,
        enable_thinking=True if arguments.thinking else False if arguments.no_thinking else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
