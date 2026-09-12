"""Run physical baseline episodes under the shared artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading
from typing import Any

from .artifacts import write_json


# The published Table I instructions, imported so the batch path and the
# single-episode path cannot state the task differently, and so every method
# receives identical text.
from mujoco_scenes.benchmark_task_instructions import (  # noqa: E402
    KITCHEN_TASK_INSTRUCTION as KITCHEN_GOAL,
    LIVING_ROOM_TASK_INSTRUCTION as LIVING_ROOM_GOAL,
    WORKSHOP_TASK_INSTRUCTION as WORKSHOP_GOAL,
)

GOALS = {
    "kitchen": KITCHEN_GOAL,
    "living_room": LIVING_ROOM_GOAL,
    "workshop": WORKSHOP_GOAL,
}
VARIANTS = {
    "kitchen": tuple(f"K{index}" for index in range(1, 13)),
    "living_room": tuple(f"L{index}" for index in range(1, 11)),
    "workshop": tuple(f"W{index}" for index in range(1, 11)),
}
# The default pair is the two model-driven baselines, because they are what a
# grid is usually re-run for.  Retrieval (CLIP, no language model) and
# ViLaIn-TAMP exist for all three scenes and are opt-in via --methods.
METHODS = ("vlm_tamp", "owl_tamp")
ENVIRONMENT_METHODS = {
    "kitchen": ("vlm_tamp", "owl_tamp", "retrieval", "vilain_tamp", "robust_tamp"),
    "living_room": ("vlm_tamp", "owl_tamp", "retrieval", "vilain_tamp", "robust_tamp"),
    "workshop": ("vlm_tamp", "owl_tamp", "retrieval", "vilain_tamp", "robust_tamp"),
}


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", choices=tuple(GOALS), default="kitchen"
    )
    parser.add_argument("--methods", type=_csv, default=METHODS)
    parser.add_argument("--variants", type=_csv)
    parser.add_argument("--camera-counts", type=_csv, default=("5",))
    parser.add_argument("--seeds", type=_csv, default=("0",))
    parser.add_argument("--protocol", choices=("native", "single_call", "receding_horizon"), default="native")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument(
        "--max-model-calls", type=int, default=5,
        help=(
            "Replanning budget per episode.  Measured on the completed Living "
            "Room grid, which ran with 10: OWL-TAMP used exactly 1 call in all "
            "60 episodes, and only 6 of 121 VLM-TAMP episodes exceeded 5.  Of "
            "those 6, five hit the ceiling and mostly failed anyway, and they "
            "were the slowest episodes in the grid (999 s mean against 166 s "
            "for single-call episodes).  Capping at 5 trims the expensive tail "
            "while changing at most one episode's outcome."
        ),
    )
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="model-native",
        help=(
            "Applied to every model-driven method so the comparison holds "
            "decoding fixed; recorded in protocol_manifest.json."
        ),
    )
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument(
        "--replan-on-no-plan", action="store_true",
        help="Forwarded to OWL-TAMP receding-horizon episodes; see that runner.",
    )
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument(
        "--max-sketch-actions",
        type=int,
        default=24,
        help=(
            "OWL-TAMP only: bound on its per-action constraint requests, so a "
            "sketch that degenerates into repetition cannot bill one request "
            "per repeat."
        ),
    )
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=3600.0,
        help=(
            "Seconds before an episode is abandoned and the batch moves on. "
            "Without it a single hung episode stalls an unattended grid "
            "indefinitely; the longest episode measured so far is well under "
            "an hour."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Episodes to run concurrently.  Each episode is a separate "
            "process: MuJoCo stepping is single-threaded, and the model calls "
            "are remote, so an episode spends much of its life idle waiting on "
            "the inference server.  Concurrency overlaps that waiting and the "
            "server batches the requests.  Keep it at or below the core count "
            "and well below what the served model can hold at once."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _set_aside(output: Path) -> Path:
    """Move a partially written episode directory out of the way."""
    for index in range(1, 1000):
        candidate = output.with_name(f"{output.name}.interrupted_{index:03d}")
        if not candidate.exists():
            output.rename(candidate)
            return candidate
    raise RuntimeError(f"Too many interrupted attempts for {output}")


def _command(
    method: str,
    variant: str,
    camera_count: int,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    common = [
        sys.executable, "-m",
        "--output-dir", str(output_dir),
        "--goal", args.goal or GOALS[args.environment],
        "--base-url", args.base_url,
        "--model", args.model,
        "--max-tokens", str(args.max_tokens),
        "--seed", str(seed),
        "--camera-count", str(camera_count),
        "--protocol", args.protocol,
    ]
    if args.environment == "kitchen":
        # Only the Kitchen runners own an interactive viewer to suppress; the
        # Living Room physical runtime is constructed headless.
        common.extend(("--headless", "--close-on-complete"))
    module = f"{method}_baseline.run_{args.environment}"
    if method == "vlm_tamp":
        if args.protocol == "receding_horizon":
            raise ValueError("VLM-TAMP has no receding_horizon protocol")
        max_calls = 1 if args.protocol == "single_call" else args.max_model_calls
        # Kitchen selects its physical variant with a dedicated flag; the
        # Living Room runner takes --variant plus --physical-execution.
        variant_flags = (
            ["--physical-variant", variant]
            if args.environment == "kitchen"
            # Workshop's runner grew from a planning-only script, so its
            # execution switch is --execute rather than --physical-execution.
            else ["--variant", variant, "--execute"]
            if args.environment == "workshop"
            else ["--variant", variant, "--physical-execution"]
        )
        return [
            *common[:2], module,
            *variant_flags,
            *common[2:],
            "--max-model-calls", str(max_calls),
            "--max-total-actions", str(args.max_actions),
            "--decoding", args.decoding,
        ]
    if method == "owl_tamp":
        return [
            *common[:2], module,
            "--variant", variant,
            "--execute" if args.environment == "workshop" else "--physical-execution",
            *common[2:],
            "--max-replans", str(args.max_replans),
            *(["--replan-on-no-plan"] if args.replan_on_no_plan else []),
            "--max-total-actions", str(args.max_actions),
            "--max-sketch-actions", str(args.max_sketch_actions),
            "--decoding", args.decoding,
        ]
    if method == "vilain_tamp":
        # This baseline owns its own CLI vocabulary: --domain rather than a
        # module per scene, --output-directory, and a model condition in place
        # of the native/single_call protocol.  It also has no --camera-count
        # --base-url/--model/--max-tokens (the endpoint and checkpoint come
        # from its own configuration), so the shared `common` block does not
        # apply and the command is built explicitly -- but --camera-count is
        # passed through, because exposure parity is a property of the grid
        # cell, not of the method.  Its runner defaults to 3; omitting the flag
        # would silently pin every camera-count cell to that default while
        # labelling the artifacts as though they differed.
        return [
            sys.executable, "-m", "mujoco_scenes.run_vilain_tamp_baseline",
            "--domain", args.environment,
            "--variant", variant,
            "--model-condition", "vilain_tamp_qwen",
            # Stated explicitly rather than relying on the config, so a
            # grid cannot silently run without exposure parity.
            "--observation-mode", "fixed_full_inspection",
            "--camera-count", str(camera_count),
            # Same reason as --camera-count: the decoding condition belongs to
            # the table, not the method.  This baseline defaults to its own
            # greedy condition, which Qwen's card advises against and which
            # would make its column non-comparable with the other three.
            "--decoding", args.decoding,
            "--live",
            "--execute",
            "--output-directory", str(output_dir),
            "--seed", str(seed),
        ]
    if method == "robust_tamp":
        # ROBUST-TAMP lives under mujoco_scenes rather than a
        # `<method>_baseline` package, so the shared `module` name does not
        # apply.  It is wired into *this* runner rather than kept behind
        # `run_discovery_execution_batch.py` so it faces the same protocol as
        # the other baselines: the same --decoding, --camera-count,
        # --max-model-calls, --max-actions, --seeds, --resume and --workers.
        # Running it from its own driver was how its decoding condition and
        # camera parity drifted from the table's in the first place.
        max_calls = 1 if args.protocol == "single_call" else args.max_model_calls
        # These runners predate `--close-on-complete`, and Workshop's renders
        # offscreen so it has no viewer to suppress either.  Passing a flag a
        # runner does not define is an argparse error, not a warning, so the
        # tail is filtered rather than assumed.
        unsupported = {"--close-on-complete"}
        if args.environment == "workshop":
            unsupported.add("--headless")
        tail = [item for item in common[2:] if item not in unsupported]
        # The shared `common` block adds --headless for Kitchen only, on the
        # grounds that "the Living Room physical runtime is constructed
        # headless".  That is true of vlm_tamp's and owl_tamp's Living Room
        # runners but NOT of run_living_room_discovery_replanning.py, which
        # owns a --headless flag defaulting to viewer-on.  Without this,
        # ROBUST-TAMP Living Room launched mujoco.viewer.launch_passive per
        # episode and stepped at render rate: measured 1.4-1.6% CPU against
        # 22-25% for the headless Workshop episodes, the same throttle that
        # made retrieval Kitchen look like a 240 h cell.
        if args.environment == "living_room" and "--headless" not in tail:
            tail.append("--headless")
        return [
            sys.executable, "-m",
            f"mujoco_scenes.run_{args.environment}_discovery_replanning",
            "--variant", variant,
            *tail,
            "--max-model-calls", str(max_calls),
            # The replanning budget follows the grid's planning budget, not
            # OWL-TAMP's --max-replans (which defaults to 8 and bounds a
            # different thing -- its sketch retries).  ROBUST-TAMP published
            # 10; 5 is what the table runs and 10 is the reported ablation.
            # Under single_call this correctly collapses to 1.
            "--max-replans", str(max_calls),
            "--max-actions", str(args.max_actions),
            "--decoding", args.decoding,
        ]
    if method == "retrieval":
        # Kitchen's retrieval runner always executes and selects its variant
        # with --physical-variant, matching the other Kitchen runners;
        # Workshop opts in with --execute; Living Room with
        # --physical-execution.
        if args.environment == "kitchen":
            return [
                *common[:2], module,
                "--physical-variant", variant,
                *common[2:],
            ]
        if args.environment == "workshop":
            return [
                *common[:2], module,
                "--variant", variant, "--execute",
                *common[2:],
            ]
        # Retrieval calls no model, so --base-url/--model/--max-tokens are
        # accepted and ignored; --protocol and the render size still apply.
        return [
            *common[:2], module,
            "--variant", variant,
            "--physical-execution",
            *common[2:],
        ]
    raise ValueError(f"Unsupported method {method!r}")


def _validate(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    methods = tuple(args.methods)
    allowed = set(ENVIRONMENT_METHODS.get(args.environment, METHODS))
    if not set(methods) <= allowed:
        raise ValueError(f"--methods supports only {', '.join(sorted(allowed))}")
    if args.protocol == "receding_horizon" and "vlm_tamp" in methods:
        raise ValueError("receding_horizon is currently available only for owl_tamp")
    allowed_variants = VARIANTS[args.environment]
    variants = tuple(args.variants) if args.variants else allowed_variants
    if not variants or any(value not in allowed_variants for value in variants):
        raise ValueError(
            f"--variants must be {args.environment} labels: "
            f"{allowed_variants[0]}-{allowed_variants[-1]}"
        )
    cameras = tuple(int(value) for value in args.camera_counts)
    seeds = tuple(int(value) for value in args.seeds)
    if not cameras or not set(cameras) <= {1, 3, 5}:
        raise ValueError("--camera-counts supports 1,3,5")
    if not seeds or any(value < 0 for value in seeds):
        raise ValueError("--seeds must be non-negative")
    return methods, variants, cameras, seeds


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        methods, variants, cameras, seeds = _validate(args)
    except ValueError as error:
        parser.error(str(error))
    root = args.output_root.resolve()
    # A run root is normally filled by several legs running at once, one per
    # method.  Both root-level files used to be written to fixed names, so the
    # last leg to write simply replaced what the others had recorded: the
    # 480-episode four-method `fixed_20260910` root ends up claiming
    # `methods: ["vlm_tamp"], variants: ["K11"], seeds: [4]` -- a single-episode
    # repair leg -- and `budgetfix_20260912` records 46 owl_tamp runs for a
    # root holding three methods.  No analysis read them, and the per-episode
    # `method_manifest.json` always held the truth, but the files that document
    # a run root were false, which is the kind of thing a reader trusts.
    # Naming them per leg removes the collision instead of serialising writers.
    leg = "-".join(sorted(methods))
    summary_path = root / f"batch_summary.{leg}.json"
    legacy_summary = root / "batch_summary.json"
    rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    source = summary_path if summary_path.is_file() else legacy_summary
    if args.resume and source.is_file():
        for row in json.loads(source.read_text(encoding="utf-8")).get("runs", ()):
            if str(row["method"]) not in methods:
                continue
            rows[(str(row["method"]), str(row["variant"]), int(row["camera_count"]), int(row["seed"]))] = row

    write_json(root / f"protocol_manifest.{leg}.json", {
        "schema_version": 1,
        "environment": args.environment,
        "methods": list(methods),
        "protocol": args.protocol,
        "model": args.model,
        "decoding": args.decoding,
        "max_model_calls": args.max_model_calls,
        # Every knob that bounds a method's search is part of the reported
        # condition, not a convenience default.  max_sketch_actions in
        # particular caps OWL-TAMP's constraint requests, and an episode that
        # hits it stops generating constraints part-way through its sketch --
        # so a grid run under a different value is a different condition and
        # must not be pooled with this one.
        "max_sketch_actions": args.max_sketch_actions,
        "max_replans": args.max_replans,
        "max_actions": args.max_actions,
        "max_tokens": args.max_tokens,
        "episode_timeout_s": args.episode_timeout,
        "variants": list(variants),
        "camera_counts": list(cameras),
        "seeds": list(seeds),
        "goal": args.goal,
        "physical_execution": True,
        "shared_result": "benchmark_execution_result.json",
    })
    # Episodes are independent processes writing to their own directories, so
    # they can overlap.  The plan is built first, and only episodes that
    # actually need running are dispatched; --resume bookkeeping and the
    # not-empty guard stay sequential so they behave exactly as before.
    pending: list[tuple] = []
    for method in methods:
        for variant in variants:
            for camera_count in cameras:
                for seed in seeds:
                    key = (method, variant, camera_count, seed)
                    output = root / method / variant / f"images_{camera_count}" / f"seed_{seed:03d}"
                    result_path = output / "benchmark_execution_result.json"
                    if args.resume and result_path.is_file():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        rows[key] = _row(args.environment, method, variant, camera_count, seed, output, 0, result)
                        continue
                    if output.exists() and any(output.iterdir()):
                        if not args.resume:
                            parser.error(
                                f"incomplete output directory is not empty: {output}"
                            )
                        # An episode interrupted mid-write leaves a directory
                        # with no result file.  Under --resume the whole point
                        # is to carry on, so set the partial artifacts aside
                        # and re-run the episode rather than aborting a grid of
                        # hundreds because one was killed.  They are moved, not
                        # deleted: a partial episode is still evidence about
                        # why the run stopped.
                        aside = _set_aside(output)
                        print(
                            f"[batch] re-running interrupted {method} "
                            f"{args.environment} {variant} images={camera_count} "
                            f"seed={seed}; partial artifacts moved to {aside.name}",
                            flush=True,
                        )
                    pending.append((key, method, variant, camera_count, seed, output, result_path))

    summary_lock = threading.Lock()
    failures: list[int] = []

    def run_episode(entry: tuple) -> None:
        key, method, variant, camera_count, seed, output, result_path = entry
        print(f"[baseline-execution] {method} {variant} images={camera_count} seed={seed}", flush=True)
        try:
            return_code = subprocess.run(
                _command(method, variant, camera_count, seed, output, args),
                check=False,
                timeout=args.episode_timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            # Abandon the episode rather than stall the grid.  The
            # partial directory is kept: a hang is worth diagnosing.
            return_code = 124
            print(
                f"[batch] TIMEOUT after {args.episode_timeout:.0f}s: "
                f"{method} {variant} images={camera_count} seed={seed}",
                flush=True,
            )
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file() else None
        )
        row = _row(args.environment, method, variant, camera_count, seed, output, return_code, result)
        # The summary is rewritten after every episode so an interrupted grid
        # still describes what completed.  Serialise that, since episodes now
        # finish concurrently.
        with summary_lock:
            rows[key] = row
            write_json(summary_path, {"schema_version": 1, "runs": list(rows.values())})
            if result is None and return_code:
                failures.append(return_code)

    workers = max(1, int(args.workers))
    if workers == 1:
        for entry in pending:
            run_episode(entry)
            if failures and not args.continue_on_error:
                raise SystemExit(failures[0])
    else:
        # Concurrency does not change any episode's condition: each is the same
        # command in its own process and its own output directory.  Under
        # --continue-on-error the grid runs to completion either way; without
        # it, already-dispatched episodes are allowed to finish before the
        # non-zero exit, so no episode is left half-written.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_episode, pending))
        if failures and not args.continue_on_error:
            raise SystemExit(failures[0])
    write_json(summary_path, {"schema_version": 1, "runs": list(rows.values())})


def _row(
    environment: str,
    method: str,
    variant: str,
    camera_count: int,
    seed: int,
    output: Path,
    return_code: int,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "environment": environment,
        "variant": variant,
        "camera_count": camera_count,
        "seed": seed,
        "return_code": return_code,
        "result_present": result is not None,
        "task_success": result.get("success") if result else None,
        "output_dir": str(output),
    }


if __name__ == "__main__":
    main()
