"""Command-line entry point for the isolated ViLaIn-TAMP baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from baseline_common.physical_benchmark import (
    GOAL_COMPLETE_STATUS,
    write_execution_result,
)
from .baselines.vilain_tamp.config import (
    BaselineConfig,
    Domain,
    ExternalToolPaths,
    ModelCondition,
    ObservationMode,
)
from .baselines.vilain_tamp.runner import (
    BaselineRunner,
    RunOptions,
    RunnerComponents,
)


PACKAGE_ROOT = Path(__file__).resolve().parent / "baselines" / "vilain_tamp"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    ModelCondition.PAPER_FAITHFUL: PACKAGE_ROOT / "configs" / "paper_faithful.yaml",
    ModelCondition.MODEL_MATCHED: PACKAGE_ROOT / "configs" / "model_matched.yaml",
    ModelCondition.QWEN_ONLY: PACKAGE_ROOT / "configs" / "qwen_only.yaml",
}
ComponentFactory = Callable[[BaselineConfig, RunOptions], RunnerComponents]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated ViLaIn-TAMP baseline. Planning-only is the "
            "default; physical execution requires --execute."
        )
    )
    parser.add_argument(
        "--domain",
        choices=tuple(item.value for item in Domain),
        required=True,
        help="Benchmark domain.",
    )
    parser.add_argument("--variant", required=True, help="Internal or paper variant ID.")
    parser.add_argument(
        "--observation-mode",
        choices=tuple(item.value for item in ObservationMode),
        default=None,
        help="Override the configured observation condition.",
    )
    parser.add_argument(
        "--model-condition",
        choices=tuple(item.value for item in ModelCondition),
        default=ModelCondition.QWEN_ONLY.value,
        help="Select the paper-faithful or optional model-matched condition.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Compose the baseline-owned live planning runtime.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit baseline YAML; otherwise selected by model condition.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Exact run directory; defaults below the configured output root.",
    )
    parser.add_argument(
        "--offline-model-fixtures",
        type=Path,
        help="Absolute directory containing recorded model responses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved run without model, planner, or simulation calls.",
    )
    parser.add_argument(
        "--cp-limit",
        type=int,
        help="Maximum corrective problem revisions (0-3).",
    )
    parser.add_argument(
        "--fast-downward",
        type=Path,
        help="Absolute Fast Downward executable path.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        help="Absolute VAL executable path.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--planning-only",
        action="store_true",
        help="Stop after a validated, refined execution projection (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly permit scored physical execution and terminal evaluation.",
    )
    parser.add_argument(
        "--camera-count", type=int, choices=(1, 3, 5), default=3,
        help=(
            "Canonical RGB-D views exposed to the model.  Five is this "
            "baseline's documented boundary; the grid runs three to hold "
            "exposure equal with the other methods, and because the "
            "five-view Kitchen prompt under full inspection exceeds the "
            "served context window."
        ),
    )
    parser.add_argument(
        "--decoding", choices=("paper", "model-native"), default="paper",
        help=(
            "Decoding condition.  'paper' is this baseline's own greedy "
            "setting (temperature 0, thinking off) and is the default; "
            "'model-native' is the shared thinking-mode sampling the other "
            "baselines use, and is required for a single reported table."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded random seed.")
    return parser


def resolve_run(
    args: argparse.Namespace,
) -> tuple[BaselineConfig, Path, RunOptions]:
    model_condition = ModelCondition(args.model_condition)
    config_path = (args.config or DEFAULT_CONFIGS[model_condition]).resolve()
    config = BaselineConfig.from_yaml(config_path)
    if config.model_condition is not model_condition:
        raise ValueError(
            "explicit config model condition differs from --model-condition"
        )
    domain = Domain(args.domain)
    observation_mode = (
        ObservationMode(args.observation_mode)
        if args.observation_mode
        else config.observation_mode
    )
    cp_limit = config.max_cp_corrections if args.cp_limit is None else args.cp_limit
    tools = ExternalToolPaths(
        fast_downward=(
            args.fast_downward.resolve()
            if args.fast_downward is not None
            else config.external_tools.fast_downward
        ),
        val=(
            args.val.resolve() if args.val is not None else config.external_tools.val
        ),
        fast_downward_version=config.external_tools.fast_downward_version,
        val_version=config.external_tools.val_version,
    )
    output_root = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else (
            REPOSITORY_ROOT
            / config.output_root
            / domain.value
            / args.variant
            / observation_mode.value
            / model_condition.value
        ).resolve()
    )
    fixture_root = (
        args.offline_model_fixtures.resolve()
        if args.offline_model_fixtures is not None
        else None
    )
    if fixture_root is not None and not fixture_root.is_dir():
        raise ValueError(
            f"offline model fixture directory is missing: {fixture_root}"
        )
    config = replace(
        config,
        domain=domain,
        observation_mode=observation_mode,
        model_condition=model_condition,
        max_cp_corrections=cp_limit,
        output_root=output_root,
        external_tools=tools,
    )
    options = RunOptions(
        domain=domain,
        variant=args.variant,
        observation_mode=observation_mode,
        model_condition=model_condition,
        output_directory=output_root,
        cp_limit=cp_limit,
        execute=bool(args.execute),
        offline_fixture_root=fixture_root,
        fast_downward_path=tools.fast_downward,
        val_path=tools.val,
        random_seed=args.seed,
        camera_count=int(args.camera_count),
        decoding=str(args.decoding),
    )
    return config, config_path, options



def _write_shared_execution_result(result: Any, options: Any) -> None:
    """Emit the cross-method artifact the batch summarizer reads.

    This baseline keeps its own rich result contract, which is what its own
    audits use.  `summarize_execution_batch` and `make_paper_tables` read only
    `benchmark_execution_result.json`, so without this the baseline cannot
    appear in a table next to the methods it is meant to be compared with.
    Nothing here reinterprets the run: every field is copied from the
    baseline's own metrics and its hidden-benchmark evaluation.
    """
    metrics = dict(result.metrics)
    run_directory = Path(options.output_directory)

    # Feasibility verdicts live in the benchmark evaluation artifact, not in
    # the run result, so read them back rather than inferring them.
    benchmark: dict[str, Any] = {}
    benchmark_path = result.artifact_paths.get("benchmark_goal_evaluation")
    if benchmark_path:
        try:
            benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            benchmark = {}

    execution_metrics = dict(metrics.get("execution_metrics") or {})
    executed_actions = int(
        execution_metrics.get("executed_actions")
        or execution_metrics.get("action_count")
        or 0
    )
    # `success` in this artifact is physical goal satisfaction only.  An
    # infeasible variant cannot produce it, and a correct rejection is carried
    # by expected/predicted_outcome instead -- the same rule the other
    # baselines and the ground-truth runners follow.
    success = bool(benchmark.get("actual_task_success", False))
    ground_truth_feasible = benchmark.get("ground_truth_feasibility")
    predicted_infeasible = bool(benchmark.get("predicted_infeasible", False))

    write_execution_result(
        run_directory,
        scene=options.domain.value,
        method="vilain_tamp",
        # This baseline's condition is its model condition, not the
        # native/single_call protocol the other runners take.
        protocol=str(options.model_condition.value),
        variant=str(options.variant),
        # Five canonical RGB-D views, fixed by the baseline's own observation
        # boundary rather than selectable per run.
        camera_count=int(options.camera_count),
        seed=int(options.random_seed),
        success=success,
        executed_actions=executed_actions,
        model_calls=int(metrics.get("model_call_count") or 0),
        raw_vlm_requests=int(metrics.get("model_call_count") or 0),
        replans=int(metrics.get("cp_calls") or 0),
        planning_latency_s=float(metrics.get("symbolic_planning_seconds") or 0.0),
        elapsed_seconds=float(metrics.get("end_to_end_seconds") or 0.0),
        terminal_status=(
            GOAL_COMPLETE_STATUS if success else str(result.run_status)
        ),
        terminal_failure=(
            None
            if success
            else {
                "run_status": result.run_status,
                "planning_status": result.planning_status,
                "refinement_status": result.refinement_status,
                "execution_status": result.execution_status,
                "benchmark_status": result.benchmark_status,
            }
        ),
        expected_outcome=(
            None
            if ground_truth_feasible is None
            else ("FEASIBLE" if ground_truth_feasible else "INFEASIBLE")
        ),
        predicted_outcome=("INFEASIBLE" if predicted_infeasible else "FEASIBLE"),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    component_factory: ComponentFactory | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, config_path, options = resolve_run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(options.to_dict(), indent=2, sort_keys=True))
        return 0
    if component_factory is None and args.live:
        from .baselines.vilain_tamp.runtime import build_live_components

        component_factory = build_live_components
    if component_factory is None:
        parser.error(
            "runtime adapters are required; pass --live or invoke main with a "
            "baseline-owned component factory"
        )
    assert component_factory is not None
    components = component_factory(config, options)
    runner = BaselineRunner(
        config=config,
        config_path=config_path,
        repository_root=REPOSITORY_ROOT,
    )
    result = runner.run(options, components)
    if args.execute:
        _write_shared_execution_result(result, options)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.run_status in {"SUCCESS", "PLANNING_COMPLETE", "INFEASIBLE_CORRECT"} else 1


if __name__ == "__main__":
    sys.exit(main())
