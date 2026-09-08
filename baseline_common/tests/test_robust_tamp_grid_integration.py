"""ROBUST-TAMP must face the same grid protocol as the other baselines.

It previously ran only from `run_discovery_execution_batch.py`, its own driver,
which is how its decoding condition and camera handling drifted from the
table's.  Wiring it into the shared runner means one place decides
--decoding, --camera-count, --max-model-calls, --max-actions, --seeds,
--resume and --workers for every method.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from baseline_common.run_baseline_execution_batch import (
    ENVIRONMENT_METHODS,
    _command,
    build_parser,
)

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("kitchen", "living_room", "workshop")
RUNNERS = {
    scene: ROOT / f"mujoco_scenes/run_{scene}_discovery_replanning.py"
    for scene in SCENES
}


def _args(scene: str, variant: str, **overrides) -> argparse.Namespace:
    argv = [
        "--environment", scene,
        "--methods", "robust_tamp",
        "--variants", variant,
        "--output-root", "/tmp/robust-tamp-test",
        "--base-url", "http://127.0.0.1:18000/v1",
        "--model", "qwen35-9b",
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return build_parser().parse_args(argv)


def _accepted_flags(path: Path) -> set[str]:
    """Every long option the runner's argparse actually defines."""
    flags: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    flags.add(str(arg.value))
    return flags


@pytest.mark.parametrize("scene", SCENES)
def test_a_robust_tamp_is_offered_for_every_scene(scene):
    assert "robust_tamp" in ENVIRONMENT_METHODS[scene]


@pytest.mark.parametrize("scene", SCENES)
def test_b_every_scene_has_a_runner(scene):
    """Workshop had none, which is why that column was empty."""
    assert RUNNERS[scene].exists(), f"missing runner for {scene}"


@pytest.mark.parametrize(
    "scene,variant", [("kitchen", "K1"), ("living_room", "L1"), ("workshop", "W1")]
)
def test_c_the_grid_passes_only_flags_the_runner_defines(scene, variant):
    """An undefined flag is an argparse error, not a warning.

    The grid appends --headless and --close-on-complete for Kitchen, and these
    runners define neither `--close-on-complete` nor, for Workshop, --headless.
    """
    command = _command("robust_tamp", variant, 3, 0, Path("/tmp/e"), _args(scene, variant))
    passed = {item for item in command if item.startswith("--")}
    unknown = passed - _accepted_flags(RUNNERS[scene])
    assert not unknown, f"{scene}: grid passes flags the runner rejects: {sorted(unknown)}"


@pytest.mark.parametrize(
    "scene,variant", [("kitchen", "K1"), ("living_room", "L1"), ("workshop", "W1")]
)
def test_d_the_replanning_budget_is_five_not_owl_tamps_eight(scene, variant):
    """--max-replans defaults to 8 for OWL-TAMP and bounds a different thing.

    ROBUST-TAMP published 10; the table runs 5, matching the grid's planning
    budget, and 10 is the reported ablation.
    """
    command = _command("robust_tamp", variant, 3, 0, Path("/tmp/e"), _args(scene, variant))
    assert command[command.index("--max-replans") + 1] == "5"
    assert command[command.index("--max-model-calls") + 1] == "5"


@pytest.mark.parametrize(
    "scene,variant", [("kitchen", "K1"), ("living_room", "L1"), ("workshop", "W1")]
)
def test_e_single_call_collapses_both_budgets_to_one(scene, variant):
    command = _command(
        "robust_tamp", variant, 3, 0, Path("/tmp/e"),
        _args(scene, variant, protocol="single_call"),
    )
    assert command[command.index("--max-replans") + 1] == "1"
    assert command[command.index("--max-model-calls") + 1] == "1"


@pytest.mark.parametrize(
    "scene,variant", [("kitchen", "K1"), ("living_room", "L1"), ("workshop", "W1")]
)
def test_f_the_tables_decoding_condition_is_forwarded(scene, variant):
    command = _command(
        "robust_tamp", variant, 3, 0, Path("/tmp/e"),
        _args(scene, variant, decoding="model-native"),
    )
    assert command[command.index("--decoding") + 1] == "model-native"


@pytest.mark.parametrize("scene", SCENES)
def test_g_every_runner_records_the_feasibility_verdict(scene):
    """`success` is false by construction on a GT-infeasible variant.

    Without the verdict in the shared artifact, correctly rejecting an
    impossible task scores the same as blundering through it -- 6 of Kitchen's
    12 variants, 4 of Living Room's 10, 2 of Workshop's 10.
    """
    source = RUNNERS[scene].read_text(encoding="utf-8")
    assert "write_robust_tamp_artifacts" in source, (
        f"{scene}: never writes the shared benchmark_execution_result.json"
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "write_robust_tamp_artifacts"
    ]
    assert calls, f"{scene}: helper imported but never called"
    for call in calls:
        names = {kw.arg for kw in call.keywords}
        assert "expected_outcome" in names, f"{scene}: no expected_outcome"


@pytest.mark.parametrize("scene", SCENES)
def test_h_raw_requests_are_counted_not_inferred(scene):
    """A retried transport fault is real HTTP traffic with no completion.

    `raw_vlm_requests` used to be set to `executive.model_calls`, which was
    true only while every request produced a completion.  Once transport faults
    gained their own retry budget that stopped holding, and this column is what
    the planner-cost comparison rests on -- VLM-TAMP and OWL-TAMP both count
    from their transports.
    """
    source = RUNNERS[scene].read_text(encoding="utf-8")
    assert "raw_vlm_requests=planner.call_count" in source, (
        f"{scene}: raw request count is inferred rather than measured"
    )
    assert '"raw_vlm_requests": executive.model_calls' not in source, (
        f"{scene}: still equates requests with completions"
    )


@pytest.mark.parametrize("scene", SCENES)
def test_i_every_runner_that_owns_a_viewer_is_told_to_suppress_it(scene):
    """A discovery runner with a --headless flag must receive it.

    The shared `common` block adds --headless for Kitchen only, because "the
    Living Room physical runtime is constructed headless" -- true of vlm_tamp's
    and owl_tamp's Living Room runners, but not of
    run_living_room_discovery_replanning.py, which owns the flag and defaults
    to viewer-on.  ROBUST-TAMP Living Room therefore launched
    mujoco.viewer.launch_passive per episode and stepped at render rate:
    1.4-1.6% CPU against 22-25% for the headless Workshop episodes.  This is
    the third time this exact bug has appeared, after retrieval Kitchen.
    """
    command = _command("robust_tamp", "K1" if scene == "kitchen" else ("L1" if scene == "living_room" else "W1"),
                       3, 0, Path("/tmp/e"), _args(scene, "K1" if scene == "kitchen" else ("L1" if scene == "living_room" else "W1")))
    defines_flag = "--headless" in _accepted_flags(RUNNERS[scene])
    passed = "--headless" in command
    assert passed == defines_flag, (
        f"{scene}: runner defines --headless={defines_flag} but grid passes it={passed}; "
        "a runner that owns a viewer and is not told to suppress it throttles "
        "stepping to render rate"
    )
