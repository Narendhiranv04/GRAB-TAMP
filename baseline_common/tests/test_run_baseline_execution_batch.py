from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from baseline_common.run_baseline_execution_batch import _command, _validate, main


def _args(**overrides):
    values = {
        "environment": "kitchen",
        "methods": ("vlm_tamp", "owl_tamp"),
        "protocol": "native",
        "variants": ("K1",),
        "camera_counts": ("5",),
        "seeds": ("0",),
        "goal": "test goal",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "model",
        "max_tokens": 24576,
        "max_model_calls": 10,
        "max_replans": 8,
        "replan_on_no_plan": False,
        "max_actions": 80,
        "max_sketch_actions": 24,
        "decoding": "model-native",
    }
    values.update(overrides)
    return Namespace(**values)


def test_vlm_tamp_command_uses_physical_variant():
    command = _command("vlm_tamp", "K1", 5, 0, Path("runs/out"), _args())
    assert command[2] == "vlm_tamp_baseline.run_kitchen"
    assert command[command.index("--physical-variant") + 1] == "K1"
    assert command[command.index("--max-model-calls") + 1] == "10"


def test_owl_tamp_command_enables_physical_execution():
    command = _command("owl_tamp", "K1", 5, 0, Path("runs/out"), _args())
    assert command[2] == "owl_tamp_baseline.run_kitchen"
    assert "--physical-execution" in command
    # A degenerate sketch bills one constraint request per repeated action, so
    # every OWL-TAMP runner must receive the bound.
    assert command[command.index("--max-sketch-actions") + 1] == "24"


def test_retrieval_executes_in_every_scene_and_needs_no_model_flags():
    """Retrieval exists for all three scenes, each with its own execution flag.

    It was Living-Room-only until the Kitchen and Workshop runners landed, and
    the execution switch is not spelled the same way in all three: Kitchen's
    runner always executes and selects with --physical-variant, Workshop opts
    in with --execute, Living Room with --physical-execution.  A regression
    here would silently run a scene planning-only.
    """
    expected_flags = {
        "living_room": ("L1", ["--variant", "--physical-execution"]),
        "kitchen": ("K1", ["--physical-variant"]),
        "workshop": ("W1", ["--variant", "--execute"]),
    }
    for environment, (variant, flags) in expected_flags.items():
        command = _command(
            "retrieval", variant, 3, 0, Path("runs/out"),
            _args(
                environment=environment,
                methods=("retrieval",),
                variants=(variant,),
            ),
        )
        assert command[2] == f"retrieval_baseline.run_{environment}"
        for flag in flags:
            assert flag in command, (environment, flag)
        # No language model is involved, so no inference budget may be billed
        # to it -- these would be silently accepted and ignored otherwise.
        for flag in ("--max-model-calls", "--decoding", "--max-replans"):
            assert flag not in command, (environment, flag)
        _validate(
            _args(
                environment=environment,
                methods=("retrieval",),
                variants=(variant,),
            )
        )


def test_receding_horizon_is_not_available_for_vlm_tamp():
    with pytest.raises(ValueError, match="owl_tamp"):
        _validate(_args(protocol="receding_horizon"))


def test_main_dispatches_every_requested_trial(tmp_path, monkeypatch):
    commands = []

    class Completed:
        returncode = 0

    timeouts = []

    def fake_run(command, check=False, timeout=None):
        commands.append(command)
        timeouts.append(timeout)
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_baseline_execution_batch",
            "--output-root", str(tmp_path / "batch"),
            "--base-url", "http://127.0.0.1:8000/v1",
            "--model", "model",
            "--methods", "vlm_tamp,owl_tamp",
            "--variants", "K1,K2",
            "--seeds", "0,1",
        ],
    )

    main()

    assert len(commands) == 8
    # Every episode is bounded, so one hang cannot stall an unattended grid.
    assert timeouts and all(value == 3600.0 for value in timeouts)
    summary = json.loads((tmp_path / "batch" / "batch_summary.json").read_text())
    assert len(summary["runs"]) == 8


def test_living_room_vlm_tamp_uses_variant_plus_physical_execution():
    args = _args(environment="living_room", variants=("L1",))
    command = _command("vlm_tamp", "L1", 3, 0, Path("runs/out"), args)
    assert command[2] == "vlm_tamp_baseline.run_living_room"
    assert command[command.index("--variant") + 1] == "L1"
    assert "--physical-execution" in command
    # Kitchen's dedicated selector must not leak into the Living Room command.
    assert "--physical-variant" not in command


def test_living_room_owl_tamp_enables_physical_execution():
    args = _args(environment="living_room", variants=("L1",))
    command = _command("owl_tamp", "L1", 3, 0, Path("runs/out"), args)
    assert command[2] == "owl_tamp_baseline.run_living_room"
    assert "--physical-execution" in command
    assert command[command.index("--variant") + 1] == "L1"


def test_living_room_variants_are_validated_against_the_environment():
    with pytest.raises(ValueError, match="living_room"):
        _validate(_args(environment="living_room", variants=("K1",)))


def test_living_room_goal_defaults_to_the_frozen_goal():
    args = _args(environment="living_room", variants=("L1",), goal=None)
    command = _command("owl_tamp", "L1", 3, 0, Path("runs/out"), args)
    goal = command[command.index("--goal") + 1]
    assert "side table" in goal and "coffee table" in goal


def test_receding_horizon_forwards_the_no_plan_retry_flag():
    """The flag must reach the runner, and only when asked for.

    Without it a single unsatisfiable sketch ends a receding-horizon episode
    at round two, so `--max-replans` is unreachable and the protocol
    degenerates to single-shot -- which is the whole thing this row exists to
    avoid.
    """
    base = dict(
        environment="workshop", methods=("owl_tamp",), variants=("W1",),
        protocol="receding_horizon", max_replans=14, max_actions=40,
    )
    on = _command("owl_tamp", "W1", 3, 0, Path("runs/out"),
                  _args(**base, replan_on_no_plan=True))
    off = _command("owl_tamp", "W1", 3, 0, Path("runs/out"),
                   _args(**base, replan_on_no_plan=False))
    assert "--replan-on-no-plan" in on
    assert "--replan-on-no-plan" not in off
    assert on[on.index("--protocol") + 1] == "receding_horizon"
    assert on[on.index("--max-replans") + 1] == "14"
