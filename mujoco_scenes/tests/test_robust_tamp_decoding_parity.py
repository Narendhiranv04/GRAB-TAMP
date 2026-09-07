"""ROBUST-TAMP must be able to run the comparison table's decoding condition.

`discovery_planner.OpenAIPlannerConfig` hardcoded `temperature 0.0` and
`max_tokens 4096`.  That is ROBUST-TAMP's own published condition and remains
the default, but it is not the condition the table runs: VLM-TAMP, OWL-TAMP
and ViLaIn-TAMP all send `model-native` sampling drawn from
`baseline_common.inference`, and BASELINE_FIDELITY.md requires one decoding
condition per table.  Without a selector this column could not go in the same
table as the others -- the same defect ViLaIn-TAMP had.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from baseline_common.inference import QWEN_THINKING_SAMPLING
from mujoco_scenes.tamp.discovery_planner import (
    DECODING_CONDITIONS,
    OpenAIPlannerConfig,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "mujoco_scenes/run_kitchen_discovery_replanning.py",
    "mujoco_scenes/run_living_room_discovery_replanning.py",
)


def _config(**kwargs):
    return OpenAIPlannerConfig(
        base_url="http://127.0.0.1:18000/v1",
        model="qwen35-9b",
        scene="kitchen",
        **kwargs,
    )


def test_a_paper_condition_is_the_published_one_and_stays_the_default():
    """temperature 0 / 4096 tokens is what ROBUST-TAMP published."""
    assert _config().decoding == "paper"
    assert _config().resolved_sampling() == {"temperature": 0.0, "max_tokens": 4096}


def test_b_model_native_matches_the_other_baselines_exactly():
    """Byte-identical to the shared source, or the columns drift apart."""
    sampling = dict(_config(decoding="model-native").resolved_sampling())
    budget = sampling.pop("max_tokens")
    assert sampling == dict(QWEN_THINKING_SAMPLING)
    assert budget == 24576


def test_c_an_unknown_condition_is_rejected_rather_than_silently_applied():
    with pytest.raises(ValueError, match="decoding must be one of"):
        _config(decoding="greedy").resolved_sampling()
    assert DECODING_CONDITIONS == ("paper", "model-native")


@pytest.mark.parametrize("runner", RUNNERS)
def test_d_each_runner_exposes_and_forwards_the_condition(runner):
    """A flag that is parsed but dropped is worse than no flag at all.

    `retrieval_baseline/run_kitchen.py` did exactly that with --headless and
    silently launched a viewer for every episode, so this is asserted
    structurally rather than trusted.
    """
    source = (ROOT / runner).read_text(encoding="utf-8")
    assert '"--decoding"' in source, f"{runner}: no --decoding flag"
    assert "decoding=decoding" in source, (
        f"{runner}: the flag never reaches OpenAIPlannerConfig"
    )
    tree = ast.parse(source)
    forwarded = any(
        isinstance(node, ast.Call)
        and any(kw.arg == "decoding" for kw in node.keywords)
        and getattr(node.func, "id", "") == "run_episode"
        for node in ast.walk(tree)
    )
    assert forwarded, f"{runner}: main() does not forward --decoding to run_episode"
