"""Every scene runner must record the feasible/infeasible verdict.

`success` is false by construction on an infeasible variant, so the only
credit it can earn is `predicted_outcome == expected_outcome`.  Both the
Living Room and Kitchen runners for VLM-TAMP and OWL-TAMP *computed* the
verdict and wrote it to `gt_sequence_comparison.json`, but never passed it to
`write_execution_result` -- which is the artifact the metrics read.  The
result was 46 recorded Living Room episodes carrying three nulls, with every
infeasible variant scoring as a plain failure.

A missing keyword argument cannot fail loudly at runtime, so it is asserted
structurally across all nine runners.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = tuple(
    f"{method}_baseline/run_{scene}.py"
    for method in ("vlm_tamp", "owl_tamp", "retrieval")
    for scene in ("living_room", "kitchen", "workshop")
)


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_passes_the_verdict_to_the_shared_artifact(runner):
    path = ROOT / runner
    assert path.exists(), runner
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_execution_result"
    ]
    assert calls, f"{runner}: no write_execution_result call"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        for required in ("expected_outcome", "predicted_outcome"):
            assert required in keywords, f"{runner}: missing {required}"
