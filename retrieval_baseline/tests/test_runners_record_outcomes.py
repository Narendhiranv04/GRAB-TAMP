"""Every retrieval runner must record the feasible/infeasible verdict.

`success` is false by construction on an infeasible variant, so the only
credit such a variant can earn is `predicted_outcome == expected_outcome`.
The Living Room runner omitted both fields, and its whole column reported
`outcome_match: null` -- contributing nothing to `outcome_correct_percent` or
`infeasible_rejection_percent`.  A missing keyword argument cannot fail
loudly here, so it is asserted structurally.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUNNERS = ("run_living_room.py", "run_kitchen.py", "run_workshop.py")


def _write_execution_result_keywords(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_execution_result"
        ):
            return {keyword.arg for keyword in node.keywords if keyword.arg}
    return set()


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_records_the_outcome_verdict(runner):
    keywords = _write_execution_result_keywords(
        Path(__file__).resolve().parents[1] / runner
    )
    assert keywords, f"{runner}: no write_execution_result call found"
    for required in ("expected_outcome", "predicted_outcome"):
        assert required in keywords, f"{runner}: missing {required}"
