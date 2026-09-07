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


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_verdict_variables_are_bound_on_the_executing_path(runner):
    """A verdict computed only under --planning-only cannot be written by the
    executing path.

    `vlm_tamp_baseline/run_kitchen.py` assigned `expected` and
    `predicted_outcome` inside `if arguments.planning_only:` while passing
    `expected["intended_outcome"]` to `write_execution_result` at the end of
    the *physical* path.  Every executing Kitchen episode therefore died with
    `UnboundLocalError` after completing 20+ minutes of work, writing no
    artifact.  It stayed latent because until the region-anonymisation fix no
    Kitchen episode reached that line at all -- Workshop and Living Room
    compute theirs unconditionally and completed 200/200.

    A name that write_execution_result depends on must not be bound only
    inside a planning-only branch.
    """
    import ast

    source = (ROOT / runner).read_text(encoding="utf-8")
    tree = ast.parse(source)

    needed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "write_execution_result"
        ):
            for kw in node.keywords:
                for sub in ast.walk(kw.value):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        needed.add(sub.id)
    if not needed:
        pytest.skip(f"{runner} does not call write_execution_result")

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "planning_only" in ast.dump(node.test):
            bound_here = {
                s.id
                for s in ast.walk(node)
                if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store)
            }
            # Names bound in the branch AND read by the shared-artifact write.
            trapped = bound_here & needed
            assert not trapped, (
                f"{runner}: {sorted(trapped)} bound only under planning_only "
                "but read by write_execution_result on the executing path"
            )
