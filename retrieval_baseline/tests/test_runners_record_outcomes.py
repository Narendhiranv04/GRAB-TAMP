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


def test_a_every_kitchen_runner_honours_headless():
    """--headless must reach show_viewer, not be parsed and dropped.

    `retrieval_baseline/run_kitchen.py` declared the flag and then omitted
    `show_viewer` from its `BaselineKitchenRuntime.from_variant` call, so it
    inherited the show_viewer=True default and launched an interactive passive
    viewer per episode.  That throttles MuJoCo stepping to render rate: under
    the batch runner -- which passes --headless precisely to avoid this -- four
    concurrent episodes held 43% CPU on an idle 24-core host and produced no
    output for 54 minutes.  It also made retrieval's wall-clock times
    incomparable with the other methods'.
    """
    import ast
    import pathlib

    for runner in ("retrieval_baseline/run_kitchen.py",
                   "vlm_tamp_baseline/run_kitchen.py",
                   "owl_tamp_baseline/run_kitchen.py"):
        source = pathlib.Path(runner).read_text()
        tree = ast.parse(source)
        constructions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and "BaselineKitchenRuntime" in ast.dump(node.func)
        ]
        assert constructions, f"{runner}: no BaselineKitchenRuntime construction found"
        for call in constructions:
            names = {kw.arg for kw in call.keywords}
            assert "show_viewer" in names, (
                f"{runner}: a BaselineKitchenRuntime construction omits "
                "show_viewer and so silently inherits the viewer"
            )


def test_b_kitchen_retrieval_reads_the_frame_directory_the_runtime_wrote():
    """The observation dir must follow the runtime's numbered-frame contract.

    `BaselineKitchenRuntime._images` writes to
    `observations/{capture_index:04d}` and increments the counter as it writes.
    The runner hardcoded `observations/initial`, which Kitchen never creates, so
    every episode completed its five region inspections and then died on
    FileNotFoundError -- which is why that cell had no results at all.
    """
    import pathlib

    source = pathlib.Path("retrieval_baseline/run_kitchen.py").read_text()
    assert '"observations" / "initial"' not in source, (
        "Kitchen writes numbered frame dirs, never 'initial'"
    )
    assert 'f"{runtime.capture_index:04d}"' in source, (
        "must read the frame the runtime just captured"
    )
