"""Every model-driven method must audit its prompt before it leaves.

`assert_no_prompt_leakage` is what caught Kitchen publishing the oracle's own
region names (B1/C1/C2/D1/D2) to the model -- a real fairness defect that had
been silently live, and that the shared observation contract had been claiming
was impossible.

VLM-TAMP, OWL-TAMP and ROBUST-TAMP all reach the model through
`baseline_common.inference.OpenAITransport`, which audits in `complete()`.
ViLaIn-TAMP builds its own OpenAI client, so it was the one model-driven method
whose prompts were never checked. A one-off manual review is not the same
property: it does not hold for prompts nobody has looked at yet, and ViLaIn's
120 Kitchen episodes had not run.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Modules that own a model transport, and the callable that must audit.
TRANSPORTS = (
    ("baseline_common/inference.py", "OpenAITransport", "complete"),
    ("mujoco_scenes/baselines/vilain_tamp/live_fm.py", "VLLMQwenTransport", "complete"),
)


@pytest.mark.parametrize("module,cls,method", TRANSPORTS)
def test_a_the_transport_audits_before_the_request_leaves(module, cls, method):
    source = (ROOT / module).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    target = item
    assert target is not None, f"{module}: {cls}.{method} not found"
    body = ast.get_source_segment(source, target) or ""
    assert "assert_no_prompt_leakage" in body, (
        f"{cls}.{method} sends a model request without auditing it"
    )


def test_b_vilain_audits_before_sending_not_after():
    """Noticing a contaminated prompt after sending it is not a guard."""
    from mujoco_scenes.baselines.vilain_tamp.live_fm import VLLMQwenTransport

    source = inspect.getsource(VLLMQwenTransport.complete)
    audit_at = source.index("assert_no_prompt_leakage")
    send_at = source.index("_completion_with_truncation_retry")
    assert audit_at < send_at


def test_c_no_model_driven_baseline_bypasses_an_audited_transport():
    """A new baseline must not quietly construct its own unaudited client."""
    offenders = []
    for pattern in ("*_baseline/*.py", "mujoco_scenes/tamp/*.py", "mujoco_scenes/baselines/*/*.py"):
        for path in ROOT.glob(pattern):
            if "test" in path.name:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            builds_client = "OpenAI(" in source or "urllib.request.urlopen" in source
            if not builds_client:
                continue
            if "assert_no_prompt_leakage" in source:
                continue
            offenders.append(str(path.relative_to(ROOT)))
    # baseline_common/inference.py is the audited transport itself; retrieval
    # makes no model call at all.
    allowed = {"baseline_common/inference.py"}
    assert not (set(offenders) - allowed), (
        "these build a model client without auditing the prompt: "
        f"{sorted(set(offenders) - allowed)}"
    )
