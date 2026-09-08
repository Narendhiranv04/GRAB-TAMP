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


def test_d_a_refused_prompt_says_where_the_token_appeared():
    """Naming the token is not enough to fix a leak.

    A Kitchen episode was refused for `regions=['D1']` and locating the source
    cost an hour: the observation (verified across 106 episodes), the effect
    strings and the failure-feedback table were all clean, so the message gave
    nothing to act on. The guard already knows what it matched; it should also
    quote where.
    """
    from baseline_common.inference import (
        PromptLeakageError,
        assert_no_prompt_leakage,
    )

    payload = {
        "model": "qwen35-9b",
        "messages": [
            {"role": "system", "content": "plan over the visible state"},
            {"role": "user", "content": "the mug is inside D1, retrieve it"},
        ],
    }
    with pytest.raises(PromptLeakageError) as caught:
        assert_no_prompt_leakage(payload)
    message = str(caught.value)
    assert "regions=['D1']" in message, "still reports what it found"
    assert "found at:" in message, "must also report where"
    assert "retrieve it" in message, "excerpt must include surrounding context"


def test_e_the_excerpt_cannot_dump_an_entire_prompt():
    """A 30k-token prompt must not end up in a log line."""
    from baseline_common.inference import (
        PromptLeakageError,
        _LEAKAGE_CONTEXT_RADIUS,
        assert_no_prompt_leakage,
    )

    filler = "x" * 40000
    payload = {"model": "m", "messages": [{"role": "user", "content": f"{filler} D1 {filler}"}]}
    with pytest.raises(PromptLeakageError) as caught:
        assert_no_prompt_leakage(payload)
    message = str(caught.value)
    assert len(message) < 4 * _LEAKAGE_CONTEXT_RADIUS + 400, (
        f"excerpt is unbounded: {len(message)} chars"
    )
