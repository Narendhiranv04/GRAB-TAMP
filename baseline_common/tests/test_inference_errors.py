from __future__ import annotations

from types import SimpleNamespace
import urllib.error

import pytest

from baseline_common.inference import (
    InvalidCompletionError,
    ModelTransportError,
    OpenAITransport,
    response_content,
    PromptLeakageError,
    assert_no_prompt_leakage,
    _auditable_text,
    PlanningError,
)


def test_invalid_completion_content_is_not_a_transport_failure() -> None:
    response = {
        "choices": [
            {"message": {"content": "not json"}, "finish_reason": "stop"}
        ]
    }

    with pytest.raises(InvalidCompletionError, match="not valid JSON"):
        response_content(response)


def test_unreachable_server_is_a_transport_failure(monkeypatch) -> None:
    def unreachable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)
    transport = OpenAITransport(
        SimpleNamespace(
            base_url="http://127.0.0.1:1/v1",
            api_key="",
            timeout_seconds=1.0,
        )
    )

    with pytest.raises(ModelTransportError, match="Cannot reach model server"):
        transport.complete({"model": "test"})


def _leak_payload(text):
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


# The no-leakage property every baseline claims was asserted in prose and never
# checked: `audit_prompt_leakage` existed for a long time with no caller. It is
# now enforced at the shared transport, and these tests exist so that it cannot
# quietly stop being enforced again.


def test_a_clean_prompt_is_audited_and_passes():
    verdict = assert_no_prompt_leakage(
        _leak_payload('{"id":"region_0001","alias":"shared_table"}')
    )
    assert verdict["audited"]
    assert verdict["zero_leakage"]


def test_canonical_storage_region_names_are_refused():
    # D1/D2/B1/C1/C2 are the Kitchen's closed storage; naming one tells the
    # model where to search and destroys the point of the inspection task.
    with pytest.raises(PromptLeakageError):
        assert_no_prompt_leakage(_leak_payload("the mug is inside C2"))


def test_oracle_symbols_are_refused():
    with pytest.raises(PromptLeakageError):
        assert_no_prompt_leakage(_leak_payload("consult GTSpecProvider"))


def test_checker_predicates_are_refused():
    with pytest.raises(PromptLeakageError):
        assert_no_prompt_leakage(_leak_payload("verify OPEN_CAVITY holds"))


def test_leakage_is_not_a_planning_error():
    # It must not be caught by the executives' PlanningError handlers, retried,
    # or scored against the method: the episode is contaminated, not badly
    # planned.
    assert not issubclass(PromptLeakageError, PlanningError)


def test_image_payloads_are_not_scanned_byte_by_byte():
    # Aliases in these frames are drawn into the pixels, not written into the
    # base64, so the encoding cannot leak a token and auditing it would cost
    # ~1.5 MB of string work per request for nothing.
    big = "data:image/png;base64," + "A" * 500_000
    payload = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "region_0001"},
                    {"type": "image_url", "image_url": {"url": big}},
                ],
            }
        ],
    }
    text = _auditable_text(payload)
    assert big not in text
    assert "<image>" in text
    assert "region_0001" in text
