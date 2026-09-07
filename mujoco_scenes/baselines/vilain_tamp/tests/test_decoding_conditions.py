"""Decoding conditions must be SDK-legal and match the other baselines.

This transport goes through the OpenAI SDK, which validates keyword arguments,
while the other three baselines POST raw JSON and so can pass vLLM's sampling
extensions as ordinary fields.  top_k, min_p and repetition_penalty therefore
have to travel in extra_body here; at the top level they raise TypeError on the
first request of every episode.
"""
from __future__ import annotations

import inspect

import pytest

from mujoco_scenes.baselines.vilain_tamp.config import DECODING_CONDITIONS
from mujoco_scenes.baselines.vilain_tamp.live_fm import decoding_arguments


@pytest.mark.parametrize("condition", DECODING_CONDITIONS)
def test_arguments_are_accepted_by_the_openai_sdk(condition):
    openai = pytest.importorskip("openai")
    allowed = set(
        inspect.signature(
            openai.resources.chat.completions.Completions.create
        ).parameters
    )
    for key in decoding_arguments(condition):
        assert key in allowed, f"{condition}: {key} would be rejected by the SDK"


def test_model_native_matches_the_other_baselines_exactly():
    """One decoding condition per table is a reporting requirement."""
    from baseline_common.inference import QWEN_THINKING_SAMPLING

    arguments = decoding_arguments("model-native")
    extra = dict(arguments["extra_body"])
    assert extra.pop("chat_template_kwargs") == {"enable_thinking": True}
    sent = {
        **{k: v for k, v in arguments.items() if k != "extra_body"},
        **extra,
    }
    assert sent == dict(QWEN_THINKING_SAMPLING)


def test_paper_condition_is_greedy_with_thinking_off():
    arguments = decoding_arguments("paper")
    assert arguments["temperature"] == 0
    assert arguments["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="decoding must be one of"):
        decoding_arguments("greedy")
