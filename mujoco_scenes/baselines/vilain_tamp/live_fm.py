"""Lazy, baseline-owned transports for the paper-faithful model condition."""

from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Any, Mapping, Protocol, Sequence

from .config import BaselineConfig, Domain, ModelCondition
from .contracts import CameraFrameArtifacts, ViLaInObservation
from .domains import load_domain
from .fm import (
    FMCallRecord,
    FMCallType,
    FMRequest,
    FMTransportError,
    FMTransportResponse,
    RecordedFMClient,
)
from .prompts import build_object_estimation_prompt


PAPER_QWEN_MODEL = "Qwen2.5-VL-7B-Instruct"
PAPER_QWEN_SOURCE = "Qwen/Qwen2.5-VL-7B-Instruct"
PAPER_REASONING_MODEL = "gpt-4o-2024-08-06"
from baseline_common.inference import assert_no_prompt_leakage

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:18000/v1"
VLLM_MAX_VISION_IMAGES = 8
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class QwenGeneration:
    text: str
    input_tokens: int
    output_tokens: int
    device: str
    dtype: str


class QwenBackend(Protocol):
    def generate(
        self,
        *,
        model_source: str,
        revision: str,
        messages: Sequence[Mapping[str, Any]],
        image_paths: Sequence[Path],
        max_new_tokens: int,
    ) -> QwenGeneration: ...


@dataclass(frozen=True)
class LiveModelClients:
    """Concrete clients and exact model identities for interpreter/CP wiring."""

    object_client: RecordedFMClient
    reasoning_client: RecordedFMClient
    object_estimator_model: str
    object_estimator_revision: str | None
    reasoning_model: str
    reasoning_model_revision: str | None


def build_qwen_only_clients(
    *,
    image_root: str | Path,
    served_model_id: str,
    reference_revision: str | None = None,
    base_url: str = DEFAULT_VLLM_BASE_URL,
    client: Any | None = None,
    timeout_seconds: float = 120.0,
    decoding: str = "paper",
) -> LiveModelClients:
    """Build distinct object/reasoning clients backed by one local vLLM API."""
    transport = VLLMQwenTransport(
        image_root=image_root,
        served_model_id=served_model_id,
        reference_revision=reference_revision,
        base_url=base_url,
        client=client,
        timeout_seconds=timeout_seconds,
        decoding=decoding,
    )
    return LiveModelClients(
        object_client=RecordedFMClient(transport),
        reasoning_client=RecordedFMClient(transport),
        object_estimator_model=served_model_id,
        object_estimator_revision=None,
        reasoning_model=served_model_id,
        reasoning_model_revision=None,
    )


# Decoding conditions.  ViLaIn's own published condition is greedy: temperature
# 0 with thinking disabled, which is what this transport did unconditionally.
# That is faithful to the paper but it is *not* the condition VLM-TAMP and
# OWL-TAMP run in the reported table, and BASELINE_FIDELITY.md requires one
# condition per table -- so the condition is now selectable and the paper's is
# merely the default.  The `model-native` numbers come from
# `baseline_common.inference`, the same source the other three baselines read,
# so the four columns cannot drift apart silently.
from .config import DECODING_CONDITIONS

# Output budget per condition.  4096 was sized for the paper condition, whose
# completions run a few hundred tokens with thinking disabled.  A thinking
# trace overruns that and returns finish_reason=length on an episode's first
# call.  The served window is 32768 and the Kitchen fixed-full-inspection
# prompt alone runs 24-31k, so the larger budget cannot always be honoured
# there; `_completion_with_context_retry` shrinks the request when the server
# rejects it outright.
DECODING_MAX_TOKENS = {"paper": 4096, "model-native": 16384}


def decoding_arguments(condition: str) -> dict[str, object]:
    """Sampling arguments and thinking flag for a named decoding condition."""
    if condition == "paper":
        # Qwen's card advises against greedy decoding for this checkpoint, and
        # BASELINE_FIDELITY.md records that near-greedy sampling is where plan
        # degeneration was first observed.  Retained because it is what
        # ViLaIn-TAMP published, not because it is the better setting here.
        return {
            "temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    if condition == "model-native":
        from baseline_common.inference import QWEN_THINKING_SAMPLING

        # The other baselines POST raw JSON, so vLLM accepts its sampling
        # extensions as ordinary fields.  This transport goes through the
        # OpenAI SDK, which validates keyword arguments and rejects anything
        # outside the documented API -- so top_k, min_p and repetition_penalty
        # have to travel in extra_body instead.  Passing them at the top level
        # raises TypeError on the first request of every episode.
        sampling = dict(QWEN_THINKING_SAMPLING)
        vllm_only = {
            key: sampling.pop(key)
            for key in ("top_k", "min_p", "repetition_penalty")
            if key in sampling
        }
        return {
            **sampling,
            "extra_body": {
                **vllm_only,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
    raise ValueError(
        f"decoding must be one of {DECODING_CONDITIONS}, got {condition!r}"
    )


# The prompt length vLLM reports on a context rejection is a lower bound, so
# the budget derived from it needs real headroom rather than a token or two.
_CONTEXT_MARGIN_TOKENS = 1024
_CONTEXT_RETRY_ATTEMPTS = 3
# BASELINE_FIDELITY.md: "Truncation draws on its own bounded retry budget, in
# the same way a transport fault already did."  VLM-TAMP implements that as
# `max_truncation_retries=2` in its executive; this transport had no retry at
# all and raised on the first `finish_reason == "length"`, which killed the
# whole episode and wrote no artifact -- the opposite of the same document's
# "recorded, not dropped".  Three attempts = one call plus two retries, so the
# budget matches VLM-TAMP's.  Retrying is meaningful rather than superstitious
# because the completion length is bimodal under thinking-mode sampling: the
# measurement in that document found calls that completed needed under 7400
# tokens while the ones that ran away hit the ceiling, so a fresh draw at the
# same temperature is a real second chance and does not alter the decoding
# condition.
_TRUNCATION_RETRY_ATTEMPTS = 3


class VLLMQwenTransport:
    """OpenAI-compatible Qwen transport restricted to the local SSH tunnel."""

    def __init__(
        self,
        *,
        image_root: str | Path,
        served_model_id: str,
        reference_revision: str | None = None,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        client: Any | None = None,
        timeout_seconds: float = 120.0,
        # Observed completions for these calls are a few hundred tokens, so
        # 8192 was ~30x over-provisioned -- and it was requested *in addition*
        # to the prompt, which made the fixed-full-inspection Kitchen prompt
        # exceed the served 32768-token window outright (24577 input + 8192
        # requested).  A smaller budget is still generous, and
        # `_completion_with_context_retry` recovers if a prompt grows further.
        # None means "pick from the decoding condition".  4096 was sized for
        # the paper condition, whose completions run a few hundred tokens with
        # thinking disabled; a thinking trace overruns it and the request comes
        # back finish_reason=length on the very first call of an episode.
        max_tokens: int | None = None,
        # Defaults to ViLaIn's published greedy condition so an existing caller
        # is unchanged; the grid selects "model-native" for table parity.
        decoding: str = "paper",
    ) -> None:
        if max_tokens is None:
            max_tokens = DECODING_MAX_TOKENS[decoding]
        normalized_url = base_url.rstrip("/")
        if normalized_url != DEFAULT_VLLM_BASE_URL:
            raise ValueError(
                f"vLLM base URL must be the approved local tunnel {DEFAULT_VLLM_BASE_URL}"
            )
        if not served_model_id.strip():
            raise ValueError("served_model_id must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.image_root = Path(image_root).resolve()
        self.served_model_id = served_model_id
        self.reference_revision = reference_revision
        self.base_url = normalized_url
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.decoding = decoding
        self.last_truncated_attempts = 0
        self.last_leakage_audit: dict[str, Any] | None = None
        # Resolved eagerly so an unknown condition fails at construction rather
        # than mid-episode, after a scene has already been built.
        self._decoding_arguments = decoding_arguments(decoding)
        self._client = client

    @staticmethod
    def _context_budget_from_error(message: str) -> int | None:
        """Derive a completion budget that fits, from the server's rejection.

        vLLM rejects an over-long request with both numbers it used, e.g.
        "maximum context length is 32768 tokens. However, you requested 8192
        output tokens and your prompt contains at least 24577 input tokens".
        Reading them back is exact, where guessing a smaller budget is not.
        """
        window = re.search(r"maximum context length is (\d+)", message)
        prompt = re.search(r"prompt contains at least (\d+)", message)
        if not window or not prompt:
            return None
        # The server reports the prompt length as "at least", so it is a lower
        # bound and it grew between attempts (28673 then 28930 on the same
        # call), which made a 256-token margin miss by exactly one token.
        budget = int(window.group(1)) - int(prompt.group(1)) - _CONTEXT_MARGIN_TOKENS
        return budget if budget > 0 else None

    def _completion_with_context_retry(
        self, client: Any, request_arguments: dict[str, Any]
    ) -> Any:
        """Issue the completion, shrinking the budget once if it will not fit.

        A prompt that overflows the context window is an infrastructure limit,
        not a model failure, and must not be recorded as one: the
        fixed-full-inspection Kitchen prompt failed this way and produced no
        result at all.  Retrying once with a budget the server itself says
        will fit keeps the episode alive without changing what is asked.
        """
        arguments = dict(request_arguments)
        for _ in range(_CONTEXT_RETRY_ATTEMPTS):
            try:
                return client.chat.completions.create(**arguments)
            except Exception as error:  # noqa: BLE001 - provider-specific type
                budget = self._context_budget_from_error(str(error))
                if budget is None or budget >= int(arguments["max_tokens"]):
                    raise
                arguments["max_tokens"] = budget
        # Out of attempts: the prompt itself is too large for the window, which
        # is a prompt-size problem rather than a budget one.  Let the real
        # provider error surface instead of masking it.
        return client.chat.completions.create(**arguments)

    def _completion_with_truncation_retry(
        self, client: Any, request_arguments: dict[str, Any]
    ) -> tuple[Any, int]:
        """Issue the completion, redrawing while the generation is cut off.

        Returns the response and the number of attempts that came back
        truncated, so the caller can tell "succeeded on the retry" from
        "exhausted the budget" -- the first is a recovered harness fault and
        the second is the distinct `MODEL_OUTPUT_TRUNCATED` failure mode that
        `BASELINE_FIDELITY.md` requires be kept out of a method's
        planning-failure count.

        The arguments are re-sent unchanged: a truncated draw is not evidence
        that the request was wrong, and shrinking or growing the budget here
        would silently move this baseline off the table's token limit.
        """
        truncated_attempts = 0
        response = None
        for _ in range(_TRUNCATION_RETRY_ATTEMPTS):
            response = self._completion_with_context_retry(
                client, request_arguments
            )
            choices = getattr(response, "choices", ())
            finish_reason = (
                getattr(choices[0], "finish_reason", None) if choices else None
            )
            if finish_reason != "length":
                return response, truncated_attempts
            truncated_attempts += 1
        return response, truncated_attempts

    def complete(self, request: FMRequest) -> FMTransportResponse:
        if request.model != self.served_model_id:
            raise FMTransportError(
                f"request model {request.model!r} differs from served model "
                f"{self.served_model_id!r}"
            )
        if request.revision is not None:
            raise FMTransportError(
                "the running vLLM server does not expose an immutable revision"
            )
        messages, image_metadata = self._messages_with_metadata(request)
        client = self._client
        if client is None:
            require_vilain_environment()
            try:
                from openai import OpenAI
            except ImportError as error:
                raise FMTransportError(
                    "OpenAI-compatible client dependency is unavailable"
                ) from error
            client = OpenAI(
                base_url=self.base_url,
                api_key="EMPTY",
                timeout=self.timeout_seconds,
                max_retries=0,
            )
            self._client = client
        request_arguments: dict[str, Any] = {
            "model": self.served_model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **self._decoding_arguments,
        }
        if request.response_format == "json":
            request_arguments["response_format"] = {"type": "json_object"}
        # Audited before the request leaves, exactly as OpenAITransport does for
        # VLM-TAMP, OWL-TAMP and ROBUST-TAMP.  This baseline builds its own
        # OpenAI client, so it was the only model-driven method whose prompts
        # were never checked for privileged evaluator information -- and that
        # guard is what caught Kitchen publishing the oracle's own region names
        # (B1/C1/C2/D1/D2) to the model.  Auditing here rather than trusting a
        # one-off manual review means Kitchen's 120 ViLaIn episodes, which have
        # not run yet, cannot leak silently.
        self.last_leakage_audit = assert_no_prompt_leakage(request_arguments)
        response, truncated_attempts = self._completion_with_truncation_retry(
            client, request_arguments
        )
        choices = getattr(response, "choices", ())
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        if truncated_attempts >= _TRUNCATION_RETRY_ATTEMPTS:
            raise FMTransportError(
                "MODEL_OUTPUT_TRUNCATED: vLLM reached max_tokens on all "
                f"{_TRUNCATION_RETRY_ATTEMPTS} attempts"
            )
        # Recorded even when zero, so an artifact shows whether the accepted
        # completion needed a redraw.
        self.last_truncated_attempts = truncated_attempts
        provider_model = str(getattr(response, "model", ""))
        if provider_model != self.served_model_id:
            raise FMTransportError(
                f"vLLM returned unexpected served model {provider_model!r}"
            )
        usage = getattr(response, "usage", None)
        call_id = str(getattr(response, "id", ""))
        if not call_id:
            raise FMTransportError("vLLM response has no call ID")
        return FMTransportResponse(
            raw_text=_openai_text(response),
            call_id=call_id,
            model=provider_model,
            revision=None,
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            provider_metadata={
                "provider": "local_vllm_openai_compatible",
                "base_url": self.base_url,
                "served_model_id": self.served_model_id,
                "configured_reference_revision": self.reference_revision,
                "served_revision_verified": False,
                "decoding": self.decoding,
                "sampling": {
                    key: value
                    for key, value in self._decoding_arguments.items()
                    if key != "extra_body"
                },
                "max_tokens": self.max_tokens,
                "thinking_enabled": bool(
                    self._decoding_arguments.get("extra_body", {})
                    .get("chat_template_kwargs", {})
                    .get("enable_thinking", False)
                ),
                "finish_reason": finish_reason,
                **image_metadata,
            },
        )

    def _messages(self, request: FMRequest) -> list[dict[str, Any]]:
        messages, _ = self._messages_with_metadata(request)
        return messages

    def _messages_with_metadata(
        self, request: FMRequest
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rendered = [dict(message) for message in request.messages]
        if request.call_type not in (
            FMCallType.OBJECT_ESTIMATION,
            FMCallType.INITIAL_STATE,
        ):
            if request.image_artifacts:
                raise FMTransportError("reasoning calls must not contain images")
            return rendered, {}
        if not request.image_artifacts:
            if request.call_type is FMCallType.OBJECT_ESTIMATION:
                raise FMTransportError("object estimation requires RGB images")
            return rendered, {}
        user_index = next(
            (index for index, item in enumerate(rendered) if item.get("role") == "user"),
            None,
        )
        if user_index is None:
            raise FMTransportError(f"{request.call_type.value} requires a user message")
        text_content = str(rendered[user_index].get("content", ""))
        image_urls, image_metadata = self._model_image_urls(request.image_artifacts)
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        ]
        content.append({"type": "text", "text": text_content})
        rendered[user_index] = {"role": "user", "content": content}
        return rendered, image_metadata

    def _model_image_urls(
        self, image_artifacts: Sequence[str]
    ) -> tuple[list[str], dict[str, Any]]:
        resolved = [(value, self._resolve_image(value)) for value in image_artifacts]
        if len(resolved) <= VLLM_MAX_VISION_IMAGES:
            return [self._path_data_url(path) for _, path in resolved], {
                "source_image_count": len(resolved),
                "model_image_count": len(resolved),
                "vision_image_packing": "none",
            }

        stages: OrderedDict[str, list[Path]] = OrderedDict()
        for value, path in resolved:
            parts = Path(value).parts
            stage = next((part for part in parts if part.startswith("stages")), None)
            if stage == "stages":
                stage_index = parts.index(stage)
                stage = parts[stage_index + 1] if stage_index + 1 < len(parts) else stage
            stages.setdefault(stage or path.parent.name, []).append(path)
        if len(stages) > VLLM_MAX_VISION_IMAGES:
            raise FMTransportError(
                "MULTIMODAL_IMAGE_LIMIT: observation has more than "
                f"{VLLM_MAX_VISION_IMAGES} stages"
            )
        urls = [self._contact_sheet_data_url(paths) for paths in stages.values()]
        return urls, {
            "source_image_count": len(resolved),
            "model_image_count": len(urls),
            "vision_image_packing": "one_contact_sheet_per_observation_stage",
            "packed_stage_ids": list(stages),
        }

    def _resolve_image(self, value: str) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.image_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.image_root)
        except ValueError as error:
            raise FMTransportError(f"image escapes observation root: {value}") from error
        if not resolved.is_file():
            raise FMTransportError(f"object-estimation image is missing: {value}")
        return resolved

    @staticmethod
    def _path_data_url(resolved: Path) -> str:
        mime = mimetypes.guess_type(resolved.name)[0] or "image/png"
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _contact_sheet_data_url(paths: Sequence[Path]) -> str:
        try:
            from PIL import Image, ImageOps
        except ImportError as error:
            raise FMTransportError(
                "Pillow is required to pack multi-stage observation images"
            ) from error
        tile_size = (448, 336)
        columns = min(3, len(paths))
        rows = (len(paths) + columns - 1) // columns
        sheet = Image.new("RGB", (tile_size[0] * columns, tile_size[1] * rows))
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                tile = ImageOps.contain(source.convert("RGB"), tile_size)
                x = (index % columns) * tile_size[0]
                y = (index // columns) * tile_size[1]
                sheet.paste(tile, (x, y))
        encoded = BytesIO()
        sheet.save(encoded, format="JPEG", quality=90, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode(
            "ascii"
        )

    def _data_url(self, value: str) -> str:
        return self._path_data_url(self._resolve_image(value))


def build_paper_faithful_clients(
    *,
    config: BaselineConfig,
    image_root: str | Path,
    qwen_revision: str,
    qwen_model_source: str = PAPER_QWEN_SOURCE,
    qwen_backend: QwenBackend | None = None,
    openai_client: Any | None = None,
) -> LiveModelClients:
    """Resolve the paper condition without initializing either model backend."""
    if config.model_condition is not ModelCondition.PAPER_FAITHFUL:
        raise ValueError("live paper-faithful clients require paper_faithful config")
    if config.object_estimator_model != PAPER_QWEN_MODEL:
        raise ValueError("configuration does not select the paper Qwen model")
    if config.reasoning_model != PAPER_REASONING_MODEL:
        raise ValueError("configuration does not select the pinned GPT-4o snapshot")
    if not _FULL_COMMIT.fullmatch(qwen_revision):
        raise ValueError("qwen_revision must be an exact 40-character commit")
    return LiveModelClients(
        object_client=RecordedFMClient(
            QwenVLTransport(
                model_source=qwen_model_source,
                image_root=image_root,
                backend=qwen_backend,
            )
        ),
        reasoning_client=RecordedFMClient(
            OpenAIReasoningTransport(
                timeout_seconds=config.timeouts.model_seconds,
                client=openai_client,
            )
        ),
        object_estimator_model=PAPER_QWEN_MODEL,
        object_estimator_revision=qwen_revision,
        reasoning_model=PAPER_REASONING_MODEL,
        reasoning_model_revision=PAPER_REASONING_MODEL,
    )


class QwenVLTransport:
    """Local Qwen2.5-VL transport loaded only when ``complete`` is called."""

    def __init__(
        self,
        *,
        model_source: str = PAPER_QWEN_SOURCE,
        image_root: str | Path,
        max_new_tokens: int = 2048,
        backend: QwenBackend | None = None,
        require_dedicated_venv: bool = True,
    ) -> None:
        if not model_source.strip():
            raise ValueError("model_source must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        self.model_source = model_source
        self.image_root = Path(image_root).resolve()
        self.max_new_tokens = max_new_tokens
        self._backend = backend
        self.require_dedicated_venv = require_dedicated_venv

    def complete(self, request: FMRequest) -> FMTransportResponse:
        if request.call_type is not FMCallType.OBJECT_ESTIMATION:
            raise FMTransportError("Qwen transport accepts only object-estimation calls")
        if request.model != PAPER_QWEN_MODEL:
            raise FMTransportError(
                f"paper-faithful object model must be {PAPER_QWEN_MODEL!r}"
            )
        if request.revision is None or not _FULL_COMMIT.fullmatch(request.revision):
            raise FMTransportError("Qwen revision must be an exact 40-character commit")
        image_paths = tuple(self._resolve_image(item) for item in request.image_artifacts)
        if not image_paths:
            raise FMTransportError("object estimation requires at least one RGB image")

        backend = self._backend
        if backend is None:
            if self.require_dedicated_venv:
                require_vilain_environment()
            backend = _TransformersQwenBackend()
            self._backend = backend
        generated = backend.generate(
            model_source=self.model_source,
            revision=request.revision,
            messages=request.messages,
            image_paths=image_paths,
            max_new_tokens=self.max_new_tokens,
        )
        return FMTransportResponse(
            raw_text=generated.text,
            call_id=f"local-qwen-{uuid.uuid4()}",
            model=request.model,
            revision=request.revision,
            usage={
                "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens,
            },
            provider_metadata={
                "provider": "local_transformers",
                "model_source": self.model_source,
                "resolved_revision": request.revision,
                "device": generated.device,
                "dtype": generated.dtype,
                "max_new_tokens": self.max_new_tokens,
            },
        )

    def _resolve_image(self, value: str) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.image_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.image_root)
        except ValueError as error:
            raise FMTransportError(f"image escapes observation root: {value}") from error
        if not resolved.is_file():
            raise FMTransportError(f"object-estimation image is missing: {value}")
        return resolved


class _TransformersQwenBackend:
    """Actual Transformers backend; heavyweight imports remain call-local."""

    def __init__(self) -> None:
        self._loaded_key: tuple[str, str] | None = None
        self._model: Any | None = None
        self._processor: Any | None = None

    def generate(
        self,
        *,
        model_source: str,
        revision: str,
        messages: Sequence[Mapping[str, Any]],
        image_paths: Sequence[Path],
        max_new_tokens: int,
    ) -> QwenGeneration:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise FMTransportError(
                "Qwen runtime dependencies are unavailable in .venv-vilain-tamp"
            ) from error

        key = (model_source, revision)
        if self._loaded_key != key:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                revision=revision,
                torch_dtype="auto",
            )
            self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoProcessor.from_pretrained(
                model_source, revision=revision
            )
            self._loaded_key = key
        model = self._model
        processor = self._processor
        assert model is not None and processor is not None
        rendered_messages = _qwen_messages(messages, image_paths)
        prompt = processor.apply_chat_template(
            rendered_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(rendered_messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parameter = next(model.parameters())
        return QwenGeneration(
            text=text,
            input_tokens=int(inputs.input_ids.shape[-1]),
            output_tokens=int(trimmed[0].shape[-1]),
            device=str(parameter.device),
            dtype=str(parameter.dtype),
        )


class OpenAIReasoningTransport:
    """Pinned GPT-4o transport with environment-owned credentials."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
        require_dedicated_venv: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.timeout_seconds = timeout_seconds
        self._client = client
        self.require_dedicated_venv = require_dedicated_venv

    def complete(self, request: FMRequest) -> FMTransportResponse:
        if request.call_type is FMCallType.OBJECT_ESTIMATION:
            raise FMTransportError("GPT reasoning transport does not estimate objects")
        if request.model != PAPER_REASONING_MODEL:
            raise FMTransportError(
                f"paper-faithful reasoning model must be {PAPER_REASONING_MODEL!r}"
            )
        if request.revision not in (None, PAPER_REASONING_MODEL):
            raise FMTransportError("GPT revision must equal the immutable model snapshot")
        if request.image_artifacts:
            raise FMTransportError("reasoning calls must not receive observation images")

        client = self._client
        if client is None:
            if self.require_dedicated_venv:
                require_vilain_environment()
            try:
                from openai import OpenAI
            except ImportError as error:
                raise FMTransportError(
                    "OpenAI runtime dependency is unavailable in .venv-vilain-tamp"
                ) from error
            client = OpenAI(timeout=self.timeout_seconds)
            self._client = client

        response = client.chat.completions.create(
            model=request.model,
            messages=[dict(message) for message in request.messages],
            temperature=0,
        )
        provider_model = str(response.model)
        if provider_model != PAPER_REASONING_MODEL:
            raise FMTransportError(
                f"provider returned unexpected model snapshot {provider_model!r}"
            )
        raw_text = _openai_text(response)
        usage = getattr(response, "usage", None)
        call_id = getattr(response, "id", None)
        if not isinstance(call_id, str) or not call_id.strip():
            raise FMTransportError("OpenAI response has no call ID")
        return FMTransportResponse(
            raw_text=raw_text,
            call_id=call_id,
            model=provider_model,
            revision=PAPER_REASONING_MODEL,
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            provider_metadata={
                "provider": "openai",
                "response_model": provider_model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "temperature": 0,
                "timeout_seconds": self.timeout_seconds,
            },
        )


def require_vilain_environment() -> None:
    """Require live model initialization to occur in the dedicated venv."""
    active = os.environ.get("VIRTUAL_ENV")
    environment = Path(active).name if active else Path(sys.prefix).name
    if environment != ".venv-vilain-tamp":
        raise FMTransportError(
            "live model calls require the dedicated .venv-vilain-tamp environment"
        )


def _qwen_messages(
    messages: Sequence[Mapping[str, Any]], image_paths: Sequence[Path]
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    attached = False
    for message in messages:
        role = str(message.get("role", ""))
        content: list[dict[str, str]] = []
        if role == "user" and not attached:
            content.extend({"type": "image", "image": str(path)} for path in image_paths)
            attached = True
        content.append({"type": "text", "text": str(message.get("content", ""))})
        rendered.append({"role": role, "content": content})
    if not attached:
        raise FMTransportError("Qwen request has no user message for image attachment")
    return rendered


def _openai_text(response: Any) -> str:
    choices = getattr(response, "choices", ())
    if not choices:
        raise FMTransportError("OpenAI response has no choices")
    content = getattr(choices[0].message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise FMTransportError("OpenAI response has no text content")
    return content


def _load_observations(path: Path) -> tuple[ViLaInObservation, ...]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    rows = loaded.get("observations") if isinstance(loaded, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("observation manifest must contain observations")
    observations = []
    for row in rows:
        frames = tuple(CameraFrameArtifacts(**frame) for frame in row["camera_frames"])
        observations.append(ViLaInObservation(**{**row, "camera_frames": frames}))
    return tuple(observations)


def validate_standalone_object_estimation(
    *,
    observation_manifest: str | Path,
    task_instruction: str,
    domain: Domain | str,
    model_source: str,
    revision: str,
    output_directory: str | Path,
    transport: QwenVLTransport | None = None,
) -> FMCallRecord:
    """Make exactly one recorded object-estimation call from captured views."""
    if not task_instruction.strip():
        raise ValueError("task_instruction must not be empty")
    manifest = Path(observation_manifest).resolve()
    observations = _load_observations(manifest)
    domain_key = domain.value if isinstance(domain, Domain) else Domain(domain).value
    if any(item.domain != domain_key for item in observations):
        raise ValueError("observation manifest domain does not match --domain")
    bundle = build_object_estimation_prompt(
        task_instruction=task_instruction,
        domain=load_domain(domain_key),
        observations=observations,
    )
    live_transport = transport or QwenVLTransport(
        model_source=model_source, image_root=manifest.parent
    )
    _, record = RecordedFMClient(live_transport).invoke(
        FMRequest(
            call_type=FMCallType.OBJECT_ESTIMATION,
            model=PAPER_QWEN_MODEL,
            revision=revision,
            messages=bundle.messages(),
            image_artifacts=bundle.image_artifacts,
            response_format="json",
            metadata={
                "model_source": model_source,
                "observation_manifest": str(manifest),
            },
        ),
        output_directory,
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one standalone, recorded ViLaIn object-estimation call."
    )
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--domain", choices=[item.value for item in Domain], required=True)
    parser.add_argument("--model-source", default=PAPER_QWEN_SOURCE)
    parser.add_argument("--revision", required=True, help="Exact 40-character HF commit")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    record = validate_standalone_object_estimation(
        observation_manifest=args.observation_manifest,
        task_instruction=args.task,
        domain=args.domain,
        model_source=args.model_source,
        revision=args.revision,
        output_directory=args.output_directory,
    )
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
