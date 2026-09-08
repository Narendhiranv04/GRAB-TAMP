"""Frozen-model transport and profile helpers shared by comparison methods."""

from __future__ import annotations

import base64
import binascii
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


MODEL_REGISTRY = Path(__file__).parents[1] / "inference_server" / "models.json"
# Fallback decoding when the model registry is unavailable; keep it in step
# with inference_server/models.json.  These are Qwen3.5-9B's published
# thinking-mode figures for precise coding rather than the general-task ones
# (which set presence_penalty 1.5): every planner request here is constrained
# JSON against a schema, and the general-task penalty measurably makes this
# checkpoint run past its stopping point on that kind of output -- 18 to 64
# actions for a 10-action Living Room task, with mutually contradictory goal
# literals.  Both profiles are prescribed by the model authors; this is the one
# whose task description matches.  repetition_penalty 1.05 is the single
# documented deviation from the published figures: it is the smallest value
# that holds the sketch to its correct length here (1.03 still degenerates to
# the 64-action cap, 1.05 and 1.10 both return exactly the 10 right actions),
# and Qwen's own guidance warns this checkpoint produces endless repetitions
# without a repetition guard.  temperature, top_p, top_k and min_p are the
# authors' values unchanged.
QWEN_THINKING_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.05,
}


class PromptLeakageError(RuntimeError):
    """A model-facing request contained privileged evaluator information.

    Deliberately **not** a `PlanningError`: this is not something the method
    did wrong and it must not be caught, retried, or scored as a planning
    failure. It means the harness was about to hand the model a checker
    predicate, a canonical storage-region name or an oracle symbol, and any
    episode built on such a request is contaminated. Failing the episode loudly
    is cheaper than discovering afterwards that a reported number was produced
    with privileged input.
    """


class PlanningError(RuntimeError):
    """The model request or completion could not produce valid output."""


class ModelTransportError(PlanningError):
    """The inference service could not return a usable completion response."""


class InvalidCompletionError(PlanningError):
    """The service responded, but the completion violates the output contract."""


class TruncatedCompletionError(InvalidCompletionError):
    """Generation hit the token ceiling before it finished its answer.

    Distinguished from every other invalid completion because no answer was
    produced: the caller's budget cut the model off mid-generation.  A caller
    that charges its per-episode model-call budget for this would score its own
    ceiling as the method's planning failure.
    """


class TransportConfig(Protocol):
    base_url: str
    api_key: str
    timeout_seconds: float


def _auditable_text(payload: Mapping[str, object]) -> str:
    """Serialize a request for leakage auditing, minus the image bytes.

    A request carries a base64 data URL per camera, which is ~1.5 MB and can
    contain no forbidden token as literal text -- the semantic aliases in these
    frames are drawn into the pixels, not written into the encoding. Dropping
    them keeps the audit on the part that could actually leak, and keeps it
    cheap enough to run on every request rather than sampling.
    """
    def strip(value: Any) -> Any:
        if isinstance(value, str):
            return "<image>" if value.startswith("data:image/") else value
        if isinstance(value, Mapping):
            return {key: strip(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    return json.dumps(strip(dict(payload)), sort_keys=True)


def assert_no_prompt_leakage(payload: Mapping[str, object]) -> dict[str, Any]:
    """Refuse to send a request carrying privileged evaluator information.

    `audit_prompt_leakage` has existed since the functional pipeline was
    written but nothing ever called it, so the no-leakage property every
    baseline claims was asserted rather than enforced. Auditing here -- at the
    single transport every model-driven method shares -- makes it hold by
    construction for all of them at once.

    Import is local so `baseline_common` keeps no module-level dependency on
    the functional pipeline, and so a checkout without it still runs: an
    unavailable auditor reports itself as unavailable rather than silently
    passing, which is the distinction that matters when reading a trace.
    """
    try:
        from mujoco_scenes.functional_tamp_pipeline.audit import audit_prompt_leakage
    except ImportError as error:  # pragma: no cover - environment guard
        return {"audited": False, "audit_status": "SKIPPED_AUDITOR_UNAVAILABLE",
                "reason": str(error)}
    auditable = _auditable_text(payload)
    verdict = audit_prompt_leakage({"request": auditable})
    if verdict.get("audited") and not verdict.get("zero_leakage", True):
        found = [
            *(verdict.get("forbidden_checkers_found") or ()),
            *(verdict.get("forbidden_regions_found") or ()),
            *(verdict.get("forbidden_oracle_symbols_found") or ()),
        ]
        raise PromptLeakageError(
            "Refusing to send a model request containing privileged evaluator "
            f"information: checkers={verdict.get('forbidden_checkers_found')} "
            f"regions={verdict.get('forbidden_regions_found')} "
            f"oracles={verdict.get('forbidden_oracle_symbols_found')}"
            f"{_leakage_context(auditable, found)}"
        )
    return verdict


# Long enough to identify the field and the surrounding structure, short enough
# that a 30k-token prompt does not end up in a log line.
_LEAKAGE_CONTEXT_RADIUS = 90


def _leakage_context(auditable: str, tokens: Sequence[str]) -> str:
    """Quote where each forbidden token actually appears in the payload.

    Naming the token alone is not enough to fix a leak: a Kitchen episode was
    refused for `regions=['D1']` and locating the source cost an hour of
    reading, because the observation, the effect strings and the failure
    feedback were all verifiably clean.  The guard already knows what it
    matched, so it should also say where.
    """
    if not tokens:
        return ""
    excerpts: list[str] = []
    for token in dict.fromkeys(tokens):
        index = auditable.find(token)
        if index < 0:
            continue
        start = max(0, index - _LEAKAGE_CONTEXT_RADIUS)
        end = min(len(auditable), index + len(token) + _LEAKAGE_CONTEXT_RADIUS)
        excerpt = auditable[start:end].replace("\n", " ")
        excerpts.append(f"{token!r} in ...{excerpt}...")
    if not excerpts:
        return ""
    return " | found at: " + " ;; ".join(excerpts)


class OpenAITransport:
    def __init__(self, config: TransportConfig):
        self.config = config
        self.last_leakage_audit: dict[str, Any] | None = None

    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        # Audited before the request leaves, so a contaminated prompt is never
        # sent rather than merely noticed afterwards.
        self.last_leakage_audit = assert_no_prompt_leakage(payload)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                try:
                    result = json.load(response)
                except (json.JSONDecodeError, UnicodeError) as error:
                    raise ModelTransportError(
                        "Model server returned invalid JSON"
                    ) from error
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ModelTransportError(
                f"Model server returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise ModelTransportError(
                f"Cannot reach model server: {error.reason}"
            ) from error
        except (OSError, TimeoutError) as error:
            raise ModelTransportError(f"Model request failed: {error}") from error
        if not isinstance(result, Mapping):
            raise ModelTransportError("Model server response must be a JSON object")
        return result


def validate_images(
    images: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    if not 1 <= len(images) <= 8:
        raise ValueError("images must contain between 1 and 8 views")
    result = []
    cameras: set[str] = set()
    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            raise ValueError(f"images[{index}] must be an object")
        camera_value = image.get("camera", "")
        data_url = image.get("data_url", "")
        if not isinstance(camera_value, str):
            raise ValueError(f"images[{index}].camera must be a string")
        camera = camera_value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", camera):
            raise ValueError(f"Invalid camera name {camera!r}")
        if camera in cameras:
            raise ValueError(f"Duplicate camera name {camera!r}")
        if not isinstance(data_url, str):
            raise ValueError(f"Camera {camera!r} data_url must be a string")
        if not data_url.startswith("data:image/") or ";base64," not in data_url[:128]:
            raise ValueError(f"Camera {camera!r} needs a base64 image data URL")
        try:
            base64.b64decode(data_url.split(",", 1)[1], validate=True)
        except (binascii.Error, IndexError) as error:
            raise ValueError(f"Camera {camera!r} has invalid base64 image data") from error
        cameras.add(camera)
        result.append({"camera": camera, "data_url": data_url})
    return tuple(result)


def response_content(
    response: Mapping[str, object],
    reasoning_markers: tuple[str, str] | None = None,
) -> Mapping[str, Any]:
    try:
        choice = response["choices"][0]  # type: ignore[index]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise InvalidCompletionError(
            "Completion response has no message content"
        ) from error
    if isinstance(content, Mapping):
        return content
    reason = str(choice.get("finish_reason", "unknown"))
    if content is None:
        if reason == "length":
            raise TruncatedCompletionError(
                "Generation reached the token ceiling before emitting any "
                "final JSON content"
            )
        raise InvalidCompletionError(
            "Completion has no final JSON content "
            f"(finish_reason={reason}); increase the baseline token budget"
        )
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    if not isinstance(content, str):
        raise InvalidCompletionError("Completion content must be JSON text")
    content = content.strip()
    if reasoning_markers and reasoning_markers[1] in content:
        content = content.split(reasoning_markers[1], 1)[1].strip()
    if content.startswith("```") and content.endswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        if reason == "length":
            raise TruncatedCompletionError(
                "Generation reached the token ceiling part-way through its "
                "JSON content"
            ) from error
        raise InvalidCompletionError(
            "Completion content is not valid JSON"
        ) from error
    if not isinstance(result, Mapping):
        raise InvalidCompletionError("Completion JSON must be an object")
    return result


def load_model_profile(name: str) -> Mapping[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Inference profile name must not be empty")
    if not MODEL_REGISTRY.is_file():
        if name == "qwen35-9b":
            return {
                "served_name": name,
                "planner": {
                    "thinking_mode": "toggle",
                    "max_tokens": 24576,
                    "sampling": {"thinking": QWEN_THINKING_SAMPLING},
                },
            }
        raise ValueError(f"Model registry not found: {MODEL_REGISTRY}")
    document = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("Model registry must be a JSON object")
    profiles = document.get("models", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("Model registry must define a models object")
    profile = profiles.get(name)
    if (
        not isinstance(profile, Mapping)
        or profile.get("available", True) is not True
    ):
        raise ValueError(f"Unknown or unavailable inference profile {name!r}")
    planner = profile.get("planner", {})
    if not isinstance(planner, Mapping):
        raise ValueError(f"Inference profile {name!r} has invalid planner settings")
    return profile
