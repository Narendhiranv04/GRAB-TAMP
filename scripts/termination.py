"""Classify why an episode ended, so budget and harness faults stop reading as method failures.

`terminal_status` collapses every non-completion to `FAILED`.  That bucket is
not one thing.  Measured over the canonical grid, ROBUST-TAMP's `FAILED`
episodes are:

    kitchen      120  ->  66 budget, 35 prompt-leakage guard, 19 server unreachable,
                           0 method failures
    workshop     100  ->  47 budget, 53 server unreachable, 0 method failures
    living_room   46  ->  39 budget,  4 server unreachable, 3 method failures

So Kitchen and Workshop ROBUST-TAMP contain no method failures at all, and
publishing them as 0/120 and 0/100 task success reports the harness.

The distinguishing text is already in `terminal_failure.message`, so this
classifies episodes that have already been run; nothing needs re-running.
`BASELINE_FIDELITY.md` states the rule for OWL-TAMP receding-horizon --
"budget-terminated and must not be reported as the method failing the task" --
and it applies to every method equally.
"""
from __future__ import annotations

import json
from pathlib import Path

SUCCESS = "SUCCESS"
BUDGET = "BUDGET"
INFRASTRUCTURE = "INFRASTRUCTURE"
GUARD = "HARNESS_GUARD"
TRUNCATED = "TOKEN_CEILING"
METHOD = "METHOD"

#: Reasons that are properties of the protocol or the machine, not the method.
#: Reasons that are properties of the protocol or the machine, not the method.
#: `TRUNCATED` is deliberately NOT here.  A decode that runs past `max_tokens`
#: is the method's own framework failing to produce a usable plan within the
#: stated budget, the same way an unparseable reply is, so it counts as a
#: failure of that method and is reported as one.  The label is kept so the
#: cause stays visible in the artifacts: 27 of 97 Kitchen ROBUST-TAMP episodes
#: and 20 of 105 Kitchen VLM-TAMP episodes end this way, and a reader should be
#: able to see that rather than infer a planning failure.
NOT_A_METHOD_FAILURE = (BUDGET, INFRASTRUCTURE, GUARD)


def classify(result: dict) -> str:
    """Return why this episode ended, from its benchmark result mapping."""
    if result.get("success"):
        return SUCCESS
    status = str(result.get("terminal_status") or "")
    if status == "MODEL_CALL_BUDGET_EXHAUSTED":
        return BUDGET
    message = str((result.get("terminal_failure") or {}).get("message") or "")
    if "budget exhausted" in message or "ceiling reached" in message:
        return BUDGET
    # Check truncation first: ROBUST-TAMP reports a decode that ran past
    # `max_tokens` as "Inference service unreachable after 2 retries:
    # MODEL_OUTPUT_TRUNCATED: ...", so matching "unreachable" first called 27
    # of 97 Kitchen episodes an outage.  They are not -- they ran a median of
    # 43.6 minutes and the endpoint was up throughout.  The 24576-token ceiling
    # is a stated framework limitation, so these are neither an outage nor the
    # method failing to plan, and they need their own bucket.
    if "MODEL_OUTPUT_TRUNCATED" in message or "token ceiling" in message:
        return TRUNCATED
    if "unreachable" in message or "Connection" in message:
        return INFRASTRUCTURE
    # An outage does not always leave a message.  When the endpoint went down
    # mid-run on 2026-09-13, Workshop VLM-TAMP wrote 100 episodes with
    # `terminal_status: INFERENCE_FAILED` and an empty `terminal_failure`, 99 of
    # them finishing in under 30 s, and this function called them method
    # failures.  A model-call failure is never the method's result, and an
    # episode that ends in seconds has not run the task.
    if status == "INFERENCE_FAILED":
        return INFRASTRUCTURE
    elapsed = result.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed < 30 and not result.get("executed_actions"):
        return INFRASTRUCTURE
    if "Refusing to send" in message:
        return GUARD
    return METHOD


def classify_dir(episode_dir: str | Path) -> str:
    path = Path(episode_dir) / "benchmark_execution_result.json"
    try:
        return classify(json.loads(path.read_text()))
    except (OSError, ValueError):
        return METHOD


def reportable(result: dict) -> bool:
    """True when the outcome is attributable to the method under test."""
    return classify(result) not in NOT_A_METHOD_FAILURE
