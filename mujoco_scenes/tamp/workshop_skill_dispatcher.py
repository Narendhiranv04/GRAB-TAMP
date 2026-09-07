"""Adapt the Workshop executor to the stepped SkillDispatcher protocol.

`DiscoveryReplanningExecutive` drives actions asynchronously -- `start(action)`
then `update()` until a terminal `SkillResult` -- because the Kitchen and
Living Room runtimes animate motion across simulator steps.  The Workshop
executors are synchronous: `execute(action)` returns an `ActionResult` when the
motion is already finished.

Rather than give the executive a second execution path, this wrapper presents
the synchronous executor through the protocol it already speaks.  `start`
records the action, the first `update` runs it to completion and returns the
result.  Reporting completion on the first poll is honest here: by the time
`execute` returns there is nothing left to advance.
"""

from __future__ import annotations

from typing import Any

from baseline_common.models import Action

from .skills import FailureCode, SkillAction, SkillResult, SkillStartError

# Argument names each Workshop skill reads, so a plan that omits one fails as a
# precondition rather than silently executing against an empty string.
_REQUIRED_ARGUMENTS = {
    "INSPECT": ("region_id",),
    "PICK": ("object_id",),
    "PLACE": ("object_id", "region_id"),
    "INSERT": ("fastener_id", "target_id"),
    "FASTEN": ("tool_id", "fastener_id", "target_id"),
}


class WorkshopSkillDispatcher:
    """Stepped SkillDispatcher over a synchronous Workshop executor."""

    def __init__(self, executor: Any):
        self.executor = executor
        self._pending: SkillAction | None = None
        self.executed_actions = 0

    def start(self, action: SkillAction) -> None:
        if self._pending is not None:
            raise SkillStartError(
                FailureCode.INTERNAL_ERROR,
                "a Workshop action is already in flight",
            )
        skill = str(action.name).upper()
        required = _REQUIRED_ARGUMENTS.get(skill)
        if required is None:
            raise SkillStartError(
                FailureCode.PRECONDITION_FAILED,
                f"Workshop cannot apply {action.name}",
            )
        missing = [
            name
            for name in required
            if not str(action.arguments.get(name, "")).strip()
        ]
        if missing:
            raise SkillStartError(
                FailureCode.PRECONDITION_FAILED,
                f"{skill} is missing {', '.join(missing)}",
            )
        self._pending = action

    def update(self) -> SkillResult | None:
        if self._pending is None:
            return None
        action = self._pending
        self._pending = None
        outcome = self.executor.execute(
            Action(
                skill=str(action.name).upper(),
                arguments={
                    key: str(value) for key, value in action.arguments.items()
                },
            )
        )
        self.executed_actions += 1
        # ActionResult.failure_code is a plain string; SkillResult wants a
        # FailureCode.  An executor-specific code with no enum member becomes
        # INTERNAL_ERROR rather than being dropped, and the original text is
        # preserved in `details` so a trace still names it.
        code: FailureCode | None = None
        if not outcome.success:
            try:
                code = FailureCode(str(outcome.failure_code))
            except ValueError:
                code = FailureCode.INTERNAL_ERROR
        return SkillResult(
            success=bool(outcome.success),
            effects=tuple(outcome.effects),
            failure_code=code,
            message=str(outcome.message),
            recoverable=bool(outcome.recoverable),
            details={
                **dict(outcome.details or {}),
                "executor_failure_code": outcome.failure_code,
            },
        )
