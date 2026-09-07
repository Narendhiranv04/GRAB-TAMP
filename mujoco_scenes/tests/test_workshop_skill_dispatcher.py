"""The Workshop executor is synchronous; the executive drives actions stepped.

`DiscoveryReplanningExecutive` calls `start(action)` then polls `update()`
until a terminal result, because Kitchen and Living Room animate motion across
simulator steps.  Workshop's executors return an `ActionResult` when the motion
is already done.  This adapter bridges the two rather than giving the executive
a second execution path.
"""

from __future__ import annotations

import pytest

from baseline_common.models import ActionResult
from mujoco_scenes.tamp.skills import FailureCode, SkillAction, SkillStartError
from mujoco_scenes.tamp.workshop_skill_dispatcher import WorkshopSkillDispatcher


class _Executor:
    def __init__(self, outcome: ActionResult):
        self.outcome = outcome
        self.seen: list[dict] = []

    def execute(self, action):
        self.seen.append(action.as_dict())
        return self.outcome


def test_a_a_successful_action_reports_once_then_goes_idle():
    executor = _Executor(ActionResult.succeeded("holding object_0001"))
    dispatcher = WorkshopSkillDispatcher(executor)
    dispatcher.start(SkillAction("PICK", {"object_id": "object_0001"}))
    result = dispatcher.update()
    assert result is not None and result.success
    assert result.effects == ("holding object_0001",)
    assert dispatcher.executed_actions == 1
    # Nothing left to advance, so a further poll must not re-run the action.
    assert dispatcher.update() is None
    assert len(executor.seen) == 1


def test_b_the_skill_name_is_upper_cased_for_the_executor():
    """WorkshopPhysicalExecutor.execute dispatches on `action.skill.upper()`."""
    executor = _Executor(ActionResult.succeeded())
    dispatcher = WorkshopSkillDispatcher(executor)
    dispatcher.start(SkillAction("pick", {"object_id": "object_0001"}))
    dispatcher.update()
    assert executor.seen[0]["skill"] == "PICK"


@pytest.mark.parametrize(
    "action,reason",
    [
        (SkillAction("PICK", {}), "missing object_id"),
        (SkillAction("PLACE", {"object_id": "o"}), "missing region_id"),
        (SkillAction("PICK", {"object_id": "   "}), "blank object_id"),
        (SkillAction("POUR", {"source_id": "o"}), "Workshop has no POUR"),
    ],
)
def test_c_an_unusable_action_is_rejected_before_the_executor_sees_it(action, reason):
    """A blank argument would otherwise execute against an empty string."""
    executor = _Executor(ActionResult.succeeded())
    dispatcher = WorkshopSkillDispatcher(executor)
    with pytest.raises(SkillStartError):
        dispatcher.start(action)
    assert executor.seen == [], reason


def test_d_an_unknown_executor_failure_code_is_kept_not_dropped():
    """`ActionResult.failure_code` is a free string; SkillResult wants an enum.

    The Workshop executor emits codes such as `unsupported_planning_action`
    that have no FailureCode member.  Mapping them to INTERNAL_ERROR keeps the
    result usable while `details` preserves the original for the trace.
    """
    executor = _Executor(
        ActionResult.failed("unsupported_planning_action", "nope", recoverable=False)
    )
    dispatcher = WorkshopSkillDispatcher(executor)
    dispatcher.start(SkillAction("PICK", {"object_id": "object_0001"}))
    result = dispatcher.update()
    assert not result.success
    assert result.failure_code is FailureCode.INTERNAL_ERROR
    assert result.recoverable is False
    assert result.details["executor_failure_code"] == "unsupported_planning_action"


def test_e_a_known_failure_code_maps_to_its_enum_member():
    executor = _Executor(ActionResult.failed("grasp_failed", "slipped"))
    dispatcher = WorkshopSkillDispatcher(executor)
    dispatcher.start(SkillAction("PICK", {"object_id": "object_0001"}))
    assert dispatcher.update().failure_code is FailureCode.GRASP_FAILED


def test_f_two_actions_cannot_be_in_flight_at_once():
    executor = _Executor(ActionResult.succeeded())
    dispatcher = WorkshopSkillDispatcher(executor)
    dispatcher.start(SkillAction("PICK", {"object_id": "object_0001"}))
    with pytest.raises(SkillStartError):
        dispatcher.start(SkillAction("PICK", {"object_id": "object_0002"}))
