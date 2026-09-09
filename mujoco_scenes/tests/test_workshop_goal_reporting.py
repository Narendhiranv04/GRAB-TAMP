"""The Workshop goal has three clauses; only one of them is the joint state."""

from __future__ import annotations

from dataclasses import dataclass

from mujoco_scenes.baseline_workshop_runtime import workshop_goal_reached


@dataclass
class _Physical:
    """Stands in for `WorkshopPhysicalExecutor.goal_satisfied`."""

    goal_satisfied: bool


@dataclass
class _Runtime:
    """Stands in for `WorkshopPlanningRuntime.goal_verifier`.

    The real predicate is `fastened is not None and held_object is None and
    locations[tool] == WORK_SURFACE` -- the fastening plus the two clauses the
    physical joint cannot see.
    """

    satisfied: bool

    def goal_verifier(self, _observation=None) -> bool:
        return self.satisfied


def test_a_fastened_joint_alone_is_not_the_goal():
    """The bug this guards: 8 of 400 episodes fastened, all still holding it.

    Scoring the joint alone credited them, and because the executive returns as
    soon as its verifier passes, it also stopped them before the driver could
    be put back.
    """
    assert workshop_goal_reached(_Physical(True), _Runtime(False)) is False


def test_the_full_goal_is_reported_when_every_clause_holds():
    assert workshop_goal_reached(_Physical(True), _Runtime(True)) is True


def test_the_physical_scene_stays_authoritative_on_the_fastening():
    """A symbolic rollout that drifted ahead of the physics cannot pass."""
    assert workshop_goal_reached(_Physical(False), _Runtime(True)) is False


def test_without_physical_execution_the_runtime_predicate_decides():
    """Planning-only episodes have no joint to read, so the rollout is all
    there is -- and it already carries the whole goal."""
    assert workshop_goal_reached(None, _Runtime(True)) is True
    assert workshop_goal_reached(None, _Runtime(False)) is False
