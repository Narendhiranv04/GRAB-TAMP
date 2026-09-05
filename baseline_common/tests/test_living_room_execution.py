"""Guards on the adapter that turns baseline actions into physical motion.

Reported physical success rests on an invariant that is easy to break by
accident: the observable location of a payload may only change after the
simulator has confirmed the placement. ``LivingRoomDiscoveryRuntime._place``
enforces that, assigning into ``locations`` only after the controller, the
settle check and the physical ON-relation verification have all passed.

This executor is the only thing standing between a baseline and that runtime.
If it ever mutated observable state itself, or reached past
``execute_phase2_action`` into the runtime's internals, a placement the physics
rejected could still be reported as a success and no existing test would
notice. The tests here pin that boundary.
"""

from __future__ import annotations

import unittest

from baseline_common.living_room_execution import (
    FAILURE_CODES,
    LivingRoomPhysicalExecutor,
)
from baseline_common.models import Action


class RecordingRuntime:
    """Minimal stand-in that records what the executor asked it to do."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[dict] = []
        self.locations = {"object_0001": "staging_area"}

    def execute_phase2_action(self, action):
        self.calls.append(action)
        return self.outcome

    def observe_state(self):  # required by the executor's constructor check
        return None


class PhysicalExecutorBoundaryTests(unittest.TestCase):
    def test_actions_are_routed_through_the_runtime_not_applied_directly(self):
        runtime = RecordingRuntime({"success": True, "effects": ["placed(a,b)"]})
        executor = LivingRoomPhysicalExecutor(runtime)
        result = executor.execute(
            Action("PLACE", {"object_id": "object_0001", "region_id": "region_0002"})
        )
        self.assertTrue(result.success)
        # The single call is the whole point: the executor must delegate, so
        # the runtime's verification cannot be bypassed.
        self.assertEqual(
            runtime.calls,
            [{"action": "PLACE", "arguments": ["object_0001", "region_0002"]}],
        )

    def test_executor_never_mutates_observable_location_itself(self):
        """A failed placement must leave the observable state untouched."""
        runtime = RecordingRuntime(
            {"success": False, "failure_code": "PLACE", "message": "not verified"}
        )
        before = dict(runtime.locations)
        executor = LivingRoomPhysicalExecutor(runtime)
        result = executor.execute(
            Action("PLACE", {"object_id": "object_0001", "region_id": "region_0002"})
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FAILURE_CODES["PLACE"])
        self.assertEqual(runtime.locations, before)

    def test_a_rejected_placement_is_recoverable_so_the_executive_can_replan(self):
        # A skill that refuses a placement leaves the world observable and
        # unchanged, so the episode should continue rather than abort.
        runtime = RecordingRuntime(
            {"success": False, "failure_code": "PLACE", "message": "not verified"}
        )
        result = LivingRoomPhysicalExecutor(runtime).execute(
            Action("PLACE", {"object_id": "object_0001", "region_id": "region_0002"})
        )
        self.assertTrue(result.recoverable)

    def test_unsupported_skills_are_refused_rather_than_silently_dropped(self):
        runtime = RecordingRuntime({"success": True})
        result = LivingRoomPhysicalExecutor(runtime).execute(
            Action("POUR", {"object_id": "object_0001"})
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, "unsupported_subgoal")
        self.assertFalse(result.recoverable)
        self.assertEqual(runtime.calls, [])

    def test_incomplete_arguments_do_not_reach_the_runtime(self):
        runtime = RecordingRuntime({"success": True})
        result = LivingRoomPhysicalExecutor(runtime).execute(
            Action("PLACE", {"object_id": "object_0001"})  # no region
        )
        self.assertFalse(result.success)
        self.assertEqual(runtime.calls, [])

    def test_a_runtime_without_the_execution_contract_is_rejected_at_construction(self):
        class NotARuntime:
            pass

        with self.assertRaises(TypeError):
            LivingRoomPhysicalExecutor(NotARuntime())


if __name__ == "__main__":
    unittest.main()
