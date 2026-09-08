"""Phase-C operator admissibility under the two policies.

The existing Phase-C suite cannot run without `runs/` artifacts that are not
in version control, so these cover the policy split directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mujoco_scenes.kitchen_phase_c_execution import (
    FROZEN_PLAN_ADMISSIBILITY,
    GEOMETRIC_ADMISSIBILITY,
    KitchenPhaseCExecutionDispatcher,
)
from mujoco_scenes.kitchen_pour_stir_manipulation import (
    PhaseCExecutionLedger,
    VerifiedMotionPhaseCLedger,
)


FROZEN_PLAN = [
    {"step": 5, "action": "POUR", "arguments": ["object_0007", "object_0002", "water"]},
    {"step": 9, "action": "STIR", "arguments": ["object_0004", "object_0002"]},
]
REGISTRY = {
    "objects": {
        f"object_{index:04d}": {
            "generic_object_id": f"object_{index:04d}",
            "source_region": "countertop",
        }
        for index in range(1, 10)
    }
}
# a vessel that is still inside a drawer
REGISTRY["objects"]["object_0006"]["source_region"] = "D1"


def make_dispatcher(admissibility: str, plan: list[dict] | None = None):
    phase_b = SimpleNamespace(
        scene=SimpleNamespace(),
        inventory_by_id={},
        binding_by_id={f"object_{index:04d}": {} for index in range(1, 10)},
    )
    return KitchenPhaseCExecutionDispatcher(
        phase_b,
        REGISTRY,
        FROZEN_PLAN if plan is None else plan,
        admissibility=admissibility,
    )


def test_unknown_admissibility_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="admissibility"):
        make_dispatcher("WHATEVER")


def test_frozen_plan_policy_admits_only_planned_pairs() -> None:
    phase_c = make_dispatcher(FROZEN_PLAN_ADMISSIBILITY)
    assert phase_c._pair_admissible("POUR", "object_0007", "object_0002")
    # a functionally valid pair the ground-truth plan did not name
    assert not phase_c._pair_admissible("POUR", "object_0007", "object_0003")
    assert isinstance(phase_c.ledger, PhaseCExecutionLedger)


def test_geometric_policy_admits_any_bound_pair() -> None:
    """A baseline grounds its own pair, so plan membership must not gate it."""

    phase_c = make_dispatcher(GEOMETRIC_ADMISSIBILITY, plan=[])
    assert phase_c._pair_admissible("POUR", "object_0007", "object_0003")
    assert phase_c._pair_admissible("STIR", "object_0004", "object_0003")
    # binding is still required: an unobserved object cannot be poured into
    assert not phase_c._pair_admissible("POUR", "object_0007", "object_9999")
    # and the cupboard rule still applies, as it does to a frozen plan
    assert not phase_c._pair_admissible("POUR", "object_0007", "object_0006")
    assert isinstance(phase_c.ledger, VerifiedMotionPhaseCLedger)


def test_geometric_policy_would_otherwise_admit_nothing_without_a_plan() -> None:
    """The regression this policy exists for: an empty plan blocks every pair."""

    frozen = make_dispatcher(FROZEN_PLAN_ADMISSIBILITY, plan=[])
    assert frozen.expected_pairs == {"POUR": {}, "STIR": {}}
    assert not frozen._pair_admissible("POUR", "object_0007", "object_0002")


def test_verified_motion_ledger_requires_verified_motion() -> None:
    ledger = VerifiedMotionPhaseCLedger()
    assert not ledger.commit(None, {"success": True, "pour_motion_verified": False})
    assert not ledger.commit(None, {"success": False, "pour_motion_verified": True})
    assert ledger.commit(None, {"success": True, "pour_motion_verified": True})
    assert ledger.commit(None, {"success": True, "stir_motion_verified": True})
    assert ledger.summary()["verified_event_count"] == 2


def test_a_retrieved_vessel_becomes_a_legal_pour_target() -> None:
    """A baseline may take a vessel out of a cupboard and then pour into it."""

    phase_c = make_dispatcher(GEOMETRIC_ADMISSIBILITY, plan=[])
    assert not phase_c._pair_admissible("POUR", "object_0007", "object_0006")
    phase_c.phase_b.place = lambda object_id, destination: {"success": True}
    phase_c.place("object_0006", "countertop")
    assert phase_c._pair_admissible("POUR", "object_0007", "object_0006")


def test_placing_a_vessel_back_into_a_drawer_makes_it_illegal_again() -> None:
    phase_c = make_dispatcher(GEOMETRIC_ADMISSIBILITY, plan=[])
    phase_c.phase_b.place = lambda object_id, destination: {"success": True}
    phase_c.place("object_0002", "D1")
    assert not phase_c._pair_admissible("POUR", "object_0007", "object_0002")


def test_a_failed_place_does_not_move_the_vessel() -> None:
    phase_c = make_dispatcher(GEOMETRIC_ADMISSIBILITY, plan=[])
    phase_c.phase_b.place = lambda object_id, destination: {"success": False}
    phase_c.place("object_0006", "countertop")
    assert not phase_c._pair_admissible("POUR", "object_0007", "object_0006")
