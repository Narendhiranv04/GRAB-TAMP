"""The HOME manipulation stance is absolute, not relative to a stale anchor.

`_home_place_candidates` and the PICK stance branches both derive their
lateral coordinate from the *world* position of the target, so the stance they
return is an absolute HOME-frame pose.  Both call sites used to add it to
`self.executor.base_stance`, which is only a no-op while that anchor is zero.

PICK ends by retreating to navigation home, so it is zero there and PLACE
after PICK worked.  POUR does not: the C2 vessel primitive parks the base at
the pour stance (~0.28 m forward, ~0.26 m lateral) and never restores the
anchor.  The sum then pushed every placement candidate to ~0.5 m forward,
inside the serving table legs, so no stance could be selected and the
placement ran to the 30000-step controller timeout.  Across the Kitchen grid
this was PLACE-after-POUR failing 0/58 while PLACE-after-PICK succeeded 29/36.

These assertions are physical-state free on purpose: they pin the frame, which
is the part that regressed, without paying for a real grasp.
"""

import tempfile

import numpy as np
import pytest

from mujoco_scenes import mobile_motion
from mujoco_scenes.baseline_kitchen_runtime import BaselineKitchenRuntime

# The pour stance the C2 vessel primitive leaves behind, measured on K10.
POUR_STANCE = np.array((0.28, 0.256, 0.0))

# A countertop return target taken from a recorded failing episode.
COUNTERTOP_TARGET = np.array((0.2995, -0.3102, 0.6314))


@pytest.fixture(scope="module")
def manipulation():
    runtime = BaselineKitchenRuntime.from_variant("K10", tempfile.mkdtemp())
    manipulation = runtime.phase_b.manipulation
    manipulation.executor.held_object = next(
        str(row["physical_backend_body"])
        for row in runtime.bundle.resolution.get("accepted", ())
        if "kettle" in str(row["physical_backend_body"])
    )
    return manipulation


def _evaluated_stances(manipulation, anchor, monkeypatch):
    """Return the (forward, lateral) base poses the stance search tried."""
    seen: list[tuple[float, float]] = []
    original = mobile_motion.MuJoCoBaseCollisionChecker.is_pose_valid

    def record(self, x, y, yaw):
        seen.append((round(y - self.home_y, 4), round(-x, 4)))
        return original(self, x, y, yaw)

    monkeypatch.setattr(
        mobile_motion.MuJoCoBaseCollisionChecker, "is_pose_valid", record
    )
    manipulation.executor.base_stance = np.asarray(anchor, dtype=float).copy()
    manipulation._select_home_place_stance(COUNTERTOP_TARGET, None)
    return sorted(set(seen))


def test_place_stance_does_not_drift_with_a_stale_base_anchor(
    manipulation, monkeypatch
):
    from_home = _evaluated_stances(manipulation, np.zeros(3), monkeypatch)
    after_pour = _evaluated_stances(manipulation, POUR_STANCE, monkeypatch)
    assert after_pour == from_home, (
        "The HOME place stance moved with the base anchor a previous action "
        "left behind; it must be an absolute HOME-frame pose."
    )


def test_place_stance_stays_within_the_bounded_home_search_window(
    manipulation, monkeypatch
):
    evaluated = _evaluated_stances(manipulation, POUR_STANCE, monkeypatch)
    forwards = {forward for forward, _ in evaluated}
    assert forwards == {0.20, 0.23, 0.25, 0.28}, forwards
    # 0.48-0.51 m forward is the compounded band that drove the base into
    # `serving_leg_fl` / `serving_leg_fr` on every candidate.
    assert max(forwards) <= 0.28


def test_home_place_candidates_are_target_centred_absolute_stances():
    from mujoco_scenes.kitchen_object_manipulation import (
        KitchenObjectManipulationExecutor,
    )

    rows = KitchenObjectManipulationExecutor._home_place_candidates(
        COUNTERTOP_TARGET
    )
    laterals = {round(lateral, 4) for _, lateral, _ in rows}
    # Derived from the world target, clipped to the reachable lateral band --
    # meaningless as an offset from an arbitrary base pose.
    assert laterals == {-0.18, -0.21, -0.15}
