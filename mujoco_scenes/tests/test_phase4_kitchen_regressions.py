import numpy as np
import pytest

from mujoco_scenes.kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from mujoco_scenes.sequential_inspection import INTERFERING_OPEN_REGIONS


def test_c2_then_b1_keeps_c2_open_by_contract():
    # Opening C2 may repair the unsupported reverse order by closing B1 first,
    # but opening B1 after C2 must not imply CLOSE(C2).
    assert INTERFERING_OPEN_REGIONS == {"C2": "B1"}
    assert INTERFERING_OPEN_REGIONS.get("B1") is None


@pytest.mark.parametrize("observed_length_m", [0.16, 0.20, 0.24])
def test_stir_orientation_family_is_vertical_only(observed_length_m):
    # A coffee stir is only logically valid with the utensil axis aligned to the
    # vessel opening normal.  This guarded a three-value inclination family
    # (0/3/5 degrees) under the name _serving_utensil_orientation_family; the
    # Kitchen execution port renamed it to _stir_orientation_family and tightened
    # the family to vertical only, which satisfies the original intent more
    # strictly rather than relaxing it.  The assertion that matters is unchanged:
    # no candidate may reintroduce the old 12--45 degree rim-resting family.
    normal = np.array((0.0, 0.0, 1.0))
    local_axis = np.array((0.0, 0.0, 1.0))
    candidates = KitchenPhaseCExecutionDispatcher._stir_orientation_family(
        np.eye(3),
        local_axis,
        normal,
        np.array((1.0, 0.0, 0.0)),
    )

    assert candidates
    assert candidates[0]["inclination_deg"] == 0.0
    assert {row["inclination_deg"] for row in candidates} == {0.0}

    for row in candidates:
        axis = row["rotation"] @ local_axis
        expected = np.cos(np.deg2rad(row["inclination_deg"]))
        assert float(np.dot(axis, normal)) == pytest.approx(expected, abs=1e-7)
        # No candidate may encode the old 12-45 degree rim-resting family.
        assert row["inclination_deg"] <= 5.0
