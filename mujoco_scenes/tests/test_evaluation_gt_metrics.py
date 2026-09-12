"""The two evaluation metrics must stay two metrics.

Functional Assignment Coverage asks whether the right entities were grounded to
the task's functions.  Goal Coverage asks whether the user's goals hold at the
end.  These are different questions, and the tests that matter most here are the
ones proving they can disagree -- if they always moved together, one of them
would be redundant and the ablation table would be measuring nothing new.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujoco_scenes.evaluation_gt import (
    load_fac_gt, load_goal_gt, score_functional_assignment_coverage,
    score_goal_coverage,
)

REPO = Path(__file__).resolve().parents[2]
FEASIBLE = ([f"K{i}" for i in range(1, 7)] + [f"L{i}" for i in range(1, 7)]
            + [f"W{i}" for i in range(1, 9)])

KITCHEN_OK = {
    "water_source": ["s1i_compact_kettle"],
    "coffee_source": ["s1i_compact_coffee_jar"],
    "coffee_stirrer": ["s1i_final_long_narrow_spoon"],
    "coffee_container": ["ab3_narrow_deep_cup", "ab3_medium_deep_mug"],
    "soup_container": ["ab3_deep_bowl", "ab3_shallow_bowl"],
    "soup_eating_utensil": ["s1i_oversized_spoon", "ab3_partial_spoon"],
}
KITCHEN_BINDINGS_A = {
    "stirs": {"s1i_final_long_narrow_spoon": ["ab3_narrow_deep_cup", "ab3_medium_deep_mug"]},
    "serves_soup_in": {"s1i_oversized_spoon": "ab3_deep_bowl",
                       "ab3_partial_spoon": "ab3_shallow_bowl"},
}
KITCHEN_BINDINGS_B = {
    "stirs": {"s1i_final_long_narrow_spoon": ["ab3_narrow_deep_cup", "ab3_medium_deep_mug"]},
    "serves_soup_in": {"s1i_oversized_spoon": "ab3_shallow_bowl",
                       "ab3_partial_spoon": "ab3_deep_bowl"},
}


# --- coverage and structural guards -----------------------------------------

def test_exactly_the_twenty_feasible_variants():
    for doc in (load_fac_gt(), load_goal_gt()):
        assert sorted(doc["variants"]) == sorted(FEASIBLE)
        assert len(doc["variants"]) == 20


def test_no_infeasible_variant_leaked_in():
    infeasible = {f"K{i}" for i in range(7, 13)} | {f"L{i}" for i in range(7, 11)} \
        | {"W9", "W10"}
    for doc in (load_fac_gt(), load_goal_gt()):
        assert not (set(doc["variants"]) & infeasible)


def test_every_denominator_is_nonzero():
    for variant, spec in load_fac_gt()["variants"].items():
        assert spec["scoring"]["total_scored_slots"] > 0, variant
        assert (spec["scoring"]["unary_slot_count"]
                + spec["scoring"]["relation_slot_count"]
                == spec["scoring"]["total_scored_slots"]), variant
    expected = {"kitchen": 12, "living_room": 5, "workshop": 3}
    for variant, spec in load_goal_gt()["variants"].items():
        assert spec["goal_count"] == expected[spec["domain"]], variant
        assert spec["goal_count"] == sum(
            g["slots"] * len(g["goals"]) for g in spec["goal_groups"]), variant


def test_metadata_is_never_scored():
    """Search provenance must not appear among scored facts."""
    forbidden = ("drawer", "cabinet", "inspection", "opened", "region_inspected",
                 "objects_requiring_movement", "gt_action_count")
    for variant, spec in load_fac_gt()["variants"].items():
        scored = json.dumps({k: v for k, v in spec.items()
                             if k in ("required_role_slots", "valid_role_fillers",
                                      "required_relational_bindings",
                                      "valid_assignment_sets", "scoring")}).lower()
        for token in forbidden:
            assert token not in scored, f"{variant}: {token!r} leaked into scored facts"


def test_workshop_source_regions_are_not_scored_facts():
    for variant in [f"W{i}" for i in range(1, 9)]:
        spec = load_fac_gt()["variants"][variant]
        assert set(spec["required_role_slots"]) == {"driver", "fastener", "repair_target"}
        blob = json.dumps(spec["valid_assignment_sets"])
        for region in ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"):
            assert region not in blob, f"{variant}: {region} is search provenance"


# --- Kitchen ----------------------------------------------------------------

def test_kitchen_perfect_assignment_scores_one():
    out = score_functional_assignment_coverage("kitchen", "K1", KITCHEN_OK, KITCHEN_BINDINGS_A)
    assert out["coverage"] == 1.0
    assert out["correct_slots"] == out["total_slots"] == 13


def test_kitchen_both_soup_matchings_are_equally_valid():
    """The executed GT and the privileged oracle pick different bijections.

    Every utensil x bowl pair passes INSERTABLE_IN and REACHES_BOTTOM, so forcing
    one identity would penalise a method for agreeing with the other GT source.
    """
    a = score_functional_assignment_coverage("kitchen", "K1", KITCHEN_OK, KITCHEN_BINDINGS_A)
    b = score_functional_assignment_coverage("kitchen", "K1", KITCHEN_OK, KITCHEN_BINDINGS_B)
    assert a["coverage"] == b["coverage"] == 1.0


def test_kitchen_sharing_one_utensil_across_bowls_is_not_valid():
    """Dedicated means injective: one utensil cannot serve both bowls."""
    shared = dict(KITCHEN_BINDINGS_A)
    shared["serves_soup_in"] = {"s1i_oversized_spoon": ["ab3_deep_bowl", "ab3_shallow_bowl"]}
    out = score_functional_assignment_coverage("kitchen", "K1", KITCHEN_OK, shared)
    assert out["coverage"] < 1.0


def test_kitchen_stirrer_must_cover_both_cups_because_it_is_reusable():
    partial = dict(KITCHEN_BINDINGS_A)
    partial["stirs"] = {"s1i_final_long_narrow_spoon": ["ab3_narrow_deep_cup"]}
    out = score_functional_assignment_coverage("kitchen", "K1", KITCHEN_OK, partial)
    assert out["relational"]["correct"] == 3 and out["coverage"] < 1.0


def test_kitchen_wrong_object_for_a_role_lowers_coverage():
    wrong = dict(KITCHEN_OK, coffee_stirrer=["s1i_oversized_spoon"])
    out = score_functional_assignment_coverage("kitchen", "K1", wrong, KITCHEN_BINDINGS_A)
    assert out["coverage"] < 1.0
    assert any(i["role"] == "coffee_stirrer" for i in out["incorrect"])


def test_kitchen_semantics_identical_across_k1_to_k6():
    """Variants relocate objects; they do not change which object serves which function."""
    base = load_fac_gt()["variants"]["K1"]
    for variant in [f"K{i}" for i in range(2, 7)]:
        other = load_fac_gt()["variants"][variant]
        assert other["required_role_slots"] == base["required_role_slots"]
        assert other["valid_role_fillers"] == base["valid_role_fillers"]
        assert other["scoring"] == base["scoring"]


# --- Living room ------------------------------------------------------------

@pytest.mark.parametrize("variant", [f"L{i}" for i in range(1, 7)])
def test_living_room_preplacement_never_reduces_the_assignment(variant):
    """L2 and L5 start with a saucer already correct; it is still a role slot."""
    spec = load_fac_gt()["variants"][variant]
    assert spec["required_role_slots"]["cup"] == 2
    assert spec["required_role_slots"]["saucer"] == 2
    assert spec["required_role_slots"]["PERSONAL_CUP_SAUCER_REGION"] == 2
    assert spec["scoring"]["total_scored_slots"] == 18


def test_living_room_l2_and_l5_record_preplacement_only_as_metadata():
    for variant in ("L2", "L5"):
        meta = load_fac_gt()["variants"][variant]["metadata"]
        assert meta["objects_requiring_movement"] == 4
        assert meta["preplaced_objects"]


def test_living_room_payloads_are_cup_and_saucer_not_drink_and_snack():
    """Scene bodies are named a2_drink_*/a2_snack_*; the semantics are cup/saucer."""
    spec = load_fac_gt()["variants"]["L1"]
    assert set(spec["object_semantics"].values()) == {"cup", "saucer", "tv_remote"}
    assert "drink" not in json.dumps(spec["required_role_slots"]).lower()
    assert "snack" not in json.dumps(spec["required_role_slots"]).lower()


def test_living_room_equivalent_cup_saucer_permutation_is_accepted():
    """Which cup pairs with which saucer is a grouping policy, not a requirement."""
    spec = load_fac_gt()["variants"]["L1"]
    cups = spec["valid_role_fillers"]["cup"]
    saucers = spec["valid_role_fillers"]["saucer"]
    regions = spec["valid_role_fillers"]["PERSONAL_CUP_SAUCER_REGION"]
    assignment = {
        "PERSONAL_CUP_SAUCER_REGION": regions,
        "SHARED_REMOTE_REGION": spec["valid_role_fillers"]["SHARED_REMOTE_REGION"],
        "cup": cups, "saucer": saucers,
        "REMOTE": spec["valid_role_fillers"]["REMOTE"],
        "SEATING_POSITION": ["seat_0001", "seat_0002"],
    }
    def bindings(swap):
        c = cups[::-1] if swap else cups
        return {
            "personal_set_on_region": [[regions[0], c[0]], [regions[0], saucers[0]],
                                       [regions[1], c[1]], [regions[1], saucers[1]]],
            "region_near_seat": [[regions[0], "seat_0001"], [regions[1], "seat_0002"]],
            "remote_on_shared": [[spec["valid_role_fillers"]["REMOTE"][0],
                                  spec["valid_role_fillers"]["SHARED_REMOTE_REGION"][0]]],
        }
    straight = score_functional_assignment_coverage("living_room", "L1", assignment, bindings(False))
    swapped = score_functional_assignment_coverage("living_room", "L1", assignment, bindings(True))
    assert straight["relational"]["correct"] == swapped["relational"]["correct"]


# --- Workshop ---------------------------------------------------------------

@pytest.mark.parametrize("variant", [f"W{i}" for i in range(1, 7)])
def test_workshop_both_drivers_valid_where_both_are_present(variant):
    """first_compatible_driver_observed is a search policy, not task semantics."""
    fillers = load_fac_gt()["variants"][variant]["valid_role_fillers"]["driver"]
    assert set(fillers) == {"workshop_long_phillips_driver", "workshop_power_driver"}


def test_workshop_single_driver_variants_are_genuinely_constrained():
    assert load_fac_gt()["variants"]["W7"]["valid_role_fillers"]["driver"] == \
        ["workshop_long_phillips_driver"]
    assert load_fac_gt()["variants"]["W8"]["valid_role_fillers"]["driver"] == \
        ["workshop_power_driver"]


def test_workshop_alternative_driver_scores_full():
    for driver in ("workshop_long_phillips_driver", "workshop_power_driver"):
        out = score_functional_assignment_coverage(
            "workshop", "W1",
            {"driver": [driver], "fastener": ["workshop_medium_phillips_screw"],
             "repair_target": ["workshop_frame_joint"]},
            {"repair_tuple": [[driver, "workshop_medium_phillips_screw"],
                              [driver, "workshop_frame_joint"],
                              ["workshop_medium_phillips_screw", "workshop_frame_joint"]]})
        assert out["coverage"] == 1.0, driver


def test_workshop_incompatible_tool_is_rejected():
    out = score_functional_assignment_coverage(
        "workshop", "W1",
        {"driver": ["workshop_wooden_hammer"],
         "fastener": ["workshop_medium_phillips_screw"],
         "repair_target": ["workshop_frame_joint"]}, {})
    assert out["coverage"] < 1.0
    assert any(i["role"] == "driver" for i in out["incorrect"])


def test_workshop_absent_driver_rejected_in_single_driver_variant():
    out = score_functional_assignment_coverage(
        "workshop", "W7",
        {"driver": ["workshop_power_driver"],
         "fastener": ["workshop_medium_phillips_screw"],
         "repair_target": ["workshop_frame_joint"]}, {})
    assert any(i["role"] == "driver" for i in out["incorrect"])


def test_workshop_source_region_does_not_change_the_score():
    """Same grounding, different provenance -> identical FAC."""
    predicted = {"driver": ["workshop_long_phillips_driver"],
                 "fastener": ["workshop_medium_phillips_screw"],
                 "repair_target": ["workshop_frame_joint"]}
    bindings = {"repair_tuple": [["workshop_long_phillips_driver", "workshop_medium_phillips_screw"],
                                 ["workshop_long_phillips_driver", "workshop_frame_joint"],
                                 ["workshop_medium_phillips_screw", "workshop_frame_joint"]]}
    w1 = score_functional_assignment_coverage("workshop", "W1", predicted, bindings)
    w5 = score_functional_assignment_coverage("workshop", "W5", predicted, bindings)
    assert w1["coverage"] == w5["coverage"] == 1.0


# --- Goal coverage ----------------------------------------------------------

WORKSHOP_SEMANTICS = {"workshop_long_phillips_driver": "driver",
                      "workshop_medium_phillips_screw": "fastener"}


def _workshop_atoms(repaired=True, returned=True):
    atoms = [("inserted", "workshop_medium_phillips_screw", "workshop_frame_joint")]
    if repaired:
        atoms.append(("repaired", "workshop_frame_joint"))
    if returned:
        atoms += [("at", "workshop_long_phillips_driver", "MAIN_WORKBENCH_ZONE"),
                  ("hand_empty",)]
    return atoms


def test_goal_coverage_full():
    out = score_goal_coverage("workshop", "W1", _workshop_atoms(), WORKSHOP_SEMANTICS)
    assert out["coverage"] == 1.0 and out["satisfied_goals"] == out["total_goals"] == 3


KITCHEN_SEMANTICS = {"cup_a": "coffee_container", "cup_b": "coffee_container",
                     "bowl_a": "soup_container", "bowl_b": "soup_container",
                     "spoon_a": "soup_eating_utensil", "spoon_b": "soup_eating_utensil"}


def _kitchen_atoms(stir=True, serve=True, utensils=True):
    atoms = []
    for cup in ("cup_a", "cup_b"):
        atoms += [("contains", cup, "coffee"), ("contains", cup, "water")]
        if stir:
            atoms.append(("stirred", cup))
        if serve:
            atoms.append(("at", cup, "dining_table"))
    for bowl, spoon in (("bowl_a", "spoon_a"), ("bowl_b", "spoon_b")):
        if serve:
            atoms.append(("at", bowl, "dining_table"))
        if utensils:
            atoms.append(("at", spoon, bowl))
    return atoms


def test_kitchen_goal_denominator_is_twelve():
    out = score_goal_coverage("kitchen", "K1", _kitchen_atoms(), KITCHEN_SEMANTICS)
    assert out["total_goals"] == 12 and out["coverage"] == 1.0


def test_pouring_is_credited_even_when_stirring_never_happens():
    """The reason Goal Coverage is flat: partial progress must be visible."""
    out = score_goal_coverage("kitchen", "K1", _kitchen_atoms(stir=False),
                              KITCHEN_SEMANTICS)
    assert out["satisfied_goals"] == 10 and out["total_goals"] == 12
    assert out["goal_status"]["coffee_serving_1.coffee_poured"] is True
    assert out["goal_status"]["coffee_serving_1.water_poured"] is True
    assert out["goal_status"]["coffee_serving_1.stirred"] is False


def test_one_utensil_cannot_serve_two_bowls():
    atoms = [("at", b, "dining_table") for b in ("bowl_a", "bowl_b")]
    atoms += [("at", "spoon_a", "bowl_a"), ("at", "spoon_a", "bowl_b")]
    out = score_goal_coverage("kitchen", "K1", atoms, KITCHEN_SEMANTICS)
    dedicated = [v for k, v in out["goal_status"].items()
                 if k.endswith("dedicated_utensil_served")]
    assert sum(dedicated) == 1, "a shared utensil may only satisfy one serving"


def test_slot_assignment_is_maximised_not_greedy():
    """One finished cup and one untouched cup must score the finished one fully."""
    atoms = [("contains", "cup_a", "coffee"), ("contains", "cup_a", "water"),
             ("stirred", "cup_a"), ("at", "cup_a", "dining_table")]
    out = score_goal_coverage("kitchen", "K1", atoms, KITCHEN_SEMANTICS)
    assert out["satisfied_goals"] == 4


def test_living_room_goal_denominator_is_five():
    semantics = {"t_left": "side_table", "t_right": "side_table",
                 "shared": "coffee_table", "cup1": "cup", "cup2": "cup",
                 "s1": "saucer", "s2": "saucer", "rem": "tv_remote"}
    atoms = [("on", "cup1", "t_left"), ("on", "s1", "t_left"),
             ("on", "cup2", "t_right"), ("on", "s2", "t_right"),
             ("on", "rem", "shared")]
    out = score_goal_coverage("living_room", "L1", atoms, semantics)
    assert out["total_goals"] == 5 and out["coverage"] == 1.0


def test_goal_coverage_partial_plan_stops_after_insertion():
    out = score_goal_coverage("workshop", "W1",
                              _workshop_atoms(repaired=False, returned=False),
                              WORKSHOP_SEMANTICS)
    assert out["satisfied_goals"] == 1 and out["total_goals"] == 3
    assert out["goal_status"]["fastening_1.fastener_inserted"] is True
    assert out["goal_status"]["fastening_1.joint_repaired"] is False


def test_goal_coverage_needs_no_functional_role_vocabulary():
    """An object-direct baseline exposes entities and atoms, never role names."""
    out = score_goal_coverage("workshop", "W1", _workshop_atoms(), WORKSHOP_SEMANTICS)
    assert out["coverage"] == 1.0


def test_goal_already_true_in_the_initial_state_counts_as_satisfied():
    """Goal Coverage measures the terminal state, not the work performed."""
    out = score_goal_coverage("workshop", "W1", _workshop_atoms(), WORKSHOP_SEMANTICS)
    before = score_goal_coverage("workshop", "W1", _workshop_atoms(), WORKSHOP_SEMANTICS)
    assert out == before


def test_search_actions_do_not_affect_goal_coverage():
    noisy = _workshop_atoms() + [("opened", "LEFT_DRAWER"), ("inspected", "TOOL_CABINET")]
    assert score_goal_coverage("workshop", "W1", noisy, WORKSHOP_SEMANTICS)["coverage"] == 1.0


# --- the four independence cases --------------------------------------------

def test_case1_perfect_assignment_and_perfect_goals():
    fac = score_functional_assignment_coverage("kitchen", "K1", KITCHEN_OK, KITCHEN_BINDINGS_A)
    gc = score_goal_coverage("workshop", "W1", _workshop_atoms(), WORKSHOP_SEMANTICS)
    assert fac["coverage"] == 1.0 and gc["coverage"] == 1.0


def test_case2_perfect_assignment_but_partial_goals():
    """The discriminating case: right objects, plan stops before repairing."""
    predicted = {"driver": ["workshop_long_phillips_driver"],
                 "fastener": ["workshop_medium_phillips_screw"],
                 "repair_target": ["workshop_frame_joint"]}
    bindings = {"repair_tuple": [["workshop_long_phillips_driver", "workshop_medium_phillips_screw"],
                                 ["workshop_long_phillips_driver", "workshop_frame_joint"],
                                 ["workshop_medium_phillips_screw", "workshop_frame_joint"]]}
    fac = score_functional_assignment_coverage("workshop", "W1", predicted, bindings)
    gc = score_goal_coverage("workshop", "W1",
                             _workshop_atoms(repaired=False, returned=False),
                             WORKSHOP_SEMANTICS)
    assert fac["coverage"] == 1.0
    assert gc["coverage"] < 1.0


def test_case3_no_functional_assignment_but_perfect_goals():
    """An object-direct baseline exposes no roles yet satisfies every goal."""
    fac = score_functional_assignment_coverage("workshop", "W1", {}, {})
    gc = score_goal_coverage("workshop", "W1", _workshop_atoms(), WORKSHOP_SEMANTICS)
    assert fac["coverage"] == 0.0 and gc["coverage"] == 1.0


def test_case4_partial_assignment_and_partial_goals():
    partial = {"driver": ["workshop_long_phillips_driver"],
               "fastener": ["workshop_medium_phillips_screw"]}
    fac = score_functional_assignment_coverage("workshop", "W1", partial, {})
    gc = score_goal_coverage("workshop", "W1",
                             _workshop_atoms(repaired=False), WORKSHOP_SEMANTICS)
    assert 0.0 < fac["coverage"] < 1.0
    assert 0.0 < gc["coverage"] < 1.0


def test_the_two_metrics_are_not_aliases():
    """If they always agreed, one of them would be measuring nothing new."""
    predicted = {"driver": ["workshop_long_phillips_driver"],
                 "fastener": ["workshop_medium_phillips_screw"],
                 "repair_target": ["workshop_frame_joint"]}
    bindings = {"repair_tuple": [["workshop_long_phillips_driver", "workshop_medium_phillips_screw"],
                                 ["workshop_long_phillips_driver", "workshop_frame_joint"],
                                 ["workshop_medium_phillips_screw", "workshop_frame_joint"]]}
    fac = score_functional_assignment_coverage("workshop", "W1", predicted, bindings)
    gc = score_goal_coverage("workshop", "W1", _workshop_atoms(repaired=False),
                             WORKSHOP_SEMANTICS)
    assert fac["coverage"] != gc["coverage"]


# --- leakage ----------------------------------------------------------------

def test_gt_files_are_not_imported_by_runtime_modules():
    """Neither GT artifact may be reachable from the online pipeline."""
    online = list((REPO / "mujoco_scenes" / "functional_tamp_pipeline").rglob("*.py"))
    online = [p for p in online if "tests" not in p.parts]
    assert online, "expected to find pipeline modules"
    for path in online:
        text = path.read_text()
        for token in ("GT_VALID_ROLE_ASSIGNMENTS", "GT_GOAL_COVERAGE",
                      "gt_valid_role_assignments", "gt_goal_coverage",
                      "evaluation_gt"):
            assert token not in text, f"{path.relative_to(REPO)} references {token}"
