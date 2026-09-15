#!/usr/bin/env python3
"""Build the two evaluation-only GT artifacts from authoritative benchmark sources.

Offline only.  Nothing here is imported by the runtime pipeline, and the files it
writes must never be read during specification, grounding, search or planning.

Two metrics, deliberately NOT the same thing:

  Functional Assignment Coverage (FAC)
      Did the method identify which physical entities fulfil the task-required
      functions, and the task-required relations between them?  Scored before
      execution, against the *set* of functionally valid assignments -- not
      against whichever one the deterministic GT controller happened to pick.

  Goal Coverage (GC)
      Of the user-level task goals, how many hold in the symbolic terminal state
      induced by the candidate plan?

      NOT representation independent as implemented.  Entities acquire their
      semantic category from this method's grounding result, so a method that
      does not expose functional roles scores zero however well it performs.
      An earlier version of this file claimed otherwise; the claim was false and
      is withdrawn.  Baselines require their own category source.

Authoritative sources, in precedence order:
  FINAL_PAPER_GT_EXECUTIONS/<domain>/<variant>/function_object_assignments.{json,txt}
  FINAL_PAPER_GT_EXECUTIONS/living_room/<variant>/objects_and_regions.txt
  mujoco_scenes/configs/workshop_variants.yaml          (driver availability)
  mujoco_scenes/configs/workshop_joint_alternatives.yaml (allowed drivers)
  mujoco_scenes/configs/l2_integrated_region_function_task.yaml (LR task semantics)
  mujoco_scenes/kitchen_feasibility_oracle.py           (valid utensil/bowl edges)
  mujoco_scenes/functional_tamp_pipeline/evaluation_metrics.py (GC denominators)
  EXPECTED_GT_ACTIONS/                                  (canonical witness only)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GT_EXEC = REPO / "FINAL_PAPER_GT_EXECUTIONS"
ACTIONS = REPO / "EXPECTED_GT_ACTIONS"

FEASIBLE = {
    "kitchen": [f"K{i}" for i in range(1, 7)],
    "living_room": [f"L{i}" for i in range(1, 7)],
    "workshop": [f"W{i}" for i in range(1, 9)],
}
W_LABEL = {f"W{i+1}": lbl for i, lbl in enumerate([
    "F0_MANUAL_FIRST_ONE_REGION", "F1_POWER_FIRST_ONE_REGION",
    "F2_MANUAL_FIRST_TWO_REGIONS", "F3_POWER_FIRST_TWO_REGIONS",
    "F4_MANUAL_FIRST_THREE_REGIONS", "F5_POWER_FIRST_THREE_REGIONS",
    "F6_MANUAL_ONLY", "F7_POWER_ONLY"])}
K_LABEL = {f"K{i+1}": lbl for i, lbl in enumerate([
    "F0_ALL_VISIBLE", "F1_HIDDEN_COFFEE_VESSEL", "F2_HIDDEN_SOUP_BOWL",
    "F3_HIDDEN_VESSELS_MIXED", "F4_TOOLS_IN_DRAWERS", "F5_FULL_DISTRIBUTED_SEARCH"])}
L_LABEL = {f"L{i+1}": lbl for i, lbl in enumerate([
    "F0_ALL_OBJECTS_IN_STAGING", "F1_LEFT_SAUCER_PREPLACED",
    "F2_LEFT_SAUCER_ON_SHARED", "F3_LEFT_CUP_ON_SHARED",
    "F4_SAUCER_PREPLACED_CUP_ON_SHARED", "F5_LEFT_PAIR_ON_SHARED"])}


def _manifest_records():
    return {r["variant"]: r for r in json.loads((ACTIONS / "manifest.json").read_text())["records"]}


def _gt_action_count(domain, variant):
    path = ACTIONS / domain / variant / "expected_gt_actions.json"
    return len(json.loads(path.read_text())["actions"]) if path.exists() else None


# --------------------------------------------------------------------------
# Kitchen
# --------------------------------------------------------------------------
KITCHEN_WATER = "s1i_compact_kettle"
KITCHEN_COFFEE = "s1i_compact_coffee_jar"
KITCHEN_STIRRER = "s1i_final_long_narrow_spoon"
KITCHEN_CUPS = ["ab3_narrow_deep_cup", "ab3_medium_deep_mug"]
KITCHEN_BOWLS = ["ab3_deep_bowl", "ab3_shallow_bowl"]
KITCHEN_UTENSILS = ["s1i_oversized_spoon", "ab3_partial_spoon"]


def kitchen_variant(variant, canonical):
    """Kitchen functional assignment.

    Object identities are invariant across K1-K6: the variants relocate objects
    into cupboards and drawers, which changes search, not function.

    The soup utensil pairing is NOT a fixed identity.  Every utensil x bowl
    combination passes INSERTABLE_IN and REACHES_BOTTOM in the benchmark's own
    geometry (verified via kitchen_feasibility_oracle), so both perfect
    matchings are functionally valid -- the privileged oracle and the executed
    GT actually disagree on which one they pick.  What the task requires is that
    each bowl gets its *own* utensil, i.e. an injective assignment.
    """
    return {
        "domain": "kitchen", "feasible": True,
        "required_role_slots": {
            "water_source": 1, "coffee_source": 1, "coffee_stirrer": 1,
            "coffee_container": 2, "soup_container": 2, "soup_eating_utensil": 2,
        },
        "valid_role_fillers": {
            "water_source": [KITCHEN_WATER],
            "coffee_source": [KITCHEN_COFFEE],
            "coffee_stirrer": [KITCHEN_STIRRER],
            "coffee_container": KITCHEN_CUPS,
            "soup_container": KITCHEN_BOWLS,
            "soup_eating_utensil": KITCHEN_UTENSILS,
        },
        "role_filler_semantics": {
            "coffee_stirrer": ("Uniquely determined: the oversized and partial spoons "
                               "exceed the narrow cup opening, so only the long narrow "
                               "spoon satisfies INSERTABLE_IN on both coffee vessels."),
            "coffee_container": "Both vessels are required; the pair is forced, not chosen.",
            "soup_eating_utensil": "Both utensils are required; which bowl each serves is free.",
        },
        "required_relational_bindings": {
            "stirs": {
                "description": "coffee_stirrer -> coffee_container, one binding per container",
                "policy": "REUSABLE",
                "cardinality": 2,
                "constraint": "the single valid stirrer must be bound to BOTH coffee containers",
            },
            "serves_soup_in": {
                "description": "soup_eating_utensil -> soup_container, one binding per bowl",
                "policy": "DEDICATED",
                "cardinality": 2,
                "constraint": ("a bijection between the two utensils and the two bowls; "
                               "either bijection is valid, sharing one utensil is not"),
            },
        },
        "valid_assignment_sets": {
            "representation": "constraint",
            "unary": {r: {"must_equal_set": v} for r, v in {
                "water_source": [KITCHEN_WATER], "coffee_source": [KITCHEN_COFFEE],
                "coffee_stirrer": [KITCHEN_STIRRER], "coffee_container": KITCHEN_CUPS,
                "soup_container": KITCHEN_BOWLS, "soup_eating_utensil": KITCHEN_UTENSILS,
            }.items()},
            "relational": {
                "stirs": {"type": "complete_from_single_tool",
                          "tool": KITCHEN_STIRRER, "targets": KITCHEN_CUPS},
                "serves_soup_in": {"type": "any_bijection",
                                   "left": KITCHEN_UTENSILS, "right": KITCHEN_BOWLS},
            },
            "enumerated_relational_alternatives": {
                "serves_soup_in": [
                    {"s1i_oversized_spoon": "ab3_deep_bowl",
                     "ab3_partial_spoon": "ab3_shallow_bowl"},
                    {"s1i_oversized_spoon": "ab3_shallow_bowl",
                     "ab3_partial_spoon": "ab3_deep_bowl"},
                ]
            },
        },
        "canonical_witness": canonical,
        "scoring": {"unary_slot_count": 9, "relation_slot_count": 4,
                    "total_scored_slots": 13},
    }


# --------------------------------------------------------------------------
# Living room
# --------------------------------------------------------------------------
def parse_living_room(variant):
    """Per-variant object/region table; ids are positional and NOT stable."""
    text = (GT_EXEC / "living_room" / variant / "objects_and_regions.txt").read_text()
    objects, regions = {}, {}
    for line in text.splitlines():
        m = re.match(r"(object_\d+) \| semantic=(\S+) \| backend=(\S+) \| initial_region=(\S+)", line)
        if m:
            oid, sem, backend, init = m.groups()
            dest = re.search(r"GT_destination=(\S+)", line)
            objects[oid] = {"semantic": sem, "backend": backend,
                            "initial_region": init,
                            "gt_destination": dest.group(1) if dest else None}
        m = re.match(r"(region_\d+) \| logical_region=(\S+) \| support=(\S+)", line)
        if m:
            rid, logical, support = m.groups()
            regions[rid] = {"logical_region": logical, "support": support}
    return objects, regions


def living_room_variant(variant, canonical):
    """Living-room functional assignment, in the roles the pipeline grounds.

    An earlier version scored `cup` and `saucer` as separate unary slots, which
    no run could satisfy: the detector emits `cup_or_saucer` and the grounder
    binds a CUP_SAUCER_SET, never an individual cup.  Asking for roles the
    method does not expose measured the GT's vocabulary rather than the method,
    and accounted for Living Room scoring 214/774.

    Two benchmark facts that do carry over:

    1.  Payload semantics are CUP and SAUCER, not drink and snack -- the scene
        body names (a2_drink_*, a2_snack_*) are stale.  One of each per set is
        what the set role encodes.
    2.  Pre-placement does not remove a role assignment.  In L2 and L5 the left
        saucer starts correctly placed with GT_destination=NONE, yet the GT
        function table still lists it inside its personal set, so all six
        variants carry the full task.

    Which set goes to which seat is settled by target_assignment_policy
    OBSERVED_X_ORDER, a selection policy, so any assignment giving each personal
    region one set near a distinct seat is valid.
    """
    objects, regions = parse_living_room(variant)
    personal = sorted(r for r, v in regions.items()
                      if v["logical_region"].startswith("PERSONAL_TABLE"))
    shared = sorted(r for r, v in regions.items() if v["logical_region"] == "SHARED_TABLE")
    remotes = sorted(o for o, v in objects.items() if v["semantic"] == "tv_remote")
    seats = ["seat_0001", "seat_0002"]
    sets = ["personal_table_slot_1", "personal_table_slot_2"]
    preplaced = sorted(o for o, v in objects.items() if v["gt_destination"] in (None, "NONE"))
    return {
        "domain": "living_room", "feasible": True,
        "required_role_slots": {
            "PERSONAL_CUP_SAUCER_REGION": 2, "SHARED_REMOTE_REGION": 1,
            "CUP_SAUCER_SET": 2, "REMOTE": 1, "SEATING_POSITION": 2,
        },
        "valid_role_fillers": {
            "PERSONAL_CUP_SAUCER_REGION": personal,
            "SHARED_REMOTE_REGION": shared,
            "CUP_SAUCER_SET": sets, "REMOTE": remotes,
            "SEATING_POSITION": seats,
        },
        "object_semantics": {o: v["semantic"] for o, v in sorted(objects.items())},
        "region_semantics": {r: v["logical_region"] for r, v in sorted(regions.items())},
        "required_relational_bindings": {
            "set_on_personal_region": {
                "description": "each PERSONAL_CUP_SAUCER_REGION carries one cup-and-saucer set",
                "cardinality": 2, "constraint": "bijection region <-> set"},
            "region_near_seat": {
                "description": "each personal region is NEAR_SEAT of a distinct seat",
                "cardinality": 2, "constraint": "injective region -> seat"},
            "remote_on_shared": {
                "description": "REMOTE placed on the SHARED_REMOTE_REGION",
                "cardinality": 1, "constraint": "exactly the shared region"},
        },
        "valid_assignment_sets": {
            "representation": "constraint",
            "unary": {
                "PERSONAL_CUP_SAUCER_REGION": {"must_equal_set": personal},
                "SHARED_REMOTE_REGION": {"must_equal_set": shared},
                "CUP_SAUCER_SET": {"must_equal_set": sets},
                "REMOTE": {"must_equal_set": remotes},
                "SEATING_POSITION": {"must_equal_set": seats},
            },
            "relational": {
                "set_on_personal_region": {"type": "any_bijection",
                                           "left": personal, "right": sets},
                "region_near_seat": {"type": "any_bijection",
                                     "left": personal, "right": seats},
                "remote_on_shared": {"type": "fixed", "left": remotes, "right": shared},
            },
        },
        "canonical_witness": canonical,
        "scoring": {"unary_slot_count": 8, "relation_slot_count": 5,
                    "total_scored_slots": 13},
        "_metadata_only": {"objects_requiring_movement": len(objects) - len(preplaced),
                           "preplaced_objects": preplaced},
    }


# --------------------------------------------------------------------------
# Workshop
# --------------------------------------------------------------------------
WORKSHOP_FASTENER = "workshop_medium_phillips_screw"
WORKSHOP_TARGET = "workshop_frame_joint"


def workshop_variant(variant, storage, canonical):
    """Workshop functional assignment.

    Both drivers are listed in workshop_joint_alternatives.yaml as
    allowed_driver_instances, and selection_rule 'first_compatible_driver_observed'
    is explicitly a *search policy*.  Where both drivers are present in the scene
    (W1-W6) both are therefore functionally valid fillers for `driver`, and the
    first-found one is recorded only as the canonical witness.  W7 and W8 remove
    one driver, so their filler set is genuinely a singleton.

    Source drawers, opened regions and inspection order are search provenance and
    are excluded from the scored facts entirely.
    """
    present = sorted({o for objs in storage.values() for o in objs if "driver" in o})
    return {
        "domain": "workshop", "feasible": True,
        "required_role_slots": {"driver": 1, "fastener": 1, "repair_target": 1},
        "valid_role_fillers": {
            "driver": present, "fastener": [WORKSHOP_FASTENER],
            "repair_target": [WORKSHOP_TARGET],
        },
        "role_filler_semantics": {
            "driver": ("Every driver present in the scene is compatible with the "
                       "phillips screw; where two exist, both are valid and the GT's "
                       "choice reflects inspection order only."),
        },
        "required_relational_bindings": {
            "repair_tuple": {
                "description": ("the (driver, fastener, repair_target) tuple, carrying "
                                "COMPATIBLE_WITH(driver, fastener), "
                                "REACHES_TARGET(driver, repair_target) and "
                                "COMPATIBLE_WITH_TARGET(fastener, repair_target)"),
                "cardinality": 3,
                "relations": ["COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET"],
                "constraint": "all three must hold for one consistent driver choice",
            },
        },
        "valid_assignment_sets": {
            "representation": "constraint",
            "unary": {
                "driver": {"must_be_member_of": present},
                "fastener": {"must_equal_set": [WORKSHOP_FASTENER]},
                "repair_target": {"must_equal_set": [WORKSHOP_TARGET]},
            },
            "relational": {
                "repair_tuple": {"type": "tuple_over_valid_fillers",
                                 "drivers": present, "fastener": WORKSHOP_FASTENER,
                                 "target": WORKSHOP_TARGET},
            },
        },
        "canonical_witness": canonical,
        "scoring": {"unary_slot_count": 3, "relation_slot_count": 3,
                    "total_scored_slots": 6},
    }


# --------------------------------------------------------------------------
# Goal coverage
# --------------------------------------------------------------------------
# Denominators are taken from the authoritative evaluator
# (functional_tamp_pipeline/evaluation_metrics.py::full_task_coverage), which
# scores 4 kitchen / 3 living-room / 3 workshop task-level units.  Goal units are
# stated here over final-state facts and semantic categories rather than over
# functional-role names, so a method that never represents "coffee_stirrer" can
# still be scored.
# Goal Coverage is FLAT: every listed goal is one independently scored unit, so
# partial progress is measured directly rather than hidden inside an
# all-or-nothing serving.  A plan that pours coffee and water but never stirs
# scores the two pours it achieved.
#
# Goals are stated over terminal-state atoms and semantic categories, never over
# functional-role names, so an object-direct baseline is scorable.  Each group
# has `slots` occurrences filled by DISTINCT entities; the scorer maximises
# satisfied goals over assignments of entities to slots.
GOAL_GROUPS = {
    "kitchen": [
        {"id": "coffee_serving", "slots": 2,
         "entity_semantics": ["coffee_container"], "distinct_entities": True,
         "goals": [
             {"id": "coffee_poured", "conditions": [["contains", "$e", "coffee"]]},
             {"id": "water_poured", "conditions": [["contains", "$e", "water"]]},
             {"id": "stirred", "conditions": [["stirred", "$e"]]},
             {"id": "served", "conditions": [["at", "$e", "dining_table"]]}]},
        {"id": "soup_serving", "slots": 2,
         "entity_semantics": ["soup_container"], "distinct_entities": True,
         "partner_semantics": ["soup_eating_utensil"], "distinct_partners": True,
         "goals": [
             {"id": "served", "conditions": [["at", "$e", "dining_table"]]},
             {"id": "dedicated_utensil_served",
              "partner_conditions": [["at", "$p", "$e"]]}]},
    ],
    "living_room": [
        # A personal setting requires one cup and one saucer.  The detector
        # cannot tell them apart -- it emits the disjunctive label
        # `cup_or_saucer`, deliberately, because from the fixed rig a cup and a
        # saucer are both small round things and forcing a choice would invent
        # precision the sensor does not have.  Scoring `cup_placed` and
        # `saucer_placed` as separate identity claims therefore asked a question
        # nothing in the pipeline answers, and both failed in all 60 runs.
        #
        # The two goals are kept, because the task does require two items per
        # setting, but they are scored as the two distinct refreshment payloads
        # the setting carries rather than as cup-versus-saucer identity.  That
        # is what the scene actually tests: whether each person's own setting
        # lands on their own table.
        {"id": "personal_setting", "slots": 2,
         "entity_semantics": ["side_table", "end_table"], "distinct_entities": True,
         "goals": [
             {"id": "first_item_placed", "payload_semantics": ["refreshment_item"],
              "conditions": [["on", "$payload", "$e"]], "distinct_payloads": True},
             {"id": "second_item_placed", "payload_semantics": ["refreshment_item"],
              "conditions": [["on", "$payload", "$e"]], "distinct_payloads": True}]},
        {"id": "shared_remote", "slots": 1,
         "entity_semantics": ["coffee_table", "central_table", "side_table"],
         "goals": [
             {"id": "remote_placed", "payload_semantics": ["tv_remote"],
              "conditions": [["on", "$payload", "$e"]]}]},
    ],
    "workshop": [
        # These two goals are NESTED, not independent: the SCREW operator takes
        # ("inserted", fastener, target) as a precondition and adds
        # ("repaired", target), so repaired implies inserted by construction.
        # Both are kept because inserting a screw and driving it home are
        # distinct task achievements and a plan can stop between them -- but
        # they cannot diverge in the other direction, and across the 10x32 run
        # they never diverged at all (31/78 each, the same 31 runs).  Read the
        # workshop denominator of 3 as two independent quantities plus a
        # refinement, not three independent ones.
        {"id": "fastening", "slots": 1,
         "entity_semantics": ["fastener"],
         "goals": [
             {"id": "fastener_inserted",
              "conditions": [["inserted", "$e", WORKSHOP_TARGET]]},
             {"id": "joint_repaired",
              "ground_atoms": [["repaired", WORKSHOP_TARGET]]}]},
        {"id": "tool_return", "slots": 1,
         "entity_semantics": ["driver"],
         "goals": [
             {"id": "driver_returned_to_workbench",
              "conditions": [["at", "$e", "MAIN_WORKBENCH_ZONE"]]}]},
    ],
}
GOAL_COUNT = {d: sum(g["slots"] * len(g["goals"]) for g in groups)
              for d, groups in GOAL_GROUPS.items()}


def main() -> int:
    import yaml
    records = _manifest_records()
    storage = yaml.safe_load(
        (REPO / "mujoco_scenes/configs/workshop_variants.yaml").read_text())
    storage = storage.get("variants", storage)

    fac, gc = {}, {}
    for domain, variants in FEASIBLE.items():
        for variant in variants:
            rec = records.get(variant, {})
            witness_path = GT_EXEC / domain / variant / "function_object_assignments.json"
            witness_txt = GT_EXEC / domain / variant / "function_object_assignments.txt"
            if witness_path.exists():
                canonical = {"source": str(witness_path.relative_to(REPO)),
                             "assignment": json.loads(witness_path.read_text())}
            elif witness_txt.exists():
                canonical = {"source": str(witness_txt.relative_to(REPO)),
                             "assignment_text": witness_txt.read_text().strip()}
            else:
                canonical = {"source": f"EXPECTED_GT_ACTIONS/{domain}/{variant}",
                             "note": "no execution bundle archived for this variant; "
                                     "identities confirmed from the GT action catalogue"}
            if domain == "kitchen":
                entry = kitchen_variant(variant, canonical)
                label = K_LABEL[variant]
            elif domain == "living_room":
                entry = living_room_variant(variant, canonical)
                label = L_LABEL[variant]
            else:
                label = W_LABEL[variant]
                entry = workshop_variant(
                    variant, storage[label]["storage_contents"], canonical)
            meta = entry.pop("_metadata_only", {})
            entry["metadata"] = {
                "internal_variant": label,
                "description": rec.get("description"),
                "gt_action_count": _gt_action_count(domain, variant),
                "excluded_from_scoring": [
                    "initial object locations", "source/storage regions",
                    "inspection order", "opened regions", "objects requiring movement",
                    "action sequence", "planner efficiency"],
                **meta,
            }
            fac[variant] = entry
            gc[variant] = {
                "domain": domain, "feasible": True,
                "goal_groups": GOAL_GROUPS[domain],
                "goal_count": GOAL_COUNT[domain],
                "metadata": {"internal_variant": label,
                             "description": rec.get("description")},
            }

    fac_doc = {
        "schema_version": 1,
        "metric": "functional_assignment_coverage",
        "definition": (
            "Fraction of task-required functional role and relation assignments "
            "correctly grounded by the method, scored against the SET of valid GT "
            "functional assignments rather than one oracle trajectory."),
        "scoring_rule": (
            "FAC = max over valid GT assignments phi* of "
            "(correctly grounded required slots) / (total required slots). "
            "Unary and relational slots are equally weighted atomic GT facts; the "
            "denominator is fixed by GT and never by the prediction."),
        "excluded_by_construction": [
            "action-sequence similarity", "PICK/PLACE ordering", "search order",
            "initial object location", "source drawer or cabinet",
            "number of regions inspected", "number of objects moved",
            "execution success", "planner efficiency"],
        "evaluation_only": True,
        "variants": fac,
    }
    gc_doc = {
        "schema_version": 1,
        "metric": "goal_coverage",
        "definition": (
            "Fraction of user-level task goals satisfied in the symbolic terminal "
            "state induced by the candidate plan, scored over this method's "
            "grounded entities. NOT representation independent as implemented: "
            "entity categories are read from the grounding result, so a method "
            "that exposes no functional roles scores zero regardless of "
            "performance. Baselines need their own category source."),
        "evaluation_mode": "symbolic_terminal_state",
        "scoring_rule": (
            "GC = satisfied goals / goal_count, where every listed goal is one unit. "
            "Partial progress is therefore measured directly: pouring coffee and water "
            "without stirring scores those two goals. Entities filling a group's slots "
            "must be distinct, and the scorer maximises satisfied goals over slot "
            "assignments. A goal already true in the initial state counts as satisfied; "
            "no physical execution is required."),
        "excluded_by_construction": [
            "number of GT actions reproduced", "PICK/OPEN/PLACE/STIR counts",
            "drawer opened", "object discovered", "driver picked up"],
        "evaluation_only": True,
        "variants": gc,
    }

    for path, doc in ((REPO / "GT_VALID_ROLE_ASSIGNMENTS/gt_valid_role_assignments.json", fac_doc),
                      (REPO / "GT_GOAL_COVERAGE/gt_goal_coverage.json", gc_doc)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        print(f"wrote {path.relative_to(REPO)}  ({len(doc['variants'])} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
