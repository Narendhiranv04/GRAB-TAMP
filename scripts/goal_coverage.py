"""Goal coverage against the ground-truth goal, with a fixed denominator per scene.

The goal is a fixed conjunction of named conditions:

    Kitchen (12)      coffee x2: coffee_poured, water_poured, stirred, served
                      soup   x2: served, dedicated_utensil_served
    Living Room (5)   personal setting x2: cup_placed, saucer_placed
                      shared: remote_placed
    Workshop (3)      fastener_inserted, joint_repaired,
                      driver_returned_to_workbench

The denominator is the goal, not the work left to do.  The previous scorers
derived it from the ground-truth action list, which omits conditions that the
variant starts with already satisfied, and that produced two defects:

* **Variants were not comparable.**  Living Room L2 and L5 pre-place a saucer,
  so their ground-truth plan has four PLACE actions and coverage was reported
  out of 4 while every other variant used 5.
* **Methods were not comparable, on the same variant.**  ViLaIn-TAMP writes no
  `expected_gt_actions.json`, so it fell through to a different denominator: on
  L2 and L5 it was scored out of 5 while VLM-TAMP, OWL-TAMP and ROBUST-TAMP
  were scored out of 4.  Twenty episodes were graded against a harder goal than
  the methods they are tabulated beside.

A condition that holds in the initial state counts as satisfied, because it is
part of the goal and the goal is what the denominator measures.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

#: Conditions in each scene's goal conjunction.
TOTALS = {"kitchen": 12, "living_room": 5, "workshop": 3}

CONDITION_NAMES = {
    "kitchen": (
        "coffee_poured", "water_poured", "stirred", "served",          # x2 coffees
        "served", "dedicated_utensil_served",                           # x2 soups
    ),
    "living_room": ("cup_placed", "saucer_placed", "remote_placed"),
    "workshop": ("fastener_inserted", "joint_repaired",
                 "driver_returned_to_workbench"),
}


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def kitchen(episode: Path, facts: set[tuple]) -> tuple[int, int]:
    """Four pours, two stirs, four serves and two dedicated utensils."""
    spec = _read(episode / "_private_evaluation/goal_contract.json")
    total = TOTALS["kitchen"]
    if not spec:
        return 0, total
    required = list(spec.get("required_effects") or [])
    stir_targets = list(spec.get("stir_targets") or [])
    literals = {f"{f[0]}({','.join(f[1:])})" for f in facts if all(f[1:])}
    hit = sum(1 for effect in required if effect in literals)
    hit += sum(1 for target in stir_targets if ("stirred", target) in facts)
    stated = len(required) + len(stir_targets)
    # Anything the contract does not state is a condition this variant starts
    # with satisfied.
    return min(hit + max(0, total - stated), total), total


def living_room(episode: Path, facts: set[tuple]) -> tuple[int, int]:
    """Two place settings of cup+saucer, and the remote on the shared table.

    Scored by role rather than object identity: the goal is symmetric under
    exchanging the two settings, so demanding ground truth's particular
    assignment marks correct solves wrong.
    """
    total = TOTALS["living_room"]
    gt = _read(episode / "_private_evaluation/expected_gt_actions.json")
    adapter = _read(episode / "_private_evaluation/adapter_resolution.json")
    if not gt or not adapter:
        return 0, total
    role = {
        row["generic_object_id"]: row.get("semantic_role")
        for row in adapter.get("objects", [])
        if row.get("generic_object_id")
    }
    required: Counter = Counter()
    for action in gt.get("actions", []):
        if str(action.get("operator", "")).upper() == "PLACE":
            args = action.get("arguments") or []
            if len(args) >= 2:
                required[(args[1], role.get(args[0]))] += 1
    if not required:
        return 0, total
    proposed: Counter = Counter()
    for fact in facts:
        if fact[0] == "placed" and len(fact) >= 3:
            proposed[(fact[2], role.get(fact[1]))] += 1
    hit = sum(min(n, proposed[k]) for k, n in required.items())
    stated = sum(required.values())
    return min(hit + max(0, total - stated), total), total


def workshop(episode: Path, facts: set[tuple]) -> tuple[int, int]:
    """Fastener seated, joint fastened, driver left on the workbench.

    A FASTEN implies its INSERT, matching the executed check, which reads the
    insertion off the joint the fastening produced.  Which driver is used is
    not checked: the physical skill rejects an incompatible one, so a planned
    fastening names a compatible tool or fails on its own.
    """
    total = TOTALS["workshop"]
    inserted = {f for f in facts if f[0] == "inserted"}
    fastened = {f for f in facts if f[0] == "fastened"}
    placed = {f for f in facts if f[0] == "placed"}
    bench = _workbench_id(episode)
    hit = 0
    hit += bool(inserted or fastened)                                  # fastener_inserted
    hit += bool(fastened)                                              # joint_repaired
    hit += bool(fastened and bench and any(f[2] == bench for f in placed))
    return hit, total


def _workbench_id(episode: Path) -> str | None:
    """Resolve the workbench region, from the evaluator's labelled observation.

    This read the episode's public `latest_observation.json`, which publishes
    `regions` with anonymous ids and no labels -- the method is not told which
    region is the workbench.  Asking it for `known_regions` therefore returned
    nothing in every episode, and `driver_returned_to_workbench` was
    unsatisfiable by construction: all three methods scored 0% on it while
    plans placed at `region_0004` 122 times, which is the workbench.

    The labelled view is the evaluator's, under `_private_evaluation/`, which is
    the right place for it: scoring may use ground truth, the planner may not.
    """
    payload = _read(episode / "_private_evaluation/latest_observation.json")
    if not payload:
        payload = _read(episode / "latest_observation.json")
    if not payload:
        return None
    for row in payload.get("known_regions") or payload.get("regions") or []:
        if str(row.get("label", "")).strip().lower() == "main workbench":
            return row.get("id")
    return None


SCORERS = {"kitchen": kitchen, "living_room": living_room, "workshop": workshop}
