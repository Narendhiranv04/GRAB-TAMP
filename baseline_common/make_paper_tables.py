"""Emit the paper's LaTeX tables from run artifacts.

Every number is derived from files on disk, so the tables can be regenerated as
a grid fills in rather than transcribed by hand.  Cells with no supporting run
are emitted as ``--`` instead of being silently omitted or guessed.

Goal coverage is recomputed here rather than read from a field: the Living Room
goal is functional and symmetric -- each personal support needs some cup and
some saucer, not a named one -- so partial credit is a role-matching count that
the boolean goal verifier does not preserve.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


# Support geom -> the roles that support must hold for a satisfied goal.
LIVING_ROOM_GOAL_REQUIREMENTS = {
    "a2_personal_left_top": ("cup", "saucer"),
    "a2_personal_right_top": ("cup", "saucer"),
    "a2_control_table_top": ("tv_remote",),
}
NO_PLAN_STATUSES = {
    "NO_PLAN", "NO_SYMBOLIC_PLAN", "NO_CONTINUOUS_PLAN",
    "INVALID_MODEL_OUTPUT", "NO_RETRIEVED_ROLE_FILLER",
}
# Deliberately absent from NO_PLAN_STATUSES: these are excluded from the
# plan-rate denominator entirely rather than counted as a failure to plan.
# Both are infrastructure faults rather than planning outcomes -- a generation
# stopped by our token ceiling, and a model server that could not be reached at
# all.  Counting either against the method would report our own harness as its
# inability to plan.  See _main_row.
TRUNCATED_STATUS = "MODEL_OUTPUT_TRUNCATED"
INFRASTRUCTURE_STATUSES = {TRUNCATED_STATUS, "INFERENCE_FAILED"}
# Each row is (label, failure codes, terminal statuses).  A physical failure
# can arrive by either route: a runner that records `terminal_failure` supplies
# the lower-case code, while one that does not leaves the code absent and the
# only evidence is the terminal status -- which `physical_terminal_status`
# builds by upper-casing that same code.  Both spellings are therefore listed,
# so an episode categorises identically whichever runner produced it.  Without
# the upper-case forms every physical failure from a runner that omits
# `terminal_failure` fell through into the uncategorised bucket.
# The Living Room executor emits six failure codes and all six are covered
# here.  The shared MuJoCoSkillDispatcher, which the Kitchen path runs through,
# emits thirteen (mujoco_scenes.tamp.skills.FailureCode); nine of them had no
# category until 2026-09-05, so a Kitchen grid would have dropped most of its
# failures into the uncategorised bucket the way OWL-TAMP's did.  Kitchen has no
# execution data yet, so this is prophylactic rather than a correction to any
# reported number.
FAILURE_CATEGORIES = (
    ("No valid functional assignment",
     {"no_valid_subgoals", "no_candidate", "object_not_visible"},
     {"NO_RETRIEVED_ROLE_FILLER", "NO_CANDIDATE", "OBJECT_NOT_VISIBLE"}),
    ("Geometric incompatibility",
     {"unsupported_subgoal", "target_occupied", "function_unsatisfied"},
     {"NO_CONTINUOUS_PLAN", "UNSUPPORTED_SUBGOAL", "TARGET_OCCUPIED",
      "FUNCTION_UNSATISFIED"}),
    ("TAMP refinement failure", {"tamp_refinement_failed"}, {"NO_SYMBOLIC_PLAN"}),
    ("IK or collision failure",
     {"execution_failed", "internal_error", "ik_failed", "collision",
      "path_blocked"},
     {"EXECUTION_FAILED", "INTERNAL_ERROR", "IK_FAILED", "COLLISION",
      "PATH_BLOCKED"}),
    ("Grasp or placement failure", {"grasp_failed", "placement_failed"},
     {"GRASP_FAILED", "PLACEMENT_FAILED"}),
    ("Precondition violation", {"hand_not_empty", "precondition_failed"},
     {"HAND_NOT_EMPTY", "PRECONDITION_FAILED"}),
    ("Final-state verification failure", {"effect_not_observed"},
     {"PLAN_EXHAUSTED_GOAL_NOT_SATISFIED", "EFFECT_NOT_OBSERVED"}),
    ("Model-call budget exhausted", set(), {"MODEL_CALL_BUDGET_EXHAUSTED"}),
    ("Model output truncated", {"model_output_truncated"}, {"MODEL_OUTPUT_TRUNCATED"}),
    # Kept separate from truncation on purpose.  OWL-TAMP reports every
    # unusable completion as INVALID_MODEL_OUTPUT, and truncation is only one
    # of its causes; folding the two together would report a token-ceiling
    # fault as a malformed-output failure, or the reverse.  Whether OWL-TAMP's
    # truncations should be exempt from its planning budget the way VLM-TAMP's
    # already are is an open question, and merging these rows would hide the
    # evidence needed to answer it.
    ("Invalid model output", {"invalid_vlm_output"}, {"INVALID_MODEL_OUTPUT"}),
)


def _goal_coverage(episode_dir: Path) -> float | None:
    """Fraction of the goal's role requirements met by the final state."""
    observation = episode_dir / "latest_observation.json"
    resolution = episode_dir / "_private_evaluation" / "adapter_resolution.json"
    if not (observation.is_file() and resolution.is_file()):
        return None
    final = json.loads(observation.read_text(encoding="utf-8"))
    adapter = json.loads(resolution.read_text(encoding="utf-8"))
    roles = {
        str(row["generic_object_id"]): str(row["semantic_role"])
        for row in adapter.get("objects", ())
    }
    supports = {
        str(row["generic_region_id"]): str(row["backend_support_geom"])
        for row in adapter.get("regions", ())
    }
    held: dict[str, list[str]] = defaultdict(list)
    for item in final.get("visible_objects", ()):
        region = (item.get("facts") or {}).get("region_id")
        role = roles.get(str(item.get("id")))
        if region and role:
            held[supports.get(str(region), str(region))].append(role)
    required = satisfied = 0
    for support, needed in LIVING_ROOM_GOAL_REQUIREMENTS.items():
        available = list(held.get(support, ()))
        for role in needed:
            required += 1
            if role in available:
                available.remove(role)  # one object cannot fill two roles
                satisfied += 1
    return satisfied / required if required else None


def _load(root: Path) -> list[dict[str, Any]]:
    episodes = []
    for path in sorted(root.rglob("benchmark_execution_result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        episode = path.parent
        planning = episode / "episode_result.json"
        comparison = {}
        if planning.is_file():
            comparison = json.loads(planning.read_text(encoding="utf-8")).get(
                "gt_comparison"
            ) or {}
        record["_expected_outcome"] = comparison.get("expected_outcome")
        record["_predicted_outcome"] = comparison.get("predicted_outcome")
        record["_outcome_match"] = comparison.get("outcome_match")
        record["_goal_coverage"] = _goal_coverage(episode)
        record["_failure_code"] = (record.get("terminal_failure") or {}).get("code")
        episodes.append(record)
    return episodes


def _pct(values: list[bool]) -> str:
    return f"{100.0 * mean(values):.1f}" if values else "--"


def _avg(values: list[float]) -> str:
    return f"{mean(values):.1f}" if values else "--"


def _main_row(name: str, episodes: list[dict[str, Any]], bold: bool = False) -> str:
    feasible = [e for e in episodes if e["_expected_outcome"] == "FEASIBLE"]
    infeasible = [e for e in episodes if e["_expected_outcome"] == "INFEASIBLE"]
    scored = [e for e in episodes if e["_outcome_match"] is not None]
    # A generation stopped by the token ceiling produced no plan, but it is the
    # harness's budget that stopped it, not the planner.  Counting it as a
    # planning failure would report our own ceiling as the method's inability
    # to plan, so it is excluded from the plan-rate denominator rather than
    # counted against it -- the same treatment the executive already gives it
    # when refusing to charge the model-call budget.  These episodes are still
    # reported, in their own failure-analysis row.
    plan_scored = [
        e for e in episodes if e["terminal_status"] not in INFRASTRUCTURE_STATUSES
    ]
    cells = [
        _pct([bool(e["_outcome_match"]) for e in scored]),
        _pct([bool(e["success"]) for e in feasible]),
        _avg([100.0 * e["_goal_coverage"] for e in feasible if e["_goal_coverage"] is not None]),
        _pct([e["_predicted_outcome"] == "FEASIBLE" for e in infeasible]),
        _pct([e["terminal_status"] not in NO_PLAN_STATUSES for e in plan_scored]),
        _avg([float(e["raw_vlm_requests"]) for e in episodes]),
        _avg([float(e["replans"]) for e in episodes]),
    ]
    if bold:
        name = f"\\textbf{{{name}}}"
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return f"& {name} & " + " & ".join(cells) + r" \\"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--ablation", type=Path, help="gt_evidence_ablation summary.csv")
    parser.add_argument("--domain-label", default="Living room")
    args = parser.parse_args()

    episodes: list[dict[str, Any]] = []
    for root in args.roots:
        episodes.extend(_load(root.resolve()))
    if not episodes:
        raise SystemExit("No benchmark_execution_result.json found")

    builds = {e.get("mujoco_version", "unrecorded") for e in episodes}
    if len(builds) > 1:
        raise SystemExit(f"Refusing to pool MuJoCo builds {sorted(builds)}")

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_method[str(episode["method"])].append(episode)

    # `discovery_replanning` is ROBUST-TAMP: visible-state-only planning with
    # discovery treated as a first-class replanning event.  It was previously
    # labelled "Ours (single FM call)" and bolded, which put a comparison
    # method in the contribution row.  The proposed method is the
    # functional-requirement + geometric-verification pipeline, which has its
    # own key below and is the row that gets bolded.
    labels = {
        "vlm_tamp": "VLM-TAMP (single-shot)",
        "owl_tamp": "OWL-TAMP (single-shot)",
        "retrieval": "Retrieval (CLIP, no FM)",
        "vilain_tamp": "ViLaIn-TAMP",
        "discovery_replanning": "ROBUST-TAMP",  # legacy artifacts
        "robust_tamp": "ROBUST-TAMP",
        "functional_tamp": "Ours (functional + geometric)",
    }
    print(f"% Generated from: {', '.join(str(r) for r in args.roots)}")
    print(f"% MuJoCo {builds.pop()};  {len(episodes)} episodes")
    variants = sorted({e["variant"] for e in episodes})
    print(f"% Variants present: {', '.join(variants)}")
    if not any(e["_expected_outcome"] == "INFEASIBLE" for e in episodes):
        print("% WARNING: no GT-infeasible variants present -> outcome-correct and")
        print("%          false-completion columns are not measurable from this run.")
    print()
    # Baselines first, proposed method last and bolded, which is also the order
    # a reader expects in the table.
    for key in (
        "vlm_tamp",
        "owl_tamp",
        "retrieval",
        "vilain_tamp",
        "robust_tamp",
        "functional_tamp",
    ):
        rows = by_method.get(key, [])
        counts = Counter(e["variant"] for e in rows)
        suffix = "" if rows else "   % no runs"
        print(_main_row(labels[key], rows, bold=key == "functional_tamp") + suffix)
        if rows:
            print(f"%   n={len(rows)}  per-variant={dict(sorted(counts.items()))}")

    print()
    print("% ---- Table IV: failure analysis (unsuccessful physical trials) ----")
    failures = [e for e in episodes if not e["success"]]
    print(f"% total unsuccessful trials: {len(failures)}")
    # One episode, one category.  terminal_status records how the loop ended
    # and the failure code records why, so an episode that ran out of model
    # calls after proposing nothing refinable carries both; counting it twice
    # would make the column sum exceed the number of failed trials.  The cause
    # wins, and the ordering above puts causes before terminal conditions.
    assigned: dict[int, str] = {}
    for label, codes, statuses in FAILURE_CATEGORIES:
        for episode in failures:
            if id(episode) in assigned:
                continue
            if episode["_failure_code"] in codes or episode["terminal_status"] in statuses:
                assigned[id(episode)] = label
    tally = Counter(assigned.values())
    for label, _codes, _statuses in FAILURE_CATEGORIES:
        print(f"{label} & {tally.get(label, 0) if failures else '--'} & -- & -- " + r"\\")
    print(f"% categorised {sum(tally.values())} of {len(failures)} failed trials")
    unclaimed = [e for e in failures if id(e) not in assigned]
    if unclaimed:
        print(f"% UNCATEGORISED ({len(unclaimed)}): "
              f"{sorted({(e['terminal_status'], e['_failure_code']) for e in unclaimed})}")


if __name__ == "__main__":
    main()
