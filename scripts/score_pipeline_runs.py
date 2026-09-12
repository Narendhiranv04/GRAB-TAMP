#!/usr/bin/env python3
"""Extract both evaluation metrics from pipeline run directories.  Offline only.

Reads what a trial already wrote -- the replayed symbolic terminal state, the
grounding result, the observed scene graph, the candidate plan -- and scores:

  Goal Coverage        satisfied goals / 12 (kitchen), 5 (living room), 3 (workshop)
                       Flat, so a plan that pours but never stirs scores its pours.

  End-to-end success   the produced plan contains the same actions as the GT
                       sequence, the same number of times.  Order is deliberately
                       ignored: subgoals are independent, so serving soup before
                       coffee is not a different plan.  A consistent body ->
                       instance correspondence is still required, so a plan
                       cannot score by binding one GT body to two objects.

  Functional Assignment Coverage, when the run exposes a grounding.

No physical execution and no FM call.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from mujoco_scenes.evaluation_gt import (  # noqa: E402
    score_functional_assignment_coverage, score_goal_coverage,
)
from compare_plans_to_gt_actions import (  # noqa: E402
    EXPLORATORY, arguments, compare, compare_unordered, load_expected, load_produced,
)

FEASIBLE = {
    "kitchen": {f"K{i}" for i in range(1, 7)},
    "living_room": {f"L{i}" for i in range(1, 7)},
    "workshop": {f"W{i}" for i in range(1, 9)},
}
# Terminal-state entity -> semantic category, taken from the run's own grounding.
# This is the only place role names are touched, and only to derive categories a
# perception-based baseline would also have; the goals themselves never mention
# a role.
ROLE_TO_CATEGORY = {
    "coffee_container": "coffee_container", "soup_container": "soup_container",
    "soup_eating_utensil": "soup_eating_utensil",
    "driver": "driver", "fastener": "fastener",
    "PERSONAL_CUP_SAUCER_REGION": "side_table",
    "SHARED_REMOTE_REGION": "coffee_table", "REMOTE": "tv_remote",
}


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def terminal_state(run_dir: Path):
    """Final atoms from replaying the candidate plan; no execution."""
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
        replay_saved_plan, validation_artifacts,
    )
    val = (replay_saved_plan(run_dir) if (run_dir / "symbolic_problem.json").exists()
           else validation_artifacts(run_dir))
    if val.get("status") != "VALID":
        return [], False
    return [tuple(a) for a in val.get("final_atoms", [])], True


def entity_categories(run_dir: Path, domain: str) -> dict[str, str]:
    grounding = _read(run_dir / "graph_grounding_result.json")
    assignment = grounding.get("assignment") or {}
    categories: dict[str, str] = {}
    for role, category in ROLE_TO_CATEGORY.items():
        for entity in _as_list(assignment.get(role)):
            categories[str(entity)] = category
    if domain == "living_room":
        # Cups and saucers are payloads inside the grounded sets, not roles.
        observed = _read(run_dir / "observed_scene_graph.json")
        nodes = observed.get("nodes", {})
        if isinstance(nodes, list):
            nodes = {n.get("instance_id"): n for n in nodes}
        for node in nodes.values():
            label = str(node.get("canonical_category") or "")
            if label in ("cup", "saucer", "tv_remote", "remote_control"):
                categories[str(node.get("instance_id"))] = (
                    "tv_remote" if "remote" in label else label)
    return categories


def score_run(domain: str, variant: str, run_dir: Path) -> dict:
    row = {"domain": domain, "variant": variant, "run_dir": str(run_dir)}
    atoms, replayed = terminal_state(run_dir)
    row["plan_replay_valid"] = replayed
    categories = entity_categories(run_dir, domain)
    gc = score_goal_coverage(domain, variant, atoms, categories)
    row.update(goal_coverage=gc["coverage"],
               satisfied_goals=gc["satisfied_goals"],
               total_goals=gc["total_goals"],
               goal_status=gc["goal_status"])

    loaded = load_expected(domain, variant)
    produced = load_produced(run_dir)
    if loaded and produced is not None:
        _, expected = loaded
        produced = [a for a in produced
                    if str(a.get("operator")) not in EXPLORATORY]
        ordered_ok, _ = compare(expected, produced)
        unordered_ok, detail = ((True, "exact structural match") if ordered_ok
                                else compare_unordered(expected, produced))
        row.update(end_to_end_success=bool(unordered_ok),
                   exact_ordered_match=bool(ordered_ok),
                   gt_action_count=len(expected),
                   produced_action_count=len(produced),
                   action_count_matches=len(expected) == len(produced),
                   end_to_end_detail=detail)
    else:
        row.update(end_to_end_success=False, exact_ordered_match=False,
                   gt_action_count=len(loaded[1]) if loaded else None,
                   produced_action_count=0, action_count_matches=False,
                   end_to_end_detail="no plan produced")

    grounding = _read(run_dir / "graph_grounding_result.json")
    assignment = grounding.get("assignment") or {}
    if assignment:
        try:
            fac = score_functional_assignment_coverage(
                domain, variant, assignment,
                _relational_from_grounding(grounding, domain))
            row.update(functional_assignment_coverage=fac["coverage"],
                       fac_correct_slots=fac["correct_slots"],
                       fac_total_slots=fac["total_slots"])
        except Exception as error:
            row["fac_error"] = f"{type(error).__name__}: {error}"
    return row


def _relational_from_grounding(grounding: dict, domain: str) -> dict:
    """Re-express operation bindings as the GT's functional relation names."""
    bindings = grounding.get("operation_bindings") or {}
    pairs: dict[str, list] = {}
    for group, items in bindings.items():
        for binding in items if isinstance(items, list) else ():
            tool, target = binding.get("tool_id"), binding.get("target_id")
            if tool is None or target is None:
                continue
            name = ("stirs" if "coffee" in group or "stir" in group else
                    "serves_soup_in" if "soup" in group or "serv" in group else
                    "personal_set_on_region" if domain == "living_room" else
                    "repair_tuple")
            pairs.setdefault(name, []).append([tool, target])
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.root.glob("**/*/*/vlm")):
        variant, domain = run_dir.parent.name, run_dir.parent.parent.name
        if domain not in FEASIBLE or variant not in FEASIBLE[domain]:
            continue
        try:
            rows.append(score_run(domain, variant, run_dir))
        except Exception as error:
            rows.append({"domain": domain, "variant": variant,
                         "run_dir": str(run_dir),
                         "error": f"{type(error).__name__}: {error}"})
    scored = [r for r in rows if "error" not in r]
    if not scored:
        print(f"no feasible runs scored under {args.root}", file=sys.stderr)
        return 1
    summary = {
        "runs": len(scored),
        "mean_goal_coverage": round(
            sum(r["goal_coverage"] for r in scored) / len(scored), 4),
        "goals_satisfied": sum(r["satisfied_goals"] for r in scored),
        "goals_total": sum(r["total_goals"] for r in scored),
        "end_to_end_success": sum(bool(r.get("end_to_end_success")) for r in scored),
        "end_to_end_pct": round(
            100 * sum(bool(r.get("end_to_end_success")) for r in scored) / len(scored), 2),
        "exact_ordered_match": sum(bool(r.get("exact_ordered_match")) for r in scored),
        "action_count_matches": sum(bool(r.get("action_count_matches")) for r in scored),
    }
    fac = [r["functional_assignment_coverage"] for r in scored
           if "functional_assignment_coverage" in r]
    if fac:
        summary["mean_functional_assignment_coverage"] = round(sum(fac) / len(fac), 4)
        summary["fac_runs"] = len(fac)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pipeline_metrics.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
        keys = [k for k in scored[0] if k != "goal_status"]
        with (args.out / "pipeline_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(scored)
        print(f"\nwrote {args.out}/pipeline_metrics.{{json,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
