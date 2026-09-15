#!/usr/bin/env python3
"""All six experiment metrics, extracted from pipeline run directories.

Offline scoring only: nothing here runs during a trial, makes an FM call, or is
read by the runtime pipeline.  Everything is derived from artifacts a trial
already wrote.

  1. Goal coverage            feasible.  Flat sub-goals (12 kitchen / 5 living
                              room / 3 workshop) evaluated on the symbolic
                              terminal state of the candidate plan, so pouring
                              without stirring still scores the pours.
  2. Functional assignment    feasible.  The grounded functional graph scored
     coverage                 against the valid-assignment GT, equivalence aware.
  3. False completion         infeasible.  ANY plan is a false completion: on an
                              impossible task a partial action sequence is still
                              the system acting as though progress were possible.
  4. End-to-end success       feasible.  Same actions as GT, same number, any
                              order, under a consistent body->instance map.
  5. Regions inspected        both.
  6. Candidate checks         both.  Perception work materialised in the observed
                              scene graph: semantic (YOLO-World) labels, unary
                              point-cloud measurements, and pairwise relation
                              evaluations.  These are checks whose results were
                              recorded -- not internal call counts, which the
                              pipeline does not instrument and which could not be
                              obtained without modifying it.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from mujoco_scenes.evaluation_gt import (  # noqa: E402
    score_functional_assignment_coverage, score_goal_coverage,
)
from compare_plans_to_gt_actions import (  # noqa: E402
    EXPLORATORY, compare, compare_unordered, load_expected, load_produced,
)

DOMAINS = {"kitchen": [f"K{i}" for i in range(1, 13)],
           "living_room": [f"L{i}" for i in range(1, 11)],
           "workshop": [f"W{i}" for i in range(1, 11)]}
FEASIBLE = {"kitchen": {f"K{i}" for i in range(1, 7)},
            "living_room": {f"L{i}" for i in range(1, 7)},
            "workshop": {f"W{i}" for i in range(1, 9)}}
ROLE_TO_CATEGORY = {
    "coffee_container": "coffee_container", "soup_container": "soup_container",
    "soup_eating_utensil": "soup_eating_utensil",
    "driver": "driver", "fastener": "fastener",
    "PERSONAL_CUP_SAUCER_REGION": "side_table",
    "SHARED_REMOTE_REGION": "coffee_table", "REMOTE": "tv_remote",
}


SIGNATURES = _SIG = None


def _signatures():
    """GT body -> measured geometry, for offline identity resolution.

    The runtime deliberately never sees simulator body names, so a grounding
    names `object_0002` where the GT names `ab3_narrow_deep_cup`.  Scoring FAC by
    string identity therefore reports 0 for every kitchen run, which is a defect
    in the scorer and not in the method.  Both sides record the same measured
    geometry, so the correspondence is recovered offline by matching it.
    """
    global _SIG
    if _SIG is None:
        path = REPO / "ground_truth" / "identity_signatures.json"
        _SIG = json.loads(path.read_text()) if path.exists() else {}
    return _SIG


# Observed categories are detector labels; GT bodies are simulator names.
_CATEGORY_ALIASES = {
    "cup": "vessel_coffee", "mug": "vessel_coffee",
    "bowl": "vessel_soup",
    "spoon": "spoon", "ladle": "spoon",
    "kettle": "water_source", "teapot": "water_source",
    "coffee canister": "coffee_source", "jar": "coffee_source",
    "canister": "coffee_source", "coffee jar": "coffee_source",
}
_GT_GROUPS = {
    "vessel_coffee": ["ab3_narrow_deep_cup", "ab3_medium_deep_mug"],
    "vessel_soup": ["ab3_shallow_bowl", "ab3_deep_bowl"],
    "spoon": ["ab3_partial_spoon", "s1i_final_long_narrow_spoon", "s1i_oversized_spoon"],
    "water_source": ["s1i_compact_kettle"],
    "coffee_source": ["s1i_compact_coffee_jar"],
}
# Within a category, GT bodies are listed in ascending order of the key below and
# observed instances are ranked the same way.
_RANK_KEYS = {
    "vessel_coffee": ("opening_width_m",),
    "vessel_soup": ("cavity_depth_m",),
    # Length separates the partial spoon from the two long ones; cross-section
    # then separates the narrow stirrer from the oversized serving spoon.
    "spoon": ("total_length_m", "maximum_cross_section_m"),
}


def _resolve_identities(run_dir: Path, domain: str) -> dict[str, str]:
    """Map observed instance id -> GT body name.

    The runtime deliberately never sees simulator body names, so a grounding
    names `object_0002` where the GT names `ab3_narrow_deep_cup`; scoring FAC by
    string identity reports 0 for every kitchen run, which is a defect in the
    scorer, not the method.

    Matching is by RANK within a detector category, not by absolute geometry.
    Observed dimensions come from point clouds and deviate from the GT's exact
    geometry by more than the gaps between objects -- the long spoon measures
    0.227 m against a true 0.258 m -- so nearest-value matching silently swaps
    objects.  Ordering survives that bias: the shorter spoon stays the shorter
    spoon however imprecisely both are measured.
    """
    if domain == "workshop":
        # Workshop detector labels map one-to-one onto GT bodies, so no ranking
        # is needed. Both drivers are valid fillers in W1-W6 and only one is
        # present in W7/W8, so the FAC check does the disambiguating.
        table = {"screwdriver": "workshop_long_phillips_driver",
                 "power_driver": "workshop_power_driver",
                 "manual_driver": "workshop_long_phillips_driver",
                 "screw": "workshop_medium_phillips_screw",
                 "repair_target": "workshop_frame_joint"}
        nodes = _read(run_dir / "observed_scene_graph.json").get("nodes", {})
        nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        mapping = {}
        for node in nodes:
            body = table.get(str(node.get("canonical_category") or "").strip().lower())
            if body:
                mapping[str(node.get("instance_id"))] = body
        # The grounder names the fixed target directly rather than by instance.
        mapping.setdefault("repair_target", "workshop_frame_joint")
        return mapping
    if domain == "living_room":
        # The grounder already speaks the GT's vocabulary here -- region ids and
        # positional set slots -- so identities pass through unchanged.
        return {}
    if domain != "kitchen":
        return {}
    nodes = _read(run_dir / "observed_scene_graph.json").get("nodes", {})
    nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    grouped: dict[str, list] = {}
    for node in nodes:
        label = str(node.get("canonical_category") or "").strip().lower()
        group = _CATEGORY_ALIASES.get(label)
        if group:
            grouped.setdefault(group, []).append(node)
    mapping: dict[str, str] = {}
    claimed: set[str] = set()
    for group, observed in grouped.items():
        bodies = _GT_GROUPS.get(group, [])
        if not bodies:
            continue
        keys = _RANK_KEYS.get(group)
        if keys:
            def sort_key(node):
                geometry = node.get("geometry") or {}
                return tuple(geometry.get(k) if geometry.get(k) is not None else 0.0
                             for k in keys)
            observed = sorted(observed, key=sort_key)
        for index, node in enumerate(observed):
            if index < len(bodies):
                mapping[str(node.get("instance_id"))] = bodies[index]
                claimed.add(bodies[index])

    # Objects the detector could not label have no category to rank within, so
    # the pass above skips them -- 8.4% of kitchen observations, and 9.3% of
    # grounded fillers, which then scored as FAC misses.  The pipeline itself
    # handles these correctly: an unlabelled object yields UNKNOWN rather than
    # FALSE on the semantic check and geometry carries the decision.  The
    # scorer should be no stricter, so an unlabelled cluster is matched on
    # measured size alone against whichever GT bodies remain unclaimed.
    table = _signatures().get("kitchen") or {}
    unlabelled = [n for n in nodes
                  if not str(n.get("canonical_category") or "").strip()
                  and str(n.get("instance_id")) not in mapping]
    fields = ("opening_width_m", "cavity_depth_m", "total_length_m",
              "maximum_cross_section_m")
    for node in unlabelled:
        geometry = node.get("geometry") or {}
        best, best_cost = None, None
        for body, signature in table.items():
            if body in claimed:
                continue
            shared = [(geometry.get(f), signature.get(f)) for f in fields
                      if geometry.get(f) is not None and signature.get(f) is not None]
            if not shared:
                continue
            # Relative error, because point-cloud measurements are biased low
            # against the GT's exact geometry by more than a fixed tolerance.
            cost = sum(abs(a - b) / max(abs(b), 1e-6) for a, b in shared) / len(shared)
            if best_cost is None or cost < best_cost:
                best, best_cost = body, cost
        if best is not None and best_cost is not None and best_cost < 0.35:
            mapping[str(node.get("instance_id"))] = best
            claimed.add(best)
    return mapping


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {} if default is None else default


def _as_list(value):
    return [] if value is None else ([value] if isinstance(value, str) else list(value))


def terminal_state(run_dir: Path):
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
        replay_saved_plan, validation_artifacts)
    try:
        val = (replay_saved_plan(run_dir) if (run_dir / "symbolic_problem.json").exists()
               else validation_artifacts(run_dir))
    except Exception:
        return [], False
    if val.get("status") != "VALID":
        return [], False
    return [tuple(a) for a in val.get("final_atoms", [])], True


def entity_categories(run_dir: Path, domain: str) -> dict[str, str]:
    assignment = _read(run_dir / "graph_grounding_result.json").get("assignment") or {}
    categories = {str(e): cat for role, cat in ROLE_TO_CATEGORY.items()
                  for e in _as_list(assignment.get(role))}
    if domain == "living_room":
        # The detector emits `cup_or_saucer`, never `cup` or `saucer` alone, so
        # looking for those labels found nothing and every placement goal failed
        # in all 60 runs. Refreshment payloads are categorised by the label the
        # perception layer actually produces.
        nodes = _read(run_dir / "observed_scene_graph.json").get("nodes", {})
        nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        for node in nodes:
            label = str(node.get("canonical_category") or "").strip().lower()
            entity = str(node.get("instance_id"))
            if label in ("cup_or_saucer", "cup", "saucer", "plate"):
                categories[entity] = "refreshment_item"
            elif label in ("tv_remote", "remote_control"):
                categories[entity] = "tv_remote"
            elif label == "cup_saucer_set":
                # A set slot is not itself placed; its payloads are.
                for payload in (node.get("unary_properties") or {}).get("payload_ids", []):
                    categories.setdefault(str(payload), "refreshment_item")
    return categories


def candidate_checks(run_dir: Path) -> dict:
    """Metric 6: perception checks recorded in the observed scene graph."""
    observed = _read(run_dir / "observed_scene_graph.json")
    nodes = observed.get("nodes", {})
    nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    relations = observed.get("relations", [])
    relations = list(relations.values()) if isinstance(relations, dict) else list(relations)
    semantic = sum(1 for n in nodes if n.get("semantic_labels") or n.get("canonical_category"))
    unary = sum(1 for n in nodes if n.get("unary_properties") or n.get("unary_predicates"))
    detector = _read(run_dir / "detection_diagnostics.json")
    detector_records = len(detector.get("records", [])) if isinstance(detector, dict) else 0
    verified = _read(run_dir / "physical_relation_verification_trace.json", [])
    return {
        "observed_entities": len(nodes),
        "semantic_candidate_checks": semantic,
        "unary_geometric_checks": unary,
        "pairwise_relation_checks": len(relations),
        "total_candidate_checks": semantic + unary + len(relations),
        "detector_records": detector_records,
        "verified_relation_trace_entries": len(verified) if isinstance(verified, list) else 0,
    }


def score_run(domain: str, variant: str, run_dir: Path) -> dict:
    feasible = variant in FEASIBLE[domain]
    result = _read(run_dir / "result.json")
    status = result.get("status")
    plan = result.get("candidate_plan") or []
    inspected = _as_list(result.get("inspected_regions"))
    row = {
        "domain": domain, "variant": variant, "feasible": feasible,
        "pipeline_status": status,
        "plan_length": len(plan),
        # Metric 5
        "regions_inspected": len(inspected),
        "inspected_regions": inspected,
        "inspection_order_source": (
            "FM" if _read(run_dir / "functional_specification.json").get("region_ranking")
            else "SYSTEM_FALLBACK"),
    }
    row.update(candidate_checks(run_dir))  # Metric 6

    if not feasible:
        # Metric 3.  Any plan at all counts: on an impossible task, emitting a
        # partial action sequence is still claiming the task can be progressed.
        row["false_completion"] = bool(plan) or status == "ACTION_SEQUENCE_READY"
        row["claimed_full_completion"] = status == "ACTION_SEQUENCE_READY"
        row["correctly_rejected"] = not row["false_completion"]
        return row

    atoms, replayed = terminal_state(run_dir)
    row["plan_replay_valid"] = replayed
    gc = score_goal_coverage(domain, variant, atoms, entity_categories(run_dir, domain))
    row.update(goal_coverage=gc["coverage"], satisfied_goals=gc["satisfied_goals"],
               total_goals=gc["total_goals"], goal_status=gc["goal_status"])  # Metric 1

    assignment = _read(run_dir / "graph_grounding_result.json").get("assignment") or {}
    identities = _resolve_identities(run_dir, domain)
    row["identity_resolution_count"] = len(identities)
    if assignment:  # Metric 2
        try:
            resolved = {role: [identities.get(str(e), str(e)) for e in _as_list(v)]
                        for role, v in assignment.items()}
            relational = _relational(
                _read(run_dir / "graph_grounding_result.json"), domain)
            relational = {name: [[identities.get(str(a), str(a)),
                                  identities.get(str(b), str(b))] for a, b in pairs]
                          for name, pairs in relational.items()}
            fac = score_functional_assignment_coverage(
                domain, variant, resolved, relational)
            row.update(functional_assignment_coverage=fac["coverage"],
                       fac_correct_slots=fac["correct_slots"],
                       fac_total_slots=fac["total_slots"])
        except Exception as error:
            row["fac_error"] = f"{type(error).__name__}: {error}"
    else:
        row.update(functional_assignment_coverage=0.0, fac_correct_slots=0)

    loaded = load_expected(domain, variant)  # Metric 4
    produced = load_produced(run_dir)
    if loaded and produced is not None:
        _, expected = loaded
        produced = [a for a in produced if str(a.get("operator")) not in EXPLORATORY]
        ordered_ok, _ = compare(expected, produced)
        unordered_ok, detail = ((True, "exact structural match") if ordered_ok
                                else compare_unordered(expected, produced))
        row.update(end_to_end_success=bool(unordered_ok),
                   exact_ordered_match=bool(ordered_ok),
                   gt_action_count=len(expected), produced_action_count=len(produced),
                   action_count_matches=len(expected) == len(produced),
                   end_to_end_detail=detail)
    else:
        row.update(end_to_end_success=False, exact_ordered_match=False,
                   gt_action_count=len(loaded[1]) if loaded else None,
                   produced_action_count=0, action_count_matches=False,
                   end_to_end_detail="no plan produced")
    return row


def _relational(grounding: dict, domain: str) -> dict:
    """Re-express the grounder's operation bindings in the GT's relation names.

    An earlier version guessed binding names from substrings of the group id and
    emitted `personal_set_on_region` where the GT declares
    `set_on_personal_region`.  Every Living Room relational slot then scored zero
    in all 50 grounded runs -- a uniform 50/50 failure, which is the signature of
    a name mismatch rather than a capability the method lacks.  The mapping is
    now explicit per domain, and the seat context and remote placement are read
    from where the grounder actually records them rather than ignored.
    """
    bindings = grounding.get("operation_bindings") or {}
    assignment = grounding.get("assignment") or {}
    pairs: dict[str, list] = {}

    if domain == "living_room":
        for items in bindings.values():
            for binding in items if isinstance(items, list) else ():
                region, slot = binding.get("tool_id"), binding.get("target_id")
                if region and slot:
                    pairs.setdefault("set_on_personal_region", []).append([region, slot])
                seat = (binding.get("context") or {}).get("SEATING_POSITION")
                if region and seat:
                    pairs.setdefault("region_near_seat", []).append([region, seat])
        # The remote is grounded as a role rather than through an operation
        # group, so its placement is read from the assignment.
        for remote in _as_list(assignment.get("REMOTE")):
            for shared in _as_list(assignment.get("SHARED_REMOTE_REGION")):
                pairs.setdefault("remote_on_shared", []).append([remote, shared])
        return pairs

    if domain == "workshop":
        driver = _as_list(assignment.get("driver"))
        fastener = _as_list(assignment.get("fastener"))
        target = _as_list(assignment.get("repair_target")) or ["repair_target"]
        # The GT scores the repair tuple as three relations over one consistent
        # driver choice; the grounder asserts them by binding the group at all.
        for d in driver:
            for f in fastener:
                pairs.setdefault("repair_tuple", []).append([d, f])
                for t in target:
                    pairs.setdefault("repair_tuple", []).append([d, t])
                    pairs.setdefault("repair_tuple", []).append([f, t])
        return pairs

    # kitchen
    for group, items in bindings.items():
        for binding in items if isinstance(items, list) else ():
            tool, target = binding.get("tool_id"), binding.get("target_id")
            if tool is None or target is None:
                continue
            name = ("stirs" if ("coffee" in group or "stir" in group or "mix" in group)
                    else "serves_soup_in")
            pairs.setdefault(name, []).append([tool, target])
    return pairs


def summarize(rows: list[dict]) -> dict:
    feas = [r for r in rows if r["feasible"]]
    inf = [r for r in rows if not r["feasible"]]

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 4) if values else None
    out = {
        "trials": len(rows), "feasible_trials": len(feas), "infeasible_trials": len(inf),
        "metric1_goal_coverage_mean": mean([r.get("goal_coverage") for r in feas]),
        "metric1_goals_satisfied": sum(r.get("satisfied_goals", 0) for r in feas),
        "metric1_goals_total": sum(r.get("total_goals", 0) for r in feas),
        "metric2_fac_mean": mean([r.get("functional_assignment_coverage") for r in feas]),
        "metric2_fac_perfect": sum(1 for r in feas
                                   if r.get("functional_assignment_coverage") == 1.0),
        "metric3_false_completions": sum(1 for r in inf if r.get("false_completion")),
        "metric3_false_completion_pct": (
            round(100 * sum(1 for r in inf if r.get("false_completion")) / len(inf), 2)
            if inf else None),
        "metric3_claimed_full_completion": sum(1 for r in inf
                                               if r.get("claimed_full_completion")),
        "metric3_correctly_rejected": sum(1 for r in inf if r.get("correctly_rejected")),
        "metric4_end_to_end_success": sum(1 for r in feas if r.get("end_to_end_success")),
        "metric4_end_to_end_pct": (
            round(100 * sum(1 for r in feas if r.get("end_to_end_success")) / len(feas), 2)
            if feas else None),
        "metric4_exact_ordered_match": sum(1 for r in feas if r.get("exact_ordered_match")),
        "metric4_action_count_matches": sum(1 for r in feas if r.get("action_count_matches")),
        "metric5_regions_inspected_mean_feasible": mean([r["regions_inspected"] for r in feas]),
        "metric5_regions_inspected_mean_infeasible": mean([r["regions_inspected"] for r in inf]),
        "metric5_fm_ranked_order_runs": sum(1 for r in rows
                                            if r["inspection_order_source"] == "FM"),
        "metric6_total_candidate_checks_mean": mean([r["total_candidate_checks"] for r in rows]),
        "metric6_semantic_checks_mean": mean([r["semantic_candidate_checks"] for r in rows]),
        "metric6_unary_checks_mean": mean([r["unary_geometric_checks"] for r in rows]),
        "metric6_relation_checks_mean": mean([r["pairwise_relation_checks"] for r in rows]),
    }
    out["by_domain"] = {}
    for domain in DOMAINS:
        group = [r for r in rows if r["domain"] == domain]
        if not group:
            continue
        gf = [r for r in group if r["feasible"]]
        gi = [r for r in group if not r["feasible"]]
        out["by_domain"][domain] = {
            "trials": len(group),
            "goal_coverage_mean": mean([r.get("goal_coverage") for r in gf]),
            "fac_mean": mean([r.get("functional_assignment_coverage") for r in gf]),
            "end_to_end_success": sum(1 for r in gf if r.get("end_to_end_success")),
            "end_to_end_of": len(gf),
            "false_completions": sum(1 for r in gi if r.get("false_completion")),
            "infeasible_of": len(gi),
            "regions_inspected_mean": mean([r["regions_inspected"] for r in group]),
            "candidate_checks_mean": mean([r["total_candidate_checks"] for r in group]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.root.glob("**/*/*/vlm")):
        variant, domain = run_dir.parent.name, run_dir.parent.parent.name
        if domain not in DOMAINS or variant not in DOMAINS[domain]:
            continue
        try:
            row = score_run(domain, variant, run_dir)
        except Exception as error:
            row = {"domain": domain, "variant": variant,
                   "feasible": variant in FEASIBLE[domain],
                   "run_dir": str(run_dir),
                   "error": f"{type(error).__name__}: {error}"}
        row["run_dir"] = str(run_dir)
        rows.append(row)
    scored = [r for r in rows if "error" not in r]
    if not scored:
        print(f"no runs scored under {args.root}", file=sys.stderr)
        return 1
    summary = summarize(scored)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "full_metrics.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
        keys = [k for k in scored[0] if k not in ("goal_status", "inspected_regions")]
        with (args.out / "full_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(scored)
        print(f"\nwrote {args.out}/full_metrics.{{json,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
