"""Scoring for the two evaluation metrics.  Offline only.

Lives outside `functional_tamp_pipeline` on purpose: both metrics read ground
truth, and the no-GT-leakage audit treats any `gt_` reference inside the
pipeline package as a finding.  Nothing here may be imported by specification
generation, the semantic compiler, search, perception, grounding, the planner or
runtime execution.

The two metrics are deliberately independent and must not be conflated:

  Functional Assignment Coverage (FAC)
      Did the method ground the task-required functions and the relations
      between them?  Scored against the SET of valid assignments, so a method is
      not penalised for choosing a different member of an equivalence class than
      the deterministic GT controller did.

  Goal Coverage (GC)
      Do the user-level task goals hold in the symbolic terminal state?
      Representation independent -- a baseline that never names a functional role
      can still score 100%.

A method can score 100% FAC and 67% GC (right objects, incomplete plan), or have
no FAC at all and 100% GC (an object-direct baseline).  See
tests/test_evaluation_gt_metrics.py, which pins all four combinations.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
FAC_PATH = REPO / "GT_VALID_ROLE_ASSIGNMENTS" / "gt_valid_role_assignments.json"
GC_PATH = REPO / "GT_GOAL_COVERAGE" / "gt_goal_coverage.json"

_FAC_CACHE: dict | None = None
_GC_CACHE: dict | None = None


def load_fac_gt() -> dict:
    global _FAC_CACHE
    if _FAC_CACHE is None:
        _FAC_CACHE = json.loads(FAC_PATH.read_text())
    return _FAC_CACHE


def load_goal_gt() -> dict:
    global _GC_CACHE
    if _GC_CACHE is None:
        _GC_CACHE = json.loads(GC_PATH.read_text())
    return _GC_CACHE


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


# --------------------------------------------------------------------------
# Metric 1: Functional Assignment Coverage
# --------------------------------------------------------------------------
def _score_unary(spec: Mapping[str, Any], predicted: Mapping[str, Any]):
    """One slot per required instance of each role.

    A role requiring two fillers contributes two slots, and the denominator is
    the GT's required count -- never how many the prediction happened to emit.
    """
    correct, total, missing, incorrect = 0, 0, [], []
    for role, required in spec["required_role_slots"].items():
        valid = set(spec["valid_role_fillers"].get(role, []))
        got = _as_list(predicted.get(role))
        total += required
        seen: set[str] = set()
        for index in range(required):
            if index >= len(got):
                missing.append({"role": role, "slot_index": index})
                continue
            filler = got[index]
            if filler in valid and filler not in seen:
                seen.add(filler)
                correct += 1
            else:
                incorrect.append({"role": role, "slot_index": index,
                                  "predicted": filler,
                                  "reason": ("duplicate filler" if filler in seen
                                             else "not a valid filler")})
    return correct, total, missing, incorrect


def _pairs(value: Any) -> set[tuple[str, str]]:
    """Accept {tool: target}, {tool: [targets]} or [[tool, target], ...]."""
    out: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        for left, right in value.items():
            for item in _as_list(right):
                out.add((str(left), str(item)))
    else:
        for item in value or ():
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.add((str(item[0]), str(item[1])))
    return out


def _score_relational(spec: Mapping[str, Any], predicted: Mapping[str, Any]):
    """Relational slots, scored against the equivalence class, not one witness."""
    relational = spec["valid_assignment_sets"].get("relational", {})
    correct, total, missing, incorrect = 0, 0, [], []
    matched: dict[str, Any] = {}
    for name, rule in relational.items():
        declared = spec["required_relational_bindings"].get(name, {})
        cardinality = int(declared.get("cardinality", 1))
        total += cardinality
        got = _pairs(predicted.get(name))
        kind = rule.get("type")

        if kind == "complete_from_single_tool":
            tool, targets = rule["tool"], list(rule["targets"])
            hit = sum(1 for t in targets if (tool, t) in got)
            correct += min(hit, cardinality)
            matched[name] = {"tool": tool, "covered": hit}
            if hit < cardinality:
                missing.append({"binding": name, "expected_pairs": cardinality, "matched": hit})

        elif kind in ("any_bijection", "partition_one_of_each", "tuple_over_valid_fillers", "fixed"):
            best, best_map = 0, None
            for candidate in _enumerate_valid(rule):
                hit = sum(1 for pair in candidate if pair in got)
                if hit > best:
                    best, best_map = hit, candidate
            correct += min(best, cardinality)
            matched[name] = {"best_alternative": sorted(best_map) if best_map else None,
                             "covered": best}
            if best < cardinality:
                missing.append({"binding": name, "expected_pairs": cardinality, "matched": best})
            for pair in got:
                if best_map is None or pair not in best_map:
                    incorrect.append({"binding": name, "predicted_pair": list(pair)})
        else:
            raise ValueError(f"unknown relational rule type {kind!r} for {name!r}")
    return correct, total, missing, incorrect, matched


def _enumerate_valid(rule: Mapping[str, Any]) -> Iterable[set[tuple[str, str]]]:
    """Expand a compact constraint into the alternatives it denotes."""
    kind = rule["type"]
    if kind == "any_bijection":
        left, right = list(rule["left"]), list(rule["right"])
        for order in permutations(right, len(left)):
            yield set(zip(left, order))
    elif kind == "partition_one_of_each":
        regions, cups, saucers = list(rule["regions"]), list(rule["cups"]), list(rule["saucers"])
        for cup_order in permutations(cups, len(regions)):
            for saucer_order in permutations(saucers, len(regions)):
                yield {(r, cup_order[i]) for i, r in enumerate(regions)} | \
                      {(r, saucer_order[i]) for i, r in enumerate(regions)}
    elif kind == "tuple_over_valid_fillers":
        for driver in rule["drivers"]:
            yield {(driver, rule["fastener"]), (driver, rule["target"]),
                   (rule["fastener"], rule["target"])}
    elif kind == "fixed":
        yield {(l, r) for l in rule["left"] for r in rule["right"]}
    else:
        raise ValueError(f"unknown rule type {kind!r}")


def score_functional_assignment_coverage(
    domain: str, variant: str,
    predicted_assignment: Mapping[str, Any],
    predicted_relational_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one trial's functional grounding.  See module docstring."""
    gt = load_fac_gt()["variants"]
    if variant not in gt:
        raise KeyError(f"{variant} is not a feasible benchmark variant with FAC ground truth")
    spec = gt[variant]
    if spec["domain"] != domain:
        raise ValueError(f"{variant} belongs to {spec['domain']}, not {domain}")

    u_correct, u_total, u_missing, u_incorrect = _score_unary(spec, predicted_assignment)
    r_correct, r_total, r_missing, r_incorrect, matched = _score_relational(
        spec, predicted_relational_bindings or {})
    correct, total = u_correct + r_correct, u_total + r_total
    declared = spec["scoring"]["total_scored_slots"]
    if total != declared:
        raise AssertionError(
            f"{variant}: computed denominator {total} != declared {declared}; "
            "the GT file and the scorer disagree about what is scored")
    return {
        "coverage": round(correct / total, 4) if total else 0.0,
        "correct_slots": correct, "total_slots": total,
        "unary": {"correct": u_correct, "total": u_total},
        "relational": {"correct": r_correct, "total": r_total},
        "matched_valid_assignment": matched,
        "missing": u_missing + r_missing,
        "incorrect": u_incorrect + r_incorrect,
    }


# --------------------------------------------------------------------------
# Metric 2: Goal Coverage
# --------------------------------------------------------------------------
def score_goal_coverage(domain: str, variant: str,
                        symbolic_terminal_state: Iterable[Iterable[str]],
                        entity_semantics: Mapping[str, str] | None = None,
                        verified_relations: Iterable[Iterable[str]] | None = None,
                        ) -> dict[str, Any]:
    """Score user-level goals over a symbolic terminal state.

    Flat: every goal in the GT is one unit, so partial progress is measured
    directly -- a plan that pours coffee and water but never stirs scores the
    two pours.  `symbolic_terminal_state` is the final atom set from replaying
    the candidate plan; no physical execution is involved.

    `entity_semantics` maps entity id -> semantic category.  That is what makes
    the metric representation independent: the categories come from perception,
    not from the method's internal vocabulary, so a baseline that never names a
    "coffee_stirrer" is still scorable.
    """
    gt = load_goal_gt()["variants"]
    if variant not in gt:
        raise KeyError(f"{variant} is not a feasible benchmark variant with goal ground truth")
    spec = gt[variant]
    if spec["domain"] != domain:
        raise ValueError(f"{variant} belongs to {spec['domain']}, not {domain}")

    atoms = {tuple(str(x) for x in a) for a in symbolic_terminal_state}
    semantics = dict(entity_semantics or {})
    by_category: dict[str, list[str]] = {}
    for entity, category in semantics.items():
        by_category.setdefault(category, []).append(entity)

    status: dict[str, bool] = {}
    for group in spec["goal_groups"]:
        pool = sorted({e for cat in group["entity_semantics"]
                       for e in by_category.get(cat, [])})
        partners = sorted({p for cat in group.get("partner_semantics", [])
                           for p in by_category.get(cat, [])})
        slots = int(group["slots"])
        best = _best_slot_assignment(group, pool, partners, slots, atoms, by_category)
        for index in range(slots):
            for goal in group["goals"]:
                status[f"{group['id']}_{index + 1}.{goal['id']}"] = bool(
                    best[index].get(goal["id"], False))

    satisfied = sum(status.values())
    total = spec["goal_count"]
    if len(status) != total:
        raise AssertionError(
            f"{variant}: evaluated {len(status)} goals but goal_count is {total}")
    return {
        "coverage": round(satisfied / total, 4) if total else 0.0,
        "satisfied_goals": satisfied, "total_goals": total, "goal_status": status,
    }


def _best_slot_assignment(group, pool, partners, slots, atoms, by_category):
    """Assign distinct entities to slots so that the most goals are satisfied.

    Greedy assignment would understate a plan: committing slot 1 to a partly
    finished vessel can strand a fully finished one, so the slots are searched
    exhaustively.  The candidate sets are two or three entities wide, so this is
    cheap and exact rather than approximate.
    """
    distinct = group.get("distinct_entities", True)
    candidates = list(permutations(pool, slots)) if distinct else         [tuple(c) for c in permutations(pool, slots)] or [()]
    if not candidates:
        candidates = [tuple([None] * slots)]
    best_rows, best_score = [{} for _ in range(slots)], -1
    for combo in candidates:
        used_partners: set[str] = set()
        used_payloads: set[str] = set()
        rows, score = [], 0
        for entity in combo:
            row = {}
            for goal in group["goals"]:
                row[goal["id"]] = _goal_holds(
                    goal, entity, atoms, by_category, partners,
                    used_partners, used_payloads, group)
            rows.append(row)
            score += sum(row.values())
        if score > best_score:
            best_rows, best_score = rows, score
    while len(best_rows) < slots:
        best_rows.append({})
    return best_rows


def _goal_holds(goal, entity, atoms, by_category, partners,
                used_partners, used_payloads, group) -> bool:
    if entity is None:
        return False
    for atom in goal.get("ground_atoms", []):
        if tuple(atom) not in atoms:
            return False
    if goal.get("partner_conditions"):
        for partner in partners:
            if group.get("distinct_partners") and partner in used_partners:
                continue
            if _holds(goal["partner_conditions"], atoms, {"$e": entity, "$p": partner}):
                used_partners.add(partner)
                return True
        return False
    if goal.get("payload_semantics"):
        pool = [p for cat in goal["payload_semantics"] for p in by_category.get(cat, [])]
        for payload in pool:
            if goal.get("distinct_payloads") and payload in used_payloads:
                continue
            if _holds(goal["conditions"], atoms, {"$e": entity, "$payload": payload}):
                if goal.get("distinct_payloads"):
                    used_payloads.add(payload)
                return True
        return False
    if goal.get("conditions"):
        return _holds(goal["conditions"], atoms, {"$e": entity})
    return True


def _holds(conditions, atoms, binding) -> bool:
    for condition in conditions:
        atom = tuple(binding.get(part, part) for part in condition)
        if atom not in atoms:
            return False
    return True


def _relations_hold(required, relations, binding) -> bool:
    for relation in required:
        want = tuple(binding.get(part, part) for part in relation)
        if want[0] == "FITS_SET_ON":
            if not any(r[0] == "FITS_SET_ON" and r[1] == want[1] for r in relations):
                return False
        elif want not in relations:
            return False
    return True
