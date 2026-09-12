#!/usr/bin/env python3
"""Two offline metrics for feasible variants, scored against the GT catalogue.

Offline scoring only: nothing here runs during a trial and nothing it computes
is fed back into the pipeline.

METRIC 1 -- valid role assignment
    Did the run bind the right things to each other?  The GT action sequence
    encodes every binding the task requires: POUR(kettle, cup) is an
    object->object pairing, PLACE(cup, serving_area) an object->region one,
    STIR(spoon, mug) a tool->target one.  Each multi-argument GT action is
    therefore one required binding, and the metric is how many of them the
    produced plan actually realises.

    Reported per variant as complete (every binding present) and partial (some
    but not all), because a run that gets three of four pairings right is a
    different animal from one that gets none, and collapsing them hides which.

METRIC 2 -- end-to-end success
    Does the produced plan contain the same actions as GT, the same number of
    times?  Order is deliberately not required: subgoals are independent, so
    serving soup before coffee is not a different plan.  What must hold is the
    multiset of steps and a *consistent* body->instance correspondence, so a
    plan cannot score by binding the same GT body to two different objects.

Both metrics share one bijection search.  A produced plan names observed
instances ("object_0003") where the catalogue names scene bodies
("s1i_compact_kettle"), and the correspondence is discovered, not assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compare_plans_to_gt_actions import (  # noqa: E402
    EXPLORATORY, OBSERVED_INSTANCE, arguments, compare, compare_unordered,
    load_expected, load_produced,
)

FEASIBLE = {
    "kitchen": {f"K{i}" for i in range(1, 7)},
    "living_room": {f"L{i}" for i in range(1, 7)},
    "workshop": {f"W{i}" for i in range(1, 9)},
}


def required_bindings(expected):
    """Each multi-argument GT action is one required binding."""
    return [a for a in expected if len(arguments(a)) >= 2]


def best_binding_match(expected, produced):
    """Largest set of GT bindings realisable under one consistent correspondence.

    Maximising rather than first-fit matters: a correspondence that satisfies
    one binding early can dead-end later, and reporting that as a partial score
    would understate a run whose other correspondence satisfies more.
    """
    required = required_bindings(expected)
    best = {"matched": 0, "total": len(required), "pairs": []}
    if not required:
        return best

    def search(index, b2i, i2b, used, matched, trace):
        nonlocal best
        if matched > best["matched"]:
            best = {"matched": matched, "total": len(required), "pairs": list(trace)}
        if index == len(required) or best["matched"] == len(required):
            return
        # Skipping is a real option: an unmatched binding must not abort the
        # search, or a single missing step would zero an otherwise good plan.
        want = required[index]
        want_args = arguments(want)
        for position, candidate in enumerate(produced):
            if used[position]:
                continue
            if str(candidate.get("operator")) != str(want.get("operator")):
                continue
            got_args = arguments(candidate)
            if len(want_args) != len(got_args):
                continue
            nb2i, ni2b, ok = dict(b2i), dict(i2b), True
            for w, g in zip(want_args, got_args):
                if not OBSERVED_INSTANCE.match(g):
                    if w != g:
                        ok = False
                        break
                    continue
                if nb2i.setdefault(w, g) != g or ni2b.setdefault(g, w) != w:
                    ok = False
                    break
            if not ok:
                continue
            used[position] = True
            search(index + 1, nb2i, ni2b, used, matched + 1,
                   trace + [{"operator": str(want.get("operator")),
                             "gt_arguments": want_args,
                             "produced_arguments": got_args}])
            used[position] = False
        search(index + 1, b2i, i2b, used, matched, trace)

    search(0, {}, {}, [False] * len(produced), 0, [])
    return best


def score_one(domain: str, variant: str, run_dir: Path) -> dict | None:
    loaded = load_expected(domain, variant)
    if loaded is None:
        return None
    _, expected = loaded
    produced = load_produced(run_dir)
    row = {
        "domain": domain, "variant": variant,
        "gt_action_count": len(expected),
        "plan_found": produced is not None,
        "produced_action_count": 0 if produced is None else len(produced),
    }
    if produced is None:
        row.update(bindings_total=len(required_bindings(expected)), bindings_matched=0,
                   role_assignment_complete=False, role_assignment_partial=False,
                   role_assignment_rate=0.0, end_to_end_success=False,
                   exact_ordered_match=False, detail="no plan produced")
        return row
    produced = [a for a in produced if str(a.get("operator")) not in EXPLORATORY]
    row["produced_action_count"] = len(produced)

    binding = best_binding_match(expected, produced)
    total, matched = binding["total"], binding["matched"]
    row.update(
        bindings_total=total,
        bindings_matched=matched,
        role_assignment_rate=round(matched / total, 4) if total else 1.0,
        role_assignment_complete=bool(total and matched == total),
        role_assignment_partial=bool(0 < matched < total),
        matched_bindings=binding["pairs"],
    )
    ordered_ok, _ = compare(expected, produced)
    unordered_ok, detail = ((True, "exact structural match") if ordered_ok
                            else compare_unordered(expected, produced))
    row.update(exact_ordered_match=bool(ordered_ok),
               end_to_end_success=bool(unordered_ok), detail=detail)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Run root containing <trial>/<domain>/<variant>/vlm")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.root.glob("*/*/*/vlm")):
        variant, domain = run_dir.parent.name, run_dir.parent.parent.name
        if domain not in FEASIBLE or variant not in FEASIBLE[domain]:
            continue  # the two metrics are defined for feasible variants only
        row = score_one(domain, variant, run_dir)
        if row is None:
            continue
        row["trial"] = run_dir.parent.parent.parent.name
        rows.append(row)

    n = len(rows)
    if not n:
        print("no feasible runs found under", args.root, file=sys.stderr)
        return 1
    complete = sum(r["role_assignment_complete"] for r in rows)
    partial = sum(r["role_assignment_partial"] for r in rows)
    e2e = sum(r["end_to_end_success"] for r in rows)
    exact = sum(r["exact_ordered_match"] for r in rows)
    bt = sum(r["bindings_total"] for r in rows)
    bm = sum(r["bindings_matched"] for r in rows)
    summary = {
        "feasible_runs": n,
        "metric1_role_assignment_complete": complete,
        "metric1_role_assignment_partial": partial,
        "metric1_role_assignment_none": n - complete - partial,
        "metric1_complete_pct": round(100 * complete / n, 2),
        "metric1_binding_rate_pct": round(100 * bm / bt, 2) if bt else None,
        "metric2_end_to_end_success": e2e,
        "metric2_end_to_end_pct": round(100 * e2e / n, 2),
        "metric2_exact_ordered_match": exact,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "role_and_plan_metrics.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
        keys = [k for k in rows[0] if k != "matched_bindings"]
        with (args.out / "role_and_plan_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}/role_and_plan_metrics.{{json,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
