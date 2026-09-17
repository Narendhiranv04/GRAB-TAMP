#!/usr/bin/env python3
"""Physical opening cost per inspection-order arm.

Grounding time measures perception and phi. It does not measure the robot
opening anything, and on this hardware the opening dominates: a workshop
trial spends about 27 s grounding and about 240 s driving drawers and a
cabinet open. An inspection-order result reported in grounding seconds is
therefore reporting the small term.

This sums the *measured* actuation time for the regions each trial actually
inspected. Per-region costs come from data/metrics/workshop_open_costs.json,
recorded with contact-gated robot actuation from the home position and verified
against ground truth, so nothing here is an estimate of the physical cost --
only the composition is computed.

Scope: workshop only, and that is a property of the pipeline rather than a
gap in the measurement. run._get_exploration_actuation returns
"direct_sim_articulation" for kitchen unconditionally -- kitchen containers are
opened by setting the simulator joint, with or without --dry-run -- and
"not_applicable" for living_room, which declares no inspectable regions. Only
workshop has a robot-actuated opening path, so only workshop has an actuation
cost that exists to be measured.

Usage:
    python3 scripts/score_inspection_open_cost.py \
        --arms order_fm order_random order_fixed \
        --out results/open_cost.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

DOMAINS = ("kitchen", "living_room", "workshop")
ACTUATED = {"workshop"}


def arm_costs(root: pathlib.Path, sim_cost: dict[str, float]) -> dict:
    per = collections.defaultdict(lambda: {"trials": 0, "regions": 0, "open_seconds": 0.0})
    for result_path in root.rglob("result.json"):
        domain = next((p for p in result_path.parts if p in DOMAINS), None)
        if domain is None:
            continue
        try:
            record = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        inspected = record.get("inspected_regions") or []
        bucket = per[domain]
        bucket["trials"] += 1
        bucket["regions"] += len(inspected)
        if domain in ACTUATED:
            bucket["open_seconds"] += sum(sim_cost.get(r, 0.0) for r in inspected)
    out = {}
    for domain, bucket in per.items():
        n = max(bucket["trials"], 1)
        out[domain] = {
            "trials": bucket["trials"],
            "regions_per_trial": bucket["regions"] / n,
            "open_seconds_per_trial": bucket["open_seconds"] / n if domain in ACTUATED else None,
            "robot_actuated": domain in ACTUATED,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports-root", default="results/runs", type=pathlib.Path)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--costs", default=None, type=pathlib.Path,
                    help="per-region measured open costs (default: <reports-root>/workshop_open_costs.json)")
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()

    costs_path = args.costs or pathlib.Path("data/metrics/workshop_open_costs.json")
    measured = json.loads(costs_path.read_text())
    sim_cost = {region: entry["sim_seconds"] for region, entry in measured.items()}

    print("Measured per-region opening cost (robot actuated, from home):")
    for region, seconds in sorted(sim_cost.items(), key=lambda kv: -kv[1]):
        print(f"  {region:<16} {seconds:8.1f} s   {measured[region]['status']}")
    print()

    report = {"per_region_open_seconds": sim_cost, "arms": {}}
    header = f"{'arm':<14}{'domain':<14}{'trials':>7}{'regions/trial':>15}{'open s/trial':>14}"
    print(header)
    print("-" * len(header))
    for name in args.arms:
        arm = arm_costs(args.reports_root / name, sim_cost)
        report["arms"][name] = arm
        for domain in DOMAINS:
            if domain not in arm:
                continue
            entry = arm[domain]
            cost = entry["open_seconds_per_trial"]
            shown = f"{cost:14.1f}" if cost is not None else f"{'not actuated':>14}"
            print(f"{name:<14}{domain:<14}{entry['trials']:>7}"
                  f"{entry['regions_per_trial']:>15.2f}{shown}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
