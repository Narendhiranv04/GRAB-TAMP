#!/usr/bin/env python3
"""Paired scoring for the inspection-order ablation.

Both arms replay the same frozen specifications, so every trial in one arm has
an exact counterpart in the other: same scene, same variant, same functional
requirement graph, differing only in the order regions are inspected. That makes
the comparison paired, and paired is what the ablation needs -- an unpaired
mean-vs-mean would let differences between specifications leak into a difference
attributed to ordering.

Reported per scene:

    regions inspected   how much of the scene had to be opened
    candidate checks    objects put through semantic + geometric evaluation
    grounding seconds   region inspection + perception + phi
    planning seconds    the symbolic planner

Terminal status is compared as well. Ordering is supposed to change what the
search costs, not what it concludes; a silent change in outcomes would matter
more than any timing result, so it is checked rather than assumed.

Significance uses the Wilcoxon signed-rank test over per-pair differences, which
assumes neither normality nor equal variance. With a null result the effect size
carries the claim, so the mean paired difference and its confidence interval are
reported alongside p -- "no significant difference" is only meaningful next to
the size of difference the data could have detected.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCENES = ("kitchen", "living_room", "workshop")
METRICS = (
    ("regions_inspected", "Regions inspected", 2),
    ("total_candidate_checks", "Candidate checks", 1),
    ("grounding_seconds", "Grounding time (s)", 2),
    ("planning_seconds", "Planning time (s)", 3),
)


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def load_arm(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(replay group, domain, variant) -> one trial's measurements."""
    trials: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sidecar_path in root.rglob("shadow_search_order_timing.json"):
        sidecar = _read(sidecar_path)
        if not sidecar:
            continue
        run_dir = sidecar_path.parent
        group = run_dir.relative_to(root).parts[0]
        result = _read(run_dir / "result.json")
        observed = _read(run_dir / "observed_scene_graph.json")

        nodes = observed.get("nodes", {})
        nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        relations = observed.get("relations", {})
        relations = list(relations.values()) if isinstance(relations, dict) else list(relations)
        semantic = sum(1 for n in nodes if n.get("semantic_labels") or n.get("canonical_category"))
        unary = sum(1 for n in nodes if n.get("unary_properties") or n.get("unary_predicates"))

        trials[(group, sidecar["domain"], sidecar["variant"])] = {
            "domain": sidecar["domain"],
            "variant": sidecar["variant"],
            "search_order": sidecar.get("search_order"),
            "search_seed": sidecar.get("search_seed"),
            "regions_inspected": len(result.get("inspected_regions") or []),
            "region_order_used": list(_read(run_dir / "run_manifest.json").get("region_order_used") or []),
            "semantic_checks": semantic,
            "unary_checks": unary,
            "pairwise_checks": len(relations),
            "total_candidate_checks": semantic + unary + len(relations),
            "grounding_seconds": sidecar.get("grounding_seconds"),
            "planning_seconds": sidecar.get("planning_seconds"),
            "terminal_status": result.get("status"),
        }
    return trials


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float | None, int]:
    """Two-sided Wilcoxon signed-rank p via normal approximation.

    Zero differences are dropped, which is the standard treatment and matters
    here: an ordering that changes nothing produces exact ties, and counting
    those as evidence either way would be wrong.
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n < 6:  # normal approximation is not trustworthy this small
        return None, n
    ordered = sorted(nonzero, key=abs)
    ranks: list[float] = [0.0] * n
    index = 0
    while index < n:
        stop = index
        while stop + 1 < n and abs(ordered[stop + 1]) == abs(ordered[index]):
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = shared
        index = stop + 1
    w_plus = sum(r for value, r in zip(ordered, ranks) if value > 0)
    mean = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return None, n
    z = (w_plus - mean) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, n


def paired_stats(fm: list[float], rand: list[float]) -> dict[str, Any]:
    diffs = [b - a for a, b in zip(fm, rand)]  # random minus FM
    n = len(diffs)
    mean_diff = statistics.fmean(diffs) if diffs else 0.0
    if n > 1:
        sd = statistics.stdev(diffs)
        half = 1.96 * sd / math.sqrt(n)
    else:
        sd = half = 0.0
    p, effective = wilcoxon_signed_rank(diffs)
    return {
        "n": n, "fm_mean": statistics.fmean(fm) if fm else None,
        "random_mean": statistics.fmean(rand) if rand else None,
        "mean_paired_diff": mean_diff, "ci95_halfwidth": half,
        "p_value": p, "nonzero_pairs": effective,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm", type=Path, required=True)
    ap.add_argument("--random", dest="rand", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fm_trials, rand_trials = load_arm(args.fm), load_arm(args.rand)
    keys = sorted(set(fm_trials) & set(rand_trials))
    only_fm, only_rand = len(fm_trials) - len(keys), len(rand_trials) - len(keys)
    print(f"FM arm {len(fm_trials)} trials, random arm {len(rand_trials)}, "
          f"{len(keys)} paired (unmatched: fm {only_fm}, random {only_rand})", file=sys.stderr)

    # A random arm that silently fell back to the FM order would look like a
    # perfect null, so the shuffle is verified rather than trusted.
    orderable = [k for k in keys if k[1] != "living_room"]
    randomised = sum(1 for k in orderable if rand_trials[k]["search_order"] == "random")
    reordered = sum(1 for k in orderable
                    if rand_trials[k]["region_order_used"] != fm_trials[k]["region_order_used"])
    print(f"orderable pairs {len(orderable)}: {randomised} drew a random policy, "
          f"{reordered} actually differ in region order", file=sys.stderr)

    report: dict[str, Any] = {"paired_trials": len(keys), "scenes": {},
                              "randomised_pairs": randomised, "reordered_pairs": reordered}
    lines: list[str] = []
    for scene in SCENES + ("overall",):
        scoped = keys if scene == "overall" else [k for k in keys if k[1] == scene]
        if not scoped:
            continue
        agreement = sum(1 for k in scoped
                        if fm_trials[k]["terminal_status"] == rand_trials[k]["terminal_status"])
        scene_out: dict[str, Any] = {"n": len(scoped), "terminal_status_agreement": agreement}
        lines.append(f"\n### {scene}  (n={len(scoped)}, outcome agreement {agreement}/{len(scoped)})\n")
        lines.append("| metric | FM-ranked | Random | paired diff (rand-FM) | 95% CI | p |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for key, label, places in METRICS:
            stats = paired_stats([fm_trials[k][key] for k in scoped],
                                 [rand_trials[k][key] for k in scoped])
            scene_out[key] = stats
            p_text = "n/a" if stats["p_value"] is None else f"{stats['p_value']:.3f}"
            lines.append(
                f"| {label} | {stats['fm_mean']:.{places}f} | {stats['random_mean']:.{places}f} | "
                f"{stats['mean_paired_diff']:+.{places}f} | "
                f"±{stats['ci95_halfwidth']:.{places}f} | {p_text} |")
        report["scenes"][scene] = scene_out

    print("\n".join(lines))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
