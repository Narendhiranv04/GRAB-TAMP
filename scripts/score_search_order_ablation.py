#!/usr/bin/env python3
"""Score the region-inspection-order ablation, scene by scene.

Compares arms that differ only in the order regions are inspected:

    regions inspected   how much of the scene had to be opened
    candidate checks    objects put through semantic + geometric evaluation
    grounding seconds   region inspection + perception + phi
    planning seconds    the symbolic planner

The candidate-check definition is imported from the main scorer rather than
restated, so both experiments count the same thing.

Timing comes from the shadow sidecars. An arm recorded before the timers existed
(the frozen 320) simply has none, and is reported as absent rather than zero --
a zero would read as "instant", which is the opposite of missing.

Trials that never reached the solve phase are kept in the region and check
columns (inspecting nothing is a real outcome) but excluded from the timing
means, where they would otherwise drag both arms toward zero equally and mask
whatever difference exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from score_full_experiment import candidate_checks, _read, _as_list  # noqa: E402

SCENES = ("kitchen", "living_room", "workshop")


def collect_arm(name: str, root: Path) -> list[dict[str, Any]]:
    """One row per trial found under an arm's output root."""
    rows: list[dict[str, Any]] = []
    for result_file in sorted(root.rglob("result.json")):
        run_dir = result_file.parent
        parts = run_dir.parts
        if len(parts) < 3:
            continue
        domain, variant = parts[-3], parts[-2]
        if domain not in SCENES:
            continue
        result = _read(result_file)
        inspected = _as_list(result.get("inspected_regions"))
        row: dict[str, Any] = {
            "arm": name,
            "domain": domain,
            "variant": variant,
            "run_dir": str(run_dir),
            "pipeline_status": result.get("status"),
            "regions_inspected": len(inspected),
            "inspected_regions": inspected,
            "inspection_order_source": (
                "FM" if _read(run_dir / "functional_specification.json").get("region_ranking")
                else "SYSTEM_FALLBACK"),
        }
        row.update(candidate_checks(run_dir))

        sidecar = _read(run_dir / "shadow_search_order_timing.json")
        if sidecar:
            reached_solve = bool(sidecar.get("grounding_calls"))
            row.update({
                "timed": True,
                "reached_solve": reached_solve,
                "search_order": sidecar.get("search_order"),
                "search_seed": sidecar.get("search_seed"),
                "grounding_seconds": sidecar.get("grounding_seconds"),
                "planning_seconds": sidecar.get("planning_seconds"),
                "trial_seconds": sidecar.get("trial_seconds"),
            })
        else:
            row.update({"timed": False, "reached_solve": None, "search_order": None,
                        "search_seed": None, "grounding_seconds": None,
                        "planning_seconds": None, "trial_seconds": None})
        rows.append(row)
    return rows


def collect_arm_from_scored(name: str, scored_json: Path) -> list[dict[str, Any]]:
    """Rows for an arm whose raw trees are gone but whose scored rows survive.

    The frozen 320 was pruned to 275 surviving run dirs, so walking disk would
    quietly score a biased subset. The scored file still carries every trial's
    regions and candidate checks, already extracted by the main scorer.
    """
    payload = json.loads(scored_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload["rows"]:
        rows.append({
            "arm": name,
            "domain": row["domain"],
            "variant": row["variant"],
            "run_dir": row["run_dir"],
            "pipeline_status": row.get("pipeline_status"),
            "regions_inspected": row["regions_inspected"],
            "inspected_regions": row.get("inspected_regions", []),
            "inspection_order_source": row.get("inspection_order_source"),
            "observed_entities": row.get("observed_entities", 0),
            "semantic_candidate_checks": row["semantic_candidate_checks"],
            "unary_geometric_checks": row["unary_geometric_checks"],
            "pairwise_relation_checks": row["pairwise_relation_checks"],
            "total_candidate_checks": row["total_candidate_checks"],
            "detector_records": row.get("detector_records", 0),
            "verified_relation_trace_entries": row.get("verified_relation_trace_entries", 0),
            "timed": False, "reached_solve": None, "search_order": None,
            "search_seed": None, "grounding_seconds": None,
            "planning_seconds": None, "trial_seconds": None,
        })
    return rows


def collect_timing(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """(domain, variant) -> timing sidecars found under a measurement root."""
    found: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sidecar in sorted(root.rglob("shadow_search_order_timing.json")):
        data = _read(sidecar)
        if not data:
            continue
        found.setdefault((data["domain"], data["variant"]), []).append(data)
    return found


def attach_timing(rows: list[dict[str, Any]], timing: dict[tuple[str, str], list[dict[str, Any]]]) -> int:
    """Attach timing measured in a separate pass to an arm's rows.

    Measurement and scoring come from different passes here (the frozen arm's
    timing is replayed), so a row is matched by (domain, variant) and each
    measurement is consumed once rather than reused across repeats.
    """
    pending = {key: list(values) for key, values in timing.items()}
    attached = 0
    for row in rows:
        queue = pending.get((row["domain"], row["variant"]))
        if not queue:
            continue
        data = queue.pop(0)
        row.update({
            "timed": True,
            "reached_solve": bool(data.get("grounding_calls")),
            "search_order": data.get("search_order"),
            "search_seed": data.get("search_seed"),
            "grounding_seconds": data.get("grounding_seconds"),
            "planning_seconds": data.get("planning_seconds"),
            "trial_seconds": data.get("trial_seconds"),
        })
        attached += 1
    return attached


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scene-wise means for one arm."""
    out: dict[str, Any] = {}
    for scene in SCENES + ("overall",):
        scoped = rows if scene == "overall" else [r for r in rows if r["domain"] == scene]
        if not scoped:
            continue
        timed = [r for r in scoped if r["timed"] and r["reached_solve"]]
        out[scene] = {
            "trials": len(scoped),
            "regions_inspected_mean": _mean([r["regions_inspected"] for r in scoped]),
            "semantic_checks_mean": _mean([r["semantic_candidate_checks"] for r in scoped]),
            "unary_checks_mean": _mean([r["unary_geometric_checks"] for r in scoped]),
            "pairwise_checks_mean": _mean([r["pairwise_relation_checks"] for r in scoped]),
            "total_candidate_checks_mean": _mean([r["total_candidate_checks"] for r in scoped]),
            "timed_trials": len(timed),
            "grounding_seconds_mean": _mean([r["grounding_seconds"] for r in timed]),
            "grounding_seconds_median": (
                statistics.median([r["grounding_seconds"] for r in timed]) if timed else None),
            "planning_seconds_mean": _mean([r["planning_seconds"] for r in timed]),
            "trial_seconds_mean": _mean([r["trial_seconds"] for r in timed]),
        }
    return out


def _fmt(value: Any, places: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def render(summaries: dict[str, dict[str, Any]]) -> str:
    arms = list(summaries)
    lines: list[str] = []
    metrics = [
        ("regions_inspected_mean", "Regions inspected", 2),
        ("total_candidate_checks_mean", "Candidate checks (semantic+geometric)", 1),
        ("semantic_checks_mean", "  - semantic", 1),
        ("unary_checks_mean", "  - unary geometric", 1),
        ("pairwise_checks_mean", "  - pairwise relational", 1),
        ("grounding_seconds_mean", "Grounding time (s)", 2),
        ("planning_seconds_mean", "Planning time (s)", 3),
    ]
    for scene in SCENES + ("overall",):
        present = [a for a in arms if scene in summaries[a]]
        if not present:
            continue
        lines.append(f"\n### {scene}\n")
        header = "| metric | " + " | ".join(present) + " |"
        lines.append(header)
        lines.append("| :--- | " + " | ".join("---:" for _ in present) + " |")
        lines.append("| trials | " + " | ".join(
            str(summaries[a][scene]["trials"]) for a in present) + " |")
        for key, label, places in metrics:
            cells = [_fmt(summaries[a][scene].get(key), places) for a in present]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                    help="Arm scored by walking a run-dir tree.")
    ap.add_argument("--arm-json", action="append", default=[], metavar="NAME=PATH",
                    help="Arm scored from an existing scored-rows JSON (use when raw trees were pruned).")
    ap.add_argument("--arm-timing", action="append", default=[], metavar="NAME=PATH",
                    help="Attach timing measured in a separate pass to an already-declared arm.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.arm and not args.arm_json:
        raise SystemExit("at least one --arm or --arm-json is required")

    def _split(spec: str, flag: str) -> tuple[str, str]:
        if "=" not in spec:
            raise SystemExit(f"{flag} expects NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        return name, path

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in args.arm:
        name, path = _split(spec, "--arm")
        arm_rows[name] = collect_arm(name, Path(path))
    for spec in args.arm_json:
        name, path = _split(spec, "--arm-json")
        arm_rows[name] = collect_arm_from_scored(name, Path(path))

    for spec in args.arm_timing:
        name, path = _split(spec, "--arm-timing")
        if name not in arm_rows:
            raise SystemExit(f"--arm-timing names unknown arm {name!r}")
        timing = collect_timing(Path(path))
        attached = attach_timing(arm_rows[name], timing)
        measured = sum(len(v) for v in timing.values())
        print(f"[{name}] timing: {measured} measured, {attached} attached", file=sys.stderr)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, rows in arm_rows.items():
        if not rows:
            print(f"WARNING: arm {name!r} matched no trials", file=sys.stderr)
        all_rows.extend(rows)
        summaries[name] = summarize(rows)
        timed = sum(1 for r in rows if r["timed"])
        print(f"[{name}] {len(rows)} trials, {timed} with timing", file=sys.stderr)

    report = render(summaries)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"summaries": summaries, "rows": all_rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
