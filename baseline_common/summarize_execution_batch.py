"""Summarize physical execution batches without mixing planning-only runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FIELDS = (
    "success", "executed_actions", "model_calls", "raw_vlm_requests",
    "replans", "planning_latency_s", "elapsed_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[Path] = set()
    for root in arguments.roots:
        paths = list(root.resolve().rglob("discovery_replanning_result.json"))
        paths.extend(root.resolve().rglob("benchmark_execution_result.json"))
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            _attach_gt_outcome(row, path.parent)
            grouped[(
                str(row.get("scene")),
                str(row.get("method", "robust_tamp")),
                str(row.get("protocol", "native")),
                int(row.get("camera_count", 5)),
            )].append(row)
    if not grouped:
        raise SystemExit(
            "No discovery_replanning_result.json or benchmark_execution_result.json files found"
        )

    output_rows = []
    for (scene, method, protocol, cameras), rows in sorted(grouped.items()):
        # Physical outcomes depend on the contact solver, so episodes produced
        # by different engine builds are different result classes and must not
        # average into one number.  Older artifacts carry no version at all;
        # those are reported as unknown rather than silently treated as equal.
        engines = sorted({str(row.get("mujoco_version", "unrecorded")) for row in rows})
        if len(engines) > 1:
            raise SystemExit(
                f"Refusing to pool {scene}/{method}/{protocol}/images_{cameras}: "
                f"episodes span MuJoCo builds {', '.join(engines)}. Summarize "
                "each build separately."
            )
        # Feasible and infeasible variants are different questions and must not
        # share a success rate.  `success` is physical goal satisfaction, which
        # an infeasible variant cannot yield by construction, so pooling them
        # scores a correct rejection identically to a blundered attempt and
        # drags the headline number down by the infeasible fraction.  Feasible
        # episodes are scored on success; infeasible ones on whether the method
        # actually said INFEASIBLE.  Episodes carrying no GT verdict (older
        # artifacts, or runners with no comparison) are counted as unscored
        # rather than silently folded into either group.
        feasible = [r for r in rows if r.get("expected_outcome") == "FEASIBLE"]
        infeasible = [r for r in rows if r.get("expected_outcome") == "INFEASIBLE"]
        unscored = [r for r in rows if r.get("expected_outcome") is None]
        matched = [r for r in rows if r.get("outcome_match") is not None]
        output_rows.append({
            "scene": scene,
            "method": method,
            "protocol": protocol,
            "images": cameras,
            "mujoco_version": engines[0],
            "trials": len(rows),
            "feasible_trials": len(feasible),
            "infeasible_trials": len(infeasible),
            "unscored_trials": len(unscored),
            "success_percent": round(100 * mean(bool(row["success"]) for row in rows), 2),
            "feasible_success_percent": (
                round(100 * mean(bool(r["success"]) for r in feasible), 2)
                if feasible else None
            ),
            # On an infeasible variant the correct behaviour is to reject the
            # task, so this is the metric, not `success`.
            "infeasible_rejection_percent": (
                round(100 * mean(r.get("predicted_outcome") == "INFEASIBLE"
                                 for r in infeasible), 2)
                if infeasible else None
            ),
            "outcome_correct_percent": (
                round(100 * mean(bool(r["outcome_match"]) for r in matched), 2)
                if matched else None
            ),
            "mean_executed_actions": _mean(rows, "executed_actions"),
            "mean_model_calls": _mean(rows, "model_calls"),
            "mean_raw_vlm_requests": _mean(rows, "raw_vlm_requests"),
            "mean_replans": _mean(rows, "replans"),
            "mean_planning_latency_s": _mean(rows, "planning_latency_s"),
            "mean_elapsed_seconds": _mean(rows, "elapsed_seconds"),
        })
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "execution_summary.json").write_text(
        # 2: rows gained the feasible/infeasible partition.  `success_percent`
        # is kept for continuity but pools both kinds of variant; read
        # `feasible_success_percent` and `infeasible_rejection_percent`
        # instead once infeasible variants are present in a batch.
        json.dumps({"schema_version": 2, "rows": output_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "execution_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(output_rows, indent=2))


def _attach_gt_outcome(row: dict[str, Any], episode_dir: Path) -> None:
    """Fill the GT outcome fields from the episode's planning artifact.

    Episodes written before ``write_execution_result`` recorded these carry
    them only in the sibling ``episode_result.json`` under ``gt_comparison``.
    Reading them here makes an existing grid scorable without rewriting any
    artifact on disk, which would destroy the provenance of a completed run.
    A value already present in the scored artifact always wins.
    """
    if row.get("expected_outcome") is not None:
        return
    planning = episode_dir / "episode_result.json"
    if not planning.is_file():
        return
    try:
        comparison = json.loads(planning.read_text(encoding="utf-8")).get(
            "gt_comparison"
        ) or {}
    except (json.JSONDecodeError, OSError):
        return
    row["expected_outcome"] = comparison.get("expected_outcome")
    row["predicted_outcome"] = comparison.get("predicted_outcome")
    row["outcome_match"] = comparison.get("outcome_match")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(mean(values), 4) if values else None


if __name__ == "__main__":
    main()
