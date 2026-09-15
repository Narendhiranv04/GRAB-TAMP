#!/usr/bin/env python3
"""Score the zero-shot resolver against the frozen manual resolver's mapping.

Two evaluation sets, kept apart deliberately:

REFERENCE_RESOLVED   phrases the frozen manual resolver mapped to an active
                     canonical binary predicate. Its own score here is not an
                     independent evaluation -- it defines the reference -- so it
                     is reported as agreement, not accuracy.

FROZEN_UNRESOLVED    phrases the manual resolver could not map. These have no
                     reference label, so no accuracy is computed for them. They
                     are reported as proposal confidence only, because calling
                     them errors would assume the manual cues were complete,
                     which is the assumption under test.

Every metric is reported instance-weighted and unique-phrase-weighted: 973
instances collapse to 510 unique phrasings, and a handful of common phrases
would otherwise dominate.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ("ZS-Global", "ZS-Typed", "ZS-Context")


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {g for g, _ in pairs} | {p for _, p in pairs if p}
    scores = []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return statistics.fmean(scores) if scores else 0.0


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["domain"], r["raw_relation_text_normalized"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=HERE / "results/zero_shot_predictions.jsonl")
    ap.add_argument("--inventory", type=Path, default=HERE / "data/canonical_predicate_inventory.json")
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.predictions.read_text(encoding="utf-8").splitlines() if l.strip()]
    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    binary = {(p["domain"], p["predicate"]) for p in inv if p["arity"] == 2}

    resolved = [r for r in rows
                if r.get("existing_canonical_predicate")
                and (r["domain"], r["existing_canonical_predicate"]) in binary]
    unresolved = [r for r in rows if r["existing_pipeline_disposition"] == "UNINTERPRETABLE_REQUIRED_RELATION"]
    task_causal = [r for r in rows
                   if r.get("existing_canonical_predicate")
                   and (r["domain"], r["existing_canonical_predicate"]) not in binary]

    report: dict = {
        "dataset": {
            "relation_instances": len(rows),
            "unique_phrases": len({(r["domain"], r["raw_relation_text_normalized"]) for r in rows}),
            "reference_resolved_instances": len(resolved),
            "reference_resolved_unique": len({(r["domain"], r["raw_relation_text_normalized"]) for r in resolved}),
            "frozen_unresolved_instances": len(unresolved),
            "frozen_unresolved_unique": len({(r["domain"], r["raw_relation_text_normalized"]) for r in unresolved}),
            "task_causal_instances": len(task_causal),
            "by_domain": dict(collections.Counter(r["domain"] for r in rows)),
            "resolved_by_predicate": dict(collections.Counter(
                r["existing_canonical_predicate"] for r in resolved).most_common()),
            "lexical_variants_per_predicate": {
                pred: len({r["raw_relation_text_normalized"] for r in resolved
                           if r["existing_canonical_predicate"] == pred})
                for pred in {r["existing_canonical_predicate"] for r in resolved}
            },
        },
        "agreement": {},
        "per_predicate": {},
        "per_domain": {},
        "confidence": {},
        "unresolved_analysis": {},
    }

    for cond in CONDITIONS:
        for weighting, subset in (("instance", resolved), ("unique_phrase", dedupe(resolved))):
            pairs = [(r["existing_canonical_predicate"], r.get(f"{cond}_predicted")) for r in subset]
            correct = sum(1 for g, p in pairs if g == p)
            report["agreement"][f"{cond}|{weighting}"] = {
                "n": len(pairs),
                "top1_agreement": correct / len(pairs) if pairs else None,
                "macro_f1": macro_f1(pairs),
            }
        report["per_predicate"][cond] = {}
        for pred in sorted({r["existing_canonical_predicate"] for r in resolved}):
            sub = [r for r in resolved if r["existing_canonical_predicate"] == pred]
            hit = sum(1 for r in sub if r.get(f"{cond}_predicted") == pred)
            report["per_predicate"][cond][pred] = {
                "n": len(sub), "agreement": hit / len(sub) if sub else None,
                "unique_phrasings": len({r["raw_relation_text_normalized"] for r in sub}),
            }
        report["per_domain"][cond] = {}
        for dom in sorted({r["domain"] for r in resolved}):
            sub = [r for r in resolved if r["domain"] == dom]
            hit = sum(1 for r in sub if r.get(f"{cond}_predicted") == r["existing_canonical_predicate"])
            report["per_domain"][cond][dom] = {"n": len(sub), "agreement": hit / len(sub) if sub else None}

        ok = [r[f"{cond}_top_score"] for r in resolved
              if r.get(f"{cond}_predicted") == r["existing_canonical_predicate"] and r.get(f"{cond}_top_score")]
        bad = [r[f"{cond}_top_score"] for r in resolved
               if r.get(f"{cond}_predicted") != r["existing_canonical_predicate"] and r.get(f"{cond}_top_score")]
        curve = []
        scored = sorted((r for r in resolved if r.get(f"{cond}_top_score") is not None),
                        key=lambda r: -r[f"{cond}_top_score"])
        for k in range(1, len(scored) + 1):
            head = scored[:k]
            acc = sum(1 for r in head if r.get(f"{cond}_predicted") == r["existing_canonical_predicate"]) / k
            if k % max(1, len(scored) // 20) == 0 or k == len(scored):
                curve.append({"coverage": k / len(scored), "accuracy": acc,
                              "score_threshold": head[-1][f"{cond}_top_score"]})
        report["confidence"][cond] = {
            "mean_top_score_agreeing": statistics.fmean(ok) if ok else None,
            "mean_top_score_disagreeing": statistics.fmean(bad) if bad else None,
            "mean_margin": statistics.fmean([r[f"{cond}_margin"] for r in resolved
                                             if r.get(f"{cond}_margin") is not None]) or None,
            "accuracy_coverage_curve": curve,
        }

        # No reference labels exist here, so these are proposals, not predictions.
        buckets = collections.Counter()
        for r in unresolved:
            top, margin = r.get(f"{cond}_top_score"), r.get(f"{cond}_margin")
            if top is None:
                buckets["NO_CANDIDATES"] += 1
            elif top >= 0.70 and (margin or 0) >= 0.40:
                buckets["HIGH_CONFIDENCE_PROPOSAL"] += 1
            elif top >= 0.50:
                buckets["AMBIGUOUS"] += 1
            else:
                buckets["LOW_CONFIDENCE"] += 1
        report["unresolved_analysis"][cond] = {
            "n_instances": len(unresolved),
            "n_unique": len({(r["domain"], r["raw_relation_text_normalized"]) for r in unresolved}),
            "buckets": dict(buckets),
            "note": "No reference label exists for these; buckets describe confidence, not correctness.",
        }

    # Cross-repeat consistency: does one canonical predicate's many phrasings map alike?
    consistency = {}
    for pred in sorted({r["existing_canonical_predicate"] for r in resolved}):
        sub = dedupe([r for r in resolved if r["existing_canonical_predicate"] == pred])
        for cond in CONDITIONS:
            preds = [r.get(f"{cond}_predicted") for r in sub]
            top = collections.Counter(preds).most_common(1)
            consistency.setdefault(pred, {})[cond] = {
                "unique_phrasings": len(sub),
                "modal_prediction": top[0][0] if top else None,
                "agreement_rate": (top[0][1] / len(sub)) if sub and top else None,
            }
    report["cross_phrasing_consistency"] = consistency

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "aggregate_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    errors = [{
        "domain": r["domain"], "variant": r["variant"], "stream": r["stream"], "repeat": r["repeat"],
        "raw_relation_text": r["raw_relation_text"],
        "gold_predicate": r["existing_canonical_predicate"],
        "predicted": r.get("ZS-Typed_predicted"),
        "ranked": r.get("ZS-Typed_ranked"),
        "participant_roles": r["participant_roles"],
        "subject_entity_kind": r["subject_entity_kind"], "object_entity_kind": r["object_entity_kind"],
    } for r in resolved if r.get("ZS-Typed_predicted") != r["existing_canonical_predicate"]]
    (args.outdir / "disagreement_analysis.json").write_text(
        json.dumps({"n": len(errors), "errors": errors}, indent=2) + "\n", encoding="utf-8")

    (args.outdir / "unresolved_analysis.json").write_text(json.dumps({
        "note": report["unresolved_analysis"]["ZS-Typed"]["note"],
        "rows": [{
            "domain": r["domain"], "raw_relation_text": r["raw_relation_text"],
            "predicted": r.get("ZS-Typed_predicted"), "top_score": r.get("ZS-Typed_top_score"),
            "margin": r.get("ZS-Typed_margin"), "ranked": r.get("ZS-Typed_ranked"),
        } for r in dedupe(unresolved)],
    }, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"dataset": report["dataset"], "agreement": report["agreement"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
