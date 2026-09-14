#!/usr/bin/env python3
"""Record the frozen manual resolver's disposition for every extracted relation.

The baseline is the pipeline's own relation_interpreter, imported read-only at
the commit under study -- not a reimplementation, so there is no risk of the
baseline drifting from what actually ran. Its mapping is already captured during
extraction; this writes it out as a standalone prediction file and reports the
coverage statistic, which is the honest way to score it.

On the resolved subset the manual resolver defines the reference labels, so its
accuracy there is 1.0 by construction and is not evidence of anything. What it
can be scored on is coverage: how many of the FM's distinct phrasings it maps at
all.
"""
from __future__ import annotations

import collections
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "mujoco_scenes/functional_tamp_pipeline/relation_interpreter.py"


def main() -> int:
    rows = [json.loads(l) for l in (HERE / "data/frozen_relation_instances.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    out = HERE / "results/manual_baseline_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps({
                "domain": r["domain"], "variant": r["variant"],
                "stream": r["stream"], "repeat": r["repeat"],
                "raw_relation_text": r["raw_relation_text"],
                "raw_relation_text_normalized": r["raw_relation_text_normalized"],
                "disposition": r["existing_pipeline_disposition"],
                "predicted": r["existing_canonical_predicate"],
                "predicted_all": r["existing_canonical_predicates"],
            }, sort_keys=True) + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    digest = subprocess.run(["sha256sum", str(SOURCE)], capture_output=True, text=True).stdout.split()[0]
    unique = {(r["domain"], r["raw_relation_text_normalized"]) for r in rows}
    mapped_unique = {(r["domain"], r["raw_relation_text_normalized"]) for r in rows
                     if r["existing_canonical_predicates"]}
    summary = {
        "SOURCE_COMMIT": commit,
        "SOURCE_FILE": str(SOURCE.relative_to(REPO)),
        "SOURCE_SHA256": digest,
        "instances": len(rows),
        "instances_mapped": sum(1 for r in rows if r["existing_canonical_predicates"]),
        "unique_phrases": len(unique),
        "unique_phrases_mapped": len(mapped_unique),
        "unique_phrase_coverage": len(mapped_unique) / len(unique) if unique else None,
        "dispositions": dict(collections.Counter(r["existing_pipeline_disposition"] for r in rows).most_common()),
    }
    (HERE / "results/manual_baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
