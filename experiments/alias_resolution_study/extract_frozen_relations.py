#!/usr/bin/env python3
"""Extract every FM-authored functional relation from the frozen 320-trial run.

Read-only. No FM calls, no pipeline modification. The frozen manual resolver is
imported rather than reimplemented, so the baseline here is the same code that
ran during the benchmark, at the same commit.

Each FM relation names two of its own role ids. The manual resolver is
endpoint-sensitive -- it consults which predicates are legal between the two
declared entity kinds -- so the raw role ids are mapped to canonical roles using
the mapping the pipeline itself recorded (metadata.raw_role_to_canonical), and
entity kinds are read from the FM's own role declarations. Nothing here invents
a mapping the frozen run did not already make.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import interpret_relation  # noqa: E402

STREAMS = ("full320_s01", "full320_s02", "full320_s03")
SCORED = REPO / "benchmark_reports/FINAL_10x32_RESULTS/final_10x32.json"
OUT = Path(__file__).resolve().parent / "data" / "frozen_relation_instances.jsonl"


def _content(path: Path):
    try:
        c = json.loads(path.read_text(encoding="utf-8")).get("content")
    except Exception:
        return None
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            return None
    return c if isinstance(c, dict) else None


def main() -> int:
    # Restrict to the trials the frozen 320 report actually scored, so no
    # development or smoke run can leak into the dataset.
    scored = {(r["domain"], r["variant"], r["run_dir"]) for r in json.loads(SCORED.read_text())["rows"]}
    scored_dirs = {Path(rd) for _, _, rd in scored}

    rows, missing_raw = [], 0
    for run_dir in sorted(scored_dirs):
        raw_path = run_dir / "fm_diagnostics" / "fm_call_001.json"
        if not raw_path.exists():
            missing_raw += 1
            continue
        doc = _content(raw_path)
        if doc is None:
            missing_raw += 1
            continue
        parts = run_dir.parts
        domain, variant = parts[-3], parts[-2]
        stream = parts[1]
        repeat = parts[2]

        contract = doc.get("task_contract") or {}
        roles = {str(r.get("id")): r for r in (contract.get("functional_roles") or []) if isinstance(r, dict)}

        spec_path = run_dir / "functional_specification.json"
        raw_to_canon = {}
        if spec_path.exists():
            try:
                raw_to_canon = (json.loads(spec_path.read_text()).get("metadata") or {}).get("raw_role_to_canonical") or {}
            except Exception:
                raw_to_canon = {}

        for rel in contract.get("functional_relations") or []:
            if not isinstance(rel, dict):
                continue
            participants = [str(x) for x in (rel.get("participant_roles") or [])]
            subj_raw = participants[0] if participants else ""
            obj_raw = participants[1] if len(participants) > 1 else ""
            subj = str(raw_to_canon.get(subj_raw) or subj_raw)
            obj = str(raw_to_canon.get(obj_raw) or obj_raw)
            subj_kind = str((roles.get(subj_raw) or {}).get("entity_kind") or "OBJECT")
            obj_kind = str((roles.get(obj_raw) or {}).get("entity_kind") or "OBJECT")
            phrase = str(rel.get("relation") or "")

            try:
                res = interpret_relation(
                    domain, phrase, subj, obj, subj_kind, obj_kind,
                    required=bool(rel.get("required", True)),
                )
                status = res.status
                preds = [p.predicate_name for p in res.interpreted_predicates]
                category = res.category
            except Exception as error:  # a resolver raise is itself a disposition
                status, preds, category = f"RESOLVER_EXCEPTION:{type(error).__name__}", [], "ERROR"

            def role_ctx(raw_id: str) -> dict:
                """What the FM itself said about one endpoint role.

                The manual resolver reaches its answer from the endpoints, not
                from the phrase alone, so a fair classifier has to see them too.
                Everything here is the FM's own output -- no alias table.
                """
                r = roles.get(raw_id) or {}
                return {
                    "id": raw_id,
                    "entity_kind": str(r.get("entity_kind") or ""),
                    "function": str(r.get("function") or ""),
                    "description": str(r.get("description") or ""),
                    "candidate_categories": [str(c) for c in (r.get("candidate_categories") or [])],
                    "required_properties": [str(c) for c in (r.get("required_properties") or [])],
                }

            rows.append({
                "subject_role_context": role_ctx(subj_raw),
                "object_role_context": role_ctx(obj_raw),
                "stream": stream, "repeat": repeat, "domain": domain, "variant": variant,
                "run_dir": str(run_dir),
                "raw_relation_id": str(rel.get("id") or ""),
                "raw_relation_text": phrase,
                "raw_relation_text_normalized": " ".join(phrase.strip().split()).lower(),
                "participant_roles": participants,
                "raw_subject_role": subj_raw, "raw_object_role": obj_raw,
                "canonical_subject_role": subj, "canonical_object_role": obj,
                "subject_entity_kind": subj_kind, "object_entity_kind": obj_kind,
                "required": bool(rel.get("required", True)),
                "existing_pipeline_disposition": status,
                "existing_pipeline_category": category,
                "existing_canonical_predicate": preds[0] if len(preds) == 1 else None,
                "existing_canonical_predicates": preds,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"scored trials: {len(scored_dirs)}  raw responses missing: {missing_raw}")
    print(f"relation instances written: {len(rows)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
