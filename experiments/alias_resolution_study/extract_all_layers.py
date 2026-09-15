#!/usr/bin/env python3
"""Extract every free-form FM item that must be mapped onto a fixed system vocabulary.

Three canonicalization layers, one shape of problem: the model invents wording,
the robot needs a closed vocabulary, and today a hand-written table bridges them.

    role        FM role id + its stated function   -> canonical role
    region      FM region label + description      -> canonical search region
    capability  FM operation token + participants  -> canonical robot capability

Reference labels come from what the frozen run itself recorded, never from the
classifier: roles from metadata.raw_role_to_canonical, regions from the
pipeline's own resolver called read-only, capabilities from
metadata.fm_semantic_accounting.

Read-only. No FM calls, no pipeline modification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mujoco_scenes.kitchen_vlm_functional_graph import (  # noqa: E402
    resolve_kitchen_region_proposal, AmbiguousCanonicalizationError,
)
from mujoco_scenes.workshop_phase1.requirements import resolve_workshop_region_proposal  # noqa: E402

SCORED = REPO / "data/metrics/scored_320.json"
OUT = Path(__file__).resolve().parent / "data"


def content(path: Path):
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
    rows = json.loads(SCORED.read_text(encoding="utf-8"))["rows"]
    roles, regions, caps = [], [], []
    for row in rows:
        run_dir = Path(row["run_dir"])
        doc = content(run_dir / "fm_diagnostics" / "fm_call_001.json")
        if doc is None:
            continue
        domain, variant = row["domain"], row["variant"]
        spec = run_dir / "functional_specification.json"
        md = (json.loads(spec.read_text()).get("metadata") or {}) if spec.exists() else {}
        raw_to_canon = md.get("raw_role_to_canonical") or {}
        accounting = {str(e.get("raw_id")): e for e in (md.get("fm_semantic_accounting") or [])
                      if isinstance(e, dict)}

        contract = doc.get("task_contract") or {}
        common = {"domain": domain, "variant": variant, "run_dir": str(run_dir)}

        for role in contract.get("functional_roles") or []:
            if not isinstance(role, dict):
                continue
            rid = str(role.get("id") or "")
            roles.append({**common, "layer": "role", "raw_id": rid,
                          "raw_text": rid.replace("_", " "),
                          "entity_kind": str(role.get("entity_kind") or ""),
                          "function": str(role.get("function") or ""),
                          "description": str(role.get("description") or ""),
                          "candidate_categories": [str(c) for c in (role.get("candidate_categories") or [])],
                          "required_properties": [str(c) for c in (role.get("required_properties") or [])],
                          "reference": raw_to_canon.get(rid)})

        for reg in ((doc.get("observation_guidance") or {}).get("inspectable_regions") or []):
            if not isinstance(reg, dict):
                continue
            label, desc = str(reg.get("label") or ""), str(reg.get("visual_description") or "")
            ref = None
            try:
                if domain == "kitchen":
                    ref = resolve_kitchen_region_proposal(reg)
                elif domain == "workshop":
                    ref = resolve_workshop_region_proposal(reg)
            except AmbiguousCanonicalizationError:
                ref = None
            except Exception:
                ref = None
            regions.append({**common, "layer": "region", "raw_id": str(reg.get("id") or ""),
                            "raw_text": f"{label} — {desc}".strip(" —"),
                            "label": label, "visual_description": desc,
                            "reason": str(reg.get("reason") or ""), "reference": ref})

        for op in contract.get("operation_pairings") or []:
            if not isinstance(op, dict):
                continue
            oid = str(op.get("id") or "")
            entry = accounting.get(oid) or {}
            canon = entry.get("canonical_representation")
            ref = canon[0] if isinstance(canon, list) and len(canon) == 1 else None
            caps.append({**common, "layer": "capability", "raw_id": oid,
                         "raw_text": str(op.get("operation") or ""),
                         "participant_roles": [str(x) for x in (op.get("participant_roles") or [])],
                         "operation_count": op.get("operation_count"),
                         "disposition": entry.get("disposition"), "reference": ref})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("role", roles), ("region", regions), ("capability", caps)):
        path = OUT / f"frozen_{name}_instances.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in data:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        mapped = sum(1 for i in data if i["reference"])
        uniq = len({(i["domain"], i["raw_text"].strip().lower()) for i in data})
        uniq_mapped = len({(i["domain"], i["raw_text"].strip().lower()) for i in data if i["reference"]})
        print(f"{name:<11} instances {len(data):>5}  mapped {mapped:>5} ({mapped/max(1,len(data))*100:4.1f}%)"
              f"  |  unique phrasings {uniq:>4}  mapped {uniq_mapped:>4}"
              f" ({uniq_mapped/max(1,uniq)*100:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
