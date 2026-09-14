#!/usr/bin/env python3
"""Zero-shot canonicalization for the role, region and capability layers.

Same contract as the relation study: the classifier sees only what the system
already publishes about its own vocabulary, plus what the FM itself declared
about the item being mapped. The hand-written alias tables -- KITCHEN_REGION_
ALIASES, WORKSHOP_REGION_ALIASES, the ontology's detector aliases and each
capability's `semantic_cues` -- are never shown to it, as label text, prompt or
example. They are the thing under test.

Label text per layer, all mechanically derived:

    role        canonical name + the categories the ontology accepts for it
    region      canonical id + a phrase derived from the scene's own geometry,
                because the registry publishes no prose for regions and writing
                some by hand is the practice being tested
    capability  canonical id + the registry's semantic_description

Premise is the FM's own wording plus the context it supplied for that item.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
TEMPLATE = "This refers to {}."

# Positions read from the scene XML; the description is composed from them so
# that no region wording is authored by hand.
KITCHEN_GEOM = {
    "C1": (-0.35, 0.65, 0.96), "C2": (0.35, 0.65, 0.96),
    "D1": (-0.44, -0.30, 0.46), "D2": (0.44, -0.30, 0.46),
    "B1": (0.52, 0.18, 0.58),
}
TABLE_TOP_Z = 0.58


def region_labels(domain: str) -> list[dict]:
    if domain == "kitchen":
        out = []
        for rid, (x, y, z) in KITCHEN_GEOM.items():
            side = "left" if x < 0 else "right"
            if z > TABLE_TOP_Z + 0.2:
                where = f"a closed storage cabinet mounted high on the wall, on the {side}"
            elif z < TABLE_TOP_Z:
                where = f"a closed drawer underneath the table, on the {side}"
            else:
                where = f"a container resting on top of the table, on the {side}"
            out.append({"id": rid, "text": f"{rid}: {where}"})
        return out
    if domain == "workshop":
        return [{"id": r, "text": f"{r}: the {r.lower().replace('_', ' ')}"}
                for r in ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")]
    return []


def role_labels(domain: str) -> list[dict]:
    import yaml
    data = yaml.safe_load((REPO / "mujoco_scenes/configs/runtime_functional_semantic_ontology.yaml").read_text())
    roles = (data["domains"].get(domain) or {}).get("roles") or (data["domains"].get(domain) or {})
    out = []
    for name, body in sorted(roles.items()):
        cats = (body or {}).get("accepted_categories") or []
        pretty = name.replace("_", " ").lower()
        text = f"{name}: the {pretty}"
        if cats:
            text += f", typically a {' or '.join(cats[:4])}"
        out.append({"id": name, "text": text})
    return out


def capability_labels(domain: str) -> list[dict]:
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import CANONICAL_ROBOT_CAPABILITIES as C
    caps = C.get(domain) or []
    caps = list(caps.values()) if isinstance(caps, dict) else list(caps)
    return [{"id": c.capability_id, "text": f"{c.capability_id}: {c.semantic_description}"} for c in caps]


def premise(layer: str, row: dict) -> str:
    if layer == "role":
        bits = [f"a role the plan calls \"{row['raw_id']}\""]
        if row.get("function"):
            bits.append(f"whose job is {row['function'].rstrip('.').lower()}")
        if row.get("candidate_categories"):
            bits.append(f"and which could be a {' or '.join(row['candidate_categories'][:3])}")
        if row.get("required_properties"):
            bits.append(f"[must be: {', '.join(row['required_properties'][:3])}]")
        return " ".join(bits)
    if layer == "region":
        return f"{row.get('label','')}: {row.get('visual_description','')}".strip(": ")
    return (f"an operation the plan calls \"{row['raw_text']}\" "
            f"involving {', '.join(row.get('participant_roles') or []) or 'unspecified participants'}")


LABELS = {"role": role_labels, "region": region_labels, "capability": capability_labels}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", required=True, choices=sorted(LABELS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch, transformers
    from transformers import pipeline
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    clf = pipeline("zero-shot-classification", model=MODEL, device=-1)

    rows = [json.loads(l) for l in (HERE / f"data/frozen_{args.layer}_instances.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    out = args.out or HERE / f"results/zero_shot_{args.layer}_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    with out.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            cands = LABELS[args.layer](row["domain"])
            if not cands:
                continue
            prem = premise(args.layer, row)
            if not prem.strip():
                continue
            key = (row["domain"], prem.lower())
            if key in cache:
                ranked = cache[key]
            elif len(cands) == 1:
                ranked = [(cands[0]["id"], 1.0)]
            else:
                res = clf(prem, [c["text"] for c in cands], hypothesis_template=TEMPLATE, multi_label=False)
                back = {c["text"]: c["id"] for c in cands}
                ranked = [(back[l], float(s)) for l, s in zip(res["labels"], res["scores"])]
            cache[key] = ranked
            rec = dict(row)
            rec.update(zs_ranked=ranked, zs_predicted=ranked[0][0], zs_top=ranked[0][1],
                       zs_margin=(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else ranked[0][1],
                       zs_n_candidates=len(cands))
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if i % 200 == 0:
                print(f"  {args.layer} {i}/{len(rows)} (cache {len(cache)})", flush=True)
    print(f"{args.layer}: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
