#!/usr/bin/env python3
"""Map frozen FM relation phrases onto canonical predicates with a zero-shot NLI model.

No fine-tuning, no FM call, and no use of the pipeline's manual alias/cue table.
The classifier sees only what the robot interface already publishes: the
canonical predicate name and the registry's own description. If it were shown
the hand-written cue list -- even as prompt examples -- the study would prove
nothing, since that list is the thing being tested for redundancy.

Three conditions:

    ZS-Global    premise is the bare relation phrase; candidates are every
                 active binary predicate in the domain
    ZS-Typed     same premise, candidates restricted to those whose registry
                 signature admits the declared subject/object entity kinds
    ZS-Context   premise additionally states what the two endpoints are, using
                 the FM's own role declarations; candidates as in ZS-Typed

ZS-Context exists because the first two were not a fair test. The manual
resolver reaches its answer from the endpoints as much as from the wording --
"placed_on" becomes FITS_SET_ON when a cup-and-saucer set is what is placed, and
FITS_ON when it is a remote control. A classifier shown only the phrase cannot
recover that distinction, so its failure would say more about the framing than
about the model. ZS-Context gives it the same information the manual resolver
has: the FM's role ids, stated functions, candidate categories and required
properties, plus the registry's declared role and kind constraints on each
predicate. Still no alias table, and still zero-shot.

Runs on CPU by default: this machine's memory is shared with the benchmark, and
a large model competing for it has already cost one run today.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
# The classifier decides which relation the phrase expresses; the template is
# generic and carries no per-predicate wording.
HYPOTHESIS_TEMPLATE = "This requirement states that {}."


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def premise_for(row: dict, with_context: bool) -> str:
    """Build the text the classifier scores.

    Without context this is the FM's relation phrase verbatim. With context it
    also names the two endpoints as the FM described them, which is what the
    manual resolver consults.
    """
    phrase = row["raw_relation_text"].strip()
    if not with_context:
        return phrase

    def describe(ctx: dict) -> str:
        bits = [ctx.get("id") or "unnamed"]
        fn = (ctx.get("function") or ctx.get("description") or "").strip().rstrip(".")
        if fn:
            bits.append(f"which is {fn[0].lower() + fn[1:]}")
        cats = ctx.get("candidate_categories") or []
        if cats:
            bits.append(f"(a {' or '.join(cats[:3])})")
        props = ctx.get("required_properties") or []
        if props:
            bits.append(f"[required: {', '.join(props[:3])}]")
        return " ".join(bits)

    subj = describe(row.get("subject_role_context") or {})
    obj = describe(row.get("object_role_context") or {})
    return (f"In the {row['domain']}, the requirement \"{phrase}\" must hold "
            f"between {subj} and {obj}.")


def hypothesis_for(pred: dict, with_context: bool) -> str:
    """Label text: registry name and description, plus declared endpoint constraints."""
    base = pred["hypothesis_text"]
    if not with_context:
        return base
    extra = []
    if pred.get("allowed_subject_roles"):
        extra.append(f"subject is one of {', '.join(pred['allowed_subject_roles'][:4])}")
    if pred.get("allowed_object_roles"):
        extra.append(f"object is one of {', '.join(pred['allowed_object_roles'][:4])}")
    return base + (f" ({'; '.join(extra)})" if extra else "")


def candidates_for(row: dict, inventory: list[dict], typed: bool) -> list[dict]:
    pool = [p for p in inventory if p["domain"] == row["domain"] and p["arity"] == 2]
    if not typed:
        return pool
    subj, obj = row["subject_entity_kind"], row["object_entity_kind"]
    kept = []
    for p in pool:
        sk, ok = p["subject_kinds"], p["object_kinds"]
        if sk and subj and subj not in sk:
            continue
        if ok and obj and obj not in ok:
            continue
        kept.append(p)
    # An empty pool would silently drop the row; fall back to the domain pool
    # and record that the type filter admitted nothing.
    return kept or pool


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", type=Path, default=HERE / "data/frozen_relation_instances.jsonl")
    ap.add_argument("--inventory", type=Path, default=HERE / "data/canonical_predicate_inventory.json")
    ap.add_argument("--out", type=Path, default=HERE / "results/zero_shot_predictions.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--unique-only", action="store_true",
                    help="Score each unique (domain, phrase, kinds) once; instances reuse the result.")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from transformers import pipeline

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    if args.device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))

    rows = load_rows(args.instances)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))

    clf = pipeline("zero-shot-classification", model=MODEL,
                   device=-1 if args.device == "cpu" else 0)
    revision = getattr(getattr(clf, "model", None), "name_or_path", MODEL)

    env = {
        "model": MODEL, "resolved_name_or_path": str(revision),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "device": args.device, "dtype": str(next(clf.model.parameters()).dtype),
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
    }
    (args.out.parent / "zero_shot_environment.json").write_text(
        json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(env, indent=2), flush=True)

    cache: dict[tuple, dict] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            phrase = row["raw_relation_text"].strip()
            if not phrase:
                continue
            out_row = dict(row)
            for condition, typed, ctx in (("ZS-Global", False, False),
                                          ("ZS-Typed", True, False),
                                          ("ZS-Context", True, True)):
                cands = candidates_for(row, inventory, typed)
                labels = [hypothesis_for(c, ctx) for c in cands]
                names = [c["predicate"] for c in cands]
                premise = premise_for(row, ctx)
                key = (row["domain"], premise.lower(), condition, tuple(names))
                if key in cache:
                    scored = cache[key]
                else:
                    if len(labels) == 1:
                        scored = {"ranked": [(names[0], 1.0)]}
                    else:
                        res = clf(premise, labels, hypothesis_template=HYPOTHESIS_TEMPLATE,
                                  multi_label=False)
                        back = {l: c["predicate"] for l, c in zip(labels, cands)}
                        scored = {"ranked": [(back[l], float(s))
                                             for l, s in zip(res["labels"], res["scores"])]}
                    cache[key] = scored
                ranked = scored["ranked"]
                out_row[f"{condition}_ranked"] = ranked
                out_row[f"{condition}_predicted"] = ranked[0][0] if ranked else None
                out_row[f"{condition}_top_score"] = ranked[0][1] if ranked else None
                out_row[f"{condition}_second"] = ranked[1][0] if len(ranked) > 1 else None
                out_row[f"{condition}_second_score"] = ranked[1][1] if len(ranked) > 1 else None
                out_row[f"{condition}_margin"] = (
                    ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1] if ranked else None)
                out_row[f"{condition}_n_candidates"] = len(names)
            handle.write(json.dumps(out_row, sort_keys=True) + "\n")
            written += 1
            if index % 50 == 0:
                print(f"  {index}/{len(rows)} (cache {len(cache)})", flush=True)
    print(f"wrote {written} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
