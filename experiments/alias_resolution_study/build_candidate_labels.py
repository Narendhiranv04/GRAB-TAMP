#!/usr/bin/env python3
"""Snapshot the frozen executable predicate vocabulary as classifier labels.

Read-only against predicate_registry.py. Candidate hypothesis text is built
mechanically from the registry's own name and description; no per-predicate
paraphrase is authored here, and nothing from the manual alias/cue table is
used. Writing paraphrases by hand, or tuning them after seeing scores, would
reintroduce exactly the hand-engineering the study is testing the need for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mujoco_scenes.functional_tamp_pipeline.predicate_registry import (  # noqa: E402
    PREDICATE_REGISTRY, PredicateStatus,
)

OUT = Path(__file__).resolve().parent / "data" / "canonical_predicate_inventory.json"


def _seq(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(str(v) for v in value)
    return [str(value)]


def main() -> int:
    entries = []
    for (domain, name), sig in sorted(PREDICATE_REGISTRY.items()):
        status = getattr(sig.status, "value", str(sig.status))
        if status != PredicateStatus.ACTIVE_CANONICAL.value:
            continue
        description = str(getattr(sig, "description", "") or "")
        entries.append({
            "domain": domain,
            "predicate": name,
            "arity": int(getattr(sig, "arity", 2) or 2),
            "category": str(getattr(sig, "category", "") or ""),
            "subject_kinds": _seq(getattr(sig, "subject_kinds", None)),
            "object_kinds": _seq(getattr(sig, "object_kinds", None)),
            "allowed_subject_roles": _seq(getattr(sig, "allowed_subject_roles", None)),
            "allowed_object_roles": _seq(getattr(sig, "allowed_object_roles", None)),
            "description": description,
            "status": status,
            # Mechanical: canonical name plus the registry's own sentence.
            "hypothesis_text": f"{name}: {description}" if description else name,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    by_domain = {}
    for e in entries:
        by_domain.setdefault(e["domain"], []).append(e["predicate"])
    print(f"active canonical predicates: {len(entries)}")
    for d, preds in sorted(by_domain.items()):
        print(f"  {d}: {len(preds)} -> {sorted(preds)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
