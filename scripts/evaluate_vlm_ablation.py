#!/usr/bin/env python3
"""Run the live benchmark with one evidence channel withheld from perception.

A thin wrapper: it applies the shadow evidence mask and then calls the real
evaluator, so every trial goes through the unmodified pipeline and differs from
the full-evidence run only in what the grounder is allowed to observe.

The mask patches grounding.ground_graph in-process, which is why this exists as
its own entry point -- the repetition runner spawns the evaluator as a
subprocess, and a monkeypatch applied in the parent would not reach it.

Conditions (cumulative, so a difference between two is attributable to the one
channel that was added):

    semantic_only    YOLO-World semantics only
    semantic_unary   semantics + unary point-cloud geometry
    full             semantics + unary + binary relations   (the main run)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mujoco_scenes.fm_ablation_shadow import evidence_masked  # noqa: E402
from mujoco_scenes.fm_evidence_ablation import CONDITION_LABELS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITION_LABELS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    from evaluate_vlm_functional_tamp import (
        INFRASTRUCTURE_EXIT_CODE, InfrastructureUnavailable, evaluate_all_variants,
    )
    print(f"[ablation] condition={args.condition} ({CONDITION_LABELS[args.condition]})",
          flush=True)
    with evidence_masked(args.condition) as stats:
        try:
            evaluate_all_variants(mode="vlm", spec_source="live",
                                  output_root=args.output_root,
                                  specification_root=None, dry_run=True,
                                  resume=False, variants=args.variants)
        except InfrastructureUnavailable as error:
            print(f"\nABORTED (infrastructure): {error}", file=sys.stderr, flush=True)
            return INFRASTRUCTURE_EXIT_CODE
    # A mask that silently failed to apply would produce a second copy of the
    # full-evidence run and look like a clean null result, so the count is
    # reported rather than assumed.
    print(f"[ablation] groundings masked: {stats.get('groundings')}", flush=True)
    if not stats.get("groundings"):
        # Legitimate when every trial failed before grounding (spec failures
        # reach no grounder), so this is reported rather than treated as an
        # error -- but a whole matrix with zero masked groundings would mean the
        # patch never took, which the aggregate check downstream will catch.
        print("[ablation] WARNING: no grounding was masked in this pass",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
