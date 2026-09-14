#!/usr/bin/env python3
"""Run the benchmark with a zero-shot fallback on the canonicalization layers.

Thin wrapper in the same shape as evaluate_vlm_ablation.py: installs the shadow
and calls the real evaluator, so every trial goes through the unmodified
pipeline and differs only where the hand-written cue tables previously returned
nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from mujoco_scenes.fm_zs_canonicalization_shadow import (  # noqa: E402
    MIN_MARGIN, MIN_SCORE, zs_canonicalization,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--specification-root", type=Path, required=True)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="Run without the shadow, to pair against on the same commit.")
    args = ap.parse_args()

    import evaluate_vlm_functional_tamp as evaluator

    def run():
        evaluator.evaluate_all_variants(
            mode="vlm", spec_source="raw-replay", output_root=args.output_root,
            specification_root=args.specification_root, dry_run=True,
            resume=False, variants=args.variants,
        )

    if args.baseline:
        print("[zs] BASELINE arm (no shadow)", flush=True)
        run()
        return 0

    print(f"[zs] shadow arm, min_score={MIN_SCORE} min_margin={MIN_MARGIN}", flush=True)
    with zs_canonicalization() as stats:
        run()
    print(f"[zs] resolver stats: {json.dumps(stats, sort_keys=True)}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "zs_resolver_stats.json").write_text(
        json.dumps({"min_score": MIN_SCORE, "min_margin": MIN_MARGIN, "stats": stats},
                   indent=2) + "\n", encoding="utf-8")
    # A shadow that never fired would produce a duplicate of the baseline and
    # look like a clean null, so the counts are reported rather than assumed.
    fired = sum(v for k, v in stats.items() if k.endswith("_zs"))
    if not fired:
        print("[zs] WARNING: the fallback never fired", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
