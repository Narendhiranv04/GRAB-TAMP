#!/usr/bin/env python3
"""Run the live benchmark with the region inspection order under ablation.

A thin wrapper in the same shape as `evaluate_vlm_ablation.py`: it installs the
shadow ordering choice plus phase timers and then calls the real evaluator, so
every trial goes through the unmodified pipeline and differs from the deployed
run only in the order regions are inspected.

    auto     the deployed policy -- the FM's own region ranking, system-completed
    random   a seeded shuffle of the whole region universe, reseeded per trial

The random policy is not new behaviour; it already exists in the frozen search
contract. This entry point only selects it and supplies the per-trial seed.

The patch is applied in-process, which is why this is its own entry point: the
repetition runner spawns the evaluator as a subprocess, and a monkeypatch in the
parent would not reach it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mujoco_scenes.fm_search_order_shadow import ORDER_MODES, search_order_shadow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-mode", required=True, choices=sorted(ORDER_MODES))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Shifts every per-trial seed; lets a whole arm be redrawn reproducibly.")
    parser.add_argument("--spec-source", default="live", choices=["live", "replay"])
    parser.add_argument("--specification-root", type=Path, default=None)
    parser.add_argument("--variants", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    import evaluate_vlm_functional_tamp as evaluator

    print(f"[search-order] mode={args.order_mode} seed_base={args.seed_base} "
          f"spec_source={args.spec_source}", flush=True)
    with search_order_shadow(evaluator, order_mode=args.order_mode,
                             seed_base=args.seed_base) as stats:
        try:
            evaluator.evaluate_all_variants(
                mode="vlm",
                spec_source=args.spec_source,
                output_root=args.output_root,
                specification_root=args.specification_root,
                dry_run=True,
                resume=False,
                variants=args.variants,
            )
        except evaluator.InfrastructureUnavailable as error:
            print(f"\nABORTED (infrastructure): {error}", file=sys.stderr, flush=True)
            return evaluator.INFRASTRUCTURE_EXIT_CODE

    # A patch that silently failed to apply would produce a second copy of the
    # deployed run and look like a clean null result, so the counts are reported
    # rather than assumed.
    print(f"[search-order] trials={stats['trials']} "
          f"random={stats['random_trials']} auto={stats['auto_trials']} "
          f"untimed={stats['untimed_trials']}", flush=True)
    if args.order_mode == "random" and not stats["random_trials"]:
        print("[search-order] WARNING: no trial received a random order", file=sys.stderr, flush=True)
    # The three domains have separate solve paths; a renamed entry point would
    # quietly leave one of them untimed, which must not pass as a zero.
    if stats["missing_instrumentation"]:
        print(f"[search-order] WARNING: not instrumented: {stats['missing_instrumentation']}",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
