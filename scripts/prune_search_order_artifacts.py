#!/usr/bin/env python3
"""Prune regenerable intermediates from finished trials of a live arm.

The scorers read ten small JSON files at each run-dir root. Everything under
`observed_search/.../stages/` is per-region inspection diagnostics -- stage point
clouds, growth GIFs, per-stage graph dumps -- and the final observed graph is
already preserved at the run-dir root. On this machine a kitchen trial writes
53 MB, of which ~51 MB is that tree, and there is not enough free disk to hold
320 of them.

Only the arm passed on the command line is touched, and only trials that have
finished: the shadow timing sidecar is written in run_pipeline's `finally`, so
its presence means the trial is done and nothing is still writing. Files the
scorers read are never candidates.

vlm_inputs holds the images the FM actually saw. Those are kept for the first
repeat as evidence and pruned for the rest, where they are reproducible from the
scene by the same deterministic render.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Written by the shadow layer's `finally`; its presence means the trial is done.
DONE_MARKER = "shadow_search_order_timing.json"

# Read by score_full_experiment / score_search_order_ablation / evaluation_metrics.
PROTECTED = frozenset({
    "result.json", "functional_specification.json", "observed_scene_graph.json",
    "detection_diagnostics.json", "physical_relation_verification_trace.json",
    "graph_grounding_result.json", "symbolic_problem.json", "run_manifest.json",
    "evaluation_records.json", DONE_MARKER,
})

KEEP_IMAGES_FOR = "repeat_01"


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune_trial(run_dir: Path, *, dry_run: bool) -> int:
    freed = 0
    targets: list[Path] = []
    for stages in run_dir.rglob("stages"):
        if stages.is_dir() and "observed_search" in stages.parts:
            targets.append(stages)
    for pattern in ("*.gif", "*.ply"):
        targets.extend(p for p in run_dir.rglob(pattern) if p.is_file())
    if KEEP_IMAGES_FOR not in run_dir.parts:
        vlm_inputs = run_dir / "vlm_inputs"
        if vlm_inputs.is_dir():
            targets.append(vlm_inputs)

    for target in targets:
        if not target.exists():
            continue
        if target.is_file() and target.name in PROTECTED:
            continue
        size = target.stat().st_size if target.is_file() else _dir_bytes(target)
        freed += size
        if dry_run:
            continue
        try:
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        except OSError as error:
            print(f"  prune failed {target}: {error}", file=sys.stderr, flush=True)
            freed -= size
    return freed


def sweep(root: Path, *, dry_run: bool) -> tuple[int, int]:
    trials = freed = 0
    for marker in sorted(root.rglob(DONE_MARKER)):
        run_dir = marker.parent
        gained = prune_trial(run_dir, dry_run=dry_run)
        if gained:
            trials += 1
            freed += gained
    return trials, freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="Arm output root; only this is touched.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="Sweep repeatedly at this interval instead of once.")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"root does not exist yet: {args.root}", file=sys.stderr)
    while True:
        trials, freed = sweep(args.root, dry_run=args.dry_run)
        if trials:
            verb = "would free" if args.dry_run else "freed"
            print(f"[prune] {trials} trials, {verb} {freed / 2**30:.2f} GiB", flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
