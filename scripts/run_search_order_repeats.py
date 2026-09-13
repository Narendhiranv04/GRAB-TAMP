#!/usr/bin/env python3
"""Run N repetitions of the full matrix under one region-inspection-order arm.

Same shape as run_ablation_repeats.py -- each repetition is an independent
sample of the whole 32-variant matrix, nothing selects among them, and a
repetition that fails stays on disk rather than being silently redone. The only
difference is that the evaluator is the search-order wrapper, so the regions are
inspected in a different order while the pipeline itself is untouched.

The per-trial seed is derived from the attempt directory, so every repetition
draws a fresh ordering for every variant rather than replaying one shuffle.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
INFRASTRUCTURE_EXIT_CODE = 86
# The full 9-key sampler, imported rather than restated: an earlier ablation
# runner dropped presence_penalty and the model ran away, making the arms
# incomparable. Both arms of this ablation must sample identically.
sys.path.insert(0, str(REPO / "scripts"))
from run_live_repeat_experiment import SAMPLER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order-mode", required=True, choices=["auto", "random"])
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen35-9b")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--spec-source", default="live", choices=["live", "replay"])
    ap.add_argument("--specification-root", type=Path, default=None)
    args = ap.parse_args()

    from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
    ran, failed = [], []
    for index in range(1, args.repeats + 1):
        repeat_dir = args.output_root / f"repeat_{index:02d}"
        manifest = repeat_dir / "repeat_manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("finished"):
            print(f"[{args.order_mode} repeat {index:02d}] already finished", flush=True)
            continue
        attempts = sorted(p for p in repeat_dir.glob("attempt_*") if p.is_dir())
        attempt_dir = repeat_dir / f"attempt_{len(attempts) + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TAMP_FM_BASE_URL=args.base_url, TAMP_FM_MODEL=args.model,
                   PYTHONPATH=".", **REPRODUCIBLE_CHILD_ENV, **SAMPLER)
        cmd = [sys.executable, "scripts/evaluate_vlm_search_order.py",
               "--order-mode", args.order_mode, "--output-root", str(attempt_dir),
               "--seed-base", str(args.seed_base), "--spec-source", args.spec_source]
        if args.specification_root is not None:
            # Replay pairs an arm against a frozen specification set, so the
            # per-repeat spec root has to track the repeat being replayed.
            spec_root = args.specification_root / f"repeat_{index:02d}"
            spec_root = spec_root if spec_root.is_dir() else args.specification_root
            cmd += ["--specification-root", str(spec_root)]
        started = time.time()
        code = subprocess.run(cmd, env=env).returncode
        elapsed = round(time.time() - started, 1)
        errors = [] if code == 0 else [f"evaluator exit code {code}"]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "repeat_index": index, "order_mode": args.order_mode,
            "seed_base": args.seed_base, "spec_source": args.spec_source,
            "finished": not errors, "errors": errors,
            "aborted_infrastructure": code == INFRASTRUCTURE_EXIT_CODE,
            "authoritative_attempt": attempt_dir.name, "seconds": elapsed,
            "model": args.model, "base_url": args.base_url,
        }, indent=2) + "\n")
        (ran if not errors else failed).append(index)
        print(f"[{args.order_mode} repeat {index:02d}] "
              f"{'ok' if not errors else 'FAILED ' + str(errors)} in {elapsed}s", flush=True)
        if code == INFRASTRUCTURE_EXIT_CODE:
            print(f"[{args.order_mode}] endpoint gone; stopping", flush=True)
            break
    print(f"\n{args.order_mode}: ran {ran} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
