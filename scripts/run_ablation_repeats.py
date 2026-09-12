#!/usr/bin/env python3
"""Run N repetitions of the full matrix under one evidence-ablation condition.

Mirrors run_live_repeat_experiment.py: each repetition is an independent sample
of the whole 32-variant matrix, nothing selects among them, and a repetition
that fails stays on disk rather than being silently redone. The only difference
is that the evaluator is the ablation wrapper, so the grounder sees a masked
observation.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INFRASTRUCTURE_EXIT_CODE = 86
SAMPLER = {"TAMP_FM_ENABLE_THINKING": "true", "TAMP_FM_MAX_TOKENS": "28000"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen35-9b")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
    ran, failed = [], []
    for index in range(1, args.repeats + 1):
        repeat_dir = args.output_root / f"repeat_{index:02d}"
        manifest = repeat_dir / "repeat_manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("finished"):
            print(f"[{args.condition} repeat {index:02d}] already finished", flush=True)
            continue
        attempts = sorted(p for p in repeat_dir.glob("attempt_*") if p.is_dir())
        attempt_dir = repeat_dir / f"attempt_{len(attempts) + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TAMP_FM_BASE_URL=args.base_url, TAMP_FM_MODEL=args.model,
                   PYTHONPATH=".", **REPRODUCIBLE_CHILD_ENV, **SAMPLER)
        started = time.time()
        code = subprocess.run(
            [sys.executable, "scripts/evaluate_vlm_ablation.py",
             "--condition", args.condition, "--output-root", str(attempt_dir)],
            env=env).returncode
        elapsed = round(time.time() - started, 1)
        errors = [] if code == 0 else [f"evaluator exit code {code}"]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "repeat_index": index, "condition": args.condition,
            "finished": not errors, "errors": errors,
            "aborted_infrastructure": code == INFRASTRUCTURE_EXIT_CODE,
            "authoritative_attempt": attempt_dir.name, "seconds": elapsed,
            "model": args.model, "base_url": args.base_url,
        }, indent=2) + "\n")
        (ran if not errors else failed).append(index)
        print(f"[{args.condition} repeat {index:02d}] "
              f"{'ok' if not errors else 'FAILED ' + str(errors)} in {elapsed}s", flush=True)
        if code == INFRASTRUCTURE_EXIT_CODE:
            print(f"[{args.condition}] endpoint gone; stopping", flush=True)
            break
    print(f"\n{args.condition}: ran {ran} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
