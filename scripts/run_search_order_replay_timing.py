#!/usr/bin/env python3
"""Measure grounding and planning time for an arm by replaying frozen specs.

The frozen 320 predates the phase timers, so its grounding and planning time
were never recorded. Re-running it live would cost another full pass and, at
temperature 0.6, would return different specifications -- the arms would then
differ by FM sampling on top of ordering, which is the one thing this ablation
must not confound.

Replaying the saved `functional_specification.json` avoids both. The model is
not asked the same question twice; everything downstream of the spec --
vocabulary, open-vocabulary semantics, point-cloud geometry, G_O, region search,
grounding, A* -- still runs for real, and that downstream part is exactly what
the timers measure. The FM call was never inside the measurement.

Replay starts from the raw FM response rather than the canonicalized
specification. The canonicalization step is where region proposals are mapped
onto canonical ids, so a specification written before that mapping was corrected
has already lost most of its ranking; replaying it would carry the loss forward
and leave the FM arm barely FM-ranked. Re-canonicalizing from the raw response
rebuilds the ranking under the current mapping.

Only the trials whose raw response still exists on disk are replayed; the rest
were lost to earlier disk pressure and are reported as reduced coverage rather
than silently dropped.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from run_live_repeat_experiment import SAMPLER  # noqa: E402


def surviving_groups(scored_json: Path) -> dict[str, list[str]]:
    """Attempt root -> variants whose raw FM response is still on disk."""
    rows = json.loads(scored_json.read_text(encoding="utf-8"))["rows"]
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        run_dir = Path(row["run_dir"])
        if not (run_dir / "fm_diagnostics" / "fm_call_001.json").exists():
            continue
        groups[str(run_dir.parents[2])].append(row["variant"])
    return {k: sorted(set(v)) for k, v in sorted(groups.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored-json", type=Path,
                    default=REPO / "benchmark_reports/FINAL_10x32_RESULTS/final_10x32.json")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--order-mode", default="auto", choices=["auto", "random"])
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()

    groups = surviving_groups(args.scored_json)
    total = sum(len(v) for v in groups.values())
    print(f"[replay-timing] {total} trials across {len(groups)} attempt roots, "
          f"order_mode={args.order_mode}", flush=True)

    from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
    ok = failed = 0
    for index, (attempt_root, variants) in enumerate(groups.items(), start=1):
        tag = "__".join(Path(attempt_root).parts[-3:])
        out_dir = args.output_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=".", **REPRODUCIBLE_CHILD_ENV, **SAMPLER)
        cmd = [sys.executable, "scripts/evaluate_vlm_search_order.py",
               "--order-mode", args.order_mode, "--output-root", str(out_dir),
               "--seed-base", str(args.seed_base),
               "--spec-source", "raw-replay", "--specification-root", attempt_root,
               "--variants", ",".join(variants)]
        started = time.time()
        code = subprocess.run(cmd, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
        elapsed = round(time.time() - started, 1)
        (out_dir / "replay_group_manifest.json").write_text(json.dumps({
            "attempt_root": attempt_root, "variants": variants,
            "order_mode": args.order_mode, "seed_base": args.seed_base,
            "exit_code": code, "seconds": elapsed,
        }, indent=2) + "\n", encoding="utf-8")
        if code == 0:
            ok += 1
        else:
            failed += 1
        print(f"[replay-timing {index}/{len(groups)}] {tag} "
              f"({len(variants)} variants) exit={code} in {elapsed}s", flush=True)

    print(f"\n[replay-timing] groups ok={ok} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
