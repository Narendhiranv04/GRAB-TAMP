"""Rebuild a run root's protocol manifest from the per-episode manifests.

Concurrent legs wrote `protocol_manifest.json` and `batch_summary.json` to
fixed names, so the last writer replaced the others.  `fixed_20260910` holds
480 episodes across four methods and claims `methods: ["vlm_tamp"], variants:
["K11"], seeds: [4]`.  Nothing computed from those files, but they are the
files a reader trusts to say what a root contains.

The truth was never lost: every episode carries its own `method_manifest.json`
and `benchmark_execution_result.json`.  This reconstructs the root-level view
from them and writes `protocol_manifest.reconstructed.json`, leaving the
original in place as evidence of what happened.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def reconstruct(root: Path) -> dict:
    methods, variants, seeds, cameras = set(), set(), set(), set()
    params: dict[str, set] = defaultdict(set)
    n = 0
    for path in root.glob("*/*/images_*/seed_[0-9][0-9][0-9]/benchmark_execution_result.json"):
        result = json.loads(path.read_text())
        n += 1
        methods.add(str(result.get("method")))
        variants.add(str(result.get("variant")))
        seeds.add(result.get("seed"))
        cameras.add(result.get("camera_count"))
        manifest = path.parent / "method_manifest.json"
        if manifest.is_file():
            recorded = json.loads(manifest.read_text())
            for key in ("max_model_calls", "max_total_actions", "decoding",
                        "thinking_enabled", "model", "protocol"):
                if recorded.get(key) is not None:
                    params[key].add(json.dumps(recorded[key], sort_keys=True))
    return {
        "schema_version": 1,
        "reconstructed_from": "per-episode method_manifest.json + benchmark_execution_result.json",
        "reason": "root-level manifest was overwritten by concurrent legs",
        "episodes": n,
        "methods": sorted(m for m in methods if m != "None"),
        "variants": sorted(variants, key=lambda v: (len(v), v)),
        "seeds": sorted(s for s in seeds if s is not None),
        "camera_counts": sorted(c for c in cameras if c is not None),
        "per_episode_parameters": {
            key: sorted(json.loads(v) for v in values) for key, values in sorted(params.items())
        },
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for name in argv:
        root = Path(name)
        if not root.is_dir():
            print(f"skip (not a directory): {root}")
            continue
        rebuilt = reconstruct(root)
        if not rebuilt["episodes"]:
            print(f"skip (no episodes): {root}")
            continue
        out = root / "protocol_manifest.reconstructed.json"
        out.write_text(json.dumps(rebuilt, indent=2, sort_keys=True) + "\n")
        stale = root / "protocol_manifest.json"
        claimed = json.loads(stale.read_text()).get("methods") if stale.is_file() else None
        print(f"{root}")
        print(f"   episodes={rebuilt['episodes']} methods={rebuilt['methods']}")
        print(f"   old manifest claimed methods={claimed}")
        print(f"   wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
