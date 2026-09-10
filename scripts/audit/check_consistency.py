"""Cross-artifact agreement, seed coverage, and effect-vocabulary checks."""
from __future__ import annotations
import json, sys, collections, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_metrics_table import DEFAULT_ROOTS, REPO, _episodes

seeds = collections.defaultdict(set)
dupes = collections.Counter()
effect_preds = collections.defaultdict(collections.Counter)
mismatch = []
orphan_effects = []
roots_per_cell = collections.defaultdict(set)

for row, d in _episodes(DEFAULT_ROOTS):
    cell = (row["scene"], row["method"], row["variant"])
    seeds[cell].add(row["seed"])
    dupes[(row["scene"], row["method"], row["variant"], row["seed"], row["camera_count"])] += 1
    roots_per_cell[(row["scene"], row["method"])].add(d.relative_to(REPO).parts[2] if len(d.relative_to(REPO).parts) > 2 else "?")

    # effects actually recorded, versus the contract's vocabulary
    contract = d / "_private_evaluation/goal_contract.json"
    required = set()
    if contract.is_file():
        spec = json.loads(contract.read_text())
        required = set(spec.get("required_effects") or [])
        required |= {f"stirred({t})" for t in spec.get("stir_targets") or []}
    history = None
    for name, path_keys in (("episode_result.json", ("result", "action_history")),
                            ("discovery_replanning_result.json", ("history",))):
        p = d / name
        if not p.is_file():
            continue
        payload = json.loads(p.read_text())
        for k in path_keys:
            payload = payload.get(k) if isinstance(payload, dict) else None
        history = payload or []
        break
    if history is None:
        continue
    produced = set()
    for step in history:
        for e in step.get("effects") or []:
            effect_preds[row["scene"]][e.split("(")[0]] += 1
            produced.add(e)
    # an effect that names a goal condition but is not in this scene's contract
    if required:
        for e in produced:
            if e.startswith(("poured(", "placed(", "stirred(")) and e not in required:
                orphan_effects.append((str(d.relative_to(REPO)), e))

    # the runner's executed_actions vs the history it wrote
    claimed = row.get("executed_actions")
    if isinstance(claimed, int) and claimed != len(history):
        mismatch.append((str(d.relative_to(REPO)), f"claimed={claimed} history={len(history)}"))

print("### seed coverage (expect 10 per variant)")
bad = {k: sorted(v) for k, v in seeds.items() if v != set(range(10))}
print(f"      variants without exactly seeds 0-9: {len(bad)} of {len(seeds)}")
for k in sorted(bad)[:8]:
    v = bad[k]
    print(f"          {k} -> {v}")
print()
print(f"### duplicate (scene,method,variant,seed,cameras) after dedupe: "
      f"{sum(1 for v in dupes.values() if v > 1)}")
print()
print("### effect predicates recorded, by scene")
for s in sorted(effect_preds):
    print(f"      {s:<12}{dict(effect_preds[s])}")
print()
print(f"### executed_actions disagreeing with the written history: {len(mismatch)}")
for p, x in mismatch[:5]:
    print(f"      {p}  {x}")
print()
print(f"### goal-shaped effects produced that are NOT in the episode's contract: {len(orphan_effects)}")
for p, e in orphan_effects[:8]:
    print(f"      {e:<34}{p}")
print()
print("### which run roots feed each (scene,method) cell")
for k in sorted(roots_per_cell):
    print(f"      {k[0]:<12}{k[1]:<22}{sorted(roots_per_cell[k])}")
