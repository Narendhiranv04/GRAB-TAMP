"""Is the same task instance defined identically for every method and seed?"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_metrics_table import DEFAULT_ROOTS, REPO, _episodes

contracts = collections.defaultdict(set)     # (scene,variant) -> {contract fingerprint}
gt_actions = collections.defaultdict(set)
adapters   = collections.defaultdict(set)
by_method  = collections.defaultdict(lambda: collections.defaultdict(set))

for row, d in _episodes(DEFAULT_ROOTS):
    key = (row["scene"], row["variant"])
    priv = d / "_private_evaluation"

    c = priv / "goal_contract.json"
    if c.is_file():
        spec = json.loads(c.read_text())
        fp = (tuple(sorted(spec.get("required_effects") or [])),
              tuple(sorted(spec.get("stir_targets") or [])))
        contracts[key].add(fp)
        by_method[key]["contract"].add((row["method"], fp))

    g = priv / "expected_gt_actions.json"
    if g.is_file():
        acts = json.loads(g.read_text()).get("actions") or []
        fp = tuple((a.get("operator"), tuple(a.get("arguments") or [])) for a in acts)
        gt_actions[key].add(fp)

    a = priv / "variant_adapter.json"
    if a.is_file():
        adapters[key].add(json.loads(a.read_text()).get("internal_variant"))

def report(title, mapping, render=lambda v: v):
    bad = {k: v for k, v in mapping.items() if len(v) > 1}
    print(f"### {title}: {len(bad)} task instances defined more than one way "
          f"(of {len(mapping)})")
    for k, v in sorted(bad)[:6]:
        print(f"      {k}: {len(v)} distinct definitions")
        for item in list(v)[:2]:
            print(f"          {render(item)[:150]}")
    print()

report("goal_contract per (scene,variant)", contracts, lambda x: str(x))
report("expected_gt_actions per (scene,variant)", gt_actions, lambda x: f"{len(x)} actions: {x[:3]}")
report("variant_adapter internal_variant", adapters, lambda x: str(x))

print("### paper variant -> internal variant map")
for k in sorted(adapters):
    print(f"      {k[0]:<12}{k[1]:<5}{sorted(adapters[k])}")
