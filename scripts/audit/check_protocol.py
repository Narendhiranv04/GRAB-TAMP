"""Decoding-condition and observation-contract checks across the whole grid."""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_metrics_table import DEFAULT_ROOTS, REPO, _episodes

sampling = collections.Counter()
maxtok = collections.Counter()
models = collections.Counter()
thinking = collections.Counter()
contract_keys = collections.defaultdict(collections.Counter)
obs_fact_keys = collections.defaultdict(collections.Counter)
leak = []

for row, d in _episodes(DEFAULT_ROOTS):
    scene = row.get("scene")

    # decoding conditions, read from the requests actually sent
    for call in sorted((d / "model_calls").glob("*.json")) if (d / "model_calls").is_dir() else []:
        payload = json.loads(call.read_text())
        requests = payload.get("model_requests") or []
        if not requests and payload.get("request"):
            requests = [payload["request"]]
        for req in requests:
            if not isinstance(req, dict):
                continue
            maxtok[req.get("max_tokens")] += 1
            models[req.get("model") or payload.get("model")] += 1
            kwargs = req.get("chat_template_kwargs") or {}
            thinking[kwargs.get("enable_thinking")] += 1
            sampling[tuple(sorted(
                (k, v) for k, v in req.items()
                if k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
            ))] += 1

    # what the model was allowed to see
    obs = d / "latest_observation.json"
    if obs.is_file():
        payload = json.loads(obs.read_text())
        for item in payload.get("visible_objects") or payload.get("visible_entities") or []:
            obs_fact_keys[scene].update((item.get("facts") or {}).keys())

    contract = d / "shared_observation_contract.json"
    if contract.is_file():
        payload = json.loads(contract.read_text())
        contract_keys[scene].update(payload.keys())

    # a private artifact must never appear inside anything the model saw
    for call in sorted((d / "model_calls").glob("*.json")) if (d / "model_calls").is_dir() else []:
        text = call.read_text()
        for needle in ("_private_evaluation", "goal_contract", "expected_gt_actions",
                       "internal_variant", "gt_assignment"):
            if needle in text:
                leak.append((str(call.relative_to(REPO)), needle))

def show(title, counter, limit=8):
    print(f"### {title}")
    for k, v in counter.most_common(limit):
        print(f"      {v:>6}  {k}")
    print()

show("max_tokens across every request sent", maxtok)
show("models", models)
show("enable_thinking", thinking)
show("sampling parameter sets", sampling, 4)
print("### object fact keys published, by scene")
for scene in sorted(obs_fact_keys):
    print(f"      {scene:<12} {dict(obs_fact_keys[scene])}")
print()
print("### shared_observation_contract keys, by scene")
for scene in sorted(contract_keys):
    print(f"      {scene:<12} {sorted(contract_keys[scene])}")
print()
print(f"### private-artifact references inside model_calls: {len(leak)}")
for p, n in leak[:5]:
    print(f"      {p}  {n}")
