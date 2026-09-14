"""Hours of slack against the deadline, from measured tokens and throughput.

Episode counts are the wrong unit: a Workshop VLM-TAMP episode burns 181k
tokens and a Living Room ROBUST-TAMP one burns a fraction of that, and the
machine is throughput-bound at roughly 800-900 tok/s.  Remaining tokens divided
by measured throughput is the only projection that has held.
"""
import datetime, glob, json, statistics as st, sys, time, urllib.request
from collections import Counter

DEADLINE = datetime.datetime(2026, 9, 14, 12, 0)
METRICS = "http://127.0.0.1:18000/metrics"
N = {"kitchen": 120, "living_room": 100, "workshop": 100}
OLD = {"living_room": "newgoal_20260911", "workshop": "fixed_20260910"}
PROBE = {("workshop", "vlm_tamp"):
         "runs/workshop/execution/budget_probe_20260912/vlm_tamp/*/images_*/seed_*"}


def _gen_tokens():
    with urllib.request.urlopen(METRICS, timeout=15) as fh:
        for line in fh.read().decode().splitlines():
            if line.startswith("vllm:generation_tokens_total"):
                return float(line.split()[-1])
    return None


def throughput(window=40):
    a = _gen_tokens()
    if a is None:
        return None
    time.sleep(window)
    b = _gen_tokens()
    return None if b is None else (b - a) / window


def tokens_per_episode(pattern):
    per = []
    for episode in glob.glob(pattern):
        total = sum(
            (response.get("usage") or {}).get("completion_tokens") or 0
            for call in glob.glob(f"{episode}/model_calls/*.json")
            for response in (json.load(open(call)).get("model_responses") or [])
        )
        if total:
            per.append(total)
    return st.median(per) if per else None


def remaining_tokens(skip=()):
    done = Counter()
    for scene in N:
        for path in glob.glob(
            f"runs/{scene}/execution/final_20260913/*/*/images_3/"
            "seed_[0-9][0-9][0-9]/benchmark_execution_result.json"
        ):
            done[(scene, json.load(open(path)).get("method"))] += 1
    total = 0
    for scene in ("kitchen", "living_room", "workshop"):
        for method in ("vlm_tamp", "owl_tamp", "robust_tamp", "vilain_tamp"):
            if (scene, method) in skip:
                continue
            new = f"runs/{scene}/execution/final_20260913/{method}/*/images_3/seed_[0-9][0-9][0-9]"
            per = tokens_per_episode(new)
            if per is None and (scene, method) in PROBE:
                per = tokens_per_episode(PROBE[(scene, method)])
            if per is None and scene in OLD:
                per = tokens_per_episode(
                    f"runs/{scene}/execution/{OLD[scene]}/{method}/*/images_3/seed_[0-9][0-9][0-9]"
                )
            total += max(0, N[scene] - done.get((scene, method), 0)) * (per or 40000)
    return total, sum(done.values())


def main():
    rate = throughput()
    if not rate:
        print("MARGIN: endpoint unreachable, cannot measure")
        return 0
    total, done = remaining_tokens()
    cut, _ = remaining_tokens(skip={("workshop", "vilain_tamp")})
    now = datetime.datetime.now()
    need = total / rate / 3600
    slack = (DEADLINE - now).total_seconds() / 3600 - need
    finish = (now + datetime.timedelta(hours=need)).strftime("%a %H:%M")
    print(f"MARGIN {slack:+.1f} h | episodes {done}/1280 | {total/1e6:.1f} Mtok "
          f"at {rate:.0f} tok/s | finish {finish} | "
          f"cutting workshop ViLaIn would save {(total-cut)/rate/3600:.1f} h")
    return slack


if __name__ == "__main__":
    sys.exit(0 if main() > 0 else 1)
