"""Derive ViLaIn goal coverage from its own benchmark snapshots.

ViLaIn does record a terminal state (`benchmark/terminal_state_snapshot.json`)
and a goal decomposition (`benchmark/terminal_subgoal_evaluation.json`); the
coverage column was blank because the metrics generator looked for the
`latest_observation.json` / `_private_evaluation/` layout the other baselines
write, which ViLaIn does not use.

Its own `coverage` field is not used here.  Its subgoals demand ground truth's
specific object-to-region assignment ("a2_drink_left on a2_personal_left"),
which is the identity-based reading that marked 30 of 95 correct Living Room
solves wrong.  Required slots are taken from those subgoals but relaxed to
(region, role), so the column means the same thing it means for every other
method.
"""
from __future__ import annotations
import json, pathlib
from collections import Counter

def _role(name: str) -> str | None:
    n = name.lower()
    for key in ("drink", "snack", "remote"):
        if key in n:
            return key
    return None

def living_room(ep: pathlib.Path) -> tuple[int, int]:
    sub = ep / "benchmark/terminal_subgoal_evaluation.json"
    snap = ep / "benchmark/terminal_state_snapshot.json"
    if not (sub.is_file() and snap.is_file()):
        return 0, 0
    try:
        subgoals = json.loads(sub.read_text()).get("results") or []
        state = json.loads(snap.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    required: Counter = Counter()
    for row in subgoals:
        sg = row.get("subgoal") or {}
        if str(sg.get("predicate", "")).upper() != "ON":
            continue
        role = _role(str(sg.get("subject") or ""))
        target = sg.get("target")
        if role and target:
            required[(target, role)] += 1
    if not required:
        return 0, 0
    final: Counter = Counter()
    for name, obj in (state.get("objects") or {}).items():
        role = _role(name)
        support = obj.get("support")
        if role and support and not obj.get("held"):
            final[(support, role)] += 1
    return sum(min(n, final[k]) for k, n in required.items()), sum(required.values())

def workshop(ep: pathlib.Path) -> tuple[int, int]:
    """The same three terminal conditions scored for the other methods.

    A ViLaIn snapshot carries only `articulation` and `contained_in`; there is
    no `fastened` relation in the schema, so the fastening clause cannot be
    read from it.  That is harmless while ViLaIn executes nothing -- an episode
    with no actions cannot have fastened anything, so 0 is the true answer --
    but it would silently report 0 for an episode that did fasten.  So this
    refuses to answer rather than guess once any action has run.
    """
    result = ep / "benchmark_execution_result.json"
    try:
        if (json.loads(result.read_text()).get("executed_actions") or 0) > 0:
            return 0, 0       # not derivable: no fastening relation is recorded
    except (OSError, json.JSONDecodeError):
        return 0, 0
    snap = ep / "benchmark/terminal_state_snapshot.json"
    if not snap.is_file():
        return 0, 0
    try:
        state = json.loads(snap.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    rel = state.get("relations") or {}
    contained = rel.get("contained_in") or {}
    joint = "workshop_frame_joint"
    # seated: the screw is contained in the joint rather than in storage
    seated = any("screw" in str(o) for o in contained.get(joint, ()))
    # fastened: recorded only when a fastening actually happened
    fastened = bool(rel.get("fastened") or state.get("fastened"))
    bench = [k for k in contained if "main_workbench_zone" in k]
    driver_back = bool(
        fastened
        and not (state.get("held_objects") or [])
        and any("driver" in str(o) for k in bench for o in contained.get(k, ()))
    )
    return int(seated) + int(fastened) + int(driver_back), 3

ROOTS = {
    "living_room": ("runs/living_room/execution/vilain_20260908d", living_room),
    "workshop": ("runs/workshop/execution/vilain_20260908d", workshop),
    "kitchen": ("runs/kitchen/execution/vilain_kitchen_20260909", None),
}
EXCLUDED = (".interrupted", ".timeout_", ".transport_", ".provenance_")

if __name__ == "__main__":
    for scene, (rel, fn) in ROOTS.items():
        base = pathlib.Path(rel)
        if not base.is_dir() or fn is None:
            print(f"  {scene:12s} no derivation available yet")
            continue
        hit = req = eps = 0
        per_variant: dict[str, list[float]] = {}
        for res in base.rglob("benchmark_execution_result.json"):
            if any(k in str(res) for k in EXCLUDED):
                continue
            row = json.loads(res.read_text())
            if row.get("expected_outcome") != "FEASIBLE":
                continue
            h, r = fn(res.parent)
            if not r:
                continue
            eps += 1; hit += h; req += r
            per_variant.setdefault(row.get("variant"), []).append(100.0 * h / r)
        pct = f"{100.0*hit/req:.1f}" if req else "--"
        print(f"  {scene:12s} feasible episodes={eps:3d}  coverage={pct}%  ({hit}/{req} conditions)")
        for v in sorted(per_variant, key=lambda x: (len(x), x)):
            vals = per_variant[v]
            print(f"       {v:4s} n={len(vals):2d}  {sum(vals)/len(vals):5.1f}%")
