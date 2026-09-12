"""Plan-level goal coverage: what a method's plan intends, before physics.

`paper_metrics_table.py` scores goal coverage on the terminal simulator state,
so it charges every method for whatever the manipulation controller lost
downstream of the plan.  This module scores the same goal conditions against
the plan itself, under the counterfactual that every proposed action succeeds.

The pair separates two failure modes that a single coverage number conflates:

    planning coverage   goal conditions the method's plan would establish
    execution coverage  goal conditions the simulator actually reached

A method that decomposes the task correctly and then loses it in the
controller shows a wide gap.  A method that never proposes the right subgoals
shows a low planning coverage, and its execution number is not the
interesting one.

Scoring a plan is artifact I/O only, so this covers the whole grid in seconds
and needs no episode re-run.  Every denominator here is the same denominator
`paper_metrics_table.py` uses for the executed number, so the two columns are
directly comparable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_metrics_table import (  # noqa: E402
    COVERAGE_SCENES, DEFAULT_ROOTS, LABEL, REPO, _coverage, _episodes,
)


# --------------------------------------------------------------------------
# Reading each method's plan
#
# The four baselines record a plan in four different places and two different
# vocabularies (skill sequences and predicate subgoals), so each is read on its
# own terms and normalized to a set of predicate tuples.
# --------------------------------------------------------------------------

def _norm_subgoal(predicate: str, arguments: dict) -> tuple | None:
    """VLM-TAMP and its descendants propose predicates."""
    name = str(predicate).upper()
    get = lambda *keys: next(  # noqa: E731
        (str(arguments[k]) for k in keys if arguments.get(k) is not None), None
    )
    if name == "POURED":
        return ("poured", get("source_id"), get("target_id"))
    if name == "PLACED":
        return ("placed", get("object_id"), get("region_id"))
    if name == "STIRRED":
        return ("stirred", get("target_id"))
    if name == "FASTENED":
        return ("fastened", get("fastener_id"), get("target_id"))
    if name == "INSERTED":
        return ("inserted", get("fastener_id", "object_id"), get("target_id"))
    return None


def _norm_action(operator: str, arguments) -> tuple | None:
    """OWL-TAMP and ROBUST-TAMP emit skill sequences."""
    name = str(operator).upper()
    if isinstance(arguments, dict):
        order = {
            "PICK": ("object_id",),
            "PLACE": ("object_id", "region_id"),
            "POUR": ("source_id", "target_id"),
            "STIR": ("tool_id", "target_id"),
            "INSERT": ("fastener_id", "target_id"),
            "FASTEN": ("tool_id", "fastener_id", "target_id"),
            "INSPECT": ("region_id",),
        }.get(name, ())
        values = [str(arguments[k]) for k in order if arguments.get(k) is not None]
    else:
        values = [str(v) for v in (arguments or [])]
    if name in {"PLACE", "PLACE_SERVING_UTENSIL"} and len(values) >= 2:
        return ("placed", values[0], values[1])
    if name == "POUR" and len(values) >= 2:
        return ("poured", values[0], values[1])
    if name == "STIR" and len(values) >= 2:
        # dispatched as (tool, target); the goal contract names targets only
        return ("stirred", values[1])
    if name == "INSERT" and len(values) >= 2:
        return ("inserted", values[0], values[1])
    if name == "FASTEN" and len(values) >= 3:
        return ("fastened", values[1], values[2])
    return None


def planned(episode: Path, method: str) -> set[tuple] | None:
    """Everything the method's plan would establish, or None if it wrote none.

    None is a distinct outcome from an empty set: "recorded no plan" is not
    "planned nothing useful", and the two must not be averaged together.
    """
    facts: set[tuple] = set()
    found = False

    calls = episode / "model_calls"
    if calls.is_dir():  # VLM-TAMP, and ROBUST-TAMP's raw call log
        for call in sorted(calls.glob("*.json")):
            try:
                payload = json.loads(call.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for subgoal in (payload.get("plan") or {}).get("subgoals") or []:
                found = True
                fact = _norm_subgoal(
                    subgoal.get("predicate", ""), subgoal.get("arguments") or {}
                )
                if fact:
                    facts.add(fact)
            for action in (payload.get("response") or {}).get("actions") or []:
                found = True
                fact = _norm_action(action.get("name", ""), action.get("arguments"))
                if fact:
                    facts.add(fact)

    trace = episode / "model_trace.json"
    if trace.is_file():  # OWL-TAMP writes the sketch it hands to the refiner
        try:
            sketch = json.loads(trace.read_text()).get("sketch") or {}
        except (OSError, json.JSONDecodeError):
            sketch = {}
        for action in sketch.get("actions") or []:
            found = True
            fact = _norm_action(action.get("operator", ""), action.get("arguments"))
            if fact:
                facts.add(fact)

    events = episode / "discovery_replanning_events.jsonl"
    if events.is_file():  # ROBUST-TAMP accepts a full flat plan per replan
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "plan_accepted":
                continue
            for action in event.get("actions") or []:
                found = True
                fact = _norm_action(action.get("name", ""), action.get("arguments"))
                if fact:
                    facts.add(fact)

    plan = episode / "attempts"
    if plan.is_dir():
        # ViLaIn plans in PDDL.  Read the attempt the run selected, translating
        # its perception vocabulary ("silver_kettle_1") through the identity
        # resolution that binds it to planning IDs.  An attempt the run never
        # selected is not a plan it proposed, so only the selected one counts.
        found = True
        facts |= _vilain_plan(episode)

    return facts if found else None


# --------------------------------------------------------------------------
# Scoring a plan against each scene's goal conditions
#
# One function per scene, each mirroring the executed-coverage definition in
# paper_metrics_table.py so that planned and executed share a denominator.
# --------------------------------------------------------------------------

def _vilain_plan(episode: Path) -> set[tuple]:
    """The symbolic plan ViLaIn selected, in planning-ID terms.

    Returns an empty set when no attempt was selected, which is the honest
    reading: ViLaIn's planning stage terminated EXHAUSTED without producing a
    plan, so there is nothing for the executor to run and nothing to score.
    """
    try:
        run = json.loads((episode / "baseline_run_result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    index = run.get("selected_attempt_index")
    if index is None:
        return set()
    attempt = episode / "attempts" / f"{int(index):02d}" / "execution_plan.json"
    if not attempt.is_file():
        return set()
    try:
        payload = json.loads(attempt.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    facts: set[tuple] = set()
    for action in payload.get("normalized_actions") or []:
        fact = _norm_action(
            action.get("name") or action.get("operator", ""),
            action.get("arguments"),
        )
        if fact:
            facts.add(fact)
    return facts


def _vilain_denominator(episode: Path) -> int:
    """ViLaIn's own goal decomposition, used as its coverage denominator.

    ViLaIn writes no `_private_evaluation/goal_contract.json`; the equivalent
    list is `benchmark/terminal_subgoal_evaluation.json`, which is what
    `paper_metrics_table._vilain_coverage` already scores the executed number
    against, so using it here keeps the two columns on one denominator.
    """
    evaluation = episode / "benchmark/terminal_subgoal_evaluation.json"
    if not evaluation.is_file():
        return 0
    try:
        return len(json.loads(evaluation.read_text()).get("results") or [])
    except (OSError, json.JSONDecodeError):
        return 0


def _kitchen_planned(episode: Path, facts: set[tuple]) -> tuple[int, int]:
    contract = episode / "_private_evaluation/goal_contract.json"
    if not contract.is_file():
        # No ViLaIn episode in this grid produced a non-empty symbolic plan --
        # 117 of 120 Kitchen runs ended planning EXHAUSTED and the three that
        # did not selected a plan with zero actions -- so there is nothing to
        # match against its subgoals, only a denominator to report it over.
        return len(facts) and 0, _vilain_denominator(episode)
    try:
        spec = json.loads(contract.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    required = list(spec.get("required_effects") or [])
    stir_targets = list(spec.get("stir_targets") or [])
    literals = {
        f"{fact[0]}({','.join(fact[1:])})" for fact in facts if all(fact[1:])
    }
    hit = sum(1 for effect in required if effect in literals)
    hit += sum(1 for target in stir_targets if ("stirred", target) in facts)
    return hit, len(required) + len(stir_targets)


def _workshop_planned(episode: Path, facts: set[tuple]) -> tuple[int, int]:
    """The same three conditions the Workshop goal asks for.

    Fastener seated in the joint, joint fastened, and the driver left on the
    workbench with an empty gripper.  A FASTEN implies its INSERT, matching the
    executed check, which reads the insertion off the joint the fastening
    produced.  Which driver is not checked, for the reason given in
    `_workshop_coverage`: the physical skill already rejects an incompatible
    one, so any planned fastening names a compatible tool or fails anyway.
    """
    if not (episode / "_private_evaluation").is_dir():
        return len(facts) and 0, _vilain_denominator(episode)
    inserted = {f for f in facts if f[0] == "inserted"}
    fastened = {f for f in facts if f[0] == "fastened"}
    placed = {f for f in facts if f[0] == "placed"}
    bench = _workbench_id(episode)
    hit = 0
    hit += bool(inserted or fastened)
    hit += bool(fastened)
    hit += bool(fastened and bench and any(f[2] == bench for f in placed))
    return hit, 3


def _workbench_id(episode: Path) -> str | None:
    observed = episode / "latest_observation.json"
    if not observed.is_file():
        return None
    try:
        payload = json.loads(observed.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for row in payload.get("known_regions", []):
        if str(row.get("label", "")).strip().lower() == "main workbench":
            return row.get("id")
    return None


def _living_room_planned(episode: Path, facts: set[tuple]) -> tuple[int, int]:
    """Required (region, role) slots the plan would fill.

    Scored by role rather than by object identity, for the reason established
    for the executed number: the goal is symmetric under exchanging the two
    place settings, so demanding ground truth's specific assignment marks
    correct solves wrong.
    """
    gt = episode / "_private_evaluation/expected_gt_actions.json"
    adapter = episode / "_private_evaluation/adapter_resolution.json"
    if not (gt.is_file() and adapter.is_file()):
        return len(facts) and 0, _vilain_denominator(episode)
    try:
        actions = json.loads(gt.read_text()).get("actions", [])
        resolution = json.loads(adapter.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    role = {
        row["generic_object_id"]: row.get("semantic_role")
        for row in resolution.get("objects", [])
        if row.get("generic_object_id")
    }
    required: Counter = Counter()
    for action in actions:
        if str(action.get("operator", "")).upper() == "PLACE":
            args = action.get("arguments") or []
            if len(args) >= 2:
                required[(args[1], role.get(args[0]))] += 1
    if not required:
        return 0, 0
    proposed: Counter = Counter()
    for fact in facts:
        if fact[0] == "placed" and len(fact) >= 3:
            proposed[(fact[2], role.get(fact[1]))] += 1
    return sum(min(n, proposed[k]) for k, n in required.items()), sum(required.values())


# The goal-coverage definition lives in `goal_coverage.py` so the plan-scored
# and execution-scored numbers cannot drift apart, and so the denominator is
# the goal rather than whatever the ground-truth action list happened to omit.
from goal_coverage import SCORERS, TOTALS  # noqa: E402,F401


def collect(roots) -> dict:
    cells: dict = defaultdict(lambda: {
        "n": 0, "no_plan": 0, "plan_hit": 0, "plan_req": 0,
        "exec_hit": 0, "exec_req": 0, "complete": 0,
    })
    for row, directory in _episodes(roots):
        scene, method = row.get("scene"), LABEL.get(row.get("method"))
        if scene not in COVERAGE_SCENES or row.get("expected_outcome") != "FEASIBLE":
            continue
        cell = cells[(scene, method)]
        cell["n"] += 1
        hit, req = _coverage(scene, directory)
        cell["exec_hit"] += hit
        cell["exec_req"] += req
        facts = planned(directory, row.get("method"))
        if facts is None:
            # Emitting no plan at all is a planning failure, not a missing
            # measurement, so it is scored as zero coverage and counted in its
            # own column rather than dropped from the denominator.
            cell["no_plan"] += 1
            facts = set()
        hit, req = SCORERS[scene](directory, facts)
        cell["plan_hit"] += hit
        cell["plan_req"] += req
        cell["complete"] += int(req > 0 and hit == req)
    return cells


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.1f}" if den else "--"


def render(cells) -> str:
    out = [
        f"{'scene':<12}{'method':<13}{'n':>5}{'plan cov':>10}{'exec cov':>10}"
        f"{'gap':>8}{'full plans':>12}{'no plan':>9}",
        "-" * 79,
    ]
    for (scene, method), c in sorted(cells.items()):
        scored = c["n"]
        plan = 100.0 * c["plan_hit"] / c["plan_req"] if c["plan_req"] else None
        execu = 100.0 * c["exec_hit"] / c["exec_req"] if c["exec_req"] else None
        gap = f"{plan - execu:>7.1f}" if plan is not None and execu is not None else "     --"
        out.append(
            f"{scene:<12}{method:<13}{c['n']:>5}"
            f"{_pct(c['plan_hit'], c['plan_req']):>9}%"
            f"{_pct(c['exec_hit'], c['exec_req']):>9}%"
            f"{gap}{_pct(c['complete'], scored):>11}%{c['no_plan']:>9}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*")
    args = parser.parse_args()
    print(render(collect(list(args.roots) or list(DEFAULT_ROOTS))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
