"""Compute the paper's end-to-end metrics table from execution artifacts.

One place that knows how each reported column is defined, so a number in the
paper can be traced to the artifacts it came from.

    .venv/bin/python -m scripts.paper_metrics_table            # readable
    .venv/bin/python -m scripts.paper_metrics_table --latex    # table body
    .venv/bin/python -m scripts.paper_metrics_table --roots R  # extra roots

Columns, and what each one actually reads:

  Outcome correct     `outcome_match`, over all trials.
  Feasible success    `success`, over trials whose `expected_outcome` is
                      FEASIBLE.
  Goal coverage       Satisfied goal conditions over required goal conditions.
                      Derived per scene, and every derivation is validated by
                      `--validate`: a trial the harness scored a success must
                      score 100% coverage, or the metric is wrong and is not
                      reported.
                        Living Room -- role slots, not object identity.  The
                          goal is symmetric ("a cup and a saucer on *each*
                          table"), so requiring GT's specific object-to-region
                          assignment marks a correct solve wrong; 30 of 95
                          successes failed that way before this was fixed.
                          Required slots come from GT PLACE targets paired with
                          `semantic_role` from `adapter_resolution.json`.
                        Kitchen -- `_private_evaluation/goal_contract.json`
                          gives `required_effects` and `stir_targets` outright;
                          observed effects come from the episode action history.
                        Workshop -- the three terminal conditions in the goal
                          text: fastener seated, joint fastened, driver back on
                          the workbench with an empty gripper.  An earlier
                          reading was withheld for failing the validation above
                          on all 8 known successes.  That validation was right
                          and the successes were wrong: they were scored on the
                          physical joint alone, which is one clause of a
                          three-clause goal, and all 8 ended holding the driver.
                          See `workshop_goal_reached`.
  False completion    Infeasible trials reported complete: `expected_outcome`
                      INFEASIBLE with `predicted_outcome` FEASIBLE, or `success`.
  Physical plan found Feasible trials with `executed_actions > 0`.  If an action
                      ran, an executable motion plan existed; if none ran, none
                      was obtained.
  Raw VLM requests    mean `raw_vlm_requests`.
  High-level replans  mean `replans`.

Relocated episodes (`.interrupted`, `.timeout_*`, `.transport_*`,
`.provenance_*`) are harness deaths, not results, and are excluded everywhere.

A cell is de-duplicated on (scene, method, variant, seed, camera_count), most
recent episode winning.  This mattered while Kitchen spanned several roots: one
held K1-K12 from before the POUR/STIR admissibility fix and another re-ran only
K1-K6 after it, so pooling them naively counted the feasible half twice under
two different harnesses and reported n=180 where the grid is 120.  Those
superseded roots were removed on 2026-09-12 and every canonical cell now lives
in exactly one root, but newest-wins is kept because a partial re-run produces
the same overlap again.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDED = (".interrupted", ".timeout_", ".transport_", ".provenance_")
#: A trial directory is exactly `seed_<digits>`.  Anything else is a relocated
#: episode -- interrupted, timed out, superseded, left over -- and is not a
#: result.  This is an allowlist on purpose: the blocklist above missed
#: `.mixed_version_superseded` and `.leftover_`, and seven such directories sat
#: inside the canonical roots.  Five carried a result file and were kept out of
#: the table only by losing the mtime comparison below, which is not a property
#: anyone should be relying on -- one `touch`, restore or rsync and a
#: superseded episode replaces a real one silently.
TRIAL_DIR = re.compile(r"^seed_\d+$")
# Superseded roots live in `runs/_superseded/` with the reason each was
# retired; see its README.  They are deliberately absent here rather than
# filtered, because a re-run that fails to produce an artifact would otherwise
# let the retired episode win the de-duplication below on mtime.
DEFAULT_ROOTS = (
    # The 2026-09-10/11 re-run, produced after the nine harness faults recorded
    # in BASELINE_FIDELITY.md were fixed.  Every cell here is 120/120 or
    # 100/100 with no gaps, verified by enumerating variant x seed rather than
    # by counting files.
    "runs/kitchen/execution/fixed_20260910",
    "runs/living_room/execution/newgoal_20260911",
    "runs/workshop/execution/fixed_20260910",
    # VLM-TAMP Workshop was deliberately NOT re-run for the harness fixes: an
    # import-graph check confirms it reaches none of the changed files, so its
    # existing root remains canonical.  Kitchen VLM-TAMP was re-run and lives
    # in fixed_20260910 above.  VLM-TAMP Living Room is NOT here because the
    # instruction change forces it back into newgoal_20260911 with the rest of
    # that scene.
    "runs/workshop/execution/vlm_rerun_20260909",
)
# The Living Room instruction was rewritten on 2026-09-11 -- the pipeline had
# been tested against a different one -- so every Living Room cell including
# VLM-TAMP is re-run under `newgoal_20260911`.  The old roots are retired in
# place as `*.old_goal_superseded_20260911` and must never be listed here: they
# answer a different question and would silently mix two instructions into one
# column.
LABEL = {
    "vlm_tamp": "VLM-TAMP", "owl_tamp": "OWL-TAMP", "retrieval": "Retrieval",
    "vilain_tamp": "ViLaIn-TAMP", "discovery_replanning": "ROBUST-TAMP",
    # `discovery_replanning` is the pre-2026-09-12 spelling; both map here.
    "robust_tamp": "ROBUST-TAMP",
    "robust_tamp": "ROBUST-TAMP", "functional_tamp": "Ours",
}
SCENES = ("kitchen", "living_room", "workshop")
# Table I variant counts x 10 seeds.  A cell holding fewer trials than this is
# a leg still in flight or one that died partway, and it is rendered as
# incomplete rather than as a result: a ROBUST-TAMP Kitchen cell holding 1 of
# 120 trials otherwise prints "0.0", which reads exactly like a finding.
GRID_TRIALS = {"kitchen": 120, "living_room": 100, "workshop": 100}
SCENE_LABEL = {"kitchen": "Kitchen", "living_room": "Living room", "workshop": "Workshop"}
ORDER = ("VLM-TAMP", "OWL-TAMP", "Retrieval", "ROBUST-TAMP", "ViLaIn-TAMP", "Ours")
# Goal coverage is only meaningful where the ground-truth goal is a set of
# placements in the same ID namespace the terminal observation uses.
COVERAGE_SCENES = {"living_room", "kitchen", "workshop"}


def _cell() -> dict:
    return dict(n=0, match=0, fs=0, ft=0, fc=0, it=0, ppf=0, rej=0, insp=0,
                cov_hit=0, cov_req=0, req=[], rep=[])


def _episodes(roots):
    """Yield one (row, directory) per trial, newest episode winning."""
    best: dict[tuple, tuple[float, dict, Path]] = {}
    for rel in roots:
        base = REPO / rel
        if not base.is_dir():
            continue
        for path in base.rglob("benchmark_execution_result.json"):
            if not TRIAL_DIR.match(path.parent.name):
                continue
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            scene, method = row.get("scene"), LABEL.get(row.get("method"))
            if scene not in SCENES or method is None:
                continue
            key = (scene, method, row.get("variant"), row.get("seed"), row.get("camera_count"))
            stamp = path.stat().st_mtime
            if key not in best or stamp > best[key][0]:
                best[key] = (stamp, row, path.parent)
    for _, row, directory in best.values():
        yield row, directory


def collect(roots) -> dict:
    cells = defaultdict(_cell)
    for row, directory in _episodes(roots):
        if True:
            scene, method = row.get("scene"), LABEL.get(row.get("method"))
            path = directory / "benchmark_execution_result.json"
            c = cells[(scene, method)]
            c["n"] += 1
            c["match"] += bool(row.get("outcome_match"))
            c["req"].append(row.get("raw_vlm_requests") or 0)
            c["rep"].append(row.get("replans") or 0)
            if row.get("expected_outcome") == "FEASIBLE":
                c["ft"] += 1
                c["fs"] += bool(row.get("success"))
                c["ppf"] += (row.get("executed_actions") or 0) > 0
                c["insp"] += _inspections(path.parent)
                if scene in COVERAGE_SCENES:
                    hit, req = _coverage(scene, path.parent)
                    c["cov_hit"] += hit
                    c["cov_req"] += req
            elif row.get("expected_outcome") == "INFEASIBLE":
                c["it"] += 1
                c["rej"] += row.get("predicted_outcome") == "INFEASIBLE"
                if row.get("predicted_outcome") == "FEASIBLE" or row.get("success"):
                    c["fc"] += 1
    return cells


def _coverage(scene: str, episode: Path) -> tuple[int, int]:
    if (episode / "benchmark/terminal_subgoal_evaluation.json").is_file():
        return _vilain_coverage(scene, episode)
    return {
        "kitchen": _kitchen_coverage,
        "living_room": _living_room_coverage,
        "workshop": _workshop_coverage,
    }[scene](episode)


def _vilain_coverage(scene: str, episode: Path) -> tuple[int, int]:
    """Goal coverage for ViLaIn, which writes a different artifact layout.

    This column was blank for ViLaIn, and the reason was a wrong assumption of
    mine rather than missing data: ViLaIn records both a terminal state
    (`benchmark/terminal_state_snapshot.json`) and a goal decomposition
    (`benchmark/terminal_subgoal_evaluation.json`), just not under the
    `latest_observation.json` / `_private_evaluation/` names the other
    baselines use.

    Its decomposition carries exactly the denominator the other methods are
    scored against -- 12 conditions in Kitchen against the goal contract's 10
    effects plus 2 stir targets, 3 in Workshop whose `INSERTED_IN`, `FASTENED`
    and `ON` map one-to-one onto the fastener/joint/driver conditions, and 5 in
    the Living Room -- so its own pass count is used directly, and for Workshop
    it is better than reading the snapshot, which records no fastening relation
    at all.

    The Living Room is the exception, for the reason established there: its
    goal is symmetric under exchanging the two place settings, and ViLaIn's
    subgoals name ground truth's specific assignment ("a2_drink_left on
    a2_personal_left"). Scoring those literally is the reading that marked 30
    of 95 correct solves wrong, so the slots are relaxed to (region, role).

    ViLaIn executes no action in any episode of any scene, so whatever this
    returns is the coverage of the *initial* scene, which it never changed.

    An earlier version of this docstring claimed that was fine because "every
    other method's figure likewise includes anything pre-satisfied".  That is
    wrong, and it inflated ViLaIn.  The other methods' Living Room denominator
    is built from ground truth's PLACE actions, and ground truth does not place
    an object that already sits on its target -- so L2 and L5, which start with
    one saucer already on the left personal table, give the other methods a
    denominator of 4 with the pre-placed slot excluded.  ViLaIn's own subgoal
    list keeps all 5 and marks the pre-placed one passed, which is where its
    entire Living Room coverage of 6.7% came from: 20 episodes scoring 1/5 for
    a scene they never touched.

    Subgoals already satisfied in the initial state are therefore dropped from
    both the numerator and the denominator, which puts ViLaIn on the same
    denominator as everyone else.  In Kitchen and Workshop the initial state
    satisfies nothing, so this changes neither.
    """
    evaluation = episode / "benchmark/terminal_subgoal_evaluation.json"
    try:
        results = json.loads(evaluation.read_text()).get("results") or []
    except (OSError, json.JSONDecodeError):
        return 0, 0
    presatisfied = _vilain_presatisfied(episode)
    results = [
        row for row in results
        if _subgoal_key(row.get("subgoal") or {}) not in presatisfied
    ]
    if not results:
        return 0, 0
    if scene != "living_room":
        return sum(1 for row in results if row.get("passed")), len(results)

    snapshot = episode / "benchmark/terminal_state_snapshot.json"
    try:
        state = json.loads(snapshot.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0

    def role(name: str) -> str | None:
        lowered = name.lower()
        return next((key for key in ("drink", "snack", "remote") if key in lowered), None)

    required: Counter = Counter()
    for row in results:
        subgoal = row.get("subgoal") or {}
        if str(subgoal.get("predicate", "")).upper() != "ON":
            continue
        slot = role(str(subgoal.get("subject") or ""))
        target = subgoal.get("target")
        if slot and target:
            required[(target, slot)] += 1
    if not required:
        return 0, 0
    final: Counter = Counter()
    for name, item in (state.get("objects") or {}).items():
        slot = role(name)
        support = item.get("support")
        if slot and support and not item.get("held"):
            final[(support, slot)] += 1
    return sum(min(n, final[k]) for k, n in required.items()), sum(required.values())


def _subgoal_key(subgoal: dict) -> tuple:
    return (
        str(subgoal.get("predicate") or ""),
        str(subgoal.get("subject") or ""),
        str(subgoal.get("target") or ""),
    )


def _vilain_presatisfied(episode: Path) -> set[tuple]:
    """Subgoals already true before the robot moved.

    ViLaIn records the initial evaluation alongside the terminal one, so the
    scene's starting credit can be identified exactly rather than estimated.
    """
    initial = episode / "benchmark/initial_subgoal_evaluation.json"
    try:
        rows = json.loads(initial.read_text()).get("results") or []
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        _subgoal_key(row.get("subgoal") or {}) for row in rows if row.get("passed")
    }


def _workshop_coverage(episode: Path) -> tuple[int, int]:
    """The three terminal conditions the Workshop goal actually asks for.

    The goal is "identify the compatible components required to complete the
    fastening at the marked workbench location, complete the fastening, and
    leave any reusable equipment used for the task safely on the workbench".
    Inspecting storage is the means, not a goal condition, so what is scored
    is the state the goal describes: the fastener seated in the joint, the
    joint fastened, and the driver that did it back on the workbench with an
    empty gripper.

    Which driver is deliberately not checked against ground truth.  As in the
    Living Room, demanding GT's specific choice marks a correct solve wrong
    when more than one tool is compatible; the physical FASTEN skill already
    rejects an incompatible driver, so a recorded fastening is a compatible
    one.
    """
    private = episode / "_private_evaluation/latest_observation.json"
    if not private.is_file():
        return 0, 0
    try:
        observed = json.loads(private.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    entities = {
        row["id"]: (row.get("facts") or {})
        for row in observed.get("visible_entities", [])
        if row.get("id")
    }
    # The workbench is named, not assumed to be a fixed region index.
    bench = next(
        (row["id"] for row in observed.get("known_regions", [])
         if str(row.get("label", "")).strip().lower() == "main workbench"),
        None,
    )
    joint = next(
        (facts for facts in entities.values()
         if facts.get("inserted_fasteners") or facts.get("fastened_with")),
        {},
    )
    driver = joint.get("fastened_with")
    holding = (observed.get("robot") or {}).get("holding")
    hit = 0
    hit += bool(joint.get("inserted_fasteners"))
    hit += bool(driver)
    hit += bool(
        driver
        and holding is None
        and bench is not None
        and entities.get(driver, {}).get("region_id") == bench
    )
    return hit, 3


def _kitchen_coverage(episode: Path) -> tuple[int, int]:
    """Goal-contract effects the episode actually produced."""
    contract = episode / "_private_evaluation/goal_contract.json"
    if not contract.is_file():
        return 0, 0
    try:
        spec = json.loads(contract.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    required = list(spec.get("required_effects") or [])
    stir_targets = list(spec.get("stir_targets") or [])
    if not required and not stir_targets:
        return 0, 0
    observed = _kitchen_observed_effects(episode)
    if observed is None:
        return 0, 0
    hit = sum(1 for effect in required if effect in observed)
    # any tool satisfies a stir target; the contract names the target only
    hit += sum(
        1 for target in stir_targets
        if any(e.startswith("stirred(") and e.rstrip(")").split(",")[-1] == target
               for e in observed)
    )
    return hit, len(required) + len(stir_targets)


# The Living Room hides nothing -- Table I lists its objects and regions as
# initially visible -- so an inspection count there measures nothing.
DISCOVERY_SCENES = {"kitchen", "workshop"}


def _inspections(episode: Path) -> int:
    """Successful INSPECT actions, however the episode's runner records them.

    This is the column that explains the Kitchen result.  Kitchen distributes
    required cups, bowls and utensils across five closed storage regions, and
    the baselines collectively open them almost never; without this the reader
    sees 0.0% feasible success with no way to tell that the methods never
    looked.
    """
    history = episode / "episode_result.json"
    if history.is_file():
        try:
            result = json.loads(history.read_text())
        except (OSError, json.JSONDecodeError):
            return 0
        return sum(
            1
            for action in (result.get("result") or {}).get("action_history") or []
            for effect in (action.get("effects") or [])
            if str(effect).startswith("inspected(")
        )
    # OWL-TAMP's Kitchen domain names the search operator OPEN, not INSPECT;
    # every other runner and every other scene uses INSPECT.  Counting only
    # the latter reported OWL as never having looked in a drawer when its
    # sketches contain 58 OPEN actions, so both spellings count.
    events = episode / "discovery_replanning_events.jsonl"
    if not events.is_file():
        return 0
    total = 0
    try:
        lines = events.read_text().splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("event") == "discovery_skill_finished"
            and record.get("success")
            and any(name in str(record.get("action")) for name in ("INSPECT", "OPEN"))
        ):
            total += 1
    return total


def _kitchen_observed_effects(episode: Path) -> set[str] | None:
    """Effects the episode produced, from whichever artifact its runner writes.

    VLM-TAMP and OWL-TAMP record an `episode_result.json` action history;
    ROBUST-TAMP records `discovery_replanning_events.jsonl` instead.  Reading
    only the former left Kitchen ROBUST-TAMP as the last unreported coverage
    cell, which read as a withheld metric rather than as the missing adapter it
    was.  Returns None when neither artifact is present, so a genuinely
    unreadable episode is still excluded rather than scored zero.
    """
    history = episode / "episode_result.json"
    if history.is_file():
        try:
            result = json.loads(history.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        observed: set[str] = set()
        for action in (result.get("result") or {}).get("action_history") or []:
            observed.update(action.get("effects") or [])
        return observed

    events = episode / "discovery_replanning_events.jsonl"
    if not events.is_file():
        return None
    observed = set()
    try:
        lines = events.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only committed effects count; a failed skill reports none anyway.
        if record.get("event") == "discovery_skill_finished" and record.get("success"):
            observed.update(record.get("effects") or [])
    return observed


def _living_room_coverage(episode: Path) -> tuple[int, int]:
    """Required role slots per region that are filled in the terminal state."""
    gt = episode / "_private_evaluation/expected_gt_actions.json"
    obs = episode / "latest_observation.json"
    adapter = episode / "_private_evaluation/adapter_resolution.json"
    if not (gt.is_file() and obs.is_file() and adapter.is_file()):
        return 0, 0
    try:
        actions = json.loads(gt.read_text()).get("actions", [])
        observed = json.loads(obs.read_text())
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
    final: Counter = Counter()
    for item in observed.get("visible_objects", []):
        region = (item.get("facts") or {}).get("region_id")
        if region:
            final[(region, role.get(item["id"]))] += 1
    return sum(min(n, final[k]) for k, n in required.items()), sum(required.values())


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.1f}" if den else "--"


def _mean(values) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "--"


def render(cells, latex: bool) -> str:
    out: list[str] = []
    if not latex:
        head = (f"{'scene':12s}{'method':13s}{'n':>5s}{'outcome':>9s}{'feas':>8s}"
                f"{'coverage':>10s}{'falseC':>8s}{'planFnd':>9s}{'reqs':>7s}{'replans':>9s}")
        out.append(head)
        out.append("-" * len(head))
    for scene in SCENES:
        for method in ORDER:
            c = cells.get((scene, method))
            if not c:
                continue
            expected = GRID_TRIALS[scene]
            if c["n"] < expected:
                note = f"incomplete: {c['n']}/{expected} trials"
                if latex:
                    out.append(
                        f"{method} & ${c['n']}/{expected}$ & "
                        + " & ".join([r"\multicolumn{1}{c}{--}"] * 7)
                        + r" \\"
                    )
                else:
                    out.append(f"{SCENE_LABEL[scene]:12s}{method:13s}"
                               f"{c['n']:5d}   {note:>60s}")
                continue
            vals = (
                _pct(c["match"], c["n"]), _pct(c["fs"], c["ft"]),
                _pct(c["cov_hit"], c["cov_req"]), _pct(c["fc"], c["it"]),
                _pct(c["ppf"], c["ft"]), _mean(c["req"]), _mean(c["rep"]),
            )
            if latex:
                out.append(f"{method} & ${c['n']}$ & " + " & ".join(f"${v}$" for v in vals) + r" \\")
            else:
                out.append(f"{SCENE_LABEL[scene]:12s}{method:13s}{c['n']:5d}"
                           + "".join(f"{v:>9s}" if i == 0 else
                                     f"{v:>8s}" if i in (1, 3) else
                                     f"{v:>10s}" if i == 2 else
                                     f"{v:>9s}" if i == 4 else
                                     f"{v:>7s}" if i == 5 else f"{v:>9s}"
                                     for i, v in enumerate(vals)))
    return "\n".join(out)


def render_split(cells, latex: bool) -> str:
    """The feasible and infeasible halves reported separately.

    `outcome_match` pooled both over a split that is 50/50 in Kitchen, 60/40 in
    the Living Room and 80/20 in Workshop, so it measured the split as much as
    the method: a constant "always feasible" predictor scores 80% in Workshop
    against the best method's 16%, and in Kitchen every method loses to either
    constant.  Reporting the halves separately removes that, and no column here
    can be won by guessing -- a method that rejects everything scores 100% on
    `rej` and is contradicted by 0.0 beside it in `succ`.
    """
    rows: list[str] = []
    if not latex:
        rows.append(f"  {'':30s} {'---- FEASIBLE ----':^27s}  {'-- INFEASIBLE --':^15s}  {'cost':^12s}")
        rows.append(f"  {'scene / method':30s} {'n':>3s} {'succ':>6s} {'cov':>6s} {'plan':>6s} {'insp':>5s}  "
                    f"{'n':>3s} {'rej':>6s} {'fcompl':>6s}  {'reqs':>6s} {'repl':>5s}")
        rows.append("  " + "-" * 96)
    for scene in SCENES:
        for method in ORDER:
            c = cells.get((scene, method))
            if not c:
                continue
            if c["n"] < GRID_TRIALS[scene]:
                rows.append(f"  {SCENE_LABEL[scene] + ' / ' + method:30s} incomplete: {c['n']}/{GRID_TRIALS[scene]}")
                continue
            insp = f"{c['insp'] / c['ft']:.2f}" if (c["ft"] and scene in DISCOVERY_SCENES) else "--"
            values = (
                _pct(c["fs"], c["ft"]), _pct(c["cov_hit"], c["cov_req"]),
                _pct(c["ppf"], c["ft"]), insp,
                _pct(c["rej"], c["it"]), _pct(c["fc"], c["it"]),
                _mean(c["req"]), _mean(c["rep"]),
            )
            if latex:
                rows.append(
                    f"{method} & ${c['ft']}$ & " + " & ".join(f"${v}$" for v in values[:4])
                    + f" & ${c['it']}$ & " + " & ".join(f"${v}$" for v in values[4:]) + r" \\"
                )
            else:
                rows.append(
                    f"  {SCENE_LABEL[scene] + ' / ' + method:30s} {c['ft']:3d} "
                    + " ".join(f"{v:>6s}" for v in values[:3]) + f" {insp:>5s}  "
                    + f"{c['it']:3d} " + " ".join(f"{v:>6s}" for v in values[4:6])
                    + f"  {values[6]:>6s} {values[7]:>5s}"
                )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true", help="emit LaTeX rows")
    parser.add_argument("--split", action="store_true",
                        help="report the feasible and infeasible halves separately")
    parser.add_argument("--roots", nargs="*", default=None, help="override the run roots")
    parser.add_argument("--validate", action="store_true",
                        help="assert every successful trial scores 100% coverage")
    args = parser.parse_args()
    roots = args.roots or DEFAULT_ROOTS
    if args.validate:
        _validate(roots)
    cells = collect(roots)
    print(render_split(cells, args.latex) if args.split else render(cells, args.latex))


def _validate(roots) -> None:
    """A trial the harness scored a success must cover every goal condition."""
    for scene in sorted(COVERAGE_SCENES):
        good = bad = 0
        for rel in roots:
            base = REPO / rel
            if not base.is_dir():
                continue
            for path in base.rglob("benchmark_execution_result.json"):
                if any(k in str(path) for k in EXCLUDED):
                    continue
                try:
                    row = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if row.get("scene") != scene or not row.get("success"):
                    continue
                hit, req = _coverage(scene, path.parent)
                if not req:
                    continue
                good, bad = (good + 1, bad) if hit == req else (good, bad + 1)
        verdict = "OK" if bad == 0 else "FAILS -- do not report this scene"
        print(f"[validate] {scene}: {good} successes at 100%, {bad} below -- {verdict}")


if __name__ == "__main__":
    main()
