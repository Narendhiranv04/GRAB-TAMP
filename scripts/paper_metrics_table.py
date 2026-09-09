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
                        Workshop -- withheld.  A GT-derived reading was built
                          against the fastening facts the privileged
                          observation records, and it fails the validation
                          above on all 8 known successes, so it measures
                          something other than the goal.
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
recent episode winning.  Kitchen needs this: `grid_20260907b` holds K1-K12 from
before the POUR/STIR admissibility fix and `pour_rerun_20260908` re-ran K1-K6
after it, so pooling the roots naively counts the feasible half twice under two
different harnesses and reports n=180 where the grid is 120.  Newest-wins keeps
the re-run for K1-K6 and the original for the infeasible K7-K12, which never
needed re-running because rejecting an infeasible variant requires no pour.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDED = (".interrupted", ".timeout_", ".transport_", ".provenance_")
DEFAULT_ROOTS = (
    "runs/kitchen/execution/grid_20260907b",
    "runs/kitchen/execution/pour_rerun_20260908",
    "runs/kitchen/execution/robust_kitchen_20260909",
    "runs/kitchen/execution/vilain_kitchen_20260909",
    "runs/workshop/execution/grid_20260907b",
    "runs/workshop/execution/robust_tamp_20260908",
    "runs/workshop/execution/vilain_20260908d",
    "runs/living_room/execution/grid_20260907b",
    "runs/living_room/execution/robust_tamp_20260908",
    "runs/living_room/execution/vilain_20260908d",
    "runs/retrieval_grid_20260908_headless",
)
LABEL = {
    "vlm_tamp": "VLM-TAMP", "owl_tamp": "OWL-TAMP", "retrieval": "Retrieval",
    "vilain_tamp": "ViLaIn-TAMP", "discovery_replanning": "ROBUST-TAMP",
    "robust_tamp": "ROBUST-TAMP", "functional_tamp": "Ours",
}
SCENES = ("kitchen", "living_room", "workshop")
SCENE_LABEL = {"kitchen": "Kitchen", "living_room": "Living room", "workshop": "Workshop"}
ORDER = ("VLM-TAMP", "OWL-TAMP", "Retrieval", "ROBUST-TAMP", "ViLaIn-TAMP", "Ours")
# Goal coverage is only meaningful where the ground-truth goal is a set of
# placements in the same ID namespace the terminal observation uses.
COVERAGE_SCENES = {"living_room", "kitchen"}   # workshop fails validation; see the docstring


def _cell() -> dict:
    return dict(n=0, match=0, fs=0, ft=0, fc=0, it=0, ppf=0,
                cov_hit=0, cov_req=0, req=[], rep=[])


def _episodes(roots):
    """Yield one (row, directory) per trial, newest episode winning."""
    best: dict[tuple, tuple[float, dict, Path]] = {}
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
                if scene in COVERAGE_SCENES:
                    hit, req = _coverage(scene, path.parent)
                    c["cov_hit"] += hit
                    c["cov_req"] += req
            elif row.get("expected_outcome") == "INFEASIBLE":
                c["it"] += 1
                if row.get("predicted_outcome") == "FEASIBLE" or row.get("success"):
                    c["fc"] += 1
    return cells


def _coverage(scene: str, episode: Path) -> tuple[int, int]:
    return _kitchen_coverage(episode) if scene == "kitchen" else _living_room_coverage(episode)


def _kitchen_coverage(episode: Path) -> tuple[int, int]:
    """Goal-contract effects the episode actually produced."""
    contract = episode / "_private_evaluation/goal_contract.json"
    history = episode / "episode_result.json"
    if not (contract.is_file() and history.is_file()):
        return 0, 0
    try:
        spec = json.loads(contract.read_text())
        result = json.loads(history.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0
    required = list(spec.get("required_effects") or [])
    stir_targets = list(spec.get("stir_targets") or [])
    if not required and not stir_targets:
        return 0, 0
    observed: set[str] = set()
    for action in (result.get("result") or {}).get("action_history") or []:
        observed.update(action.get("effects") or [])
    hit = sum(1 for effect in required if effect in observed)
    # any tool satisfies a stir target; the contract names the target only
    hit += sum(
        1 for target in stir_targets
        if any(e.startswith("stirred(") and e.rstrip(")").split(",")[-1] == target
               for e in observed)
    )
    return hit, len(required) + len(stir_targets)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true", help="emit LaTeX rows")
    parser.add_argument("--roots", nargs="*", default=None, help="override the run roots")
    parser.add_argument("--validate", action="store_true",
                        help="assert every successful trial scores 100% coverage")
    args = parser.parse_args()
    roots = args.roots or DEFAULT_ROOTS
    if args.validate:
        _validate(roots)
    print(render(collect(roots), args.latex))


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
