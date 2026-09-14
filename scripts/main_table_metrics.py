"""Per-cell metrics for the main results table, split feasible / infeasible.

The pooled cost columns hid the thing that matters: a method that answers an
infeasible variant by exhausting its replan budget spends several times what it
spends solving a feasible one, and averaging the two makes both unreadable.
Every cost column here is reported separately for the two halves.

`Infeasible plan commitment` is the fraction of the scene's goal relations a
method asserts in the plan it builds for a variant that cannot be satisfied.
It is measured on the plan, not the execution, because the failure being
described is committing to a complete-looking plan for an impossible task --
an executor that then fails is beside the point.  The denominator is the
scene's relation count (Kitchen 12, Living Room 5, Workshop 3); the numerator
is the distinct goal-shaped relations the plan asserts, capped at it.
"""
from __future__ import annotations
import json, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_metrics_table import DEFAULT_ROOTS, LABEL, _episodes, _coverage, COVERAGE_SCENES
from planning_metrics import planned, SCORERS

RELATIONS = {"kitchen": 12, "living_room": 5, "workshop": 3}
GOAL_SHAPED = ("poured", "placed", "stirred", "inserted", "fastened")


def _inspections(directory: Path) -> int:
    from paper_metrics_table import _inspections as f
    return f(directory)


def collect(roots):
    cell = defaultdict(lambda: {
        "fN": 0, "iN": 0, "cov_h": 0, "cov_r": 0, "succ": 0,
        "commit_h": 0, "commit_r": 0, "recognised": 0, "no_answer": 0,
        "f_req": [], "f_rep": [], "f_time": [], "f_insp": [],
        "i_req": [], "i_rep": [], "i_time": [], "i_insp": [],
    })
    for row, d in _episodes(roots):
        scene, method = row.get("scene"), LABEL.get(row.get("method"))
        c = cell[(scene, method)]
        feasible = row.get("expected_outcome") == "FEASIBLE"
        p = "f" if feasible else "i"
        c[f"{p}_req"].append(row.get("raw_vlm_requests") or 0)
        c[f"{p}_rep"].append(row.get("replans") or 0)
        t = row.get("planning_latency_s")
        if t is not None:
            c[f"{p}_time"].append(float(t))
        c[f"{p}_insp"].append(_inspections(d))
        if feasible:
            c["fN"] += 1
            c["succ"] += bool(row.get("success"))
            if scene in COVERAGE_SCENES:
                facts = planned(d, row.get("method"))
                hit, req = SCORERS[scene](d, facts if facts is not None else set())
                c["cov_h"] += hit
                c["cov_r"] += req
        else:
            c["iN"] += 1
            facts = planned(d, row.get("method")) or set()
            asserted = {f for f in facts if f and f[0] in GOAL_SHAPED}
            # Eq. 13 is trial-level: c_i in {0,1}, averaged over N_I.  This
            # counted the *fraction of the scene's goal relations* a plan
            # asserted, which is a different quantity and read very
            # differently: Kitchen VLM-TAMP scored 55.7% that way against
            # 100.0% under the equation, because its plans assert 4 to 12 of
            # the 12 relations and every one of the 60 trials produced a plan.
            c["commit_h"] += bool(asserted)
            c["commit_r"] += 1
            # Kept separately because `Reject = 100 - Commit` otherwise credits
            # a method for failing: 25% of Kitchen ROBUST-TAMP's infeasible
            # trials produce no plan because they hit the token ceiling or the
            # replan budget, not because they recognised the task as
            # impossible.  Only Workshop OWL-TAMP genuinely rejects (80%).
            if not asserted:
                if row.get("predicted_outcome") == "INFEASIBLE":
                    c["recognised"] += 1
                else:
                    c["no_answer"] += 1
    return cell


def pct(n, d):
    return f"{100.0*n/d:.1f}" if d else "--"


def mean(v):
    return f"{statistics.mean(v):.2f}" if v else "--"


def mean0(v):
    """Planning time, or `--` when the runner never recorded it.

    `planning_latency_s` is written as 0.0 by several runners rather than
    omitted: OWL-TAMP and VLM-TAMP Workshop record 0 in all 100 episodes, and
    Kitchen ROBUST-TAMP and ViLaIn-TAMP in about half of theirs.  Averaging
    those zeros would publish a measurement that was never taken, so a cell is
    reported only when at least 90% of its episodes carry a nonzero value.
    """
    if not v:
        return "--"
    nonzero = [x for x in v if x]
    if len(nonzero) < 0.9 * len(v):
        return "--"
    return f"{statistics.mean(nonzero):.0f}"


def main():
    cells = collect(DEFAULT_ROOTS)
    order = ["VLM-TAMP", "OWL-TAMP", "ROBUST-TAMP", "ViLaIn-TAMP"]
    hdr = (f"{'scene/method':<26}{'fN':>4}{'cov':>7}{'succ':>7}"
           f"{'iN':>4}{'commit':>8}"
           f"{'insp_f':>8}{'insp_i':>8}"
           f"{'fm_f':>7}{'fm_i':>7}{'rep_f':>7}{'rep_i':>7}{'t_f':>8}{'t_i':>8}")
    print(hdr); print("-" * len(hdr))
    for scene in ("kitchen", "living_room", "workshop"):
        for m in order:
            c = cells.get((scene, m))
            if not c or not (c["fN"] or c["iN"]):
                continue
            print(f"{scene[:11]+'/'+m:<26}{c['fN']:>4}{pct(c['cov_h'],c['cov_r']):>7}"
                  f"{pct(c['succ'],c['fN']):>7}{c['iN']:>4}{pct(c['commit_h'],c['commit_r']):>8}"
                  f"{mean(c['f_insp']):>8}{mean(c['i_insp']):>8}"
                  f"{mean(c['f_req']):>7}{mean(c['i_req']):>7}"
                  f"{mean(c['f_rep']):>7}{mean(c['i_rep']):>7}"
                  f"{mean0(c['f_time']):>8}{mean0(c['i_time']):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
