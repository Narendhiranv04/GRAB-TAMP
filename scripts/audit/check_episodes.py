"""Invariant checks over every recorded episode artifact.

Each check is a named predicate over one episode.  A violation is reported
with its path so it can be opened, and the checks are deliberately mechanical:
they assert things the harness itself claims, not things a method ought to do.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_metrics_table import (
    DEFAULT_ROOTS, REPO, LABEL, SCENES, _episodes, _coverage, COVERAGE_SCENES,
)
from mujoco_scenes.benchmark_task_instructions import TASK_INSTRUCTIONS

V = collections.defaultdict(list)          # check name -> [(path, detail)]
def flag(name, path, detail=""):
    V[name].append((str(Path(path).relative_to(REPO)), detail))

REQUIRED = (
    "scene", "method", "variant", "seed", "camera_count", "success",
    "expected_outcome", "predicted_outcome", "outcome_match",
    "executed_actions", "raw_vlm_requests", "replans", "model_calls",
    "elapsed_seconds", "protocol", "schema_version",
)

for row, d in _episodes(DEFAULT_ROOTS):
    path = d / "benchmark_execution_result.json"
    scene, method = row.get("scene"), row.get("method")

    for key in REQUIRED:
        if key not in row:
            flag("missing_required_field", path, key)
        elif row[key] is None:
            flag("null_required_field", path, key)

    # 1. the goal shown to the model must be Table I's, byte for byte
    goal = row.get("goal")
    if goal is not None and goal != TASK_INSTRUCTIONS.get(scene):
        flag("goal_text_differs_from_table_I", path, repr(goal)[:80])

    # 2. outcome_match must agree with the two outcomes it summarizes
    exp, pred, match = row.get("expected_outcome"), row.get("predicted_outcome"), row.get("outcome_match")
    if exp and pred and match is not None and bool(match) != (exp == pred):
        flag("outcome_match_contradicts_outcomes", path, f"{exp}/{pred}/{match}")

    # 3. a success on a feasible task must cover every goal condition
    if row.get("success") and exp == "FEASIBLE" and scene in COVERAGE_SCENES:
        hit, req = _coverage(scene, d)
        if req and hit != req:
            flag("success_without_full_coverage", path, f"{hit}/{req}")

    # 4. full coverage that is not scored a success
    if not row.get("success") and exp == "FEASIBLE" and scene in COVERAGE_SCENES:
        hit, req = _coverage(scene, d)
        if req and hit == req:
            flag("full_coverage_without_success", path, f"{hit}/{req}")

    # 5. claiming success while never having moved
    if row.get("success") and (row.get("executed_actions") or 0) == 0:
        flag("success_with_zero_executed_actions", path)

    # 6. an infeasible trial the method reported complete
    if exp == "INFEASIBLE" and row.get("success"):
        flag("success_on_infeasible_variant", path, str(pred))

    # 7. model calls recorded on disk vs the count the runner reports
    on_disk = len(list((d / "model_calls").glob("*.json"))) if (d / "model_calls").is_dir() else None
    claimed = row.get("model_calls")
    if on_disk is not None and isinstance(claimed, int) and on_disk != claimed:
        flag("model_call_count_mismatch", path, f"disk={on_disk} claimed={claimed}")

    # 8. a method that reports planner traffic but wrote no prompt
    if (row.get("raw_vlm_requests") or 0) == 0 and (row.get("model_calls") or 0) == 0:
        flag("episode_with_no_model_traffic", path)

    # 9. negative or absurd telemetry
    for key in ("executed_actions", "raw_vlm_requests", "replans", "model_calls", "elapsed_seconds"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value < 0:
            flag("negative_telemetry", path, f"{key}={value}")

    # 10. the scene/method/variant on disk must match the recorded ones
    parts = d.parts
    if row.get("method") and row["method"] not in parts and LABEL.get(row["method"]) is None:
        flag("unknown_method_label", path, str(row.get("method")))

print(f"episodes checked: {sum(1 for _ in _episodes(DEFAULT_ROOTS))}\n")
for name in sorted(V, key=lambda k: -len(V[k])):
    rows = V[name]
    print(f"### {name}: {len(rows)}")
    for p, detail in rows[:4]:
        print(f"      {p}  {detail}")
    if len(rows) > 4:
        print(f"      ... {len(rows)-4} more")
if not V:
    print("no violations")
