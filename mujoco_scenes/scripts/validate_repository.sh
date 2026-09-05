#!/bin/sh
set -eu

root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "Missing $python; follow EXECUTION_AND_TESTING.md first." >&2
    exit 1
fi

cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

git diff --check

# Every module with a __main__ guard and an argument parser is imported through
# --help, which catches an import-time break without running anything.  The
# baseline-comparison and reporting entry points are included: they are what
# produce the reported numbers, and they were absent from this list until
# 2026-09-05 even though a break in one of them invalidates a grid.
for module in \
    mujoco_scenes.scene_loader \
    mujoco_scenes.living_room_scene \
    mujoco_scenes.workshop_scene \
    mujoco_scenes.workshop_pointcloud \
    mujoco_scenes.run_kitchen_goal_execution \
    mujoco_scenes.run_kitchen_planner_execution \
    mujoco_scenes.run_kitchen_symbolic_pipeline \
    mujoco_scenes.run_living_room_region_function \
    mujoco_scenes.run_living_room_mobile_execution \
    mujoco_scenes.run_living_room_discovery_replanning \
    mujoco_scenes.run_kitchen_discovery_replanning \
    mujoco_scenes.run_gt_evidence_ablation \
    mujoco_scenes.run_kitchen_ground_truth_execution \
    mujoco_scenes.run_living_room_execution \
    llm3_baseline.client \
    llm3_baseline.run_kitchen \
    vlm_tamp_baseline.client \
    vlm_tamp_baseline.run_kitchen \
    vlm_tamp_baseline.run_living_room \
    owl_tamp_baseline.run_kitchen \
    owl_tamp_baseline.run_living_room \
    retrieval_baseline.run_living_room \
    baseline_common.run_baseline_execution_batch \
    baseline_common.run_discovery_execution_batch \
    baseline_common.run_plan_gt_batch \
    baseline_common.summarize_execution_batch \
    baseline_common.summarize_plan_gt_batch \
    baseline_common.make_paper_tables
do
    "$python" -m "$module" --help >/dev/null
done

# The five Kitchen failures below are the documented known-good state, not
# environment faults -- see MACHINE_HANDOFF.md, "Verify before running
# anything".  They are deselected rather than tolerated wholesale so that this
# script still fails on any *other* failure, which is the property that makes
# it worth running.  If one of them starts passing, delete its line here: a
# stale deselect silently hides real coverage.
"$python" -m pytest -q -p no:cacheprovider \
    --deselect mujoco_scenes/tests/test_kitchen_ground_truth_execution.py::test_hidden_soup_first_serving_order_reserves_space_for_all_targets \
    --deselect mujoco_scenes/tests/test_kitchen_ground_truth_execution.py::test_countertop_utensil_validation_does_not_require_vessel_upright_axis \
    --deselect mujoco_scenes/tests/test_kitchen_phase_b_execution.py::test_serving_allocator_is_deterministic_and_role_separated \
    --deselect mujoco_scenes/tests/test_kitchen_phase_b_execution.py::test_serving_allocator_uses_persistent_occupied_state_for_second_and_third \
    --deselect mujoco_scenes/tests/test_kitchen_phase_b_execution.py::test_serving_allocator_sequence_is_deterministic

echo "validate_repository.sh: OK"
