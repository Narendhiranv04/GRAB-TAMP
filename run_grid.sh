#!/usr/bin/env bash
# Living Room feasible grid, i9 host. Re-runnable: --resume skips completed
# episodes and sets aside any partial one. -u keeps output unbuffered so the
# console log is live-tailable.
cd /home/longhorizon/Documents/LH_Extension/V1 || exit 1
ROOT=runs/living_room/execution/feasible_i9_20260904
env -u PYTHONPATH .venv/bin/python -u -m baseline_common.run_baseline_execution_batch \
  --environment living_room --methods vlm_tamp,owl_tamp,retrieval \
  --variants L1,L2,L3,L4,L5,L6 \
  --camera-counts 3 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-root "$ROOT" \
  --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
  --resume --continue-on-error
