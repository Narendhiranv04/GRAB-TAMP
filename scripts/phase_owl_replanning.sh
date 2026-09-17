#!/bin/bash
# Workshop OWL-TAMP under the replanning protocol.
#
# A separately named condition, written to its own root so it can never be
# confused with -- or resumed into -- the reported single-shot grid under
# final_20260913.  See BASELINE_FIDELITY.md: this is an extension beyond the
# paper, not a reproduction of it.
#
# 10 variants x 5 seeds = 50 episodes, one whole-episode budget of 15 model
# calls each.
#
# Measured, not guessed: a 10-episode leg at 10 workers took 57.7 min wall
# clock -- 150 model calls, so 2.6 calls/min aggregate.  The endpoint saturates
# rather than running out of batch slots, so per-call latency grows with
# concurrency (44 s at 1 worker, 108 s at 3, 146 s at 10) and throughput rises
# only sub-linearly.  50 episodes is therefore ~3.5 h at 20 workers, not the
# ~2 h a single-episode timing would suggest.  MuJoCo is local and the model
# server is remote, so workers are bounded by local RAM, not by the GPU.
set -u
ROOT=/home/longhorizon/Documents/LH_Extension/V1
OUT=runs/workshop/execution/owl_replanning_20260914
SEEDS=${SEEDS:-0,1,2,3,4}
WORKERS=${WORKERS:-20}

systemctl --user reset-failed owl-replanning.service 2>/dev/null
systemd-run --user --slice=lh.slice --collect --unit=owl-replanning \
  --working-directory="$ROOT" --setenv=MUJOCO_GL=egl --setenv=PYTHONPATH="$ROOT" \
  "$ROOT/.venv/bin/python" -m baseline_common.run_baseline_execution_batch \
    --environment workshop --methods owl_tamp \
    --variants W1,W2,W3,W4,W5,W6,W7,W8,W9,W10 --seeds "$SEEDS" \
    --camera-counts 3 --protocol replanning --decoding model-native \
    --output-root "$OUT" \
    --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
    --max-tokens 24576 --max-model-calls 15 \
    --max-actions 80 --max-sketch-actions 24 \
    --episode-timeout 7200 --workers "$WORKERS" --resume --continue-on-error

echo "launched: $OUT seeds=$SEEDS workers=$WORKERS $(date -Is)"
