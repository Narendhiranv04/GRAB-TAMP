#!/bin/bash
# Donate finished legs' workers to the leg still running.
#
# The scene-sequence orchestrator allocates workers once, from estimates made
# before the run.  ROBUST-TAMP's Kitchen episodes turned out to take 54.7 min
# against the 25 assumed, so at 3 workers it needed 23 h while OWL-TAMP and
# ViLaIn-TAMP finished early and their 6 workers sat idle until the scene ended.
# Re-running the 77 remaining ROBUST episodes at 9 workers takes about 8 h.
#
# Waits for the early legs to exit so the worker total never rises above 17,
# which is the memory ceiling.  --resume means completed episodes are skipped,
# so only the episodes in flight at the moment of restart are repeated.
set -u
ROOT=/home/longhorizon/Documents/LH_Extension/V1
K=K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12
while systemctl --user is-active --quiet rr5-kitchen-owl_tamp.service \
   || systemctl --user is-active --quiet rr5-kitchen-vilain_tamp.service; do
    sleep 60
done
# If ROBUST already finished on its own there is nothing to donate to.
systemctl --user is-active --quiet rr5-kitchen-robust_tamp.service || {
    echo "robust already finished; nothing to reallocate $(date -Is)"; exit 0; }
echo "owl and vilain done -- restarting robust at 9 workers $(date -Is)"
systemctl --user stop rr5-kitchen-robust_tamp.service
sleep 10
systemctl --user reset-failed rr5-kitchen-robust_tamp.service 2>/dev/null
systemd-run --user --slice=lh.slice --collect --unit=rr5-kitchen-robust_tamp \
    --working-directory="$ROOT" --setenv=MUJOCO_GL=egl --setenv=PYTHONPATH="$ROOT" \
    "$ROOT/.venv/bin/python" -m baseline_common.run_baseline_execution_batch \
        --environment kitchen --methods robust_tamp --variants "$K" \
        --seeds 0,1,2,3,4,5,6,7,8,9 --camera-counts 3 --protocol native \
        --decoding model-native --output-root runs/kitchen/execution/final_20260913 \
        --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
        --max-tokens 24576 --max-model-calls 15 --max-replans 8 \
        --max-sketch-actions 24 --max-actions 80 \
        --episode-timeout 7200 --workers 9 --resume --continue-on-error >/dev/null 2>&1
echo "robust restarted at 9 workers $(date -Is)"
