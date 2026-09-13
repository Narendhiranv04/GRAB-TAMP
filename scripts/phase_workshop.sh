#!/bin/bash
# Start Workshop only once Kitchen and Living Room are both complete, counted by
# episodes on disk.  The previous orchestrator treated "no active legs" as
# "scene complete"; a session death then made it declare Kitchen finished at
# 370/480 and run the remaining scenes into a dead endpoint.
set -u
ROOT=/home/longhorizon/Documents/LH_Extension/V1
count() { find "$ROOT/runs/$1/execution/final_20260913" \
    -path '*/seed_[0-9][0-9][0-9]/benchmark_execution_result.json' 2>/dev/null | wc -l; }
while true; do
    k=$(count kitchen); l=$(count living_room)
    [ "$k" -ge 480 ] && [ "$l" -ge 400 ] && break
    # Legs all gone but the grids are short: they died rather than finished.
    if [ -z "$(systemctl --user list-units --state=active --no-legend 'rr5-*' 2>/dev/null)" ]; then
        sleep 300
        k=$(count kitchen); l=$(count living_room)
        if [ "$k" -lt 480 ] || [ "$l" -lt 400 ]; then
            echo "ABORT: legs gone with kitchen=$k/480 living_room=$l/400 $(date -Is)"; exit 1
        fi
        break
    fi
    sleep 120
done
echo "kitchen and living_room complete -- starting workshop $(date -Is)"
W=W1,W2,W3,W4,W5,W6,W7,W8,W9,W10
go() { systemctl --user reset-failed "rr5-workshop-$2.service" 2>/dev/null
  systemd-run --user --slice=lh.slice --collect --unit="rr5-workshop-$2" \
    --working-directory="$ROOT" --setenv=MUJOCO_GL=egl --setenv=PYTHONPATH="$ROOT" \
    "$3" -m baseline_common.run_baseline_execution_batch \
      --environment workshop --methods "$2" --variants "$W" --seeds 0,1,2,3,4,5,6,7,8,9 \
      --camera-counts 3 --protocol native --decoding model-native \
      --output-root runs/workshop/execution/final_20260913 \
      --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
      --max-tokens 24576 --max-model-calls 15 --max-replans 8 --max-sketch-actions 24 \
      --max-actions 80 --episode-timeout 7200 --workers "$1" --resume --continue-on-error >/dev/null 2>&1; }
go 7 vlm_tamp "$ROOT/.venv/bin/python"
go 6 robust_tamp "$ROOT/.venv/bin/python"
go 3 vilain_tamp "$ROOT/.venv-vilain-tamp/bin/python"
go 1 owl_tamp "$ROOT/.venv/bin/python"
echo "workshop started $(date -Is)"
