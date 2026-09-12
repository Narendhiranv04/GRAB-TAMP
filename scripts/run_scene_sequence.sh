#!/bin/bash
# Run the execution grid one scene at a time.
#
# Spreading 17 workers over 12 legs wastes them: every leg needs at least one,
# so Living Room ROBUST-TAMP finishes in 8 hours and idles while Workshop
# VLM-TAMP runs 90 on a single worker.  Total work is 573 leg-hours, so the
# floor at 17 workers is 34 hours; the parallel shape took 90.  One scene at a
# time puts all 17 on four legs, and workers are allocated by measured
# leg-hours rather than by episode count.
#
# Leg units are rr5-<scene>-<method>.  This orchestrator is `rrseq`, which is
# NOT matched by that pattern -- a previous orchestrator stopped itself by
# listing units with a glob that included its own name.
set -u
ROOT=/home/longhorizon/Documents/LH_Extension/V1
V="$ROOT/.venv/bin/python"; VV="$ROOT/.venv-vilain-tamp/bin/python"
K=K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12
W=W1,W2,W3,W4,W5,W6,W7,W8,W9,W10
L=L1,L2,L3,L4,L5,L6,L7,L8,L9,L10

start_leg() { # scene variants workers method python
    systemctl --user reset-failed "rr5-$1-$4.service" 2>/dev/null
    systemd-run --user --slice=lh.slice --collect --unit="rr5-$1-$4" \
        --working-directory="$ROOT" --setenv=MUJOCO_GL=egl --setenv=PYTHONPATH="$ROOT" \
        "$5" -m baseline_common.run_baseline_execution_batch \
            --environment "$1" --methods "$4" --variants "$2" --seeds 0,1,2,3,4,5,6,7,8,9 \
            --camera-counts 3 --protocol native --decoding model-native \
            --output-root "runs/$1/execution/final_20260913" \
            --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
            --max-tokens 24576 --max-model-calls 15 --max-replans 8 \
            --max-sketch-actions 24 --max-actions 80 \
            --episode-timeout 7200 --workers "$3" --resume --continue-on-error \
        >/dev/null 2>&1
}

wait_for_scene() {
    while [ -n "$(systemctl --user list-units --state=active --no-legend 'rr5-*' 2>/dev/null)" ]; do
        sleep 60
    done
}

run_scene() { # scene variants "method:workers ..."
    local scene=$1 variants=$2; shift 2
    echo "=== $scene starting $(date -Is) ==="
    for spec in "$@"; do
        local m=${spec%%:*} w=${spec##*:} py="$V"
        [ "$m" = vilain_tamp ] && py="$VV"
        start_leg "$scene" "$variants" "$w" "$m" "$py"
        echo "    leg $m workers=$w"
    done
    sleep 20
    wait_for_scene
    local n
    n=$(find "$ROOT/runs/$scene/execution/final_20260913" \
        -path '*/seed_[0-9][0-9][0-9]/benchmark_execution_result.json' 2>/dev/null | wc -l)
    echo "=== $scene finished $(date -Is) episodes=$n ==="
}

# Workers allocated by measured leg-hours, not by episode count.
run_scene kitchen     "$K" vlm_tamp:8 robust_tamp:3 vilain_tamp:3 owl_tamp:3
run_scene workshop    "$W" vlm_tamp:7 robust_tamp:6 vilain_tamp:3 owl_tamp:1
run_scene living_room "$L" vlm_tamp:6 vilain_tamp:5 owl_tamp:4 robust_tamp:2
echo "=== ALL SCENES COMPLETE $(date -Is) ==="
