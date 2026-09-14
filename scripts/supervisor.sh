#!/bin/bash
# Drive the grid to completion without supervision.
#
# Every failure this run has hit needs a different response, and none of them
# were automatic:
#   * the endpoint died and legs kept writing instant failures -- 322 episodes
#     lost, so legs are stopped while it is down and restarted when it returns;
#   * a leg exited 0 with its grid short (ViLaIn 118/120), so any cell below
#     target with no live leg is restarted;
#   * the session died and the orchestrator read "no legs" as "scene complete",
#     so a scene is finished only when its episodes are on disk.
#
# `--resume` skips completed episodes, so restarting a leg costs only whatever
# was in flight.
set -u
ROOT=/home/longhorizon/Documents/LH_Extension/V1
V="$ROOT/.venv/bin/python"; VV="$ROOT/.venv-vilain-tamp/bin/python"
URL=http://127.0.0.1:18000/v1/models
declare -A VARIANTS=(
  [kitchen]=K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12
  [workshop]=W1,W2,W3,W4,W5,W6,W7,W8,W9,W10
  [living_room]=L1,L2,L3,L4,L5,L6,L7,L8,L9,L10 )
declare -A TARGET=( [kitchen]=120 [workshop]=100 [living_room]=100 )
# phase 1 runs kitchen and living room together; workshop follows.
# Living Room carries the phase once Kitchen drains, so it is sized for that
# rather than for the moment the phase starts.  Leaving it at 7 workers let
# concurrency -- and with it throughput -- fall as Kitchen finished: 935 tok/s
# down to 809, which turned a +2.3 h margin into -0.2 h.  Kitchen's own counts
# stay high because its remaining legs are the slowest episodes in the grid.
# Phases are split so that capacity a finishing scene releases is picked up by
# the next one, rather than idling until every scene in the phase is done.
# Kitchen held 16 workers for its last 8 episodes while Workshop waited on
# Living Room; concurrency sagged and the margin went from +2.3 h to -0.2 h.
# Kitchen runs alone because its episodes are the most expensive in the grid;
# Living Room and Workshop then share the machine.
PHASE1="kitchen:robust_tamp:8 kitchen:vlm_tamp:5 kitchen:owl_tamp:2 kitchen:vilain_tamp:2"
PHASE2="living_room:vlm_tamp:4 living_room:owl_tamp:3 living_room:vilain_tamp:3 living_room:robust_tamp:3
        workshop:vlm_tamp:7 workshop:robust_tamp:6 workshop:vilain_tamp:4 workshop:owl_tamp:3"

count() { find "$ROOT/runs/$1/execution/final_20260913/$2" \
    -path '*/seed_[0-9][0-9][0-9]/benchmark_execution_result.json' 2>/dev/null | wc -l; }
healthy() { curl -s -m 15 -o /dev/null "$URL"; }

start_leg() { # scene method workers
    local py="$V"; [ "$2" = vilain_tamp ] && py="$VV"
    systemctl --user reset-failed "rr5-$1-$2.service" 2>/dev/null
    systemd-run --user --slice=lh.slice --collect --unit="rr5-$1-$2" \
        --working-directory="$ROOT" --setenv=MUJOCO_GL=egl --setenv=PYTHONPATH="$ROOT" \
        "$py" -m baseline_common.run_baseline_execution_batch \
            --environment "$1" --methods "$2" --variants "${VARIANTS[$1]}" \
            --seeds 0,1,2,3,4,5,6,7,8,9 --camera-counts 3 --protocol native \
            --decoding model-native --output-root "runs/$1/execution/final_20260913" \
            --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
            --max-tokens 24576 --max-model-calls 15 --max-replans 8 \
            --max-sketch-actions 24 --max-actions 80 --episode-timeout 7200 \
            --workers "$3" --resume --continue-on-error >/dev/null 2>&1
    echo "  started rr5-$1-$2 (workers=$3, at $(count "$1" "$2")/${TARGET[$1]}) $(date -Is)"
}

phase_done() { # spec
    local spec s m
    for spec in $1; do
        IFS=: read -r s m _ <<< "$spec"
        [ "$(count "$s" "$m")" -ge "${TARGET[$s]}" ] || return 1
    done
    return 0
}

drive() { # spec
    local spec s m w down=0
    while ! phase_done "$1"; do
        if ! healthy; then
            if [ "$down" -eq 0 ]; then
                echo "ENDPOINT DOWN -- stopping legs $(date -Is)"
                for u in $(systemctl --user list-units --state=active --no-legend 'rr5-*' | awk '{print $1}'); do
                    systemctl --user stop "$u"
                done
                down=1
            fi
            sleep 60; continue
        fi
        [ "$down" -eq 1 ] && { echo "ENDPOINT BACK -- resuming $(date -Is)"; down=0; }
        for spec in $1; do
            IFS=: read -r s m w <<< "$spec"
            [ "$(count "$s" "$m")" -ge "${TARGET[$s]}" ] && continue
            systemctl --user is-active --quiet "rr5-$s-$m.service" && continue
            start_leg "$s" "$m" "$w"
            sleep 5
        done
        sleep 120
    done
}

echo "=== supervisor start $(date -Is) ==="
drive "$PHASE1"; echo "=== phase 1 complete (kitchen) $(date -Is) ==="
drive "$PHASE2"; echo "=== phase 2 complete (living room + workshop) $(date -Is) ==="
echo "=== ALL COMPLETE $(date -Is) ==="
