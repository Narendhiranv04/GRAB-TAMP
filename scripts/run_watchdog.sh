#!/bin/bash
# Halt the execution legs when the inference endpoint dies.
#
# On 2026-09-13 the endpoint went down mid-run and the legs kept going, writing
# 322 episodes that terminated the moment a model call failed -- 99 of the
# Workshop ones in under 30 seconds.  Nothing was watching, so the grid was
# consumed before anyone looked.  An outage should cost minutes, not a cell.
#
# Stops on the second consecutive failed probe, so one dropped request does not
# halt a healthy run.
URL="${1:-http://127.0.0.1:18000/v1/models}"
PATTERN="${2:-rr4-*}"
misses=0
while true; do
    if curl -s -m 10 -o /dev/null "$URL"; then
        misses=0
    else
        misses=$((misses + 1))
        echo "ENDPOINT PROBE FAILED ($misses) $(date -Is)"
        if [ "$misses" -ge 2 ]; then
            echo "ENDPOINT DOWN -- stopping legs matching $PATTERN"
            # The orchestrator first: otherwise it simply starts the next scene
            # into a dead endpoint.  It is stopped by exact name, never by a
            # glob -- a previous orchestrator matched its own unit and killed
            # itself.
            systemctl --user stop rrseq.service 2>/dev/null && echo "  stopped rrseq.service"
            for u in $(systemctl --user list-units --state=active --no-legend "$PATTERN" | awk '{print $1}'); do
                systemctl --user stop "$u" && echo "  stopped $u"
            done
            echo "LEGS HALTED $(date -Is)"
            exit 1
        fi
    fi
    # Exit quietly once every leg has finished on its own.
    # Quiet exit only when the orchestrator is done too; between scenes there
    # are briefly no legs, and exiting then would disarm the guard.
    if [ -z "$(systemctl --user list-units --state=active --no-legend "$PATTERN" 2>/dev/null)" ] \
       && ! systemctl --user is-active --quiet rrseq.service; then
        echo "ALL LEGS FINISHED $(date -Is)"; exit 0
    fi
    sleep 30
done
