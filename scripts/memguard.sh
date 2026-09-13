#!/bin/bash
# Shed the least valuable leg rather than let the host OOM.
#
# Episode processes hold ~1.27 GB each, so the worker count is bounded by RAM,
# not by the GPU (vLLM accepts 32 concurrent and we feed it ~20).  Pushing the
# count up to meet a deadline moves us toward that ceiling; an OOM kill would
# lose episodes silently, which is the failure this project keeps paying for.
#
# Sheds in reverse order of value: Living Room ROBUST-TAMP is the cheapest leg
# to restart (2.6 min/episode) and Workshop VLM-TAMP the most expensive.
set -u
SHED_ORDER="rr5-living_room-robust_tamp rr5-living_room-vlm_tamp rr5-living_room-owl_tamp"
while true; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt 900 ]; then
        for u in $SHED_ORDER; do
            if systemctl --user is-active --quiet "$u.service"; then
                echo "MEMORY LOW (${avail}MB) -- shedding $u $(date -Is)"
                systemctl --user stop "$u.service"
                break
            fi
        done
        sleep 120
    fi
    [ -z "$(systemctl --user list-units --state=active --no-legend 'rr5-*' 2>/dev/null)" ] \
        && ! systemctl --user is-active --quiet rrworkshop.service && exit 0
    sleep 20
done
