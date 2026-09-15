#!/usr/bin/env bash
# One trial end to end: task instruction and three rendered views in, validated
# action sequence out.
#
#   ./run_demo.sh                    # kitchen K3
#   ./run_demo.sh workshop W2
#   ./run_demo.sh living_room L1
#
# Replays the archived foundation-model response for that variant, so it needs
# no GPU and makes no model call; everything after the call runs for real.
#
#   --live       call a served model instead of replaying
#   --physical   open workshop containers by driving the robot rather than
#                setting the simulator joint (~4 minutes per region)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false
export TAMP_ZS_MIN_SCORE=0.0 TAMP_ZS_MIN_MARGIN=0.0

DOMAIN="kitchen"; VARIANT="K3"
[ $# -ge 1 ] && case "$1" in -*) ;; *) DOMAIN="$1" ;; esac
[ $# -ge 2 ] && case "$2" in -*) ;; *) VARIANT="$2" ;; esac

SPEC=(--specification-root data/reference_run)
EXTRA=()
for a in "$@"; do
  [ "$a" = "--live" ] && SPEC=()
  [ "$a" = "--physical" ] && EXTRA+=(--physical)
done

OUT="results/runs/demo_${DOMAIN}_${VARIANT}"
echo "domain=${DOMAIN} variant=${VARIANT}  ->  ${OUT}"
echo

python3 scripts/evaluate_vlm_zs_canonicalization.py \
  "${SPEC[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --variants "$VARIANT" --output-root "$OUT"

echo
python3 - "$OUT" "$DOMAIN" "$VARIANT" <<'PY'
import json, pathlib, sys
out, domain, variant = sys.argv[1:4]
hits = sorted(pathlib.Path(out).rglob(f"{domain}/{variant}/*/result.json"))
if not hits:
    print(f"no result.json under {out}"); raise SystemExit(1)
d = json.loads(hits[-1].read_text())
print(f"status          {d.get('status')}")
print(f"outcome         {d.get('outcome_category')}")
print(f"regions opened  {d.get('inspected_regions')}")
print(f"role binding    {d.get('assignment')}")
print("action sequence:")
for i, s in enumerate(d.get("plan") or d.get("candidate_plan") or [], 1):
    if isinstance(s, dict):
        name = s.get("operator") or s.get("action") or s.get("action_instance_id")
        args = ", ".join(map(str, s.get("arguments") or []))
        print(f"  {i:2d}. {name}({args})")
    else:
        print(f"  {i:2d}. {s}")
PY
