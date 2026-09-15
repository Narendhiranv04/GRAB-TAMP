#!/usr/bin/env bash
# One GRAB-TAMP trial, end to end: instruction and three rendered views in,
# validated action sequence out.
#
#   ./run_demo.sh                  # kitchen K3
#   ./run_demo.sh workshop W2      # robot-actuated container opening
#   ./run_demo.sh living_room L1   # no inspectable regions, no search stage
#
# Replays the archived foundation-model response, so it needs no GPU and makes
# no model call. The response is recompiled from the model's own words, so the
# whole pipeline downstream of the call runs for real.
#
#   --live       call a served model instead of replaying
#   --physical   drive containers open with the robot instead of setting the
#                simulator joint (workshop only; kitchen always sets the joint)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

DOMAIN="kitchen"; VARIANT="K3"
[ $# -ge 1 ] && case "$1" in -*) ;; *) DOMAIN="$1" ;; esac
[ $# -ge 2 ] && case "$2" in -*) ;; *) VARIANT="$2" ;; esac

SPEC_ARGS=(--spec-source raw-replay --specification-root examples)
DRY=(--dry-run)
for arg in "$@"; do
  case "$arg" in
    --live)     SPEC_ARGS=(--spec-source live) ;;
    --physical) DRY=() ;;
  esac
done

# Fall back to the full archive when the variant is not one of the examples.
if [ "${SPEC_ARGS[1]}" = "raw-replay" ] \
   && [ ! -f "examples/${DOMAIN}/${VARIANT}/vlm/fm_diagnostics/fm_call_001.json" ]; then
  HIT=$(find benchmark_reports/full320_s0* \
          -path "*/${DOMAIN}/${VARIANT}/vlm/fm_diagnostics/fm_call_001.json" \
          -print -quit 2>/dev/null || true)
  if [ -z "$HIT" ]; then
    echo "No archived response for ${DOMAIN}/${VARIANT}; re-run with --live." >&2
    exit 1
  fi
  SPEC_ARGS=(--spec-source raw-replay --specification-root "${HIT%/${DOMAIN}/${VARIANT}/vlm/fm_diagnostics/fm_call_001.json}")
fi

OUT="results/demo/${DOMAIN}_${VARIANT}"
echo "domain=${DOMAIN} variant=${VARIANT} ${SPEC_ARGS[*]}"
echo "output=${OUT}"
echo

python3 scripts/evaluate_vlm_functional_tamp.py \
  --mode vlm "${SPEC_ARGS[@]}" "${DRY[@]}" \
  --variants "$VARIANT" --output-root "$OUT"

echo
echo "Terminal status and validated plan:"
python3 - "$OUT" "$DOMAIN" "$VARIANT" <<'PY'
import json, pathlib, sys
out, domain, variant = sys.argv[1:4]
hits = sorted(pathlib.Path(out).rglob(f"{domain}/{variant}/*/result.json"))
if not hits:
    print(f"  (no result.json under {out})")
    raise SystemExit(0)
doc = json.loads(hits[-1].read_text())
print(f"  status           {doc.get('status')}")
print(f"  outcome          {doc.get('outcome_category')}")
print(f"  regions opened   {doc.get('inspected_regions')}")
print(f"  role assignment  {doc.get('assignment')}")
plan = doc.get("plan") or doc.get("candidate_plan") or []
for i, step in enumerate(plan, 1):
    print(f"  {i:2d}. {step.get('action_instance_id', step) if isinstance(step, dict) else step}")
PY
