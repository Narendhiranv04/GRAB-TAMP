#!/usr/bin/env bash
# Regenerate the reported tables, or re-run the evaluation.
#
#   ./run_benchmark.sh              # Tables III and IV from the scored run (seconds)
#   ./run_benchmark.sh --replay     # re-run all 32 variants from the archived responses
#   ./run_benchmark.sh --live       # fresh model calls (needs a served model)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false
export TAMP_ZS_MIN_SCORE=0.0 TAMP_ZS_MIN_MARGIN=0.0

case "${1:---tables}" in
  --tables)
    python3 scripts/make_paper_tables.py --out results/tables
    ;;
  --replay)
    python3 scripts/evaluate_vlm_zs_canonicalization.py \
      --specification-root data/reference_run --output-root results/runs/replay
    python3 scripts/score_full_experiment.py \
      --root results/runs/replay --out results/runs/replay
    ;;
  --live)
    : "${TAMP_MODEL:=qwen35-9b}" "${TAMP_BASE_URL:=http://127.0.0.1:8000/v1}"
    python3 scripts/run_live_repeat_experiment.py \
      --output-root results/runs/live --repeats 10 \
      --base-url "$TAMP_BASE_URL" --model "$TAMP_MODEL"
    python3 scripts/score_full_experiment.py --root results/runs/live --out results/runs/live
    ;;
  *) sed -n '2,7p' "$0" >&2; exit 2 ;;
esac
