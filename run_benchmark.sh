#!/usr/bin/env bash
# Regenerate the GRAB-TAMP evaluation, or just its tables.
#
#   ./run_benchmark.sh --tables-only   # tables from the frozen run (seconds, no GPU)
#   ./run_benchmark.sh --replay        # recompile all 320 from the raw responses
#   ./run_benchmark.sh --ablations     # evidence + inspection-order ablations
#   ./run_benchmark.sh --live          # fresh model calls (needs a served model)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

SCORED=benchmark_reports/FINAL_10x32_RESULTS/rescored_current_scorer.json
MODE="${1:---tables-only}"

tables() {
  echo "==> paper tables"
  python3 scripts/make_paper_tables.py --assume-unscored-successful --out results/tables
}

case "$MODE" in
  --tables-only)
    tables
    ;;

  --replay)
    echo "==> recompiling every archived attempt from its raw response"
    echo "    (reported configuration: zero-shot canonicalization fallback on)"
    export TAMP_ZS_MIN_SCORE=0.0 TAMP_ZS_MIN_MARGIN=0.0 TOKENIZERS_PARALLELISM=false
    for root in benchmark_reports/full320_s0*/repeat_*/attempt_*; do
      [ -d "$root" ] || continue
      python3 scripts/evaluate_vlm_zs_canonicalization.py \
        --specification-root "$root" \
        --output-root "results/runs/replay/$(echo "$root" | tr '/' '_')"
    done
    echo "==> scoring"
    python3 scripts/score_full_experiment.py \
      --root results/runs/replay --out results/runs/replay
    tables
    ;;

  --ablations)
    echo "==> evidence ablation (semantic / unary / binary)"
    python3 scripts/run_fm_evidence_ablation.py --output-root results/runs/evidence

    echo "==> inspection order: FM-ranked, seeded-random, worst case"
    for arm in auto random worst; do
      python3 scripts/run_search_order_replay_timing.py \
        --scored-json "$SCORED" \
        --output-root "results/runs/order_$arm" --order-mode "$arm"
    done

    echo "==> scoring the inspection-order arms"
    python3 scripts/score_inspection_order_ablation.py \
      --fm results/runs/order_auto --random results/runs/order_random \
      --out results/order_fm_vs_random.json
    python3 scripts/score_inspection_order_ablation.py \
      --fm results/runs/order_auto --random results/runs/order_worst \
      --out results/order_fm_vs_worst.json
    python3 scripts/score_inspection_open_cost.py \
      --reports-root results/runs \
      --arms order_auto order_random order_worst \
      --costs benchmark_reports/workshop_open_costs.json \
      --out results/order_open_cost.json

    echo
    echo "Run the arms back to back on one machine with nothing else competing:"
    echo "grounding time is only interpretable within a machine. Read the"
    echo "living_room cells first -- that domain has no inspectable regions, so"
    echo "every arm runs it identically and whatever it still shows is the floor."
    ;;

  --live)
    : "${TAMP_MODEL:=qwen35-9b}"
    : "${TAMP_BASE_URL:=http://127.0.0.1:8000/v1}"
    echo "==> live run against ${TAMP_BASE_URL} (${TAMP_MODEL})"
    python3 scripts/run_live_repeat_experiment.py \
      --output-root results/runs/live --repeats 10 \
      --base-url "$TAMP_BASE_URL" --model "$TAMP_MODEL"
    python3 scripts/score_full_experiment.py \
      --root results/runs/live --out results/runs/live
    ;;

  *)
    sed -n '2,9p' "$0" >&2
    exit 2
    ;;
esac
