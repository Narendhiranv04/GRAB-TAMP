set -u
cd /home/naren/RA_iiith
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false
export TAMP_FM_SCHEMA_VERSION=3 TAMP_FM_MAX_TOKENS=28000
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 TAMP_ZS_MIN_SCORE=0.0
run_one() {
  arm=$1; tag=$2; attempt=$3; variants=$4
  out=benchmark_reports/zs4_${arm}/${tag}
  # already finished cleanly -> skip
  if grep -qx "$arm $tag exit=0" /home/naren/zs_transfer/progress/zs4.txt 2>/dev/null; then
    echo "skip $arm $tag (done)"; return 0
  fi
  rm -rf "$out"; mkdir -p "$out"
  if [ "$arm" = "thresh" ]; then export TAMP_ZS_MIN_MARGIN=0.15; else export TAMP_ZS_MIN_MARGIN=0.0; fi
  python3 scripts/evaluate_vlm_zs_canonicalization.py \
    --output-root "$out" --specification-root "$attempt" --variants "$variants" > "$out/run.log" 2>&1
  rc=$?
  echo "$arm $tag exit=$rc" >> /home/naren/zs_transfer/progress/zs4.txt
  echo "done $arm $tag exit=$rc"
}
export -f run_one
for ARM in top1 thresh; do
  echo "=== ARM $ARM ==="
  awk -v a="$ARM" '{print a" "$0}' /home/naren/zs_transfer/jobs.txt | xargs -P 1 -n 4 bash -c 'run_one "$@"' _
done
echo "=== ZS4 RESUME COMPLETE ==="
