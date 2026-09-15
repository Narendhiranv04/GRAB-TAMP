set -u
cd /home/naren/RA_iiith
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false
export TAMP_FM_SCHEMA_VERSION=3 TAMP_FM_MAX_TOKENS=28000
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
mkdir -p /home/naren/zs_transfer/progress
: > /home/naren/zs_transfer/progress/done.txt
python3 - > /home/naren/zs_transfer/jobs.txt <<'PY'
import json
from pathlib import Path
from collections import defaultdict
rows=json.load(open('benchmark_reports/FINAL_10x32_RESULTS/final_10x32.json'))['rows']
g=defaultdict(list)
for r in rows:
    rd=Path(r['run_dir'])
    if (rd/'fm_diagnostics'/'fm_call_001.json').exists(): g[str(rd.parents[2])].append(r['variant'])
for attempt,variants in sorted(g.items()):
    print(f"{'__'.join(Path(attempt).parts[-3:])} {attempt} {','.join(sorted(set(variants)))}")
PY
echo "jobs: $(wc -l < /home/naren/zs_transfer/jobs.txt)"
run_one() {
  tag=$1; attempt=$2; variants=$3
  out=benchmark_reports/zs_arm/${tag}
  mkdir -p "$out"
  python3 scripts/evaluate_vlm_zs_canonicalization.py \
    --output-root "$out" --specification-root "$attempt" --variants "$variants" \
    > "$out/run.log" 2>&1
  echo "$tag exit=$?" >> /home/naren/zs_transfer/progress/done.txt
}
export -f run_one
xargs -P 2 -n 3 bash -c 'run_one "$@"' _ < /home/naren/zs_transfer/jobs.txt
echo "=== ZS ARM COMPLETE ==="
