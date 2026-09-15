set -u
cd ~/zsrun && . .venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export TAMP_FM_SCHEMA_VERSION=3 TAMP_FM_MAX_TOKENS=28000
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p ~/zsrun/progress ~/zsrun/out
: > ~/zsrun/progress/done.txt
python - <<'PY' > ~/zsrun/groups.json
import json
from pathlib import Path
from collections import defaultdict
rows=json.load(open('benchmark_reports/FINAL_10x32_RESULTS/final_10x32.json'))['rows']
g=defaultdict(list)
for r in rows:
    rd=Path(r['run_dir'])
    if (rd/'fm_diagnostics'/'fm_call_001.json').exists(): g[str(rd.parents[2])].append(r['variant'])
json.dump({k:sorted(set(v)) for k,v in sorted(g.items())}, open(1,'w'))
PY
python - <<'PY' > ~/zsrun/jobs.txt
import json, os
from pathlib import Path
g=json.load(open(os.path.expanduser('~/zsrun/groups.json')))
for arm in ('baseline','zs'):
    for attempt,variants in g.items():
        tag="__".join(Path(attempt).parts[-3:])
        print(f"{arm} {tag} {attempt} {','.join(variants)}")
PY
echo "jobs: $(wc -l < ~/zsrun/jobs.txt)"
run_one() {
  arm=$1; tag=$2; attempt=$3; variants=$4
  out=~/zsrun/out/zs_${arm}/${tag}
  mkdir -p "$out"
  if [ "$arm" = "baseline" ]; then
    python scripts/evaluate_vlm_zs_canonicalization.py --baseline \
      --output-root "$out" --specification-root "$attempt" --variants "$variants" > "$out/run.log" 2>&1
  else
    python scripts/evaluate_vlm_zs_canonicalization.py \
      --output-root "$out" --specification-root "$attempt" --variants "$variants" > "$out/run.log" 2>&1
  fi
  echo "$arm $tag exit=$?" >> ~/zsrun/progress/done.txt
}
export -f run_one
xargs -P 10 -n 4 bash -c 'run_one "$@"' _ < ~/zsrun/jobs.txt
echo "=== SERVER RUN COMPLETE ==="
