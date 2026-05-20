#!/usr/bin/env bash
# =============================================================================
# Resume ONLY 05_comparison.ipynb — does not re-run 01–04 or rewrite their CSVs.
#
# Use after a full-base sweep (N_SWEEP=1281167) completed notebooks 02–04 but
# notebook 05 was interrupted or is stale vs current results/full/*.csv.
#
#   LAB_N_SWEEP=1281167 LAB_SCALING_FULL=1 ./scripts/resume_from_05.sh
#
# Keeps: results/full/ivf_*.csv, hnsw_*.csv, lsh.csv (from 02–04)
# Refreshes: results/full/best_configs.csv, scaling.csv, docs/img/full/05_*.png
#
# Env (match your 02–04 sweep unless you plan to re-run those notebooks):
#   LAB_N_SWEEP=1281167   — must match n_base in loaded CSVs
#   LAB_SCALING_FULL=1    — include 1.28M point in scaling.csv
#   LAB_QUERY_N=5000      — sweeps used 5000 unless you re-ran 02–04 with 10000
#   SKIP_SCALING_REBUILD=1 — load results/full/scaling.csv instead of rebuilding (2nd pass)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

LAB_N_SWEEP="${LAB_N_SWEEP:-1281167}"
LAB_SCALING_FULL="${LAB_SCALING_FULL:-1}"
LAB_BATCH_ADD="${LAB_BATCH_ADD:-50000}"
LAB_QPS_REPEAT="${LAB_QPS_REPEAT:-3}"
LAB_QPS_WARMUP="${LAB_QPS_WARMUP:-1}"
LAB_QUERY_N="${LAB_QUERY_N:-5000}"
LAB_NOTEBOOK_TIMEOUT="${LAB_NOTEBOOK_TIMEOUT:-168h}"
LAB_CELL_TIMEOUT="${LAB_CELL_TIMEOUT:-86400}"
NB_KERNEL="${JUPYTER_KERNEL:-python3}"

export LAB_N_SWEEP LAB_SCALING_FULL LAB_BATCH_ADD LAB_QPS_REPEAT LAB_QPS_WARMUP LAB_QUERY_N
export SKIP_SCALING_REBUILD="${SKIP_SCALING_REBUILD:-0}"
export LAB_LIGHT=0

echo "==> resume_from_05: N_SWEEP=${LAB_N_SWEEP}  LAB_SCALING_FULL=${LAB_SCALING_FULL}"
echo "==> Does NOT execute notebooks 01–04"

# Ensure dirs only — no cleanup_legacy_outputs (would not remove CSVs anyway)
.venv/bin/python - <<'PY'
import utils
utils.results_dir()
utils.plots_dir()
PY

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="results/full/_backup_before_05_${STAMP}"
mkdir -p "$BACKUP"
for f in best_configs.csv best_configs_scenarios.csv scaling.csv; do
  [[ -f "results/full/$f" ]] && cp -a "results/full/$f" "$BACKUP/"
done
shopt -s nullglob
for f in docs/img/full/05_*.png; do
  cp -a "$f" "$BACKUP/"
done
echo "==> Backed up prior 05 outputs → $BACKUP/"

if [[ ! -f results/full/ivf_all.csv || ! -f results/full/hnsw_all.csv ]]; then
  echo "ERROR: Missing results/full/ivf_all.csv or hnsw_all.csv — run notebooks 02–03 first." >&2
  exit 1
fi

echo ""
echo "==================== Executing 05_comparison.ipynb ===================="
echo "  Started: $(date)"
timeout --foreground "${LAB_NOTEBOOK_TIMEOUT}" .venv/bin/jupyter nbconvert \
  --to notebook --inplace \
  --execute \
  --ExecutePreprocessor.timeout="${LAB_CELL_TIMEOUT}" \
  --ExecutePreprocessor.kernel_name="$NB_KERNEL" \
  05_comparison.ipynb

echo ""
echo "==> Done. Check:"
echo "    results/full/best_configs.csv"
echo "    results/full/best_configs_scenarios.csv"
echo "    results/full/scaling.csv  (should include n=${LAB_N_SWEEP} if LAB_SCALING_FULL=1)"
echo "    docs/img/full/05_*.png"
echo "==> Next: LAB_N_SWEEP=1000000 ./scripts/record_flame.sh"
