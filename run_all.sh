#!/bin/bash
# Execute every notebook in order, in-place, writing outputs back to the .ipynb.
# Set LAB_N_SWEEP to control how many base vectors are used by the parameter sweeps.
# Do NOT set LAB_LIGHT here (full grids + full Flat GT check in notebook 01).
#
#   LAB_N_SWEEP=200000  ./run_all.sh   # ~30 min total - quick validation
#   LAB_N_SWEEP=500000  ./run_all.sh   # ~3-5 h  - default
#   LAB_N_SWEEP=1281167 ./run_all.sh   # ~10-15 h - full 1.28 M base
#
# RAM budget (with stream_add — no parallel `xb` array):
#   IVFFlat / HNSW peak ≈ n * 8 KB + small overhead
#       500 K → ~5  GB, 1 M → ~9  GB, 1.28 M → ~11 GB
#   Flat GT (notebook 01)  ≈ same as IVFFlat
#   IVF+PQ / IVF+SQ / LSH  ≪ 1 GB
# Add ~2 GB for Python/Jupyter/plotting. Close browsers when running at full size.
#
# Tunables:
#   LAB_BATCH_ADD   — vectors per stream_add() batch (default 50000, ~400 MB peak)
#   LAB_QPS_REPEAT / LAB_QPS_WARMUP — qps timing passes (default 1 / 0 for full)
#   LAB_QUERY_N       — queries per sweep timing (default 5000 for full, all for light)
#   LAB_NOTEBOOK_TIMEOUT — wall-clock cap per notebook (default 12h)
#   LAB_CELL_TIMEOUT  — ExecutePreprocessor per-cell timeout in seconds (default 14400)
#
# For a short smoke run use ./run_light.sh instead.
#
set -e
cd "$(dirname "$0")"

# Drop stale flat results/*.csv from before light/full split; ensure subdirs exist.
.venv/bin/python - <<'PY'
import utils
n = utils.cleanup_legacy_outputs()
print(f"==> removed {n} legacy file(s) from results/ and docs/img/ roots")
utils.results_dir()
utils.plots_dir()
PY

NB_KERNEL="${JUPYTER_KERNEL:-python3}"
LAB_N_SWEEP="${LAB_N_SWEEP:-500000}"
LAB_BATCH_ADD="${LAB_BATCH_ADD:-50000}"
LAB_QPS_REPEAT="${LAB_QPS_REPEAT:-1}"
LAB_QPS_WARMUP="${LAB_QPS_WARMUP:-0}"
LAB_QUERY_N="${LAB_QUERY_N:-5000}"
LAB_NOTEBOOK_TIMEOUT="${LAB_NOTEBOOK_TIMEOUT:-12h}"
LAB_CELL_TIMEOUT="${LAB_CELL_TIMEOUT:-14400}"

export LAB_N_SWEEP LAB_BATCH_ADD LAB_QPS_REPEAT LAB_QPS_WARMUP LAB_QUERY_N
export LAB_LIGHT=0
echo "==> N_SWEEP=${LAB_N_SWEEP}  BATCH_ADD=${LAB_BATCH_ADD}  LAB_LIGHT=${LAB_LIGHT}"
echo "==> QPS timing: repeat=${LAB_QPS_REPEAT}  warmup=${LAB_QPS_WARMUP}  QUERY_N=${LAB_QUERY_N}"
echo "==> Per-notebook wall timeout: ${LAB_NOTEBOOK_TIMEOUT}  ·  cell timeout: ${LAB_CELL_TIMEOUT}s"
echo "==> Free RAM: $(free -h | awk '/Mem:/ {print $7}')"

for nb in 01_data_preparation.ipynb \
          02_ivf_benchmark.ipynb \
          03_hnsw_benchmark.ipynb \
          04_lsh_benchmark.ipynb \
          05_comparison.ipynb; do
    echo ""
    echo "===================="
    echo "  Executing $nb"
    echo "  Started: $(date)"
    echo "===================="
    timeout --foreground "${LAB_NOTEBOOK_TIMEOUT}" .venv/bin/jupyter nbconvert \
        --to notebook --inplace \
        --execute \
        --ExecutePreprocessor.timeout="${LAB_CELL_TIMEOUT}" \
        --ExecutePreprocessor.kernel_name="$NB_KERNEL" \
        "$nb"
done

echo ""
echo "==> All notebooks complete."
echo "==> Plots: docs/img/full/ (full) or docs/img/light/ (light)  ·  CSVs: results/full/ or results/light/"
