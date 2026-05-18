#!/bin/bash
# Light benchmark pass — minutes, not hours.
# Does NOT index the full 1.28M base for Flat GT verification; uses smaller parameter grids.
#
#   ./run_light.sh              # default: 100k base vectors, LAB_LIGHT=1
#   LAB_N_SWEEP=50000 ./run_light.sh
#
set -e
cd "$(dirname "$0")"

.venv/bin/python - <<'PY'
import utils
n = utils.cleanup_legacy_outputs()
print(f"==> removed {n} legacy file(s) from results/ and docs/img/ roots")
PY

export LAB_LIGHT=1
export LAB_N_SWEEP="${LAB_N_SWEEP:-100000}"
export LAB_QPS_REPEAT="${LAB_QPS_REPEAT:-2}"
export LAB_QPS_WARMUP="${LAB_QPS_WARMUP:-1}"
LAB_NOTEBOOK_TIMEOUT="${LAB_NOTEBOOK_TIMEOUT:-180m}"
LAB_CELL_TIMEOUT="${LAB_CELL_TIMEOUT:-3600}"

echo "==> LAB_LIGHT=$LAB_LIGHT  LAB_N_SWEEP=$LAB_N_SWEEP"
echo "==> QPS timing: repeat=${LAB_QPS_REPEAT}  warmup=${LAB_QPS_WARMUP}"
echo "==> Per-notebook wall timeout: ${LAB_NOTEBOOK_TIMEOUT}  ·  cell timeout: ${LAB_CELL_TIMEOUT}s"
echo "==> Free RAM: $(free -h 2>/dev/null | awk '/Mem:/ {print $7}' || echo '?')"

NB_KERNEL="${JUPYTER_KERNEL:-python3}"

for nb in 01_data_preparation.ipynb \
          02_ivf_benchmark.ipynb \
          03_hnsw_benchmark.ipynb \
          04_lsh_benchmark.ipynb \
          05_comparison.ipynb; do
    echo ""
    echo "====================  $nb  ($(date -Iseconds))"
    echo "===================="
    timeout --foreground "${LAB_NOTEBOOK_TIMEOUT}" .venv/bin/jupyter nbconvert \
        --to notebook --inplace \
        --execute \
        --ExecutePreprocessor.timeout="${LAB_CELL_TIMEOUT}" \
        --ExecutePreprocessor.kernel_name="$NB_KERNEL" \
        "$nb"
done

echo ""
echo "==> Light run complete. Plots: docs/img/light/  CSVs: results/light/"
echo "==> Full benchmarks: unset LAB_LIGHT; export LAB_N_SWEEP=1281167; ./run_all.sh  → results/full/ docs/img/full/"
