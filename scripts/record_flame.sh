#!/bin/bash
# =============================================================================
# AGENTS — Lab3 flame-only run (py-spy HNSW build). Run after ./run_all.sh / meta_run_all.
# -----------------------------------------------------------------------------
# Does not run notebooks. Writes where notebook 05 embeds the SVG:
#   docs/img/full/hnsw_build_flame.svg  (or docs/img/light/ when LAB_LIGHT=1)
# Also copies a marker to results/flame/run.txt
#
#   LAB_N_SWEEP=1000000 ./scripts/record_flame.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

LAB_N_SWEEP="${LAB_N_SWEEP:-500000}"
LAB_LIGHT="${LAB_LIGHT:-0}"
if [[ "$LAB_LIGHT" == "1" ]]; then
  OUT_DIR="docs/img/light"
  RESULTS_SUB="light"
else
  OUT_DIR="docs/img/full"
  RESULTS_SUB="full"
fi
mkdir -p "$OUT_DIR" "results/flame"
OUT="$OUT_DIR/hnsw_build_flame.svg"
{
  echo "flame-only run LAB_N_SWEEP=$LAB_N_SWEEP LAB_LIGHT=$LAB_LIGHT"
  echo "output=$OUT"
} > "results/flame/run.txt"

PY_SPY="${PY_SPY:-}"
if [[ -z "$PY_SPY" && -x ".venv/bin/py-spy" ]]; then
  PY_SPY=".venv/bin/py-spy"
elif [[ -z "$PY_SPY" ]] && command -v py-spy >/dev/null 2>&1; then
  PY_SPY="$(command -v py-spy)"
fi

if [[ -z "$PY_SPY" || ! -x "$PY_SPY" ]]; then
  echo "py-spy not found. Install: .venv/bin/pip install py-spy"
  exit 1
fi

export LAB_N_SWEEP
echo "==> Recording HNSW build flame (N_SWEEP=$LAB_N_SWEEP) → $OUT"

# py-spy: --format flamegraph (default) writes SVG; there is no "svg" format name.
"$PY_SPY" record -o "$OUT" --format flamegraph -- \
  .venv/bin/python - <<'PY'
import gc
import os
from pathlib import Path

import faiss
import h5py
import numpy as np

import utils

DATA = Path("data")
with h5py.File(DATA / "imagenet1m.h5") as h:
    dim = int(h.attrs["dim"])
    base_path = str(h.attrs["base_path"])
if not Path(base_path).exists():
    base_path = str((DATA / "imagenet_base.fvecs").resolve())

n = int(os.environ.get("LAB_N_SWEEP", "500000"))
idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
idx.hnsw.efConstruction = 200
utils.stream_add(idx, base_path, n)
print("built", idx.ntotal)
del idx
gc.collect()
PY

echo "==> Wrote $OUT"
