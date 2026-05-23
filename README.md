# DB Algorithms Lab 3 — FAISS ANN Benchmarks

Comprehensive Jupyter-notebook benchmark suite comparing **IVF / IVF+PQ / IVF+SQ / HNSW / LSH**
approximate-nearest-neighbour indexes from [FAISS](https://github.com/facebookresearch/faiss)
on the **ImageNet-1M** (ZJU) dataset.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

### Light vs full benchmarks

| Script | Purpose |
|--------|---------|
| `./run_light.sh` | **Default for laptops:** `LAB_LIGHT=1`, `LAB_N_SWEEP=100000`, smaller parameter grids, subset Flat GT check in notebook 01 (no full-base index). Finishes in minutes. Writes **`results/light/`** and **`docs/img/light/`**. |
| `./run_all.sh` | Full experiment matrix + full-base Flat vs published GT in notebook 01. Can take many hours. Writes **`results/full/`** and **`docs/img/full/`**. |

Environment variables:

- `LAB_N_SWEEP` — how many **first** base vectors indexes use (and recomputed exact GT cache `data/gt_n{N}_k100.npy`).
- `LAB_LIGHT=1` — shrink grids in notebooks 02–04; skip heavy scaling curve in 05 (single snapshot); notebook 01 uses a 100k-vector Flat + numpy spot-check instead of indexing all 1.28M vectors.
- `LAB_BATCH_ADD` — batch size for `utils.stream_add()` (default `50000` in `run_all.sh`).
- `LAB_QPS_REPEAT` / `LAB_QPS_WARMUP` — qps timing passes (defaults: `3` / `1` in `run_all.sh`; `2` / `1` in `run_light.sh`).
- `LAB_QUERY_N` — queries used in sweep timing (default `5000` in full runs).
- `LAB_SCALING_FULL` — set to `1` to include the full 1.28M base in notebook 05 scaling.
- `LAB_NOTEBOOK_TIMEOUT` / `LAB_CELL_TIMEOUT` — wall-clock cap per notebook and per-cell seconds for `nbconvert` (defaults: `12h` / `14400` in `run_all.sh`; `180m` / `3600` in `run_light.sh`).

Run notebooks in order (or use the scripts above):

1. `01_data_preparation.ipynb` — downloads the dataset, validates the ground truth,
   produces dataset-exploration plots, and stores everything in `data/imagenet1m.h5`.
2. `02_ivf_benchmark.ipynb` — IVFFlat / IVF+PQ / IVF+ScalarQuantizer sweeps.
3. `03_hnsw_benchmark.ipynb` — HNSW `M` / `efConstruction` / `efSearch` sweeps.
4. `04_lsh_benchmark.ipynb` — random-projection LSH `nbits` sweep.
5. `05_comparison.ipynb` — cross-algorithm Pareto, scaling, anomaly analysis, final pick.

After the notebooks finish, generate the consolidated review:

```bash
python3 scripts/analyze_and_report.py --run full              # default
python3 scripts/analyze_and_report.py --run full --english    # also emit REPORT_*.md
```

It reads `results/{run}/*.csv`, refreshes the cross-algorithm / memory-budget /
anomaly / cross-CSV-consistency charts under `docs/img/{run}/`, writes derived
stats next to the CSVs and emits two Russian reports:

- `docs/OTCHET_polnyj_{run}.md` — подробный (draft, full explanations + proofs)
- `docs/OTCHET_kratkij_{run}.md` — краткий (same charts + tables, minimal prose)

Pass `--english` to additionally generate the legacy English `REPORT_{run}.md`.
Independent of FAISS — runs in seconds.

## Dataset

- **Source:** http://www.cad.zju.edu.cn/home/dengcai/Data/ANNS/ANNSData.html
- **Base set:** 1 281 167 vectors × 2048 dims (≈ 9.4 GB float32)
- **Queries:** 25 000 vectors (all used; the task requires ≥ 10 000)
- **Ground truth:** precomputed 100-NN per query (verified in notebook 01 via FAISS `IndexFlatL2`)

## Layout

```
.
├── 01_data_preparation.ipynb
├── 02_ivf_benchmark.ipynb
├── 03_hnsw_benchmark.ipynb
├── 04_lsh_benchmark.ipynb
├── 05_comparison.ipynb
├── utils.py                # shared helpers (readers, recall, timing, pareto)
├── data/                   # raw .fvecs / .ivecs / imagenet1m.h5 (.gitignored)
├── results/light/          # CSV/JSON from ./run_light.sh (.gitignored)
├── results/full/           # CSV/JSON from ./run_all.sh (.gitignored)
├── docs/img/light/         # plots from light runs (.gitignored)
└── docs/img/full/          # plots from full runs (.gitignored)
```

## Metrics

Every configuration logs: build time (s), index size (MB), RSS after build (MB),
RSS delta and peak RSS during build (MB), mean and p99 query latency (ms), QPS,
Recall@1, Recall@10, Recall@100.

Optional flame graph after benchmarks:

```bash
.venv/bin/pip install py-spy   # once
LAB_N_SWEEP=1000000 ./scripts/record_flame.sh
```

## Hardware target

Designed to fully exercise a ~30 GB-RAM machine. Scaling runs 100 K → 1 M by default;
set `LAB_SCALING_FULL=1` to measure the full 1.28 M base against the 28 GB target.
