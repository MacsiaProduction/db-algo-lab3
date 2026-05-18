# Lab task summary

Consolidated from the conversation: user prompts, remarks, requirements, and implementation status.

## Goal

Implement and run the **FAISS ANN benchmark lab** on ImageNet-1M (ZJU) per `TODO.md`: notebooks `01`–`05`, metrics, plots, and reproducible scripts for **light** (smoke) and **full** (reporting) runs.

## User prompts (chronological)

1. Implement the lab per `TODO.md` with graphs and metrics.
2. Run **light** mode first; remove Docker; run **natively** with `.venv`.
3. **Reduce RAM** — full run OOM’d when loading the entire base array; use streaming adds instead.
4. **Review results** — fix inconsistencies (missing LSH in summaries, bad RSS, missing plots, chart gaps); add **timeouts**.
5. **Separate outputs** — light vs full into different folders (`results/{light,full}/`, `docs/img/{light,full}/`).
6. **Fix `run_all.sh`** — full run failed on `02_ivf_benchmark.ipynb` with `CellTimeoutError` after 7200s on the monolithic IVFFlat sweep cell.

## Requirements and remarks

### Environment and execution

- No Docker; use project `.venv` and `./run_light.sh` / `./run_all.sh`.
- `run_light.sh`: `LAB_LIGHT=1`, shorter grids, legacy cleanup, bounded notebook/cell timeouts.
- `run_all.sh`: `LAB_LIGHT=0`, full parameter grids, `LAB_N_SWEEP` default 500000 (configurable up to full 1.28M base).

### Memory

- Do **not** hold the full base `xb` in RAM alongside indexes.
- Use `utils.stream_add()` and a small `train_x` slice for IVF training only.
- Track peak RSS during builds (`RssPeakMonitor`, `rss_peak_mb`).

### Outputs and layout

- **Light**: `results/light/`, `docs/img/light/`
- **Full**: `results/full/`, `docs/img/full/`
- Remove stale flat `results/*.csv` and `docs/img/*.png` from before the split (via `utils.cleanup_legacy_outputs()`).
- Notebook `05` must include **all** index families (including LSH) in comparison and `best_configs.csv`.

### Metrics and quality

- Per config: build time, index size, QPS, latency, R@1 / R@10 / R@100, peak RSS.
- Ground truth: recompute exact Flat GT for `N_SWEEP` subset when needed; cache under `data/gt_n{n}_k{k}.npy`.
- Plots saved under `docs/img/{light|full}/`; CSVs under `results/{light|full}/`.

### Timeouts and full-run reliability

- Per-notebook wall timeout and per-cell execute timeout (nbconvert).
- Full-run IVFFlat sweep must **not** put the entire `nlist × nprobe` grid in a single cell (exceeded 7200s at `N_SWEEP=500000`).
- Sweeps should **checkpoint** to CSV so partial progress survives cell/notebook failures.

### Tunables (env)

| Variable | Role |
|----------|------|
| `LAB_N_SWEEP` | Base vectors for sweeps |
| `LAB_LIGHT` | `1` = light grids; `0` = full |
| `LAB_BATCH_ADD` | `stream_add` batch size |
| `LAB_QPS_REPEAT` / `LAB_QPS_WARMUP` | QPS measurement passes |
| `LAB_QUERY_N` | Queries used in sweep timing (full default 5000) |
| `LAB_NOTEBOOK_TIMEOUT` | Wall clock per notebook |
| `LAB_CELL_TIMEOUT` | Per-cell timeout (full default 14400s) |

## Failure observed

```
./run_all.sh
  01_data_preparation.ipynb  — OK
  02_ivf_benchmark.ipynb     — CellTimeoutError after 7200s
```

Timed-out cell: single loop over `NLIST_GRID` building IVFFlat and searching all `NPROBE_GRID` with full query set at `N_SWEEP=500000`.

## Task / fix scope

1. Split monolithic sweep cells into **one notebook cell per major config** (IVFFlat `nlist`, IVF+PQ `M`, IVF+SQ variant, HNSW `M` / `efConstruction`, LSH `nbits`).
2. **Append results** after each cell via `utils.init_results_csv()` / `utils.append_results()`.
3. Use **`queries_sweep`** (`LAB_QUERY_N`, default 5000 on full) for faster QPS/recall during sweeps; slice GT as `gt_local[:QUERY_N]`.
4. Regenerate notebooks from `_build_notebooks.py`.
5. Update `run_all.sh`: export `LAB_QUERY_N`, raise default `LAB_CELL_TIMEOUT` to 14400.
6. Re-run `./run_all.sh` (or resume from `02` onward) and confirm outputs only under `results/full/` and `docs/img/full/`.

## Implementation status

| Item | Status |
|------|--------|
| Light run (`./run_light.sh`) | Completed successfully |
| `utils.py` streaming, RSS, paths, CSV append | Done |
| Light/full output dirs, `.gitignore` | Done |
| Notebook `01` fixes (paths, `DOCS_IMG`, GT) | Done |
| Notebook `05` LSH / scaling / charts | Done |
| Split sweep cells + checkpoint CSVs | Done in `_build_notebooks.py`; notebooks regenerated |
| Full run `./run_all.sh` end-to-end | **Pending user re-run** after timeout fix |

## Key files

- `TODO.md` — original lab spec
- `_build_notebooks.py` — generates `01`–`05` `.ipynb`
- `utils.py` — shared helpers
- `run_light.sh`, `run_all.sh` — execution entry points
- `results/light/` — valid light-run artifacts
- `results/full/` — populated after successful full run

## Do not edit

- `benchmark_result_cleanup_409bf66d.plan.md` (if present) — planning artifact only; not part of deliverables.
