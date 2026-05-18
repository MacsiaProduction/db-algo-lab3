## **Verdict: NOT report-ready**

The notebooks run, the CSVs exist, the structure is right — but the report does **not** meet the bar set in `TODO.md` ("графики это приоритет … если что-то не стыкуется в результатах нужно выявить причину или вывести первопричину с доказательствами").

## **Critical gaps vs the task**

1. **No flame graph anywhere.** `TODO.md` says explicitly *"should also include flame graph"*. There is no `py-spy`, `cProfile`, or `.svg` artifact in the repo.
2. **Full-base run was never done.** All sweeps run at `N_SWEEP=500 000`; the scaling experiment stops at 1 M. The conclusion's claim "scaling plots verify that the chosen best configurations stay within the 28 GB RAM target on the full 1.28 M base" is unverified — HNSW at 1 M already peaks at 20.4 GB.
3. `docs/img/` **folder doesn't exist on disk.** Plots only live as base64 inside the .ipynb files. The README points users to a folder that isn't there.

## **Real inconsistencies the report does not flag**


| **#** | **Anomaly**                                                                                                             | **Evidence**                                                                                                                                                                                                                            |
| ----- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `rss_mb` ≡ `rss_peak_mb` in every CSV row                                                                               | Generator does `rss_mb = rss_peak_mb` — there is no "RSS after build" column anywhere. The "memory sanity" plot in 05 is comparing peak vs peak.                                                                                        |
| 2     | Peak RSS is mostly `mmap` page-cache, not the index                                                                     | `IVFPQ M=32` index = 23 MB → peak RSS = **9 069 MB**. `LSH nbits=128` index = 8.6 MB → peak RSS = **6 395 MB**. The "Peak RSS" bar/scaling charts read as index memory but are tracking how much of `imagenet_base.fvecs` got paged in. |
| 3     | `IVFPQ` build time non-monotonic in M: 42.7 s (M=32) → **26.7 s (M=64)** → 38.8 s (M=128)                               | Caused by `QPS_REPEAT=1, QPS_WARMUP=0` in `run_all.sh`; no warmup, single timing pass. Not flagged in anomaly section.                                                                                                                  |
| 4     | HNSW recall non-monotonic in efC at low efS: efC=40 gives **R@100=0.749** at efS=10, while efC=200 gives only **0.637** | Real and explainable (looser graph keeps long-range shortcut edges), but the report says nothing about it.                                                                                                                              |
| 5     | `IVFFlat nlist=256, nprobe=1` QPS outlier (1 694) vs nlist=4096/nprobe=1 (24 110)                                       | First-config-of-family cache-cold artefact from `QPS_REPEAT=1`.                                                                                                                                                                         |
| 6     | `IVFPQ` sweep covers **only nlist=256**                                                                                 | The `_ivfpq_m_cell` reuses `best_nlist` from the IVFFlat sweep, `idxmax` ties to 256. So there is no PQ × nlist study, contradicting the task's "изучить настройки построения … в зависимости от различных настроек".                   |
| 7     | `best_configs.csv` LSH row picks `nbits=128` (R@100=0.22) instead of `nbits=4096` (R@100=0.42)                          | Threshold cascade falls to 0.2 floor, then picks highest QPS — which always picks the worst recall. Same logic gives IVFFlat `nlist=16384/nprobe=64` (R=0.957) when `nlist=4096/nprobe=64` (R=0.988) was also available.                |


## **Chart-quality issues**

- **Cross-algo Pareto (05 cell 4):** "HNSW HNSW HNSW HNSW" labels stack into an illegible column near recall ≈ 1.0.
- **Build/Size/RSS bars (05 cell 8):** each subplot uses a *different family order*, mixes linear/log y, and the "Peak RSS" header is misleading per the page-cache issue.
- **Memory sanity scatter (05 cell 8):** 4 points clustered, y=x reference goes to 12 000 MB into empty whitespace — uninformative because `rss == peak`.
- **Scaling Recall@100 panel:** y-axis 0.0–1.0 but only LSH/IVFPQ ever drop below 0.6, so HNSW/IVFFlat/IVFSQ collapse onto a single line near 1.0. Needs `ylim(0.4, 1.02)`.
- **Scaling RSS panel:** 28 GB target line floats above the data because x-axis stops at 1.0 M instead of 1.28 M.
- **HNSW Pareto:** efS labels overlap.
- **IVF Pareto:** non-Pareto points carry no marker/shape per nlist — chart unreadable without re-deriving.
- **IVFFlat build-time bar (02 cell 25):** 22 → 78 → 303 → 1 205 s on linear y; 16384 looks "just a bit slow" instead of 50× slower.
- **LSH recall vs nbits:** x-axis log-scale with only "10³" tick — six configs not labelled.
- **Notebook 01 GT histogram:** single tall bar at 1.0 with no spread — no information.
- **No latency p99 anywhere.** Only mean latency reported.

## **Cosmetic, but very visible**

The anomaly section in `05` prints literal `\n` characters because of double-escaping:

=== ANOMALY CHECKLIST ===\n

[A] IVFFlat nlist=  256 ...

\n[B] HNSW Recall@100 saturation per M:

...

\n>>> Winner overall: HNSW  (score=2.000)

Source is `print('... \\\\n')` inside `r"""..."""` cells in `_build_notebooks.py`.

## **Methodology issues that distort numbers**

- `QPS_REPEAT=1, QPS_WARMUP=0` (full run) → single timing pass, no variance, first-config cache-cold (anomalies 3 and 5 above).
- `LAB_QUERY_N=5000` of 25 000 → fine-grained recall above 0.99 is noisy.
- `TRAIN_N=200 000` for `nlist=16384` → ~12 points per centroid; FAISS itself prints "please provide at least 159 744 training points" in the scaling cell (visible in `05_comparison.ipynb` cell output, lines 521, 574, 627).
- No CPU/thread count or seed captured per row → re-runs aren't comparable.

## **Composite "winner score" is fragile**

HNSW    score=1.9998

IVFSQ   score=1.9381

IVFPQ   score=1.6265

IVFFlat score=1.1676

LSH     score=1.0000

HNSW edges IVFSQ by 0.06 — well inside measurement noise — and the three columns aren't computed from the same row (`qps_at_0p9` picks the highest-QPS config above recall 0.9, `size_min_mb` picks the family-minimum across *all* sweep rows). Either pick one row per family, or drop the composite and present three Pareto-optimal picks (recall, speed-at-0.95, size-at-0.5).

## **Minimum patch list to get this to a defendable state**

Ordered by impact / effort ratio (full ordered list in `~/context/db-algo-lab3-review/REVIEW.md` §8):

1. Fix `rss_mb` to be `tb.rss_after / 1024 / 1024` (currently aliased to peak). Add a `delta_rss` column.
2. Set `QPS_REPEAT=3, QPS_WARMUP=1` in `run_all.sh` and rerun sweeps. Kills two of the anomalies (§2.3, §2.5) and gives a variance estimate.
3. Add an IVFPQ `nlist × M` sweep ({1024, 4096} × {32, 64, 128}). One nlist value is not a study.
4. Run the scaling at `n=1 281 167` so the 28 GB claim is *measured*, not extrapolated.
5. Add a flame graph (`py-spy record` on the HNSW build at n=1 M, embed the SVG).
6. Fix the `\\n` print bug in six sites of `_build_notebooks.py`.
7. Extend the anomaly checklist to include: HNSW recall vs efC at fixed efS, IVFPQ build-time vs M monotonicity, first-config-of-family QPS spike, `rss_mb == rss_peak_mb` degeneracy.
8. Chart cleanup: limit cross-algo Pareto annotations to ≤ 5 hull points, fix the scaling Recall ylim, log y on IVFFlat build bar, explicit ticks on LSH `nbits` chart, drop the "memory sanity" scatter or rebuild it on the fixed `rss_mb`.
9. Add a latency panel (mean + p99 from `measure_qps`).
10. Either repair the composite score (same row for all three columns) or replace it with quadrant winners.

Full per-section breakdown lives at `~/context/db-algo-lab3-review/REVIEW.md`. Want me to start applying the patches? I'd suggest the order: (a) `\n` typo + `rss_mb` fix in `_build_notebooks.py`, (b) `run_all.sh` QPS knobs and PQ grid, (c) regenerate notebooks + rerun, (d) chart cleanup.