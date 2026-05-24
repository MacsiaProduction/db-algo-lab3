#!/usr/bin/env python3
"""IVF+PQ tuner — pushes Recall@100 toward 1.0.

Standalone script: does **not** touch the regular sweep CSVs and does **not**
need ``run_all.sh``. Run after ``01_data_preparation.ipynb`` has produced
``data/imagenet1m.h5`` and ``data/imagenet_base.fvecs``.

Usage:
    python3 scripts/tune_ivfpq.py                 # full tune at n=500K
    python3 scripts/tune_ivfpq.py --n 200000      # quick smoke
    python3 scripts/tune_ivfpq.py --variants base,opq,refine
    python3 scripts/tune_ivfpq.py --nlist 1024 --m 128 --nbits 8

Why pure IVF+PQ tops out below 0.80 on this dataset:
    At 2048-D with M=128, nbits=8 the codebook stores 16 bytes/vector
    (125× compression). The quantisation error blurs the boundary between
    near neighbours, so even with `nprobe=nlist` we cannot push Recall@100
    above ~0.79. The standard fixes — in order of cost — are:

      1. OPQ rotation       — optimises the basis before PQ; +5–8 R@100
                              points, same memory budget as plain PQ.
      2. IndexRefineFlat    — keeps the raw vectors and reranks the top
                              `k*k_factor` PQ candidates with exact L2;
                              recall reaches 0.99–1.00 but RAM ≈ baseline
                              + 4 × n × dim bytes.
      3. OPQ + Refine       — combined; the gold standard for high-recall
                              PQ-based search.

Each variant is benchmarked with the same QPS / recall harness as the main
sweep (utils.measure_qps + utils.compute_recalls), and one row per
(variant, nprobe) is written to ``results/<mode>/ivf_pq_tuned.csv``.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import utils  # noqa: E402


DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Data loading (same layout as 02_ivf_benchmark.ipynb)
# ---------------------------------------------------------------------------

def load_queries_and_base(n_sweep: int) -> tuple[np.ndarray, int, int, str]:
    with h5py.File(DATA / "imagenet1m.h5", "r") as h:
        queries = np.array(h["query"], dtype=np.float32)
        dim = int(h.attrs["dim"])
        n_total = int(h.attrs["n_base"])
        base_path = str(h.attrs["base_path"])
    local = DATA / "imagenet_base.fvecs"
    if not Path(base_path).exists() and local.exists():
        base_path = str(local.resolve())
    n_sweep = min(int(n_sweep), n_total)
    return queries, dim, n_sweep, base_path


def load_or_make_ground_truth(base_path: str, n_sweep: int,
                              queries: np.ndarray, k: int = 100) -> np.ndarray:
    cache = DATA / f"gt_n{n_sweep}_k{k}.npy"
    if cache.exists():
        return np.load(cache).astype(np.int32)
    print(f"  ! cached GT missing — recomputing via IndexFlatL2 (n={n_sweep:,}, k={k})")
    flat = faiss.IndexFlatL2(queries.shape[1])
    utils.stream_add(flat, base_path, n_sweep, progress=True)
    _, gt = flat.search(queries, k)
    np.save(cache, gt)
    del flat
    gc.collect()
    return gt.astype(np.int32)


# ---------------------------------------------------------------------------
# Index factories
# ---------------------------------------------------------------------------

def _ivfpq_cp(idx: faiss.IndexIVFPQ, nlist: int, n_sweep: int) -> None:
    # Same kmeans-CP params as 02_ivf_benchmark — keeps cross-CSV results
    # comparable to results/full/ivf_pq.csv.
    idx.cp.min_points_per_centroid = 5
    idx.cp.max_points_per_centroid = max(256, n_sweep // nlist)


def _make_ivfpq(dim: int, nlist: int, M: int, nbits: int, n_sweep: int):
    quant = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quant, dim, nlist, M, nbits)
    _ivfpq_cp(idx, nlist, n_sweep)
    return idx


def _make_opq_ivfpq(dim: int, nlist: int, M: int, nbits: int, n_sweep: int):
    opq = faiss.OPQMatrix(dim, M)
    base = _make_ivfpq(dim, nlist, M, nbits, n_sweep)
    return faiss.IndexPreTransform(opq, base)


def _make_ivfpq_refine(dim: int, nlist: int, M: int, nbits: int,
                       n_sweep: int, k_factor: float):
    base = _make_ivfpq(dim, nlist, M, nbits, n_sweep)
    ref = faiss.IndexRefineFlat(base)
    ref.k_factor = float(k_factor)
    return ref


def _make_opq_ivfpq_refine(dim: int, nlist: int, M: int, nbits: int,
                           n_sweep: int, k_factor: float):
    opq = faiss.OPQMatrix(dim, M)
    base = _make_ivfpq(dim, nlist, M, nbits, n_sweep)
    pre = faiss.IndexPreTransform(opq, base)
    ref = faiss.IndexRefineFlat(pre)
    ref.k_factor = float(k_factor)
    return ref


# ---------------------------------------------------------------------------
# nprobe drilling (IVFPQ may be wrapped in PreTransform / RefineFlat)
# ---------------------------------------------------------------------------

def set_nprobe(idx, nprobe: int) -> None:
    cur = idx
    seen: set[int] = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "nprobe"):
            try:
                cur.nprobe = int(nprobe)
            except Exception:
                pass
        nxt = None
        if hasattr(cur, "base_index") and cur.base_index is not None:
            nxt = cur.base_index
        elif hasattr(cur, "index") and cur.index is not None:
            nxt = cur.index
        if nxt is None:
            break
        try:
            cur = faiss.downcast_index(nxt)
        except Exception:
            cur = nxt


# ---------------------------------------------------------------------------
# Per-variant build + benchmark
# ---------------------------------------------------------------------------

def build_and_bench(
    *,
    label: str,
    make_index: Callable[[], faiss.Index],
    train_x: np.ndarray,
    base_path: str,
    n_sweep: int,
    queries: np.ndarray,
    gt: np.ndarray,
    nprobe_grid: List[int],
    k: int,
    qps_repeat: int,
    qps_warmup: int,
    extra_meta: Dict[str, object],
) -> List[dict]:
    rows: List[dict] = []
    print(f"\n=== {label} ===")
    print(f"  train_x={train_x.shape}  n_sweep={n_sweep:,}")
    with utils.timed(label, sample_rss_peak=True) as tb:
        idx = make_index()
        idx.train(train_x)
        utils.stream_add(idx, base_path, n_sweep)
    size_mb = utils.index_size_mb(idx)
    print(f"  build {tb.elapsed:.1f}s  size {size_mb:.1f} MB  "
          f"rss_after {tb.rss_after_mb/1024:.2f} GB  peak {tb.rss_peak_mb/1024:.2f} GB")
    meta = utils.bench_meta()
    for nprobe in nprobe_grid:
        set_nprobe(idx, nprobe)
        qps, lat_ms, p99_ms, I = utils.measure_qps(
            lambda q, kk: idx.search(q, kk),
            queries, k,
            repeat=qps_repeat, warmup=qps_warmup,
        )
        recalls = utils.compute_recalls(I, gt, (1, 10, 100))
        row = dict(
            variant=label,
            nprobe=int(nprobe),
            build_s=tb.elapsed,
            size_mb=size_mb,
            rss_mb=tb.rss_after_mb,
            rss_peak_mb=tb.rss_peak_mb,
            rss_delta_mb=tb.rss_delta_mb,
            **meta,
            qps=qps,
            latency_ms=lat_ms,
            latency_p99_ms=p99_ms,
            recall_1=recalls[1],
            recall_10=recalls[10],
            recall_100=recalls[100],
            n_base=n_sweep,
            **extra_meta,
        )
        rows.append(row)
        print(f"  nprobe={nprobe:5d}  R@100={recalls[100]:.4f}  "
              f"R@10={recalls[10]:.4f}  QPS={qps:8.1f}  "
              f"mean {lat_ms*1000:.0f}µs  p99 {p99_ms*1000:.0f}µs")
    del idx
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Variant assembly
# ---------------------------------------------------------------------------

def assemble_variants(args: argparse.Namespace,
                      dim: int, n_sweep: int) -> List[dict]:
    """List of variant specs (dict with `label`, `make`, `extra`)."""
    selected = {v.strip() for v in args.variants.split(",") if v.strip()}
    nlist, M, nbits = args.nlist, args.m, args.nbits
    kfs = [float(x) for x in args.k_factors.split(",") if x.strip()]
    common = dict(nlist=nlist, M=M, nbits=nbits)
    variants: List[dict] = []

    if "base" in selected or "all" in selected:
        variants.append(dict(
            label=f"IVFPQ_nlist{nlist}_M{M}_nb{nbits}",
            make=lambda: _make_ivfpq(dim, nlist, M, nbits, n_sweep),
            extra=dict(**common, k_factor=0.0, opq=False, refine=False),
        ))
    if "opq" in selected or "all" in selected:
        variants.append(dict(
            label=f"OPQ+IVFPQ_nlist{nlist}_M{M}_nb{nbits}",
            make=lambda: _make_opq_ivfpq(dim, nlist, M, nbits, n_sweep),
            extra=dict(**common, k_factor=0.0, opq=True, refine=False),
        ))
    if "refine" in selected or "all" in selected:
        for kf in kfs:
            variants.append(dict(
                label=f"IVFPQ+Refine_kf{int(kf)}_nlist{nlist}_M{M}_nb{nbits}",
                make=(lambda _kf=kf: _make_ivfpq_refine(dim, nlist, M, nbits, n_sweep, _kf)),
                extra=dict(**common, k_factor=kf, opq=False, refine=True),
            ))
    if "opq_refine" in selected or "all" in selected:
        for kf in kfs:
            variants.append(dict(
                label=f"OPQ+IVFPQ+Refine_kf{int(kf)}_nlist{nlist}_M{M}_nb{nbits}",
                make=(lambda _kf=kf: _make_opq_ivfpq_refine(dim, nlist, M, nbits, n_sweep, _kf)),
                extra=dict(**common, k_factor=kf, opq=True, refine=True),
            ))
    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--n", type=int, default=int(os.environ.get("LAB_N_SWEEP", "500000")),
                    help="n_base for the sweep (default 500_000; same as the main IVF sweep)")
    ap.add_argument("--query-n", type=int, default=10_000,
                    help="number of query vectors used for QPS / recall (default 10_000)")
    ap.add_argument("--nlist", type=int, default=1024,
                    help="IVF coarse partitions (default 1024 — best from existing sweep)")
    ap.add_argument("--m", type=int, default=128,
                    help="PQ sub-quantisers (default 128; divides 2048 evenly)")
    ap.add_argument("--nbits", type=int, default=8,
                    help="bits per PQ code (default 8 → 256 codes/sub-quantiser)")
    ap.add_argument("--nprobe", default="16,64,256,1024",
                    help="comma-separated nprobe grid")
    ap.add_argument("--k-factors", default="5,10,20",
                    help="comma-separated k_factor values for Refine variants")
    ap.add_argument("--variants", default="all",
                    help="comma-separated subset of {base,opq,refine,opq_refine,all}")
    ap.add_argument("--qps-repeat", type=int, default=3)
    ap.add_argument("--qps-warmup", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="output CSV path (default results/<mode>/ivf_pq_tuned.csv)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else utils.results_dir() / "ivf_pq_tuned.csv"
    nprobe_grid = [int(x) for x in args.nprobe.split(",") if x.strip()]

    print(f"=== IVF+PQ tuner ===")
    print(f"  n_sweep={args.n:,}  nlist={args.nlist}  M={args.m}  nbits={args.nbits}")
    print(f"  nprobe_grid={nprobe_grid}  k_factors={args.k_factors}  variants={args.variants}")
    print(f"  out={out_path}")

    queries, dim, n_sweep, base_path = load_queries_and_base(args.n)
    print(f"  dim={dim}  base_path={base_path}")

    train_n = max(200_000, 30 * args.nlist)
    train_x = utils.load_train_subset(base_path, train_n)
    print(f"  train_x {train_x.shape}  "
          f"RSS={psutil.Process().memory_info().rss/1e9:.2f} GB free")

    queries_sweep = queries[: args.query_n]
    gt = load_or_make_ground_truth(base_path, n_sweep, queries_sweep, k=100)
    print(f"  gt {gt.shape}")

    variants = assemble_variants(args, dim, n_sweep)
    if not variants:
        print("no variants selected; pass --variants base,opq,refine,opq_refine,all")
        return 1

    rows: List[dict] = []
    for spec in variants:
        try:
            rows.extend(build_and_bench(
                label=spec["label"],
                make_index=spec["make"],
                train_x=train_x,
                base_path=base_path,
                n_sweep=n_sweep,
                queries=queries_sweep,
                gt=gt,
                nprobe_grid=nprobe_grid,
                k=100,
                qps_repeat=args.qps_repeat,
                qps_warmup=args.qps_warmup,
                extra_meta=spec["extra"],
            ))
        except MemoryError as e:
            print(f"  ! MemoryError in {spec['label']}: {e} — skipping")
        except RuntimeError as e:
            # FAISS Refine variants need raw vectors; if mmap can't materialise
            # at this n we still want to keep tuner progress.
            print(f"  ! RuntimeError in {spec['label']}: {e} — skipping")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    df.to_json(out_path.with_suffix(".json"), orient="records", indent=2)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")

    if not df.empty:
        print("\nBest Recall@100 per variant:")
        for variant, sub in df.groupby("variant"):
            b = sub.sort_values("recall_100", ascending=False).iloc[0]
            print(f"  {variant:50s}  R@100={b.recall_100:.4f}  "
                  f"QPS={b.qps:8.1f}  nprobe={int(b.nprobe):5d}  "
                  f"size={b.size_mb:6.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
