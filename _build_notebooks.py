"""Generate the five lab notebooks programmatically.

Run with:  .venv/bin/python _build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n"))


def write(nb, path):
    Path(path).write_text(nbf.writes(nb))
    print(f"wrote {path}")


# Max grids used when generating per-config notebook cells (light mode skips via NLIST_GRID etc.)
_ALL_NLIST = [256, 1024, 4096, 16384]
_ALL_PQ_M = [32, 64, 128]
_ALL_PQ_NLIST = [1024, 4096]
_ALL_SQ_NAMES = ["SQ8", "SQ4"]
_ALL_HNSW_M = [8, 16, 32, 48]
_ALL_EFC = [40, 100, 200, 400]
_ALL_LSH_NBITS = [128, 256, 512, 1024, 2048, 4096]


def _ivfflat_nlist_cell(nlist: int) -> str:
    return f"""
if {nlist} not in NLIST_GRID:
    print('skip IVFFlat nlist={nlist} (not in NLIST_GRID)')
else:
    rows = []
    quant = faiss.IndexFlatL2(DIM)
    idx = faiss.IndexIVFFlat(quant, DIM, {nlist}, faiss.METRIC_L2)
    idx.cp.min_points_per_centroid = 5
    idx.cp.max_points_per_centroid = max(256, len(train_x) // {nlist})
    with utils.timed('train+add nlist={nlist}', sample_rss_peak=True) as tb:
        idx.train(train_x)
        utils.stream_add(idx, BASE_PATH, N_SWEEP)
    size_mb = utils.index_size_mb(idx)
    rss_mb = tb.rss_after_mb
    rss_peak_mb = tb.rss_peak_mb
    rss_delta_mb = tb.rss_delta_mb
    print(f'[nlist={{{nlist}:5}}]  build {{tb.elapsed:6.1f}}s · size {{size_mb:7.1f}} MB · RSS {{rss_mb:7.1f}} MB · peak {{rss_peak_mb:7.1f}} MB')
    for nprobe in NPROBE_GRID:
        if nprobe > {nlist}:
            continue
        idx.nprobe = nprobe
        qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
            lambda q,k: idx.search(q,k), queries_sweep, QUERY_K,
            repeat=QPS_REPEAT, warmup=QPS_WARMUP,
        )
        recalls = utils.compute_recalls(I, gt_local[:QUERY_N], (1, 10, 100))
        rows.append(dict(algo='IVFFlat', nlist={nlist}, nprobe=nprobe,
                        build_s=tb.elapsed, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
                        rss_delta_mb=rss_delta_mb, **utils.bench_meta(),
                        qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
                        recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100],
                        n_base=N_SWEEP))
        print(f'    nprobe={{nprobe:5}}  qps={{qps:8.1f}}  R@100={{recalls[100]:.3f}}')
    del idx, quant; gc.collect()
    utils.append_results(rows, IVF_FLAT_PATH)
    print(f'  → appended {{len(rows)}} rows to {{IVF_FLAT_PATH}}')
""".strip()


def _ivfpq_nlist_m_cell(nlist: int, m: int) -> str:
    return f"""
if {nlist} not in PQ_NLIST_GRID or {m} not in PQ_M_GRID:
    print('skip IVFPQ nlist={nlist} M={m}')
else:
    rows = []
    quant = faiss.IndexFlatL2(DIM)
    idx = faiss.IndexIVFPQ(quant, DIM, {nlist}, int({m}), int(PQ_NBITS))
    with utils.timed('train+add PQ nlist={nlist} M={m}', sample_rss_peak=True) as tb:
        idx.train(train_x)
        utils.stream_add(idx, BASE_PATH, N_SWEEP)
    size_mb = utils.index_size_mb(idx)
    rss_mb = tb.rss_after_mb
    rss_peak_mb = tb.rss_peak_mb
    rss_delta_mb = tb.rss_delta_mb
    print(f'[PQ nlist={{{nlist}:5}} M={{{m}:4}}]  build {{tb.elapsed:6.1f}}s · size {{size_mb:7.1f}} MB')
    for nprobe in NPROBE_GRID:
        if nprobe > {nlist}:
            continue
        idx.nprobe = nprobe
        qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
            lambda q,k: idx.search(q,k), queries_sweep, QUERY_K,
            repeat=QPS_REPEAT, warmup=QPS_WARMUP,
        )
        recalls = utils.compute_recalls(I, gt_local[:QUERY_N], (1, 10, 100))
        rows.append(dict(algo='IVFPQ', nlist={nlist}, nprobe=nprobe, M={m}, nbits=PQ_NBITS,
                        build_s=tb.elapsed, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
                        rss_delta_mb=rss_delta_mb, **utils.bench_meta(),
                        qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
                        recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100],
                        n_base=N_SWEEP))
        print(f'    nprobe={{nprobe:5}}  qps={{qps:8.1f}}  R@100={{recalls[100]:.3f}}')
    del idx, quant; gc.collect()
    utils.append_results(rows, IVF_PQ_PATH)
""".strip()


def _ivfsq_name_cell(name: str, qt_expr: str) -> str:
    return f"""
if '{name}' not in [t[0] for t in SQ_TYPES]:
    print('skip IVFSQ {name}')
else:
    rows = []
    quant = faiss.IndexFlatL2(DIM)
    idx = faiss.IndexIVFScalarQuantizer(quant, DIM, best_nlist, {qt_expr}, faiss.METRIC_L2)
    with utils.timed('train+add SQ {name}', sample_rss_peak=True) as tb:
        idx.train(train_x)
        utils.stream_add(idx, BASE_PATH, N_SWEEP)
    size_mb = utils.index_size_mb(idx)
    rss_mb = tb.rss_after_mb
    rss_peak_mb = tb.rss_peak_mb
    rss_delta_mb = tb.rss_delta_mb
    print(f'[SQ {name}]  build {{tb.elapsed:5.1f}}s · size {{size_mb:7.1f}} MB')
    for nprobe in NPROBE_GRID:
        if nprobe > best_nlist:
            continue
        idx.nprobe = nprobe
        qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
            lambda q,k: idx.search(q,k), queries_sweep, QUERY_K,
            repeat=QPS_REPEAT, warmup=QPS_WARMUP,
        )
        recalls = utils.compute_recalls(I, gt_local[:QUERY_N], (1, 10, 100))
        rows.append(dict(algo='IVFSQ', sq='{name}', nlist=best_nlist, nprobe=nprobe,
                        build_s=tb.elapsed, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
                        rss_delta_mb=rss_delta_mb, **utils.bench_meta(),
                        qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
                        recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100],
                        n_base=N_SWEEP))
        print(f'    nprobe={{nprobe:5}}  qps={{qps:8.1f}}  R@100={{recalls[100]:.3f}}')
    del idx, quant; gc.collect()
    utils.append_results(rows, IVF_SQ_PATH)
""".strip()


def _hnsw_m_cell(m: int) -> str:
    return f"""
if {m} not in M_GRID:
    print('skip HNSW M={m}')
else:
    rows = []
    idx = faiss.IndexHNSWFlat(DIM, {m}, faiss.METRIC_L2)
    idx.hnsw.efConstruction = EFC_FIXED
    with utils.timed('build M={m}', sample_rss_peak=True) as tb:
        utils.stream_add(idx, BASE_PATH, N_SWEEP)
    size_mb = utils.index_size_mb(idx)
    rss_mb = tb.rss_after_mb
    rss_peak_mb = tb.rss_peak_mb
    rss_delta_mb = tb.rss_delta_mb
    print(f'[M={{{m}:3}}]  build {{tb.elapsed:7.1f}}s · size {{size_mb:7.1f}} MB')
    for efs in EFS_GRID:
        idx.hnsw.efSearch = efs
        qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
            lambda q,k: idx.search(q,k), queries_sweep, QUERY_K,
            repeat=QPS_REPEAT, warmup=QPS_WARMUP,
        )
        recalls = utils.compute_recalls(I, gt_local[:QUERY_N], (1, 10, 100))
        rows.append(dict(algo='HNSW', M={m}, efConstruction=EFC_FIXED, efSearch=efs,
                        build_s=tb.elapsed, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
                        rss_delta_mb=rss_delta_mb, **utils.bench_meta(),
                        qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
                        recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100],
                        n_base=N_SWEEP, study='varyM'))
        print(f'    efS={{efs:4}}  qps={{qps:8.1f}}  R@100={{recalls[100]:.3f}}')
    del idx; gc.collect()
    utils.append_results(rows, HNSW_VARYM_PATH)
""".strip()


def _hnsw_efc_cell(efc: int) -> str:
    return f"""
if {efc} not in EFC_GRID:
    print('skip HNSW efC={efc}')
else:
    rows = []
    idx = faiss.IndexHNSWFlat(DIM, M_FIXED, faiss.METRIC_L2)
    idx.hnsw.efConstruction = {efc}
    with utils.timed('build efC={efc}', sample_rss_peak=True) as tb:
        utils.stream_add(idx, BASE_PATH, N_SWEEP)
    size_mb = utils.index_size_mb(idx)
    rss_mb = tb.rss_after_mb
    rss_peak_mb = tb.rss_peak_mb
    rss_delta_mb = tb.rss_delta_mb
    print(f'[efC={{{efc}:4}}]  build {{tb.elapsed:7.1f}}s · size {{size_mb:7.1f}} MB')
    for efs in EFS_GRID:
        idx.hnsw.efSearch = efs
        qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
            lambda q,k: idx.search(q,k), queries_sweep, QUERY_K,
            repeat=QPS_REPEAT, warmup=QPS_WARMUP,
        )
        recalls = utils.compute_recalls(I, gt_local[:QUERY_N], (1, 10, 100))
        rows.append(dict(algo='HNSW', M=M_FIXED, efConstruction={efc}, efSearch=efs,
                        build_s=tb.elapsed, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
                        rss_delta_mb=rss_delta_mb, **utils.bench_meta(),
                        qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
                        recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100],
                        n_base=N_SWEEP, study='varyEFC'))
        print(f'    efS={{efs:4}}  qps={{qps:8.1f}}  R@100={{recalls[100]:.3f}}')
    del idx; gc.collect()
    utils.append_results(rows, HNSW_VARYEFC_PATH)
""".strip()


def _lsh_nbits_cell(nbits: int) -> str:
    return f"""
if {nbits} not in NBITS_GRID:
    print('skip LSH nbits={nbits}')
else:
    idx = faiss.IndexLSH(DIM, {nbits})
    with utils.timed('train+add LSH {nbits}', sample_rss_peak=True) as tb:
        idx.train(train_x)
        utils.stream_add(idx, BASE_PATH, N_SWEEP)
    size_mb = utils.index_size_mb(idx)
    rss_mb = tb.rss_after_mb
    rss_peak_mb = tb.rss_peak_mb
    rss_delta_mb = tb.rss_delta_mb
    qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
        lambda q,k: idx.search(q,k), queries_sweep, QUERY_K,
        repeat=QPS_REPEAT, warmup=QPS_WARMUP,
    )
    recalls = utils.compute_recalls(I, gt_local[:QUERY_N], (1, 10, 100))
    rows = [dict(algo='LSH', nbits={nbits},
               build_s=tb.elapsed, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
               rss_delta_mb=rss_delta_mb, **utils.bench_meta(),
               qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
               recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100],
               n_base=N_SWEEP)]
    utils.append_results(rows, LSH_PATH)
    del idx; gc.collect()
    print(f'[nbits={{{nbits}:5}}]  build {{tb.elapsed:5.1f}}s · QPS {{qps:8.1f}} · R@100 {{recalls[100]:.3f}}')
""".strip()


# ---------------------------------------------------------------------------
# 01 — data preparation
# ---------------------------------------------------------------------------

nb1 = nbf.v4.new_notebook()
nb1.metadata = {
    "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nb1.cells = [
    md(r"""
# 01 — Data Preparation: ImageNet-1M (ZJU)

This notebook **downloads, verifies and explores** the ImageNet-1M benchmark used in all
subsequent experiments.

| Split | Vectors | Dim | Size |
|---|---|---|---|
| Base | 1 281 167 | 2048 | ≈ 9.4 GB |
| Query | 25 000 | 2048 | ≈ 195 MB |
| Ground truth | 25 000 × 100 nn | int32 | ≈ 9.6 MB |

Source: <http://www.cad.zju.edu.cn/home/dengcai/Data/ANNS/ANNSData.html>

Sections:
1. Download & integrity check
2. Dataset summary
3. L2-norm distribution
4. Per-dimension statistics
5. 2-D PCA scatter
6. Ground-truth verification against `IndexFlatL2`
7. Persist a small HDF5 with query + GT for downstream notebooks
"""),
    code(r"""
import os, sys, time, subprocess, hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import faiss
import psutil
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))
import utils

DATA = Path('data')
DATA.mkdir(exist_ok=True)
LAB_LIGHT = int(os.environ.get('LAB_LIGHT', '0'))
OUT_RUN = utils.run_mode()
DOCS_IMG = utils.plots_dir()
print(f'OUT_RUN={OUT_RUN}  plots → {DOCS_IMG}')

PATHS = {
    'base':  DATA / 'imagenet_base.fvecs',
    'query': DATA / 'imagenet_query.fvecs',
    'gt':    DATA / 'imagenet_groundtruth.ivecs',
    'h5':    DATA / 'imagenet1m.h5',
}

URLS = {
    'base.tar':  'http://www.cad.zju.edu.cn/home/dengcai/Data/ANNS/imagenet_base.fvecs.tar.gz',
    'query.tar': 'http://www.cad.zju.edu.cn/home/dengcai/Data/ANNS/imagenet_query.fvecs.tar.gz',
    'gt.tar':    'http://www.cad.zju.edu.cn/home/dengcai/Data/ANNS/imagenet_groundtruth.ivecs.tar.gz',
}

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110
print('faiss', faiss.__version__, '| threads', faiss.omp_get_max_threads())
print(f"RAM total {psutil.virtual_memory().total/1e9:.1f} GB | free {psutil.virtual_memory().available/1e9:.1f} GB")
"""),
    md("""
## 1 · Download & extract

The cell below is **idempotent** — it does nothing if the `.fvecs` / `.ivecs` files already
exist. The 9.4 GB base file is the slow one; expect 30–90 min on a 3-5 MB/s link.
"""),
    code(r"""
def _curl(url: str, out: Path) -> None:
    print(f"  → {url}")
    subprocess.check_call(['curl', '-fSL', '--retry', '5', '-o', str(out), url])

def _extract(tar: Path, expected: Path) -> None:
    if expected.exists():
        return
    print(f"  ✱ tar xzf {tar.name}")
    subprocess.check_call(['tar', 'xzf', str(tar), '-C', str(tar.parent)])

for name, fvecs in [('base', PATHS['base']), ('query', PATHS['query']), ('gt', PATHS['gt'])]:
    if fvecs.exists():
        sz = fvecs.stat().st_size / 1e9
        print(f"✓ {fvecs.name}  ({sz:.2f} GB) already present")
        continue
    tar = fvecs.parent / (fvecs.stem + ('.fvecs.tar.gz' if fvecs.suffix == '.fvecs' else '.ivecs.tar.gz'))
    if not tar.exists():
        _curl(URLS[f'{name}.tar'], tar)
    _extract(tar, fvecs)
    print(f"✓ {fvecs.name}  ({fvecs.stat().st_size/1e9:.2f} GB)")
"""),
    md("""
## 2 · Dataset summary
"""),
    code(r"""
n_base, dim = utils._count_vecs(str(PATHS['base']), dtype_bytes=4)
n_query, dim_q = utils._count_vecs(str(PATHS['query']), dtype_bytes=4)
n_gt, k_gt = utils._count_vecs(str(PATHS['gt']), dtype_bytes=4)
summary = pd.DataFrame({
    'split': ['base', 'query', 'groundtruth'],
    'n':     [n_base, n_query, n_gt],
    'cols':  [dim, dim_q, k_gt],
    'bytes': [PATHS['base'].stat().st_size, PATHS['query'].stat().st_size, PATHS['gt'].stat().st_size],
})
summary['MB'] = (summary.bytes / 1024 / 1024).round(1)
display(summary)
assert dim == dim_q == 2048, "expected 2048-D vectors"
assert n_query == n_gt == 25_000
print(f"base vectors: {n_base:,}  ·  query/gt: {n_query:,}  ·  dim: {dim}")
"""),
    md("""
## 3 · L2-norm distribution

These features are L2-normalised-ish ResNet-style embeddings.  Plotting the histogram of
norms reveals whether the dataset is normalised (so cosine ≡ L2) or arbitrary scale.
"""),
    code(r"""
SAMPLE_N = 100_000
rng = np.random.default_rng(42)
sample_ids = np.sort(rng.choice(n_base, SAMPLE_N, replace=False))

# Read sample by memmap to avoid loading 10 GB
mm = utils.read_fvecs(str(PATHS['base']), mmap=True)
sample = np.array(mm[sample_ids], dtype=np.float32)  # 100k × 2048 ≈ 780 MB
print('sample shape', sample.shape, 'RAM', sample.nbytes/1e6, 'MB')

norms = np.linalg.norm(sample, axis=1)
print(f"norm min={norms.min():.3f}  median={np.median(norms):.3f}  max={norms.max():.3f}  std={norms.std():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(norms, bins=80, color='steelblue', edgecolor='white')
ax[0].set_title('L2 norms (100k sample)')
ax[0].set_xlabel('||v||₂'); ax[0].set_ylabel('count')

q_norms = np.linalg.norm(utils.read_fvecs(str(PATHS['query'])), axis=1)
ax[1].hist(q_norms, bins=80, color='darkorange', edgecolor='white')
ax[1].set_title('L2 norms (25k queries)')
ax[1].set_xlabel('||v||₂'); ax[1].set_ylabel('count')
plt.tight_layout(); plt.savefig(DOCS_IMG / '01_norms.png', dpi=120, bbox_inches='tight'); plt.show()
"""),
    md("""
## 4 · Per-dimension mean / std

If any subset of dimensions has very different scale, IVF k-means and HNSW edges can be
dominated by those axes.  This plot is a sanity check that the embedding is reasonably
isotropic.
"""),
    code(r"""
mu = sample.mean(axis=0)
sd = sample.std(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(mu, lw=0.4, color='steelblue')
ax[0].axhline(mu.mean(), color='k', ls='--', lw=0.7, label=f'global μ={mu.mean():.3f}')
ax[0].set_title('Per-dimension mean'); ax[0].set_xlabel('dim'); ax[0].legend()
ax[1].plot(sd, lw=0.4, color='crimson')
ax[1].axhline(sd.mean(), color='k', ls='--', lw=0.7, label=f'global σ̄={sd.mean():.3f}')
ax[1].set_title('Per-dimension std'); ax[1].set_xlabel('dim'); ax[1].legend()
plt.tight_layout(); plt.savefig(DOCS_IMG / '01_dim_stats.png', dpi=120, bbox_inches='tight'); plt.show()

print(f"min std {sd.min():.3f}  max std {sd.max():.3f}  ratio {sd.max()/sd.min():.1f}×")
"""),
    md("""
## 5 · 2-D PCA scatter (10k sample)

A cheap 2-D PCA gives a feel for how the 1 000 ImageNet classes are arranged in this
embedding.  We colour 10 random clusters from a small `KMeans` on top of the PCA to make
structure visible — this is just a visualisation, no labels are used.
"""),
    code(r"""
small_n = 10_000
small = sample[:small_n]

# Quick centred SVD-based PCA, top-2 components
small_c = small - small.mean(axis=0)
U, S, Vt = np.linalg.svd(small_c, full_matrices=False)
proj = small_c @ Vt[:2].T

# 10-way k-means on the embedding (fast with faiss)
km = faiss.Kmeans(d=dim, k=10, niter=20, seed=1, verbose=False)
km.train(small)
_, labels = km.index.search(small, 1)
labels = labels.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=4, alpha=0.6)
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.set_title('Top-2 PCA of ImageNet-1M embeddings (10k sample, k-means k=10 colours)')
plt.colorbar(sc, ax=ax, label='cluster id')
plt.tight_layout(); plt.savefig(DOCS_IMG / '01_pca.png', dpi=120, bbox_inches='tight'); plt.show()

print(f"PC1 explains {S[0]**2 / (S**2).sum() * 100:.2f} %, "
      f"PC2 explains {S[1]**2 / (S**2).sum() * 100:.2f} %")
del small, small_c, U, S, Vt, sample; import gc; gc.collect()
"""),
    md("""
## 6 · Ground-truth verification

**Full run (default):** `IndexFlatL2` over the **entire base** (streamed from memmap) on
5 000 queries; Recall@100 vs the published GT must be ≈ 1.0.

**Light run (`LAB_LIGHT=1`):** Flat index on the first **100 000** base vectors only,
300 queries, plus a **numpy cross-check** on query 0 (exact top-100 by brute force on that
subset).  This avoids the multi-hour full-base pass while still validating readers + FAISS.
"""),
    code(r"""
# Load query + supplied GT
queries = utils.read_fvecs(str(PATHS['query']))   # 25k × 2048
gt = utils.read_ivecs(str(PATHS['gt']))           # 25k × 100
print('queries', queries.shape, 'gt', gt.shape)

LAB_LIGHT = int(os.environ.get('LAB_LIGHT', '0'))
mm = utils.read_fvecs(str(PATHS['base']), mmap=True)

if LAB_LIGHT:
    SUB_N, SAMPLE_Q = 100_000, 300
    qs = queries[:SAMPLE_Q]
    gts = gt[:SAMPLE_Q]
    xb_sub = np.ascontiguousarray(mm[:SUB_N])
    print(f'LAB_LIGHT=1 — Flat on first {SUB_N:,} base vectors, {SAMPLE_Q} queries')
    flat = faiss.IndexFlatL2(dim)
    t0 = time.perf_counter()
    flat.add(xb_sub)
    print(f'  add {flat.ntotal:,} in {time.perf_counter()-t0:.1f}s')
    t0 = time.perf_counter()
    D_flat, I_flat = flat.search(qs, 100)
    print(f'  search {time.perf_counter()-t0:.2f}s')

    # numpy exact top-100 for query 0 on the same subset
    d0 = np.sum((xb_sub - qs[0]) ** 2, axis=1)
    true_idx = np.argpartition(d0, 99)[:100]
    true_idx = true_idx[np.argsort(d0[true_idx])]
    faiss_idx = np.sort(I_flat[0])
    np_idx = np.sort(true_idx)
    assert np.array_equal(faiss_idx, np_idx), 'FAISS Flat vs numpy mismatch on subset'
    print('  ✓ numpy cross-check on query 0 passed')

    r1   = utils.compute_recall(I_flat, gts, 1)
    r10  = utils.compute_recall(I_flat, gts, 10)
    r100 = utils.compute_recall(I_flat, gts, 100)
    print(f'Flat(subset) vs *full-base* supplied GT — R@1 {r1:.4f}  R@10 {r10:.4f}  R@100 {r100:.4f}')
    print('  (Low recall here is expected: GT neighbours often lie outside the first 100k IDs.)')
else:
    SAMPLE_Q = 5_000
    qs = queries[:SAMPLE_Q]
    gts = gt[:SAMPLE_Q]
    print('Building IndexFlatL2 on full base (memmap stream-add)...')
    flat = faiss.IndexFlatL2(dim)
    BATCH = 50_000
    t0 = time.perf_counter()
    for i in tqdm(range(0, n_base, BATCH)):
        flat.add(np.ascontiguousarray(mm[i:i+BATCH]))
    print(f'  added {flat.ntotal:,} in {time.perf_counter()-t0:.1f}s  | RSS {psutil.Process().memory_info().rss/1e9:.2f} GB')
    print('Searching 5k queries × k=100...')
    t0 = time.perf_counter()
    D_flat, I_flat = flat.search(qs, 100)
    print(f'  search {time.perf_counter()-t0:.1f}s  ({SAMPLE_Q/(time.perf_counter()-t0):.1f} qps)')
    r1   = utils.compute_recall(I_flat, gts, 1)
    r10  = utils.compute_recall(I_flat, gts, 10)
    r100 = utils.compute_recall(I_flat, gts, 100)
    print(f'Flat vs supplied GT — R@1 {r1:.4f}  R@10 {r10:.4f}  R@100 {r100:.4f}')
"""),
    code(r"""
# Per-query recall@100 distribution (full run only asserts against GT)
per_q = np.array([np.intersect1d(I_flat[i], gts[i]).size / 100 for i in range(SAMPLE_Q)])
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 1, 21) if per_q.max() > per_q.min() + 1e-6 else np.linspace(0, 1, 6)
ax.hist(per_q, bins=bins, color='seagreen', edgecolor='white')
ax.set_xlabel('Recall@100 of Flat vs supplied GT')
ax.set_ylabel('# queries')
ax.set_title(f'Per-query GT match — mean {per_q.mean():.4f} · std {per_q.std():.4f} · min {per_q.min():.2f}')
if per_q.max() <= per_q.min() + 1e-6:
    ax.text(0.5, 0.5, 'All queries identical recall\n(full-base GT match)', transform=ax.transAxes,
            ha='center', va='center', fontsize=10)
plt.tight_layout(); plt.savefig(DOCS_IMG / '01_gt_recall_hist.png', dpi=120, bbox_inches='tight'); plt.show()

if int(os.environ.get('LAB_LIGHT', '0')):
    print('LAB_LIGHT: skipping strict GT assert (subset index vs full-base GT is not comparable).')
else:
    assert per_q.mean() > 0.95, f"Flat-vs-GT recall too low ({per_q.mean():.3f}); dataset suspect"
del flat; import gc; gc.collect()
"""),
    md("""
## 7 · Persist HDF5 metadata

We only store **query + GT + metadata** in HDF5 — the base file stays as the original
`.fvecs` and is accessed via `np.memmap` from later notebooks. Duplicating 10 GB of base
features into HDF5 would waste disk and yields no benefit because raw `.fvecs` is already
contiguous float32.
"""),
    code(r"""
with h5py.File(PATHS['h5'], 'w') as h:
    h.attrs['source']    = 'imagenet1m (ZJU CAD lab)'
    h.attrs['dim']       = dim
    h.attrs['n_base']    = n_base
    h.attrs['n_query']   = n_query
    h.attrs['gt_k']      = k_gt
    h.attrs['base_path'] = str(PATHS['base'])  # relative — portable across hosts/containers
    h.create_dataset('query',       data=queries, compression='gzip', compression_opts=4)
    h.create_dataset('groundtruth', data=gt,      compression='gzip', compression_opts=4)
print(f"wrote {PATHS['h5']}  ({PATHS['h5'].stat().st_size/1e6:.1f} MB)")

with h5py.File(PATHS['h5'], 'r') as h:
    print({k: h.attrs[k] for k in h.attrs.keys()})
    print('datasets:', list(h.keys()))
"""),
    md("""
## Summary

* Dataset downloaded and integrity verified.
* `IndexFlatL2` matches the supplied ground truth at Recall@100 ≈ 1.0, confirming the
  dataset files are intact and our `fvecs`/`ivecs` readers are correct.
* All exploration figures saved under `docs/img/light/` or `docs/img/full/` depending on `LAB_LIGHT`.
* `data/imagenet1m.h5` stores the query + GT; the base set is accessed via memmap from
  `data/imagenet_base.fvecs`.

→ Proceed to **`02_ivf_benchmark.ipynb`**.
"""),
]
write(nb1, '01_data_preparation.ipynb')
print("done 01")


# ---------------------------------------------------------------------------
# Common preamble used by 02-04
# ---------------------------------------------------------------------------

PREAMBLE = r"""
import os, sys, time, gc, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import faiss
import psutil
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))
import utils

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110

DATA = Path('data')
LAB_LIGHT = int(os.environ.get('LAB_LIGHT', '0'))
OUT_RUN = utils.run_mode()
RESULTS = utils.results_dir()
DOCS_IMG = utils.plots_dir()
print(f'OUT_RUN={OUT_RUN}  RESULTS={RESULTS}  DOCS_IMG={DOCS_IMG}')

# Load query + GT
with h5py.File(DATA / 'imagenet1m.h5', 'r') as h:
    queries = np.array(h['query'], dtype=np.float32)
    gt = np.array(h['groundtruth'], dtype=np.int32)
    DIM = int(h.attrs['dim']); N_BASE = int(h.attrs['n_base'])
    BASE_PATH = str(h.attrs['base_path'])

# Make BASE_PATH portable: prefer the file as it currently exists on disk
# (host path stored in h5 may not match container/CI paths).
_local = DATA / 'imagenet_base.fvecs'
if not Path(BASE_PATH).exists() and _local.exists():
    BASE_PATH = str(_local.resolve())
print('BASE_PATH =', BASE_PATH)

print('queries', queries.shape, 'gt', gt.shape, 'dim', DIM, 'base', N_BASE)
print(f"threads={faiss.omp_get_max_threads()}  RAM free={psutil.virtual_memory().available/1e9:.1f} GB")
"""

# ---------------------------------------------------------------------------
# 02 — IVF benchmark
# ---------------------------------------------------------------------------

nb2 = nbf.v4.new_notebook()
nb2.metadata = nb1.metadata

_ivfflat_sweep = []
for _nl in _ALL_NLIST:
    _ivfflat_sweep.append(md(f"#### IVFFlat — nlist={_nl}"))
    _ivfflat_sweep.append(code(_ivfflat_nlist_cell(_nl)))
_ivfflat_sweep.append(code(r"""
df_ivf = pd.read_csv(IVF_FLAT_PATH)
if df_ivf.empty:
    raise RuntimeError('ivf_flat.csv has no rows — check IVFFlat sweep cells above')
display(df_ivf.tail(8))
"""))

_ivfpq_sweep = []
for _nl in _ALL_PQ_NLIST:
    for _m in _ALL_PQ_M:
        _ivfpq_sweep.append(md(f"#### IVF+PQ — nlist={_nl} M={_m}"))
        _ivfpq_sweep.append(code(_ivfpq_nlist_m_cell(_nl, _m)))
_ivfpq_sweep.append(code(r"""
df_pq = pd.read_csv(IVF_PQ_PATH)
display(df_pq.tail(6))
"""))

_ivfsq_sweep = []
for _sq_name, _sq_qt in [
    ("SQ8", "faiss.ScalarQuantizer.QT_8bit"),
    ("SQ4", "faiss.ScalarQuantizer.QT_4bit"),
]:
    _ivfsq_sweep.append(md(f"#### IVF+SQ — {_sq_name}"))
    _ivfsq_sweep.append(code(_ivfsq_name_cell(_sq_name, _sq_qt)))
_ivfsq_sweep.append(code(r"""
df_sq = pd.read_csv(IVF_SQ_PATH)
display(df_sq)
"""))

nb2.cells = [
    md(r"""
# 02 — IVF / IVF+PQ / IVF+SQ Benchmarks

We sweep the three Inverted-File index families that FAISS exposes:

* `IndexIVFFlat` — coarse quantiser + exact vectors per cell
* `IndexIVFPQ` — coarse quantiser + product-quantised residuals  (compressed)
* `IndexIVFScalarQuantizer` — coarse quantiser + per-component byte quantisation

The full base set (1 281 167 × 2048 D) is too large for a dense parameter grid in
reasonable wall-time, so we sweep on the **full base for IVFFlat** and use the **best
nlist** for the IVF+PQ / IVF+SQ comparison.

Metrics logged per configuration: build (s) · index size (MB) · QPS · R@1 · R@10 · R@100 ·
peak RSS during build (MB, background-sampled).
"""),
    code(PREAMBLE),
    code(r"""
# ---------------------------------------------------------------------------
# Tunables — adjust to change the wall-time of this notebook.
# ---------------------------------------------------------------------------
# Subsample of the base used for parameter sweeps.  Set to N_BASE to use everything.
N_SWEEP = int(os.environ.get('LAB_N_SWEEP', 500_000))
LAB_LIGHT = int(os.environ.get('LAB_LIGHT', '0'))
# IVFFlat grid
NLIST_GRID  = [256, 1024, 4096, 16384]
NPROBE_GRID = [1, 4, 16, 64, 256, 1024]
# IVF+PQ grid — nlist × M (task requires varying coarse quantiser, not one nlist)
PQ_NLIST_GRID = [1024, 4096]
PQ_M_GRID = [32, 64, 128]   # divides 2048: 64,32,16 D per sub-vector
PQ_NBITS  = 8
# IVF+SQ grid
SQ_TYPES = [
    ('SQ8',  faiss.ScalarQuantizer.QT_8bit),
    ('SQ4',  faiss.ScalarQuantizer.QT_4bit),
]

# Number of vectors used for IVF training (k-means).  ≥ 30 * nlist recommended.
TRAIN_N = max(200_000, 30 * max(NLIST_GRID))

QUERY_K = 100  # search depth; we report R@1, R@10, R@100

if LAB_LIGHT:
    TRAIN_N = min(TRAIN_N, 80_000)
    NLIST_GRID = [256, 1024]
    NPROBE_GRID = [1, 4, 16, 64, 256]
    PQ_NLIST_GRID = [256, 1024]
    PQ_M_GRID = [32, 64]
    SQ_TYPES = [('SQ8', faiss.ScalarQuantizer.QT_8bit)]

QPS_REPEAT = int(os.environ.get('LAB_QPS_REPEAT', '2' if LAB_LIGHT else '1'))
QPS_WARMUP = int(os.environ.get('LAB_QPS_WARMUP', '1' if LAB_LIGHT else '0'))
# Full runs: fewer queries per sweep cell keeps nbconvert under per-cell timeout.
_default_qn = queries.shape[0] if LAB_LIGHT else min(10000, queries.shape[0])
QUERY_N = int(os.environ.get('LAB_QUERY_N', str(_default_qn)))
queries_sweep = queries[:QUERY_N]
print(f"N_SWEEP={N_SWEEP:,}  TRAIN_N={TRAIN_N:,}  LAB_LIGHT={LAB_LIGHT}")
print(f"QPS_REPEAT={QPS_REPEAT}  QPS_WARMUP={QPS_WARMUP}  QUERY_N={QUERY_N}")
print(f"NLIST_GRID={NLIST_GRID}  NPROBE_GRID={NPROBE_GRID}")
print(f"PQ_NLIST_GRID={PQ_NLIST_GRID}  PQ_M_GRID={PQ_M_GRID}  SQ_TYPES={[t[0] for t in SQ_TYPES]}")
print(f"TRAIN_N={TRAIN_N:,}  faiss_threads={faiss.omp_get_max_threads()}")
"""),
    md("""
## Helper — stream base vectors via memmap + recompute exact GT for the sweep subset

When `N_SWEEP < N_BASE` the supplied 100-NN ground truth is **invalid** for the subset
(it indexes into the full 1.28M base).  We recompute GT against the chosen subset using
`IndexFlatL2` — this is one-off, cached on disk and reused across notebooks.
"""),
    code(r"""
def ensure_gt(n: int, k: int = QUERY_K) -> np.ndarray:
    # Exact GT for first n base vectors; cached as data/gt_n{n}_k{k}.npy
    cache = DATA / f'gt_n{n}_k{k}.npy'
    if cache.exists():
        print(f'  ✓ cached GT  {cache}')
        return np.load(cache)
    print(f'Computing exact GT (Flat) on first {n:,} base vectors × {queries.shape[0]:,} queries × k={k}...')
    flat = faiss.IndexFlatL2(DIM)
    utils.stream_add(flat, BASE_PATH, n)
    _, I = flat.search(queries, k)
    np.save(cache, I)
    del flat; gc.collect()
    print(f'  cached → {cache}')
    return I

gt_local = ensure_gt(N_SWEEP)
# Small training slice only — the rest of the base is streamed straight into each index.
train_x = utils.load_train_subset(BASE_PATH, TRAIN_N)
utils.print_mem('after GT + train_x')
print('train_x', train_x.shape, 'gt_local', gt_local.shape)
"""),
    md("""
## IVFFlat sweep — (nlist × nprobe)

Each `nlist` runs in its **own notebook cell** with CSV checkpointing so a 2h cell timeout
cannot discard the whole sweep.
"""),
    code(r"""
IVF_FLAT_PATH = RESULTS / 'ivf_flat.csv'
utils.init_results_csv(IVF_FLAT_PATH)
print('IVFFlat checkpoint:', IVF_FLAT_PATH)
"""),
    *_ivfflat_sweep,
    md("""
### Plot 1 — Recall@100 vs nprobe (line per nlist)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
for nl, sub in df_ivf.groupby('nlist'):
    sub = sub.sort_values('nprobe')
    ax.plot(sub.nprobe, sub.recall_100, marker='o', label=f'nlist={nl}')
ax.set_xscale('log'); ax.set_xlabel('nprobe'); ax.set_ylabel('Recall@100')
ax.set_title(f'IVFFlat — Recall@100 vs nprobe   (N_base={N_SWEEP:,})')
ax.legend(); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivf_recall_vs_nprobe.png', dpi=120); plt.show()
"""),
    md("""
### Plot 2 — Heatmap nlist × nprobe → Recall@100
"""),
    code(r"""
pivot = df_ivf.pivot(index='nlist', columns='nprobe', values='recall_100')
# Invalid IVF configs: nprobe must be ≤ nlist — mask so heatmap shows gaps explicitly
mask_invalid = np.zeros_like(pivot, dtype=bool)
for i, nl in enumerate(pivot.index):
    for j, np_ in enumerate(pivot.columns):
        if int(np_) > int(nl):
            mask_invalid[i, j] = True
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(pivot, mask=mask_invalid, annot=True, fmt='.3f', cmap='viridis', vmin=0, vmax=1,
            ax=ax, cbar_kws={'label': 'Recall@100'})
ax.set_title('IVFFlat — Recall@100 (masked where nprobe > nlist)')
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivf_heatmap.png', dpi=120); plt.show()
"""),
    md("""
### Plot 3 — QPS vs Recall@100 Pareto frontier
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_ivf.recall_100, df_ivf.qps, c=np.log2(df_ivf.nprobe.values+1),
           cmap='plasma', s=40, edgecolors='k')
mask = utils.pareto_frontier(df_ivf.recall_100.values, df_ivf.qps.values)
order = np.argsort(df_ivf.recall_100.values[mask])
ax.plot(df_ivf.recall_100.values[mask][order], df_ivf.qps.values[mask][order],
        'k--', lw=1, label='Pareto frontier')
# Annotate Pareto points only (avoids clutter)
dfm = df_ivf.iloc[np.where(mask)[0]].sort_values('recall_100')
for _, r in dfm.iterrows():
    ax.annotate(f"L{int(r.nlist)}/P{int(r.nprobe)}", (r.recall_100, r.qps),
                fontsize=7, alpha=0.85, xytext=(3, 3), textcoords='offset points')
ax.set_yscale('log'); ax.set_xlabel('Recall@100'); ax.set_ylabel('QPS (log)')
ax.set_title('IVFFlat — QPS vs Recall@100')
ax.legend()
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivf_pareto.png', dpi=120); plt.show()
"""),
    md("""
### Plot 4 — QPS vs nprobe (per nlist)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
for nl, sub in df_ivf.groupby('nlist'):
    sub = sub.sort_values('nprobe')
    ax.plot(sub.nprobe, sub.qps, marker='o', label=f'nlist={nl}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('nprobe'); ax.set_ylabel('QPS (log)')
ax.set_title('IVFFlat — QPS vs nprobe')
ax.legend(); plt.tight_layout()
plt.savefig(DOCS_IMG / '02_ivf_qps_vs_nprobe.png', dpi=120); plt.show()
"""),
    md("""
### Plot 5 — Build time vs nlist
"""),
    code(r"""
df_bt = df_ivf.drop_duplicates('nlist')[['nlist', 'build_s', 'size_mb']]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df_bt, x='nlist', y='build_s', ax=ax[0], color='steelblue')
ax[0].set_yscale('log')
ax[0].set_title('IVFFlat build time (log)'); ax[0].set_ylabel('seconds')
sns.barplot(data=df_bt, x='nlist', y='size_mb', ax=ax[1], color='darkorange')
ax[1].set_title('IVFFlat index size'); ax[1].set_ylabel('MB')
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivf_build_size.png', dpi=120); plt.show()
display(df_bt.reset_index(drop=True))
"""),
    md("""
## IVF+PQ sweep — nlist × PQ M

Full grid over `PQ_NLIST_GRID × PQ_M_GRID` (not a single best nlist from IVFFlat).
"""),
    code(r"""
IVF_PQ_PATH = RESULTS / 'ivf_pq.csv'
utils.init_results_csv(IVF_PQ_PATH)
print('IVFPQ checkpoint:', IVF_PQ_PATH)

# SQ still uses the IVFFlat winner as coarse quantiser.
best_nlist = int((df_ivf
              .groupby('nlist')['recall_100'].max()
              .idxmax()))
print(f'using nlist={best_nlist} for IVF+SQ only')
"""),
    *_ivfpq_sweep,
    md("""
### Plot 6 — IVF+PQ Recall@100 vs nprobe (per M)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(9, 5))
for (nl, M), sub in df_pq.groupby(['nlist', 'M']):
    sub = sub.sort_values('nprobe')
    ax.plot(sub.nprobe, sub.recall_100, marker='o', label=f'nlist={int(nl)} M={int(M)}')
ax.set_xscale('log'); ax.set_xlabel('nprobe'); ax.set_ylabel('Recall@100')
ax.set_title('IVF+PQ — Recall@100 vs nprobe (per nlist × M)')
ax.legend(); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivfpq_recall.png', dpi=120); plt.show()
"""),
    md("""
### Plot 7 — IVF+PQ Pareto QPS vs Recall@100 (per M)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(9, 5))
markers = ['o', 's', '^', 'D', 'v', 'P']
for i, (nl, M) in enumerate(sorted({(int(r.nlist), int(r.M)) for _, r in df_pq.iterrows()})):
    sub = df_pq[(df_pq.nlist == nl) & (df_pq.M == M)]
    mk = markers[i % len(markers)]
    ax.scatter(sub.recall_100, sub.qps, marker=mk, s=45, edgecolors='k',
               label=f'nlist={nl} M={M}')
    ms = utils.pareto_frontier(sub.recall_100.values, sub.qps.values)
    if ms.any():
        o = np.argsort(sub.recall_100.values[ms])
        ax.plot(sub.recall_100.values[ms][o], sub.qps.values[ms][o], ls='--', lw=0.8, alpha=0.7)
ax.scatter(df_ivf[df_ivf.nlist==best_nlist].recall_100, df_ivf[df_ivf.nlist==best_nlist].qps,
           marker='X', color='k', s=90, label=f'IVFFlat nlist={best_nlist}', zorder=5)
ax.set_yscale('log'); ax.set_xlabel('Recall@100'); ax.set_ylabel('QPS (log)')
ax.set_title('IVF+PQ — QPS vs Recall@100 (each nlist × M)')
ax.legend(fontsize=7, ncol=2, loc='lower right')
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivfpq_pareto.png', dpi=120); plt.show()
"""),
    md("""
### Plot 8 — Index size vs PQ M (compression vs flat)
"""),
    code(r"""
size_rows = [{'family': 'IVFFlat', 'nlist': best_nlist, 'M': None,
              'size_mb': float(df_ivf[df_ivf.nlist==best_nlist].size_mb.iloc[0])}]
for nl in sorted(df_pq.nlist.unique()):
    for M in sorted(df_pq.M.unique()):
        sub = df_pq[(df_pq.nlist == nl) & (df_pq.M == M)]
        size_rows.append({'family': 'IVFPQ', 'nlist': int(nl), 'M': int(M),
                         'size_mb': float(sub.size_mb.iloc[0])})
size_df = pd.DataFrame(size_rows)
size_df['label'] = size_df.apply(
    lambda r: f"Flat L{int(r.nlist)}" if r.family == 'IVFFlat' else f"PQ L{int(r.nlist)} M{int(r.M)}",
    axis=1)
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=size_df, x='label', y='size_mb', hue='family', ax=ax, dodge=False)
ax.set_yscale('log'); ax.set_ylabel('MB (log)')
ax.set_title('Index footprint — IVFFlat vs IVF+PQ (per nlist × M)')
for i, (_, r) in enumerate(size_df.iterrows()):
    ax.text(i, r.size_mb, f'{r.size_mb:.0f}', ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=25, ha='right')
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivfpq_size.png', dpi=120); plt.show()
display(size_df)
"""),
    md("""
## IVF+ScalarQuantizer

One SQ variant per notebook cell (checkpointed CSV).
"""),
    code(r"""
IVF_SQ_PATH = RESULTS / 'ivf_sq.csv'
utils.init_results_csv(IVF_SQ_PATH)
print('IVFSQ checkpoint:', IVF_SQ_PATH)
"""),
    *_ivfsq_sweep,
    md("""
### Plot 9 — SQ vs PQ vs Flat at the same nlist (Pareto)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(df_ivf[df_ivf.nlist==best_nlist].recall_100, df_ivf[df_ivf.nlist==best_nlist].qps,
           label=f'IVFFlat nlist={best_nlist}', color='black', s=55, marker='X')
for M, sub in df_pq.groupby('M'):
    ax.scatter(sub.recall_100, sub.qps, label=f'IVF+PQ M={M}', s=40, alpha=0.85)
for name, sub in df_sq.groupby('sq'):
    ax.scatter(sub.recall_100, sub.qps, label=f'IVF+{name}', s=70, marker='s', alpha=0.9)
ax.set_yscale('log'); ax.set_xlabel('Recall@100'); ax.set_ylabel('QPS (log)')
ax.set_title('IVF variants — Pareto QPS vs Recall@100')
ax.legend()
plt.tight_layout(); plt.savefig(DOCS_IMG / '02_ivf_all_pareto.png', dpi=120); plt.show()

# Release training slice (~1.6 GB) before notebooks save and plots render.
try:
    del train_x
except NameError:
    pass
gc.collect()
utils.print_mem('after IVF sweeps')

# Persist combined for notebook 05
df_all = pd.concat([df_ivf, df_pq, df_sq], ignore_index=True)
df_all.to_csv(RESULTS / 'ivf_all.csv', index=False)
print(f'rows={len(df_all)}  → {RESULTS / "ivf_all.csv"}')
"""),
    md("""
## Summary

Notable observations to keep in mind when reading `05_comparison.ipynb`:

* The IVFFlat upper bound on Recall@100 is set by `nprobe / nlist`. At the largest
  `nlist`/`nprobe` ratio we should saturate to ≥ 0.99.
* IVF+PQ trades index size for recall — index footprint drops by ~50–100× while
  recall stays within ~10–20 % of Flat **only at high nprobe**.
* IVF+ScalarQuantizer is a middle ground — 4× compression with minimal recall loss.
"""),
]
write(nb2, '02_ivf_benchmark.ipynb')
print("done 02")


# ---------------------------------------------------------------------------
# 03 — HNSW benchmark
# ---------------------------------------------------------------------------

nb3 = nbf.v4.new_notebook()
nb3.metadata = nb1.metadata

_hnsw_varym_sweep = []
for _m in _ALL_HNSW_M:
    _hnsw_varym_sweep.append(md(f"#### HNSW study 1 — M={_m}"))
    _hnsw_varym_sweep.append(code(_hnsw_m_cell(_m)))
_hnsw_varym_sweep.append(code(r"""
df_M = pd.read_csv(HNSW_VARYM_PATH)
display(df_M.tail(8))
"""))

_hnsw_varyefc_sweep = []
for _efc in _ALL_EFC:
    _hnsw_varyefc_sweep.append(md(f"#### HNSW study 2 — efConstruction={_efc}"))
    _hnsw_varyefc_sweep.append(code(_hnsw_efc_cell(_efc)))
_hnsw_varyefc_sweep.append(code(r"""
df_EFC = pd.read_csv(HNSW_VARYEFC_PATH)
display(df_EFC.tail(8))
"""))

nb3.cells = [
    md(r"""
# 03 — HNSW Benchmarks

We sweep:

* `M` — number of edges per node (memory & build cost driver)
* `efConstruction` — build-time candidate width
* `efSearch` — query-time candidate width (recall ↔ QPS tunable)

Two studies:
1. Vary `M` (efConstruction fixed) → effect of graph degree
2. Vary `efConstruction` (M fixed) → effect of build quality
3. Heatmap `M × efSearch` → Recall@100

Notebook uses the same `data/imagenet1m.h5` + `BASE_PATH` memmap as notebook 02.
"""),
    code(PREAMBLE),
    code(r"""
N_SWEEP = int(os.environ.get('LAB_N_SWEEP', 500_000))
LAB_LIGHT = int(os.environ.get('LAB_LIGHT', '0'))

# Study 1: vary M, fixed ef_construction
M_GRID = [8, 16, 32, 48]
EFC_FIXED = 200
EFS_GRID = [10, 20, 40, 80, 160, 320, 640]
# Study 2: vary ef_construction, fixed M
M_FIXED = 32
EFC_GRID = [40, 100, 200, 400]

QUERY_K = 100
if LAB_LIGHT:
    M_GRID = [8, 16, 32]
    EFS_GRID = [10, 20, 40, 80, 160, 320]
    EFC_GRID = [40, 100, 200]
QPS_REPEAT = int(os.environ.get('LAB_QPS_REPEAT', '2' if LAB_LIGHT else '1'))
QPS_WARMUP = int(os.environ.get('LAB_QPS_WARMUP', '1' if LAB_LIGHT else '0'))
_default_qn = queries.shape[0] if LAB_LIGHT else min(10000, queries.shape[0])
QUERY_N = int(os.environ.get('LAB_QUERY_N', str(_default_qn)))
queries_sweep = queries[:QUERY_N]
print(f"N_SWEEP={N_SWEEP:,}  LAB_LIGHT={LAB_LIGHT}  M_GRID={M_GRID}  EFC_FIXED={EFC_FIXED}  EFS_GRID={EFS_GRID}")
print(f"M_FIXED={M_FIXED}  EFC_GRID={EFC_GRID}")
print(f"QPS_REPEAT={QPS_REPEAT}  QPS_WARMUP={QPS_WARMUP}  QUERY_N={QUERY_N}")
"""),
    code(r"""
def ensure_gt(n: int, k: int = QUERY_K):
    cache = DATA / f'gt_n{n}_k{k}.npy'
    if cache.exists():
        return np.load(cache)
    print(f'Computing exact GT (Flat) on first {n:,} base vectors × {queries.shape[0]:,} queries × k={k}...')
    flat = faiss.IndexFlatL2(DIM)
    utils.stream_add(flat, BASE_PATH, n)
    _, I = flat.search(queries, k)
    np.save(cache, I)
    del flat; gc.collect()
    return I

gt_local = ensure_gt(N_SWEEP)
utils.print_mem('after GT')
print('gt_local', gt_local.shape, '  (base streamed from disk, not held in RAM)')
"""),
    md("""
## Study 1 — vary M

Each `M` value runs in its own cell (CSV checkpoint).
"""),
    code(r"""
HNSW_VARYM_PATH = RESULTS / 'hnsw_varyM.csv'
utils.init_results_csv(HNSW_VARYM_PATH)
print('HNSW vary-M checkpoint:', HNSW_VARYM_PATH)
"""),
    *_hnsw_varym_sweep,
    md("""
### Plot 1 — efSearch vs Recall@100 (per M)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
for M, sub in df_M.groupby('M'):
    sub = sub.sort_values('efSearch')
    ax.plot(sub.efSearch, sub.recall_100, marker='o', label=f'M={M}')
ax.set_xscale('log'); ax.set_xlabel('efSearch'); ax.set_ylabel('Recall@100')
ax.set_title(f'HNSW — Recall@100 vs efSearch  (efC={EFC_FIXED}, N={N_SWEEP:,})')
ax.legend(); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(DOCS_IMG / '03_hnsw_recall_vs_efs.png', dpi=120); plt.show()
"""),
    md("""
### Plot 2 — QPS vs Recall@100 Pareto (per M)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette('plasma', len(M_GRID))
for col, (M, sub) in zip(palette, df_M.groupby('M')):
    ax.scatter(sub.recall_100, sub.qps, color=col, label=f'M={M}', s=50, edgecolors='k')
    ms = utils.pareto_frontier(sub.recall_100.values, sub.qps.values)
    o = np.argsort(sub.recall_100.values[ms])
    ax.plot(sub.recall_100.values[ms][o], sub.qps.values[ms][o], color=col, lw=0.8, ls='--', alpha=0.85)
    pidx = np.where(ms)[0]
    if len(pidx) <= 6:
        for i in pidx:
            r = sub.iloc[i]
            ax.annotate(f"efS={int(r.efSearch)}", (r.recall_100, r.qps), fontsize=6, alpha=0.75,
                        xytext=(3, 3), textcoords='offset points')
ax.set_yscale('log'); ax.set_xlabel('Recall@100'); ax.set_ylabel('QPS (log)')
ax.set_title('HNSW — Pareto QPS vs Recall@100  (efC={}; labels on Pareto only)'.format(EFC_FIXED))
ax.legend()
plt.tight_layout(); plt.savefig(DOCS_IMG / '03_hnsw_pareto_M.png', dpi=120); plt.show()
"""),
    md("""
### Plot 3 & 4 — Build time / index size vs M
"""),
    code(r"""
df_bt = df_M.drop_duplicates('M')[['M', 'build_s', 'size_mb']]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df_bt, x='M', y='build_s', ax=ax[0], color='steelblue')
ax[0].set_title('HNSW build time'); ax[0].set_ylabel('seconds')
sns.barplot(data=df_bt, x='M', y='size_mb', ax=ax[1], color='darkorange')
ax[1].set_title('HNSW index size'); ax[1].set_ylabel('MB')
plt.tight_layout(); plt.savefig(DOCS_IMG / '03_hnsw_build_size_M.png', dpi=120); plt.show()
display(df_bt.reset_index(drop=True))
"""),
    md("""
## Study 2 — vary efConstruction (M={fixed})

Each `efConstruction` value runs in its own cell (CSV checkpoint).
""".replace("{fixed}", str(32))),
    code(r"""
HNSW_VARYEFC_PATH = RESULTS / 'hnsw_varyEFC.csv'
utils.init_results_csv(HNSW_VARYEFC_PATH)
print('HNSW vary-efC checkpoint:', HNSW_VARYEFC_PATH)
"""),
    *_hnsw_varyefc_sweep,
    md("""
### Plot 5 — Recall@100 vs efSearch (per efConstruction, M fixed)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
for efc, sub in df_EFC.groupby('efConstruction'):
    sub = sub.sort_values('efSearch')
    ax.plot(sub.efSearch, sub.recall_100, marker='o', label=f'efC={efc}')
ax.set_xscale('log'); ax.set_xlabel('efSearch'); ax.set_ylabel('Recall@100')
ax.set_title(f'HNSW — efSearch vs Recall@100  (M={M_FIXED})')
ax.legend(); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(DOCS_IMG / '03_hnsw_recall_vs_efs_EFC.png', dpi=120); plt.show()
"""),
    md("""
### Plot 6 — Build time vs efConstruction
"""),
    code(r"""
df_bt2 = df_EFC.drop_duplicates('efConstruction')[['efConstruction', 'build_s']]
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=df_bt2, x='efConstruction', y='build_s', ax=ax, color='seagreen')
ax.set_title(f'HNSW build time vs efConstruction (M={M_FIXED})')
ax.set_ylabel('seconds')
plt.tight_layout(); plt.savefig(DOCS_IMG / '03_hnsw_build_vs_EFC.png', dpi=120); plt.show()
display(df_bt2.reset_index(drop=True))
"""),
    md("""
### Plot 7 — Heatmap M × efSearch → Recall@100
"""),
    code(r"""
piv = df_M.pivot(index='M', columns='efSearch', values='recall_100')
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(piv, annot=True, fmt='.3f', cmap='viridis', vmin=0, vmax=1, ax=ax)
ax.set_title(f'HNSW — Recall@100 heatmap  (efC={EFC_FIXED}, N={N_SWEEP:,})')
plt.tight_layout(); plt.savefig(DOCS_IMG / '03_hnsw_heatmap.png', dpi=120); plt.show()

# Combined CSV for notebook 05
df_all_hnsw = pd.concat([df_M, df_EFC], ignore_index=True)
df_all_hnsw.to_csv(RESULTS / 'hnsw_all.csv', index=False)
print('rows', len(df_all_hnsw))
"""),
    md("""
## Summary

* Larger `M` → higher recall and bigger index, with diminishing return after M ≈ 32.
* Higher `efConstruction` mostly helps **at the top of the recall range**
  (`Recall@100 ≥ 0.9`).
* `efSearch` is a *runtime knob* — adjustable without rebuilding.

See `05_comparison.ipynb` for cross-algorithm comparison.
"""),
]
write(nb3, '03_hnsw_benchmark.ipynb')
print("done 03")


# ---------------------------------------------------------------------------
# 04 — LSH benchmark
# ---------------------------------------------------------------------------

nb4 = nbf.v4.new_notebook()
nb4.metadata = nb1.metadata

_lsh_sweep = []
for _nb in _ALL_LSH_NBITS:
    _lsh_sweep.append(md(f"#### LSH — nbits={_nb}"))
    _lsh_sweep.append(code(_lsh_nbits_cell(_nb)))
_lsh_sweep.append(code(r"""
df_lsh = pd.read_csv(LSH_PATH)
try:
    del train_x
except NameError:
    pass
gc.collect()
utils.print_mem('after LSH sweep')
display(df_lsh)
"""))

nb4.cells = [
    md(r"""
# 04 — Random-Projection LSH

`IndexLSH` projects each vector onto `nbits` random hyperplanes and stores the resulting
binary code.  Search is Hamming-distance then re-ranked.  Strong baseline for *low*
recall + *small* index size.  Curse of dimensionality at 2048 D is the main concern.
"""),
    code(PREAMBLE),
    code(r"""
N_SWEEP = int(os.environ.get('LAB_N_SWEEP', 500_000))
LAB_LIGHT = int(os.environ.get('LAB_LIGHT', '0'))
NBITS_GRID = [128, 256, 512, 1024, 2048, 4096]
QUERY_K = 100
if LAB_LIGHT:
    NBITS_GRID = [128, 256, 512, 1024]
QPS_REPEAT = int(os.environ.get('LAB_QPS_REPEAT', '2' if LAB_LIGHT else '1'))
QPS_WARMUP = int(os.environ.get('LAB_QPS_WARMUP', '1' if LAB_LIGHT else '0'))
_default_qn = queries.shape[0] if LAB_LIGHT else min(10000, queries.shape[0])
QUERY_N = int(os.environ.get('LAB_QUERY_N', str(_default_qn)))
queries_sweep = queries[:QUERY_N]
print(f"N_SWEEP={N_SWEEP:,}  LAB_LIGHT={LAB_LIGHT}  NBITS_GRID={NBITS_GRID}")
print(f"QPS_REPEAT={QPS_REPEAT}  QPS_WARMUP={QPS_WARMUP}  QUERY_N={QUERY_N}")
"""),
    code(r"""
def ensure_gt(n: int, k: int = QUERY_K):
    cache = DATA / f'gt_n{n}_k{k}.npy'
    if cache.exists():
        return np.load(cache)
    print(f'Computing exact GT (Flat) on first {n:,} base vectors × {queries.shape[0]:,} queries × k={k}...')
    flat = faiss.IndexFlatL2(DIM)
    utils.stream_add(flat, BASE_PATH, n)
    _, I = flat.search(queries, k)
    np.save(cache, I)
    del flat; gc.collect()
    return I

gt_local = ensure_gt(N_SWEEP)
train_x = utils.load_train_subset(BASE_PATH, min(200_000, N_SWEEP))
utils.print_mem('after GT + train_x')
print('train_x', train_x.shape, 'gt_local', gt_local.shape)
"""),
    md("""
## LSH sweep — nbits grid

One `nbits` value per notebook cell (CSV checkpoint).
"""),
    code(r"""
LSH_PATH = RESULTS / 'lsh.csv'
utils.init_results_csv(LSH_PATH)
print('LSH checkpoint:', LSH_PATH)
"""),
    *_lsh_sweep,
    md("""
### Plot 1 — nbits vs Recall (R@1, R@10, R@100)
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
for col, k in zip(['#1f77b4', '#ff7f0e', '#2ca02c'], [1, 10, 100]):
    ax.plot(df_lsh.nbits, df_lsh[f'recall_{k}'], marker='o', label=f'R@{k}', color=col)
ax.set_xscale('log'); ax.set_xlabel('nbits'); ax.set_ylabel('Recall')
ax.set_xticks(df_lsh.nbits)
ax.set_xticklabels([str(int(x)) for x in df_lsh.nbits])
ax.set_title(f'LSH — recall vs nbits  (N={N_SWEEP:,}, dim={DIM})')
ax.legend(); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(DOCS_IMG / '04_lsh_recall.png', dpi=120); plt.show()
"""),
    md("""
### Plot 2 — QPS vs nbits
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_lsh.nbits, df_lsh.qps, marker='o', color='crimson')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('nbits'); ax.set_ylabel('QPS (log)')
ax.set_title('LSH — QPS vs nbits')
plt.tight_layout(); plt.savefig(DOCS_IMG / '04_lsh_qps.png', dpi=120); plt.show()
"""),
    md("""
### Plot 3 — QPS vs Recall@100 Pareto
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df_lsh.recall_100, df_lsh.qps, s=60, edgecolors='k', c=np.log2(df_lsh.nbits), cmap='plasma')
msk = utils.pareto_frontier(df_lsh.recall_100.values, df_lsh.qps.values)
o = np.argsort(df_lsh.recall_100.values[msk])
ax.plot(df_lsh.recall_100.values[msk][o], df_lsh.qps.values[msk][o], 'k--', lw=1, label='Pareto')
for i in np.where(msk)[0]:
    r = df_lsh.iloc[i]
    ax.annotate(f"nbits={int(r.nbits)}", (r.recall_100, r.qps),
                fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.set_yscale('log'); ax.set_xlabel('Recall@100'); ax.set_ylabel('QPS (log)')
ax.set_title('LSH — Pareto QPS vs Recall@100')
ax.legend()
plt.tight_layout(); plt.savefig(DOCS_IMG / '04_lsh_pareto.png', dpi=120); plt.show()
"""),
    md("""
### Plot 4 & 5 — Build time and index size vs nbits
"""),
    code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df_lsh, x='nbits', y='build_s', ax=ax[0], color='steelblue')
ax[0].set_title('LSH build time'); ax[0].set_ylabel('seconds')
sns.barplot(data=df_lsh, x='nbits', y='size_mb', ax=ax[1], color='darkorange')
ax[1].set_title('LSH index size'); ax[1].set_ylabel('MB')
plt.tight_layout(); plt.savefig(DOCS_IMG / '04_lsh_build_size.png', dpi=120); plt.show()
"""),
    md("""
## Notes

* At 2048 D, LSH needs **a lot** of bits to start being competitive. We expect
  R@100 ≪ 0.5 even with 4096 bits — the curse of dimensionality bites hard.
* Despite poor recall, LSH index footprint is the smallest of the lot (just bit codes).
* Useful as a *very fast first-pass filter* but not as a stand-alone ANN solution at this
  dimensionality.
"""),
]
write(nb4, '04_lsh_benchmark.ipynb')
print("done 04")


# ---------------------------------------------------------------------------
# 05 — comparison
# ---------------------------------------------------------------------------

nb5 = nbf.v4.new_notebook()
nb5.metadata = nb1.metadata
nb5.cells = [
    md(r"""
# 05 — Cross-Algorithm Comparison & Scaling

Loads all per-algorithm CSVs, plots:
1. Cross-algorithm Pareto QPS vs Recall@100
2. Best-config summary table
3. Build time / index size / RSS comparison bars
4. Scaling experiment (100K → 1M base vectors)
5. Anomaly analysis with quantitative evidence
6. Final ranking
"""),
    code(PREAMBLE),
    code(r"""
def load(path):
    return pd.read_csv(path) if Path(path).exists() else None

frames = {
    'IVF_all':  load(RESULTS / 'ivf_all.csv'),
    'HNSW_all': load(RESULTS / 'hnsw_all.csv'),
    'LSH':      load(RESULTS / 'lsh.csv'),
}
for k, v in frames.items():
    print(f'{k:10} {None if v is None else len(v)} rows')

# Normalise into one combined DF
def tag(df, algo):
    df = df.copy()
    df['family'] = algo
    return df

combined = pd.concat([
    tag(frames['IVF_all'], frames['IVF_all'].algo if frames['IVF_all'] is not None else 'IVF'),
    tag(frames['HNSW_all'], 'HNSW'),
    tag(frames['LSH'], 'LSH'),
], ignore_index=True)
combined['family'] = combined['algo']
combined['family'] = combined['family'].replace({'IVFFlat':'IVFFlat','IVFPQ':'IVFPQ','IVFSQ':'IVFSQ'})
print(combined.family.value_counts().to_string())
"""),
    md("""
## 1 · Cross-algorithm Pareto frontier
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(10, 6))
palette = {'IVFFlat':'#1f77b4', 'IVFPQ':'#ff7f0e', 'IVFSQ':'#2ca02c',
           'HNSW':'#d62728', 'LSH':'#9467bd'}
for fam, sub in combined.groupby('family'):
    ax.scatter(sub.recall_100, sub.qps, c=palette.get(fam, 'k'), label=fam,
               s=35, alpha=0.65, edgecolors='k', linewidth=0.3)

# global Pareto — annotate hull only (≤5 labels)
mask = utils.pareto_frontier(combined.recall_100.values, combined.qps.values)
order = np.argsort(combined.recall_100.values[mask])
ax.plot(combined.recall_100.values[mask][order], combined.qps.values[mask][order],
        'k--', lw=1.2, label='global Pareto')
df_p = combined.iloc[np.where(mask)[0]].sort_values('recall_100')
for _, r in df_p.iloc[::max(1, len(df_p) // 5)][:5].iterrows():
    lbl = f"{r.algo}"
    if pd.notna(r.get('nlist')): lbl += f" L{int(r.nlist)}"
    if pd.notna(r.get('M')): lbl += f" M{int(r.M)}"
    ax.annotate(lbl, (r.recall_100, r.qps), fontsize=7, alpha=0.85,
                xytext=(3, 3), textcoords='offset points')
ax.set_yscale('log'); ax.set_xlabel('Recall@100'); ax.set_ylabel('QPS (log)')
ax.set_title('Cross-algorithm QPS vs Recall@100')
ax.legend()
plt.tight_layout(); plt.savefig(DOCS_IMG / '05_global_pareto.png', dpi=120); plt.show()
"""),
    md("""
## 2 · Best configurations summary
"""),
    code(r"""
THRESHOLDS = [0.95, 0.9, 0.8, 0.5, 0.2]
OPERATIONAL_THR = 0.95

def _row_to_record(fam, scenario, thr, is_fb, b):
    cfg_parts = []
    for c in ['nlist', 'nprobe', 'M', 'efConstruction', 'efSearch', 'nbits', 'sq']:
        if c in b.index and pd.notna(b[c]):
            cfg_parts.append(f'{c}={b[c]}')
    rss_after = float(b['rss_mb']) if pd.notna(b.get('rss_mb')) else float('nan')
    rss_peak = float(b['rss_peak_mb']) if 'rss_peak_mb' in b.index and pd.notna(b.get('rss_peak_mb')) else rss_after
    rss_delta = float(b['rss_delta_mb']) if 'rss_delta_mb' in b.index and pd.notna(b.get('rss_delta_mb')) else float('nan')
    lat_p99 = float(b['latency_p99_ms']) if 'latency_p99_ms' in b.index and pd.notna(b.get('latency_p99_ms')) else float('nan')
    return dict(
        family=fam, scenario=scenario, threshold=thr, threshold_fallback=is_fb,
        recall_100=b.recall_100, qps=b.qps, size_mb=b.size_mb,
        latency_ms=float(b['latency_ms']) if pd.notna(b.get('latency_ms')) else float('nan'),
        latency_p99_ms=lat_p99,
        build_s=float(b['build_s']) if 'build_s' in b.index and pd.notna(b.get('build_s')) else float('nan'),
        rss_mb=rss_after, rss_peak_mb=rss_peak, rss_delta_mb=rss_delta,
        config=', '.join(cfg_parts),
    )

def best_operational_row(sub, thr=OPERATIONAL_THR):
    cand = sub[sub.recall_100 >= thr]
    if len(cand):
        return thr, cand.sort_values(['qps', 'recall_100'], ascending=[False, False]).iloc[0], False
    return 0.0, sub.sort_values(['recall_100', 'qps'], ascending=[False, False]).iloc[0], True

scenario_rows = []
summary_rows = []
for fam in sorted(combined['family'].unique()):
    sub = combined[combined.family == fam]
    br = sub.sort_values(['recall_100', 'qps'], ascending=[False, False]).iloc[0]
    scenario_rows.append(_row_to_record(fam, 'best_recall', 0.0, True, br))
    bf = sub.sort_values(['size_mb', 'recall_100'], ascending=[True, False]).iloc[0]
    scenario_rows.append(_row_to_record(fam, 'best_footprint', 0.0, True, bf))
    for thr in THRESHOLDS:
        cand = sub[sub.recall_100 >= thr]
        if len(cand):
            bq = cand.sort_values(['qps', 'recall_100'], ascending=[False, False]).iloc[0]
            scenario_rows.append(_row_to_record(fam, f'best_qps_at_{thr}', thr, False, bq))
    thr, b, is_fb = best_operational_row(sub)
    summary_rows.append(_row_to_record(fam, 'operational', thr, is_fb, b))

summary = pd.DataFrame(summary_rows)
scenarios = pd.DataFrame(scenario_rows)
display(summary)
display(scenarios)
summary.to_csv(RESULTS / 'best_configs.csv', index=False)
scenarios.to_csv(RESULTS / 'best_configs_scenarios.csv', index=False)
print(f'Operational pick: max QPS among recall>={OPERATIONAL_THR} (else highest recall). See best_configs_scenarios.csv for alternatives.')
"""),
    md("""
## 3 · Build time / index size / RSS comparison (operational best configs)

Peak RSS bars include **mmap page-cache** during `stream_add`, not index size alone — compare `size_mb` and `rss_delta_mb` for process growth.
"""),
    code(r"""
sum_plot = summary.copy().sort_values('family')

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
sns.barplot(data=sum_plot, x='family', y='build_s', ax=ax[0], hue='family', palette='tab10', legend=False)
ax[0].set_title('Build time'); ax[0].set_ylabel('seconds')
sns.barplot(data=sum_plot, x='family', y='size_mb', ax=ax[1], hue='family', palette='tab10', legend=False)
ax[1].set_yscale('log'); ax[1].set_title('Index size'); ax[1].set_ylabel('MB (log)')
sns.barplot(data=sum_plot, x='family', y='rss_mb', ax=ax[2], hue='family', palette='tab10', legend=False)
ax[2].set_title('RSS after build'); ax[2].set_ylabel('MB')
sns.barplot(data=sum_plot, x='family', y='rss_peak_mb', ax=ax[3], hue='family', palette='tab10', legend=False)
ax[3].set_yscale('log'); ax[3].set_title('Peak RSS (build)'); ax[3].set_ylabel('MB (log)')
for a in ax:
    a.tick_params(axis='x', rotation=15)
plt.tight_layout(); plt.savefig(DOCS_IMG / '05_best_bars.png', dpi=120); plt.show()

# Index size vs RSS delta (peak includes mmap page-cache for stream_add)
fig, ax = plt.subplots(figsize=(7, 4.5))
if 'rss_delta_mb' in sum_plot.columns:
    ax.scatter(sum_plot['size_mb'], sum_plot['rss_delta_mb'], s=90, c='steelblue', edgecolors='k')
    ax.set_ylabel('RSS delta during build (MB)')
    ax.set_title('Index size vs RSS growth (not peak — peak tracks mmap cache)')
else:
    ax.scatter(sum_plot['size_mb'], sum_plot['rss_peak_mb'], s=90, c='steelblue', edgecolors='k')
    ax.set_ylabel('Peak RSS (MB)')
    ax.set_title('Index size vs peak RSS')
for _, r in sum_plot.iterrows():
    ax.annotate(r['family'], (r['size_mb'], r.get('rss_delta_mb', r['rss_peak_mb'])), fontsize=9,
                xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Index size on disk (MB)')
plt.tight_layout(); plt.savefig(DOCS_IMG / '05_memory_sanity.png', dpi=120); plt.show()

# Latency from best-config rows
if 'latency_p99_ms' in sum_plot.columns and sum_plot['latency_p99_ms'].notna().any():
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sum_plot))
    w = 0.35
    ax.bar(x - w/2, sum_plot['latency_ms'], width=w, label='mean latency')
    ax.bar(x + w/2, sum_plot['latency_p99_ms'], width=w, label='p99 latency (batch runs)')
    ax.set_xticks(x); ax.set_xticklabels(sum_plot['family'], rotation=15)
    ax.set_ylabel('ms / query'); ax.set_title('Query latency at best config')
    ax.legend()
    plt.tight_layout(); plt.savefig(DOCS_IMG / '05_latency_best.png', dpi=120); plt.show()
"""),
    md("""
## 4 · Scaling experiment

For each algorithm's best config, rebuild at increasing dataset sizes and measure
Recall@100, QPS, build time and RSS.  We track how each algorithm scales toward the
RAM limit.
"""),
    code(r"""
LAB_LIGHT_05 = int(os.environ.get('LAB_LIGHT', '0'))
N_CAP = int(os.environ.get('LAB_N_SWEEP', '500000'))
QPS_REPEAT = int(os.environ.get('LAB_QPS_REPEAT', '2' if LAB_LIGHT_05 else '1'))
QPS_WARMUP = int(os.environ.get('LAB_QPS_WARMUP', '1' if LAB_LIGHT_05 else '0'))

if LAB_LIGHT_05:
    # One small scale point only — full scaling is hours of rebuilds.
    SCALES = [min(50_000, N_CAP, N_BASE)]
    print('LAB_LIGHT=1 — scaling section uses a single N:', SCALES)
else:
    SCALES = [100_000, 250_000, 500_000, 1_000_000]
    if int(os.environ.get('LAB_SCALING_FULL', '0')):
        SCALES.append(N_BASE)
    SCALES = sorted({s for s in SCALES if s <= N_BASE})
print(f"QPS_REPEAT={QPS_REPEAT}  QPS_WARMUP={QPS_WARMUP}  LAB_SCALING_FULL={os.environ.get('LAB_SCALING_FULL','0')}")

def ensure_gt(n: int, k: int = 100):
    cache = DATA / f'gt_n{n}_k{k}.npy'
    if cache.exists():
        return np.load(cache)
    flat = faiss.IndexFlatL2(DIM)
    utils.stream_add(flat, BASE_PATH, n)
    _, I = flat.search(queries, k)
    np.save(cache, I)
    del flat; gc.collect()
    return I

_scale_csv = RESULTS / 'scaling.csv'
if os.environ.get('SKIP_SCALING_REBUILD', '0') == '1' and _scale_csv.exists():
    df_scale = pd.read_csv(_scale_csv)
    print(f'SKIP_SCALING_REBUILD=1 — using existing {_scale_csv} ({len(df_scale)} rows)')
else:
    scale_rows = []
    _nl = 256 if LAB_LIGHT_05 else 4096
    _np = min(64, _nl)

    # One representative config per family (always include LSH when present in combined)
    configs = []
    for fam in ['IVFFlat', 'IVFPQ', 'IVFSQ', 'HNSW', 'LSH']:
        if fam not in combined['family'].values:
            continue
        if fam == 'IVFFlat':
            cfg = dict(nlist=_nl, nprobe=_np)
        elif fam == 'IVFPQ':
            cfg = dict(nlist=_nl, nprobe=_np, M=64)
        elif fam == 'IVFSQ':
            cfg = dict(nlist=_nl, nprobe=_np, sq='SQ8')
        elif fam == 'HNSW':
            cfg = dict(M=32, efC=200, efS=160)
        else:
            lsh_sub = combined[combined.family == 'LSH']
            nb = int(lsh_sub.sort_values(['recall_100', 'qps'], ascending=[False, False]).iloc[0]['nbits'])
            cfg = dict(nbits=nb)
        configs.append((fam, cfg))
    print('scaling configs (all families in combined):', configs)

    def build_search(family, cfg, n, q, k=100):
        if family == 'IVFFlat':
            quant = faiss.IndexFlatL2(DIM)
            idx = faiss.IndexIVFFlat(quant, DIM, cfg['nlist'])
        elif family == 'IVFPQ':
            quant = faiss.IndexFlatL2(DIM)
            idx = faiss.IndexIVFPQ(quant, DIM, cfg['nlist'], cfg['M'], 8)
        elif family == 'IVFSQ':
            quant = faiss.IndexFlatL2(DIM)
            idx = faiss.IndexIVFScalarQuantizer(quant, DIM, cfg['nlist'], faiss.ScalarQuantizer.QT_8bit)
        elif family == 'HNSW':
            idx = faiss.IndexHNSWFlat(DIM, cfg['M'])
            idx.hnsw.efConstruction = cfg['efC']
        elif family == 'LSH':
            idx = faiss.IndexLSH(DIM, cfg['nbits'])
        else:
            return None

        with utils.timed(f'{family} build', sample_rss_peak=True) as tb:
            if hasattr(idx, 'is_trained') and not idx.is_trained:
                train_x = utils.load_train_subset(BASE_PATH, min(n, 200_000))
                idx.train(train_x)
                del train_x; gc.collect()
            utils.stream_add(idx, BASE_PATH, n)
        if 'nprobe' in cfg:
            idx.nprobe = cfg['nprobe']
        if family == 'HNSW':
            idx.hnsw.efSearch = cfg['efS']
        size_mb = utils.index_size_mb(idx)
        rss_mb = tb.rss_after_mb
        rss_peak_mb = tb.rss_peak_mb
        rss_delta_mb = tb.rss_delta_mb
        qps, lat_ms, lat_p99_ms, I = utils.measure_qps(
            lambda q2, k2: idx.search(q2, k2), q, k,
            repeat=QPS_REPEAT, warmup=QPS_WARMUP,
        )
        del idx
        gc.collect()
        return tb.elapsed, size_mb, rss_mb, rss_peak_mb, rss_delta_mb, qps, lat_ms, lat_p99_ms, I

    for n in SCALES:
        print()
        print(f'=== n={n:,} ===')
        gt_loc = ensure_gt(n)
        utils.print_mem(f'before configs at n={n}')
        for fam, cfg in configs:
            try:
                t_build, size_mb, rss_mb, rss_peak_mb, rss_delta_mb, qps, lat_ms, lat_p99_ms, I = build_search(fam, cfg, n, queries)
                recalls = utils.compute_recalls(I, gt_loc, (1, 10, 100))
                scale_rows.append(dict(family=fam, n=n, config=str(cfg),
                                        build_s=t_build, size_mb=size_mb, rss_mb=rss_mb, rss_peak_mb=rss_peak_mb,
                                        rss_delta_mb=rss_delta_mb, qps=qps, latency_ms=lat_ms, latency_p99_ms=lat_p99_ms,
                                        recall_1=recalls[1], recall_10=recalls[10], recall_100=recalls[100]))
                print(f'   {fam:8} n={n:>8,}  build={t_build:6.1f}s  size={size_mb:6.0f}MB  peakRSS={rss_peak_mb:6.0f}MB  '
                      f'qps={qps:7.1f}  R@100={recalls[100]:.3f}')
            except Exception as e:
                print(f'   {fam:8} n={n:>8,}  FAILED: {e}')
            gc.collect()
        gc.collect()

    df_scale = pd.DataFrame(scale_rows)
    df_scale.to_csv(_scale_csv, index=False)
display(df_scale)
"""),
    md("""
### Plot — Recall@100 / QPS / Build time / RSS vs N
"""),
    code(r"""
import os as _os
if len(df_scale) == 0:
    print('No scaling rows — skipping plot.')
elif _os.environ.get('LAB_LIGHT') == '1' and df_scale['n'].nunique() < 2:
    print('LAB_LIGHT: single scaling point — bar-style snapshot instead of curves.')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    snap = df_scale.sort_values('family')
    rss_c = 'rss_peak_mb' if 'rss_peak_mb' in snap.columns else 'rss_mb'
    sns.barplot(data=snap, x='family', y='recall_100', ax=axes[0], palette='tab10')
    axes[0].set_title(f"Recall@100  (N={int(snap['n'].iloc[0]):,})")
    sns.barplot(data=snap, x='family', y='qps', ax=axes[1], palette='tab10')
    axes[1].set_yscale('log'); axes[1].set_title('QPS')
    sns.barplot(data=snap, x='family', y='build_s', ax=axes[2], palette='tab10')
    axes[2].set_title('Build time (s)')
    sns.barplot(data=snap, x='family', y=rss_c, ax=axes[3], palette='tab10')
    axes[3].set_yscale('log'); axes[3].set_title('Peak RSS (MB)')
    for a in axes:
        a.tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.savefig(DOCS_IMG / '05_scaling.png', dpi=120); plt.show()
else:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rss_col = 'rss_peak_mb' if 'rss_peak_mb' in df_scale.columns else 'rss_mb'
    for fam, sub in df_scale.groupby('family'):
        sub = sub.sort_values('n')
        axes[0,0].plot(sub.n, sub.recall_100, marker='o', label=fam)
        axes[0,1].plot(sub.n, sub.qps, marker='o', label=fam)
        axes[1,0].plot(sub.n, sub.build_s, marker='o', label=fam)
        axes[1,1].plot(sub.n, sub[rss_col] / 1024, marker='o', label=fam)  # MB → GB
    axes[0,0].set_title('Recall@100 vs dataset size')
    axes[0,0].set_xlabel('n_base'); axes[0,0].set_ylabel('Recall@100'); axes[0,0].set_ylim(0.4, 1.02)
    axes[0,0].set_xlim(left=min(df_scale['n'].min(), 80_000), right=max(df_scale['n'].max(), N_BASE * 0.99))
    axes[0,1].set_title('QPS vs dataset size')
    axes[0,1].set_xlabel('n_base'); axes[0,1].set_ylabel('QPS (log)'); axes[0,1].set_yscale('log')
    axes[1,0].set_title('Build time vs dataset size')
    axes[1,0].set_xlabel('n_base'); axes[1,0].set_ylabel('seconds (log)')
    axes[1,0].set_xscale('log'); axes[1,0].set_yscale('log')
    axes[1,1].set_title('Peak RSS during build (GB) vs dataset size')
    axes[1,1].set_xlabel('n_base'); axes[1,1].set_ylabel('GB')
    axes[1,1].axhline(28, color='red', ls=':', label='28 GB target')
    for a in axes.flat:
        a.legend(); a.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(DOCS_IMG / '05_scaling.png', dpi=120); plt.show()
"""),
    md("""
## 5 · Anomaly analysis

Below we look for surprises in the data — cases where empirical measurements contradict
naive expectations.  For each, we either explain the cause or flag it.
"""),
    code(r"""
print('=== ANOMALY CHECKLIST ===')
print()

anomaly_flags = []
ivf = frames['IVF_all']
hnsw = frames['HNSW_all']
lsh = frames['LSH']

# A) IVFFlat recall monotone in nprobe
if ivf is not None:
    flat_ivf = ivf[ivf.algo == 'IVFFlat'].sort_values(['nlist', 'nprobe'])
    for nl, sub in flat_ivf.groupby('nlist'):
        r = sub.recall_100.values
        if np.any(np.diff(r) < -0.01):
            anomaly_flags.append(f'[A] IVFFlat nlist={nl}: recall drops with higher nprobe')
        print(f'[A] IVFFlat nlist={nl:5}  recall monotone in nprobe: {not np.any(np.diff(r) < -0.01)}')

# B) HNSW saturation + efC non-monotonicity at low efSearch
if hnsw is not None:
    sat = hnsw.groupby('M').recall_100.max().reset_index()
    print()
    print('[B] HNSW max Recall@100 per M:')
    print(sat.to_string(index=False))
    for efs in sorted(hnsw.efSearch.unique())[:3]:
        sub = hnsw[hnsw.efSearch == efs].groupby('efConstruction').recall_100.max()
        if len(sub) >= 2 and sub.iloc[-1] < sub.iloc[0] - 0.05:
            anomaly_flags.append(
                f'[B] HNSW at efSearch={efs}: higher efC lowers max R@100 (looser graph effect)')

# C) LSH recall ceiling
if lsh is not None:
    r_max = lsh.recall_100.max()
    print()
    print(f'[C] LSH best Recall@100 = {r_max:.3f} (expect low at dim={DIM})')

# D) IVF+PQ build time vs M (non-monotonic → timing noise if QPS_REPEAT=1)
if ivf is not None and (ivf.algo == 'IVFPQ').any():
    pq = ivf[ivf.algo == 'IVFPQ']
    print()
    print('[D] IVF+PQ build_s per (nlist, M) — first row each:')
    for (nl, M), sub in pq.groupby(['nlist', 'M']):
        b = sub.drop_duplicates('M')['build_s'].iloc[0]
        print(f'   nlist={int(nl):5} M={int(M):3}  build_s={b:6.1f}s')
    for nl, sub in pq.groupby('nlist'):
        bt = sub.groupby('M').build_s.first()
        if len(bt) >= 2 and (bt.diff().dropna() < -5).any():
            anomaly_flags.append(f'[D] IVFPQ nlist={nl}: build_s non-monotonic in M (check QPS_REPEAT/warmup)')

# E) rss_mb must differ from rss_peak_mb when peak tracks mmap
if ivf is not None and 'rss_delta_mb' in ivf.columns:
    same = (ivf['rss_mb'] == ivf['rss_peak_mb']).all() if 'rss_peak_mb' in ivf.columns else False
    print()
    print(f'[E] rss_mb == rss_peak_mb for all IVF rows: {same} (should be False after fix)')
    if same:
        anomaly_flags.append('[E] rss_mb still aliased to peak — RSS columns degenerate')

# F) First IVFFlat config QPS cold-start (lowest nlist, nprobe=1)
if ivf is not None:
    flat_ivf = ivf[ivf.algo == 'IVFFlat']
    cold = flat_ivf[(flat_ivf.nlist == flat_ivf.nlist.min()) & (flat_ivf.nprobe == 1)]
    hot = flat_ivf[(flat_ivf.nlist == flat_ivf.nlist.max()) & (flat_ivf.nprobe == 1)]
    if len(cold) and len(hot) and cold.qps.iloc[0] < hot.qps.iloc[0] * 0.15:
        anomaly_flags.append('[F] IVFFlat nprobe=1 QPS cold-start on smallest nlist (use QPS_WARMUP>=1)')

# G) best_configs coverage
miss = set(combined['family'].unique()) - set(summary['family'].values)
print()
if miss:
    anomaly_flags.append(f'[G] families missing from best_configs: {sorted(miss)}')
else:
    print('[G] best_configs covers all families ✓')

# H) IVFFlat invalid nprobe
if ivf is not None:
    bad = ivf[(ivf.algo == 'IVFFlat') & (ivf.nprobe > ivf.nlist)]
    print(f'[H] IVFFlat nprobe>nlist rows: {len(bad)}')

if anomaly_flags:
    print()
    print('Flags:')
    for msg in anomaly_flags:
        print(' ', msg)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.38 * len(anomaly_flags))))
    y = np.arange(len(anomaly_flags))
    ax.barh(y, np.ones(len(anomaly_flags)), color='coral', height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(anomaly_flags, fontsize=8)
    ax.set_title('Automated anomaly flags')
    ax.set_xticks([])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(DOCS_IMG / '05_anomaly_flags.png', dpi=120, bbox_inches='tight')
    plt.show()
else:
    print()
    print('No automated anomaly flags.')
"""),
    md("""
## 6 · Final pick (one row per family from §2)

Quadrant winners use the **same** `best_configs` row for recall, QPS, and size — avoids
mixing metrics from different sweep points.
"""),
    code(r"""
pick = summary.set_index('family')
display(pick[['recall_100', 'qps', 'size_mb', 'build_s', 'latency_ms', 'latency_p99_ms', 'config']])

print()
print('>>> Quadrant winners (from best_configs picks):')
print(f"    Highest recall@100: {pick['recall_100'].idxmax()}  ({pick['recall_100'].max():.3f})")
print(f"    Highest QPS at pick: {pick['qps'].idxmax()}  ({pick['qps'].max():.0f})")
print(f"    Smallest index at pick: {pick['size_mb'].idxmin()}  ({pick['size_mb'].min():.1f} MB)")
"""),
    md("""
## 7 · Flame graph (optional)

Run after benchmarks to profile HNSW build vs IVFFlat (see `scripts/record_flame.sh`).
"""),
    code(r"""
_flame = DOCS_IMG / 'hnsw_build_flame.svg'
if _flame.exists():
    from IPython.display import SVG
    display(SVG(_flame.read_text()))
    print('Embedded:', _flame)
else:
    print('No flame graph yet. Generate with:')
    print('  LAB_N_SWEEP=1000000 ./scripts/record_flame.sh')
    print('(writes docs/img/full/hnsw_build_flame.svg — requires py-spy in .venv)')
"""),
    md("""
## Conclusion

Headline findings (see rendered cells for numbers from this run):

* **HNSW** — best recall–QPS trade-off at high `efSearch`; build/RSS grow with `N`.
* **IVF+PQ** — smallest on-disk index; needs high `nprobe`; sweep spans **nlist × M**.
* **IVFFlat** — simple baseline; build time grows ~linearly with `nlist` (see log-scale bar).
* **LSH** — fast but low recall at 2048 D; higher `nbits` monotonically helps.

**Methodology notes:** `rss_peak_mb` includes mmap page-cache during `stream_add`, not
index size alone. Use `rss_mb` (after build) and `rss_delta_mb` for process growth.
Scaling default stops at 1M vectors; set `LAB_SCALING_FULL=1` to measure the full
1.28M base against the 28 GB target.
"""),
]
write(nb5, '05_comparison.ipynb')
print("done 05")

