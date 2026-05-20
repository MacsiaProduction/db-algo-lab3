"""Quick smoke test of the FAISS benchmark logic on synthetic data.

Verifies the core code paths in 02/03/04 work end-to-end before we invest
serious compute time on the real run.
"""

import gc
import numpy as np
import faiss

import utils

np.random.seed(0)

DIM = 128
N = 20_000
NQ = 1_000
K = 100

print(f"Synthetic dataset: N={N} dim={DIM} queries={NQ}")
xb = np.random.randn(N, DIM).astype(np.float32)
xq = np.random.randn(NQ, DIM).astype(np.float32)

# Ground truth
flat = faiss.IndexFlatL2(DIM)
flat.add(xb)
_, gt = flat.search(xq, K)
print("GT ok", gt.shape)

# --- IVFFlat ---
quant = faiss.IndexFlatL2(DIM)
idx = faiss.IndexIVFFlat(quant, DIM, 64)
idx.train(xb)
idx.add(xb)
idx.nprobe = 8
qps, lat, lat_p99, I = utils.measure_qps(lambda q, k: idx.search(q, k), xq, K, repeat=2)
r10 = utils.compute_recall(I, gt, 10)
print(f"IVFFlat: qps={qps:.0f} lat={lat:.2f}ms R@10={r10:.3f}")

# --- IVF+PQ ---
idx = faiss.IndexIVFPQ(faiss.IndexFlatL2(DIM), DIM, 64, 16, 8)
idx.train(xb)
idx.add(xb)
idx.nprobe = 8
qps, lat, lat_p99, I = utils.measure_qps(lambda q, k: idx.search(q, k), xq, K, repeat=2)
r10 = utils.compute_recall(I, gt, 10)
print(f"IVFPQ:   qps={qps:.0f} lat={lat:.2f}ms R@10={r10:.3f}")

# --- IVF+SQ ---
idx = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(DIM), DIM, 64,
                                    faiss.ScalarQuantizer.QT_8bit)
idx.train(xb)
idx.add(xb)
idx.nprobe = 8
qps, lat, lat_p99, I = utils.measure_qps(lambda q, k: idx.search(q, k), xq, K, repeat=2)
r10 = utils.compute_recall(I, gt, 10)
print(f"IVFSQ:   qps={qps:.0f} lat={lat:.2f}ms R@10={r10:.3f}")

# --- HNSW ---
idx = faiss.IndexHNSWFlat(DIM, 16)
idx.hnsw.efConstruction = 80
idx.add(xb)
idx.hnsw.efSearch = 40
qps, lat, lat_p99, I = utils.measure_qps(lambda q, k: idx.search(q, k), xq, K, repeat=2)
r10 = utils.compute_recall(I, gt, 10)
size = utils.index_size_mb(idx)
print(f"HNSW:    qps={qps:.0f} lat={lat:.2f}ms R@10={r10:.3f} size={size:.2f}MB")

# --- LSH ---
idx = faiss.IndexLSH(DIM, 256)
idx.train(xb)
idx.add(xb)
qps, lat, lat_p99, I = utils.measure_qps(lambda q, k: idx.search(q, k), xq, K, repeat=2)
r10 = utils.compute_recall(I, gt, 10)
print(f"LSH:     qps={qps:.0f} lat={lat:.2f}ms R@10={r10:.3f}")

# --- pareto frontier sanity ---
xs = np.array([0.1, 0.5, 0.9, 0.95, 0.99])
ys = np.array([10000, 5000, 200, 50, 10])
mask = utils.pareto_frontier(xs, ys)
print(f"Pareto mask: {mask}  (expect all True)")

print("\nAll smoke tests OK")
