"""Shared utilities for FAISS ANN benchmark notebooks."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import psutil


# ---------------------------------------------------------------------------
# fvecs / ivecs readers
# ---------------------------------------------------------------------------

def _count_vecs(path: str, dtype_bytes: int = 4) -> tuple[int, int]:
    """Return (n_vectors, dim) for a .fvecs/.ivecs/.bvecs file.

    Layout per vector: [int32 dim][dim * dtype_bytes payload].
    """
    size = int(os.path.getsize(path))
    with open(path, "rb") as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    # Use Python int for record — mixing np.int32 with large file sizes overflows int32.
    record = 4 + dim * dtype_bytes
    if size % record != 0:
        raise ValueError(
            f"{path} size {size} not multiple of record size {record} (dim={dim})"
        )
    return size // record, int(dim)


def read_fvecs(path: str, count: Optional[int] = None, offset: int = 0,
               mmap: bool = False) -> np.ndarray:
    """Read float32 vectors from a .fvecs file.

    Args:
        count: optional limit on number of vectors to read.
        offset: number of vectors to skip from the start.
        mmap: if True use np.memmap (does NOT load into RAM).

    Returns float32 array shape (n, dim).
    """
    n_total, dim = _count_vecs(path, dtype_bytes=4)
    if count is None:
        count = n_total - offset
    count = min(count, n_total - offset)

    record_floats = 1 + dim  # 1 int32 dim header + dim floats

    if mmap:
        arr = np.memmap(path, dtype=np.float32, mode="r",
                        offset=offset * record_floats * 4,
                        shape=(count, record_floats))
        # First column is dim header reinterpreted as float; discard.
        return arr[:, 1:]
    # Eager read
    with open(path, "rb") as f:
        f.seek(offset * record_floats * 4)
        raw = np.fromfile(f, dtype=np.float32, count=count * record_floats)
    raw = raw.reshape(count, record_floats)
    return np.ascontiguousarray(raw[:, 1:])


def stream_add(index, base_path: str, n: int, batch: Optional[int] = None,
               progress: bool = True, dtype=np.float32) -> int:
    """Add `n` vectors from a .fvecs file to a FAISS index in batches.

    Avoids materializing the whole base as a Python array (saves ~10.5 GB at
    full size for ImageNet-1M).  Each batch is a fresh contiguous float32 copy
    that is dropped before the next one is read.

    Returns the number of vectors added.
    """
    if batch is None:
        batch = int(os.environ.get("LAB_BATCH_ADD", "50000"))
    if batch <= 0:
        batch = 50000
    mm = read_fvecs(base_path, mmap=True)
    n_total = mm.shape[0]
    n = min(int(n), int(n_total))
    rng = range(0, n, batch)
    if progress:
        try:
            from tqdm import tqdm
            rng = tqdm(rng, desc=f"add (n={n:,}, batch={batch:,})", leave=False)
        except ImportError:
            pass
    added = 0
    for i in rng:
        end = min(i + batch, n)
        chunk = np.ascontiguousarray(mm[i:end], dtype=dtype)
        index.add(chunk)
        added += chunk.shape[0]
        del chunk
    # Release the memmap explicitly — otherwise some platforms hold the file
    # cached past the timed block, polluting the next build's rss_before.
    del mm
    gc.collect()
    return added


def load_train_subset(base_path: str, n: int, dtype=np.float32) -> np.ndarray:
    """Materialize the first `n` base vectors as a contiguous array.

    Use a small `n` (e.g. ~200k) for index *training* only; for `add()` use
    `stream_add()` instead.
    """
    mm = read_fvecs(base_path, mmap=True)
    n = min(int(n), int(mm.shape[0]))
    return np.ascontiguousarray(mm[:n], dtype=dtype)


def print_mem(label: str = "") -> None:
    """One-liner: RSS + system available, useful for spotting RAM creep."""
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    print(f"[mem{(' ' + label) if label else ''}]  "
          f"RSS={rss_gb:.2f} GB  ·  free={vm.available/1e9:.2f} GB  ·  "
          f"used%={vm.percent:.0f}")


def run_mode() -> str:
    """'light' when LAB_LIGHT is set, otherwise 'full'."""
    return "light" if int(os.environ.get("LAB_LIGHT", "0")) else "full"


def results_dir() -> Path:
    """results/light or results/full — never the flat results/ root."""
    root = Path("results")
    out = root / run_mode()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plots_dir() -> Path:
    """docs/img/light or docs/img/full."""
    root = Path("docs/img")
    out = root / run_mode()
    out.mkdir(parents=True, exist_ok=True)
    return out


def cleanup_legacy_outputs(verbose: bool = True) -> int:
    """Remove CSV/JSON/PNG accidentally written to flat results/ or docs/img/."""
    removed = 0
    for pattern in ("results/*.csv", "results/*.json", "docs/img/*.png"):
        for path in Path(".").glob(pattern):
            if path.is_file():
                if verbose:
                    print(f"  remove legacy {path}")
                path.unlink()
                removed += 1
    return removed


def read_ivecs(path: str, count: Optional[int] = None, offset: int = 0) -> np.ndarray:
    """Read int32 vectors from a .ivecs file."""
    n_total, dim = _count_vecs(path, dtype_bytes=4)
    if count is None:
        count = n_total - offset
    count = min(count, n_total - offset)
    record_ints = 1 + dim
    with open(path, "rb") as f:
        f.seek(offset * record_ints * 4)
        raw = np.fromfile(f, dtype=np.int32, count=count * record_ints)
    raw = raw.reshape(count, record_ints)
    return np.ascontiguousarray(raw[:, 1:])


# ---------------------------------------------------------------------------
# recall metrics
# ---------------------------------------------------------------------------

def compute_recall(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Recall@k averaged over queries.

    pred_ids, gt_ids: arrays of shape (n_queries, K_pred), (n_queries, K_gt).
    Uses first k columns of each. Each row's recall = |pred[:k] ∩ gt[:k]| / k.

    Uses per-row sorted-search membership: for each query we sort the gt row
    once (O(k log k)) then binary-search the prediction row against it
    (O(k log k)). Total is O(nq · k log k) but the inner loops are C — about
    20–30× faster than the previous np.intersect1d Python loop.
    """
    if pred_ids.shape[0] != gt_ids.shape[0]:
        raise ValueError("pred and gt must have same number of queries")
    nq = pred_ids.shape[0]
    p = pred_ids[:, :k]
    g = gt_ids[:, :k]
    g_sorted = np.sort(g, axis=1)
    hits = 0
    for i in range(nq):
        gi = g_sorted[i]
        pi = p[i]
        idx = np.searchsorted(gi, pi)
        idx = np.clip(idx, 0, k - 1)
        hits += int((gi[idx] == pi).sum())
    return hits / (nq * k)


def compute_recalls(pred_ids: np.ndarray, gt_ids: np.ndarray,
                    ks: Iterable[int] = (1, 10, 100)) -> dict[int, float]:
    return {k: compute_recall(pred_ids, gt_ids, k) for k in ks}


# ---------------------------------------------------------------------------
# timing / memory helpers
# ---------------------------------------------------------------------------

class RssPeakMonitor:
    """Background RSS sampler — catches native allocator spikes missed by end-of-block reads."""

    def __init__(self, interval_sec: float = 0.05):
        self._interval = max(0.02, float(interval_sec))
        self._lock = threading.Lock()
        self._peak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()

    def start(self) -> None:
        rb = self._proc.memory_info().rss
        with self._lock:
            self._peak = rb
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(self._interval):
                r = self._proc.memory_info().rss
                with self._lock:
                    if r > self._peak:
                        self._peak = r

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        r = self._proc.memory_info().rss
        with self._lock:
            if r > self._peak:
                self._peak = r
        return self._peak


@dataclass
class TimedBlock:
    label: str = ""
    elapsed: float = 0.0
    rss_before: int = 0
    rss_after: int = 0
    rss_peak: int = 0  # max(rss_before, rss_after, optional sampled peak)

    @property
    def rss_delta_mb(self) -> float:
        return (self.rss_after - self.rss_before) / 1024 / 1024

    @property
    def rss_after_mb(self) -> float:
        return self.rss_after / 1024 / 1024

    @property
    def rss_peak_mb(self) -> float:
        return self.rss_peak / 1024 / 1024


@contextmanager
def timed(
    label: str = "",
    sample_rss_peak: bool = False,
    rss_sample_interval: float = 0.05,
):
    """Time a block and optionally sample process RSS in the background.

    When ``sample_rss_peak=True``, ``rss_peak`` includes the maximum RSS seen
    between start and end (useful during FAISS train/add in native code).
    """
    proc = psutil.Process()
    gc.collect()
    rb = proc.memory_info().rss
    t0 = time.perf_counter()
    block = TimedBlock(label=label, rss_before=rb)
    mon: Optional[RssPeakMonitor] = None
    if sample_rss_peak:
        mon = RssPeakMonitor(interval_sec=rss_sample_interval)
        mon.start()
    try:
        yield block
    finally:
        block.elapsed = time.perf_counter() - t0
        ra = proc.memory_info().rss
        block.rss_after = ra
        sampled = mon.stop() if mon is not None else ra
        block.rss_peak = max(rb, ra, sampled)


def measure_qps(search_fn, queries: np.ndarray, k: int, repeat: int = 3,
                warmup: int = 1, chunk: int = 50) -> tuple[float, float, float, np.ndarray]:
    """Run search_fn(queries, k) -> (D, I) ``repeat`` times after ``warmup`` runs.

    QPS is derived from whole-batch wall time (median across repeats).
    The p99 latency is computed from **per-chunk timings within each pass**:
    queries are split into ``chunk``-sized mini-batches (default 50), each chunk
    is timed independently and converted to per-query ms. The full distribution
    across all chunks × repeats is used for `p99` — so `p99 > mean` carries real
    tail-latency signal rather than the previous near-degenerate "batch-retiming"
    estimate.

    Returns ``(median_qps, mean_latency_ms, p99_latency_ms, last_I)``.
    """
    nq = queries.shape[0]
    if nq == 0:
        return 0.0, 0.0, 0.0, np.empty((0, k), dtype=np.int64)

    for _ in range(warmup):
        D, I = search_fn(queries, k)

    chunk = max(1, min(int(chunk), nq))
    chunk_ms: list[float] = []
    full_batch_times: list[float] = []
    last_I = None
    for _ in range(repeat):
        t_pass = time.perf_counter()
        # Stream queries through search_fn chunk by chunk so we get
        # nq / chunk timings per pass, which together form the latency distribution.
        chunks_out_I: list[np.ndarray] = []
        for i in range(0, nq, chunk):
            j = min(nq, i + chunk)
            tc = time.perf_counter()
            _, Ic = search_fn(queries[i:j], k)
            elapsed = time.perf_counter() - tc
            chunk_ms.append(elapsed * 1000.0 / max(1, j - i))
            chunks_out_I.append(Ic)
        full_batch_times.append(time.perf_counter() - t_pass)
        last_I = np.concatenate(chunks_out_I, axis=0)

    full_batch_times.sort()
    median = full_batch_times[len(full_batch_times) // 2]
    qps = nq / median
    mean_lat = float(np.mean(chunk_ms))
    p99_lat = float(np.percentile(chunk_ms, 99))
    return qps, mean_lat, p99_lat, last_I


def bench_meta() -> dict:
    """Run metadata stamped into each benchmark row."""
    import faiss
    return dict(faiss_threads=int(faiss.omp_get_max_threads()))


# ---------------------------------------------------------------------------
# FAISS index size on disk (proxy for memory footprint of the structure)
# ---------------------------------------------------------------------------

def index_size_mb(index, tmp_path: str = "/tmp/_faiss_size.bin") -> float:
    import faiss
    faiss.write_index(index, tmp_path)
    sz = os.path.getsize(tmp_path) / 1024 / 1024
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return sz


# ---------------------------------------------------------------------------
# results IO
# ---------------------------------------------------------------------------

def save_results(rows: list[dict], path: str | Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    # JSON sidecar for any non-tabular use
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)


def init_results_csv(path: str | Path) -> Path:
    """Delete existing CSV (and JSON sidecar) so a sweep starts fresh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_suffix(".json")):
        if p.exists():
            p.unlink()
    return path


def append_results(rows: list[dict], path: str | Path) -> Path:
    """Append rows to an existing CSV, or create it. Returns the path."""
    import pandas as pd
    if not rows:
        return Path(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    return path


def load_results(path: str | Path):
    import pandas as pd
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Pareto frontier (maximize x, maximize y)
# ---------------------------------------------------------------------------

def pareto_frontier(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return bool mask of points on the upper-right Pareto frontier."""
    order = np.argsort(-xs)  # x desc
    mask = np.zeros_like(xs, dtype=bool)
    best_y = -np.inf
    for idx in order:
        if ys[idx] > best_y:
            mask[idx] = True
            best_y = ys[idx]
    return mask
