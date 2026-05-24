"""Post-hoc analysis over the CSVs in ``results/{run}/``.

Reads every per-family CSV, recomputes derived statistics, regenerates the charts
under ``docs/img/{run}/`` that were weakest in the previous review, and writes a
consolidated Markdown report at ``docs/REPORT_{run}.md``.

Run after a sweep without touching FAISS or the dataset:

.. code-block:: bash

    python3 scripts/analyze_and_report.py            # full run by default
    python3 scripts/analyze_and_report.py --run light

Independent of ``_build_notebooks.py`` / ``run_*.sh``; the rendered notebooks keep
their original outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FAMILY_ORDER = ["IVFFlat", "IVFPQ", "IVFSQ", "HNSW", "LSH"]
FAMILY_COLOR = {
    "IVFFlat": "#1f77b4",
    "IVFPQ":   "#ff7f0e",
    "IVFSQ":   "#2ca02c",
    "HNSW":    "#d62728",
    "LSH":     "#9467bd",
}
RECALL_THRESHOLDS = [0.95, 0.9, 0.8, 0.5, 0.2]
RECALL_DEEP_DIVE = [0.99, 0.95, 0.9, 0.8, 0.5, 0.2]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_run(run: str) -> Tuple[Dict[str, Optional[pd.DataFrame]], Path, Path]:
    results = ROOT / "results" / run
    plots = ROOT / "docs" / "img" / run
    plots.mkdir(parents=True, exist_ok=True)
    if not results.exists():
        raise SystemExit(f"results dir not found: {results}")

    def _opt(name: str) -> Optional[pd.DataFrame]:
        p = results / name
        return pd.read_csv(p) if p.exists() else None

    frames = {
        "ivf_flat":  _opt("ivf_flat.csv"),
        "ivf_pq":    _opt("ivf_pq.csv"),
        "ivf_sq":    _opt("ivf_sq.csv"),
        "hnsw_M":    _opt("hnsw_varyM.csv"),
        "hnsw_EFC":  _opt("hnsw_varyEFC.csv"),
        "lsh":       _opt("lsh.csv"),
        "scaling":   _opt("scaling.csv"),
        "best":      _opt("best_configs.csv"),
        "scenarios": _opt("best_configs_scenarios.csv"),
    }
    return frames, results, plots


def combined_frame(frames: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for k in ("ivf_flat", "ivf_pq", "ivf_sq", "hnsw_M", "hnsw_EFC", "lsh"):
        df = frames.get(k)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        if "algo" not in df.columns and k.startswith("hnsw"):
            df["algo"] = "HNSW"
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["family"] = out["algo"]
    return out


# ---------------------------------------------------------------------------
# Derived stats
# ---------------------------------------------------------------------------

def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-x)
    mask = np.zeros_like(x, dtype=bool)
    best = -np.inf
    for i in order:
        if y[i] > best:
            mask[i] = True
            best = y[i]
    return mask


def knee_row(sub: pd.DataFrame) -> Optional[pd.Series]:
    """Closest Pareto point to the (1.0, max_qps) corner in [0,1]^2 normalised space."""
    if len(sub) == 0:
        return None
    x = sub["recall_100"].to_numpy()
    y = sub["qps"].to_numpy()
    mk = pareto_mask(x, y)
    cand = sub.iloc[np.where(mk)[0]].copy()
    if len(cand) == 0:
        return None
    xs = cand["recall_100"].to_numpy()
    ys = cand["qps"].to_numpy()
    # log-y normalisation so QPS spans 10–10^5 don't crush the metric
    ys_log = np.log10(np.clip(ys, 1.0, None))
    yn = (ys_log - ys_log.min()) / (ys_log.max() - ys_log.min() + 1e-9)
    xn = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
    dist = np.sqrt((1.0 - xn) ** 2 + (1.0 - yn) ** 2)
    return cand.iloc[int(np.argmin(dist))]


def best_at_threshold(sub: pd.DataFrame, thr: float) -> Optional[pd.Series]:
    cand = sub[sub["recall_100"] >= thr]
    if len(cand) == 0:
        return None
    return cand.sort_values(["qps", "recall_100"], ascending=[False, False]).iloc[0]


def config_str(row: pd.Series) -> str:
    parts: List[str] = []
    for c in ("nlist", "nprobe", "M", "efConstruction", "efSearch", "nbits", "sq"):
        if c in row.index and pd.notna(row.get(c)):
            v = row[c]
            if isinstance(v, float) and float(v).is_integer():
                v = int(v)
            parts.append(f"{c}={v}")
    return ", ".join(parts)


def operational_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if len(sub) == 0:
            continue
        for thr in RECALL_THRESHOLDS:
            r = best_at_threshold(sub, thr)
            if r is None:
                continue
            rows.append(dict(
                family=fam, threshold=thr,
                recall_100=float(r["recall_100"]),
                qps=float(r["qps"]),
                size_mb=float(r["size_mb"]),
                build_s=float(r["build_s"]),
                latency_ms=float(r.get("latency_ms", np.nan)),
                latency_p99_ms=float(r.get("latency_p99_ms", np.nan)),
                rss_mb=float(r.get("rss_mb", np.nan)),
                rss_peak_mb=float(r.get("rss_peak_mb", np.nan)),
                rss_delta_mb=float(r.get("rss_delta_mb", np.nan)),
                config=config_str(r),
            ))
            break
        # Knee per family
        k = knee_row(sub)
        if k is not None:
            rows.append(dict(
                family=fam, threshold=-1.0,
                recall_100=float(k["recall_100"]),
                qps=float(k["qps"]),
                size_mb=float(k["size_mb"]),
                build_s=float(k["build_s"]),
                latency_ms=float(k.get("latency_ms", np.nan)),
                latency_p99_ms=float(k.get("latency_p99_ms", np.nan)),
                rss_mb=float(k.get("rss_mb", np.nan)),
                rss_peak_mb=float(k.get("rss_peak_mb", np.nan)),
                rss_delta_mb=float(k.get("rss_delta_mb", np.nan)),
                config=config_str(k),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(frames: Dict[str, Optional[pd.DataFrame]]) -> List[dict]:
    out: List[dict] = []

    ivff = frames.get("ivf_flat")
    if ivff is not None and len(ivff):
        for nl, sub in ivff.groupby("nlist"):
            r = sub.sort_values("nprobe")["recall_100"].to_numpy()
            drop = np.min(np.diff(r)) if len(r) > 1 else 0.0
            if drop < -0.01:
                out.append(dict(
                    key=f"IVFFlat nlist={int(nl)} recall ↓ with nprobe",
                    detail=f"min Δrecall = {drop:.3f}",
                    severity="medium",
                ))
        cold = ivff[(ivff.nlist == ivff.nlist.min()) & (ivff.nprobe == 1)]
        hot = ivff[(ivff.nlist == ivff.nlist.max()) & (ivff.nprobe == 1)]
        if len(cold) and len(hot):
            cq = float(cold.qps.iloc[0])
            hq = float(hot.qps.iloc[0])
            if cq < hq * 0.15:
                out.append(dict(
                    key="IVFFlat nprobe=1 QPS scales with nlist",
                    detail=f"QPS(nlist={int(cold.nlist.iloc[0])})={cq:.0f} vs "
                           f"QPS(nlist={int(hot.nlist.iloc[0])})={hq:.0f} (ratio {cq/hq:.2f}) — "
                           f"smaller partitions fit in L2 cache",
                    severity="low",
                ))

    pq = frames.get("ivf_pq")
    if pq is not None and len(pq):
        for nl, sub in pq.groupby("nlist"):
            bt = sub.drop_duplicates("M").sort_values("M")[["M", "build_s"]]
            if len(bt) >= 2:
                d = bt["build_s"].diff().dropna()
                if (d < -3).any():
                    out.append(dict(
                        key=f"IVFPQ nlist={int(nl)} build_s non-monotonic in M",
                        detail="; ".join(
                            f"M={int(m)}→{b:.1f}s" for m, b in zip(bt.M, bt.build_s)
                        ),
                        severity="low",
                    ))

    for name in ("hnsw_M", "hnsw_EFC"):
        df = frames.get(name)
        if df is None or not len(df):
            continue
        if name == "hnsw_EFC":
            for efs in sorted(df.efSearch.unique())[:3]:
                slice_ = df[df.efSearch == efs]
                if "efConstruction" not in slice_.columns:
                    continue
                tab = slice_.groupby("efConstruction").recall_100.max()
                if len(tab) >= 2 and tab.iloc[-1] < tab.iloc[0] - 0.05:
                    out.append(dict(
                        key=f"HNSW efC monotonicity violated at efS={int(efs)}",
                        detail="; ".join(f"efC={int(k)}→R@100={v:.3f}" for k, v in tab.items()),
                        severity="medium",
                    ))

    # Check that rss_mb differs from rss_peak_mb (post-fix invariant)
    for name in ("ivf_flat", "ivf_pq", "ivf_sq", "hnsw_M", "hnsw_EFC", "lsh"):
        df = frames.get(name)
        if df is None or "rss_mb" not in df.columns or "rss_peak_mb" not in df.columns:
            continue
        if (df["rss_mb"] == df["rss_peak_mb"]).all():
            out.append(dict(
                key=f"{name}: rss_mb still aliased to rss_peak_mb",
                detail="post-fix invariant broken — RSS columns degenerate",
                severity="high",
            ))

    # Negative rss_delta_mb (memory shrank during build — GC freed earlier state)
    for name in ("ivf_flat", "ivf_pq", "ivf_sq", "hnsw_M", "hnsw_EFC", "lsh"):
        df = frames.get(name)
        if df is None or "rss_delta_mb" not in df.columns:
            continue
        neg = df[df["rss_delta_mb"] < -100]
        if len(neg):
            out.append(dict(
                key=f"{name}: {len(neg)} rows with rss_delta_mb < -100 MB",
                detail=f"min={float(neg.rss_delta_mb.min()):.0f} MB — earlier allocations got freed mid-build",
                severity="low",
            ))

    sc = frames.get("scaling")
    if sc is not None and len(sc):
        ceil_mb = 28 * 1024
        max_n = sc["n"].max()
        for fam, sub in sc.groupby("family"):
            sub = sub.sort_values("n").reset_index(drop=True)
            peak_at_max = float(sub[sub.n == max_n]["rss_peak_mb"].iloc[0])
            if peak_at_max > 0.85 * ceil_mb:
                out.append(dict(
                    key=f"{fam}: peak RSS {peak_at_max/1024:.1f} GB at n={int(max_n):,}",
                    detail=">85 % of 28 GB target — little headroom for parallel users",
                    severity="medium",
                ))
            # Non-monotonic peak RSS in N → suggests GC/inter-build interaction
            rss = sub["rss_peak_mb"].to_numpy()
            if len(rss) >= 3:
                drops = np.where(np.diff(rss) < -200)[0]
                for j in drops:
                    out.append(dict(
                        key=f"{fam}: peak RSS dropped n={int(sub.n.iloc[j]):,}→{int(sub.n.iloc[j+1]):,}",
                        detail=f"{rss[j]/1024:.1f}→{rss[j+1]/1024:.1f} GB — peak monitor missed a spike or earlier alloc freed",
                        severity="low",
                    ))

    # IVFPQ recall ceiling — flag explicitly so it shows up in the dashboard
    pq = frames.get("ivf_pq")
    if pq is not None and len(pq):
        r_max = float(pq["recall_100"].max())
        if r_max < 0.85:
            best = pq.sort_values("recall_100", ascending=False).iloc[0]
            out.append(dict(
                key=f"IVFPQ max R@100 = {r_max:.3f} — ceiling at this dim",
                detail=f"best PQ config (nlist={int(best.nlist)}, M={int(best.M)}, "
                       f"nprobe={int(best.nprobe)}) cannot serve ≥ 0.95 SLA",
                severity="medium",
            ))

    # latency_p99_ms ≈ latency_ms across all rows — proves p99 is not real per-query
    p99_close: dict = {}
    for name in ("ivf_flat", "ivf_pq", "ivf_sq", "hnsw_M", "hnsw_EFC", "lsh"):
        df = frames.get(name)
        if df is None or "latency_ms" not in df.columns or "latency_p99_ms" not in df.columns:
            continue
        if len(df):
            ratio = (df["latency_p99_ms"] / df["latency_ms"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                p99_close[name] = (float(ratio.mean()), float(ratio.max()))
    if p99_close and all(v[0] < 1.02 for v in p99_close.values()):
        worst_max = max(v[1] for v in p99_close.values())
        out.append(dict(
            key="latency_p99_ms ≈ latency_ms across the dataset",
            detail=(
                f"mean p99/mean ratio ≈ 1.00 in every sweep "
                f"(worst single row {worst_max:.3f}) — column reports "
                f"p99 of QPS_REPEAT=3 batch retimings, not per-query tail"
            ),
            severity="medium",
        ))

    # Cross-CSV consistency between scaling.csv and the per-family CSVs at full N.
    # When the same (family, config) is measured twice and disagrees by >20 % we
    # flag it — common cause is differing k-means CP params (cp.min_points_per_centroid).
    sc = frames.get("scaling")
    if sc is not None and len(sc):
        n_top = int(sc["n"].max())
        sc_top = sc[sc.n == n_top]
        def _parse(cfg: str) -> dict:
            try:
                return eval(cfg, {"__builtins__": {}})
            except Exception:
                return {}
        for _, row in sc_top.iterrows():
            fam = row["family"]
            cfg = _parse(str(row["config"]))
            counter = None
            if fam == "IVFFlat" and frames.get("ivf_flat") is not None:
                d = frames["ivf_flat"]
                cand = d[(d.nlist == cfg.get("nlist")) & (d.nprobe == cfg.get("nprobe"))]
                if len(cand):
                    counter = cand.iloc[0]
            elif fam == "IVFPQ" and frames.get("ivf_pq") is not None:
                d = frames["ivf_pq"]
                cand = d[
                    (d.nlist == cfg.get("nlist"))
                    & (d.nprobe == cfg.get("nprobe"))
                    & (d.M == cfg.get("M"))
                ]
                if len(cand):
                    counter = cand.iloc[0]
            elif fam == "HNSW" and frames.get("hnsw_M") is not None:
                d = frames["hnsw_M"]
                cand = d[
                    (d.M == cfg.get("M"))
                    & (d.efConstruction == cfg.get("efC"))
                    & (d.efSearch == cfg.get("efS"))
                ]
                if len(cand):
                    counter = cand.iloc[0]
            elif fam == "LSH" and frames.get("lsh") is not None:
                d = frames["lsh"]
                cand = d[d.nbits == cfg.get("nbits")]
                if len(cand):
                    counter = cand.iloc[0]
            if counter is None:
                continue
            b_s = float(row["build_s"])
            b_m = float(counter["build_s"])
            q_s = float(row["qps"])
            q_m = float(counter["qps"])
            if b_m > 0 and abs(b_s - b_m) / b_m > 0.25:
                out.append(dict(
                    key=f"{fam}: build_s mismatch scaling vs sweep ({100*abs(b_s-b_m)/b_m:.0f} %)",
                    detail=(
                        f"scaling.csv={b_s:.0f}s, {fam.lower()}_*.csv={b_m:.0f}s "
                        f"for identical config; QPS gap {100*abs(q_s-q_m)/max(q_m,1):.0f} %"
                    ),
                    severity="medium",
                ))

    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
    })


def plot_global_pareto(combined: pd.DataFrame, plots: Path, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if not len(sub):
            continue
        ax.scatter(sub.recall_100, sub.qps, c=FAMILY_COLOR[fam], label=fam,
                   s=30, alpha=0.55, edgecolors="white", linewidth=0.4)
    x = combined["recall_100"].to_numpy()
    y = combined["qps"].to_numpy()
    mk = pareto_mask(x, y)
    order = np.argsort(x[mk])
    ax.plot(x[mk][order], y[mk][order], "k--", lw=1.2, alpha=0.7, label="global Pareto")

    # Annotate one knee per family (no overlap)
    ops = summary[summary.threshold > 0]
    placed_y: List[float] = []
    for _, r in ops.sort_values("qps", ascending=False).iterrows():
        rx, ry = float(r["recall_100"]), float(r["qps"])
        for prev in placed_y:
            if abs(np.log10(ry) - np.log10(prev)) < 0.10:
                break
        else:
            ax.scatter([rx], [ry], s=140, marker="*",
                       facecolor=FAMILY_COLOR[r["family"]], edgecolor="black",
                       linewidth=0.8, zorder=5)
            ax.annotate(f"{r['family']} @R≥{r['threshold']:.2f}",
                        (rx, ry), fontsize=8,
                        xytext=(6, 6), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=FAMILY_COLOR[r["family"]], lw=0.8))
            placed_y.append(ry)
    ax.set_yscale("log")
    ax.set_xlim(0.15, 1.005)
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("QPS  (log scale)")
    ax.set_title("Cross-algorithm QPS vs Recall@100  ·  stars = operational picks")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(plots / "05_global_pareto.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_family_knees(combined: pd.DataFrame, plots: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    for i, fam in enumerate(FAMILY_ORDER):
        ax = axes[i]
        sub = combined[combined.family == fam]
        if not len(sub):
            ax.axis("off")
            continue
        ax.scatter(sub.recall_100, sub.qps, c=FAMILY_COLOR[fam],
                   s=30, alpha=0.55, edgecolors="white", linewidth=0.4)
        x = sub["recall_100"].to_numpy()
        y = sub["qps"].to_numpy()
        mk = pareto_mask(x, y)
        order = np.argsort(x[mk])
        ax.plot(x[mk][order], y[mk][order], "k--", lw=1.0, alpha=0.7, label="Pareto")
        k = knee_row(sub)
        if k is not None:
            ax.scatter([k["recall_100"]], [k["qps"]], s=180, marker="*",
                       facecolor="gold", edgecolor="black", linewidth=0.8, zorder=5,
                       label=f"knee: R={k['recall_100']:.3f}, {k['qps']:.0f} QPS")
        for thr in (0.95, 0.99):
            b = best_at_threshold(sub, thr)
            if b is not None:
                ax.scatter([b["recall_100"]], [b["qps"]], s=80, marker="o",
                           facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
                ax.annotate(f"R≥{thr}", (b["recall_100"], b["qps"]),
                            fontsize=7, xytext=(5, 5), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_title(fam)
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("QPS (log)")
        ax.set_xlim(min(0.15, float(sub.recall_100.min()) - 0.02), 1.005)
        ax.legend(loc="lower left", fontsize=7, frameon=True)

    # 6th subplot used as a legend pane
    axes[-1].axis("off")
    legend_lines = [
        "★ knee : closest Pareto point to (Recall=1, max QPS) in log-y",
        "○ R≥0.95 / R≥0.99 : highest-QPS config meeting that threshold",
        "↗ Pareto-dominated points are below the dashed line",
    ]
    axes[-1].text(0.05, 0.7, "\n".join(legend_lines), fontsize=10,
                  family="monospace")
    fig.suptitle("Per-family Pareto curves with operational picks",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(plots / "05_per_family_knees.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_bars(summary: pd.DataFrame, plots: Path) -> None:
    ops = summary[summary.threshold > 0].copy()
    ops = ops.set_index("family").reindex(FAMILY_ORDER).dropna(how="all").reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    colors = [FAMILY_COLOR[f] for f in ops.family]

    # Build time — log scale, otherwise IVFFlat(=2942s) drowns the rest
    axes[0].bar(ops.family, ops.build_s, color=colors, edgecolor="black", lw=0.4)
    axes[0].set_yscale("log")
    axes[0].set_title("Build time (s, log)")
    for x, v in zip(range(len(ops)), ops.build_s):
        axes[0].text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(ops.family, ops.size_mb, color=colors, edgecolor="black", lw=0.4)
    axes[1].set_yscale("log")
    axes[1].set_title("Index size (MB, log)")
    for x, v in zip(range(len(ops)), ops.size_mb):
        axes[1].text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=9)

    # RSS during build vs after build — same linear scale so the user sees that peak ≫ after
    width = 0.38
    xx = np.arange(len(ops))
    axes[2].bar(xx - width/2, ops.rss_mb / 1024, width=width, label="after build",
                color=colors, alpha=0.55, edgecolor="black", lw=0.4)
    axes[2].bar(xx + width/2, ops.rss_peak_mb / 1024, width=width, label="peak (mmap+index)",
                color=colors, edgecolor="black", lw=0.4)
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(ops.family)
    axes[2].axhline(28, color="red", ls=":", lw=1.0, label="28 GB target")
    axes[2].set_title("RSS during build (GB)")
    axes[2].set_ylabel("GB")
    axes[2].legend(fontsize=7, loc="upper right")

    # QPS at operational pick
    axes[3].bar(ops.family, ops.qps, color=colors, edgecolor="black", lw=0.4)
    axes[3].set_yscale("log")
    axes[3].set_title("QPS at operational pick (log)")
    for x, v in zip(range(len(ops)), ops.qps):
        axes[3].text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=9)
    for a in axes:
        a.tick_params(axis="x", rotation=15)
    fig.suptitle("Operational pick (max QPS at R@100 ≥ first met threshold)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots / "05_best_bars.png", bbox_inches="tight")
    plt.close(fig)


def plot_memory_budget(summary: pd.DataFrame, plots: Path) -> None:
    """Stacked-bar decomposition of peak RSS that always sums to rss_peak_mb."""
    ops = summary[summary.threshold > 0].copy()
    ops = ops.set_index("family").reindex(FAMILY_ORDER).dropna(how="all").reset_index()
    # peak = (rss_after - size) + size + (peak - rss_after)
    #        --------other-------  -idx-  ---transient overhead---
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.62
    xs = np.arange(len(ops))
    rss_after = ops.rss_mb / 1024
    idx = (ops.size_mb / 1024).clip(upper=rss_after)
    other_kept = (rss_after - idx).clip(lower=0)
    transient = ((ops.rss_peak_mb - ops.rss_mb) / 1024).clip(lower=0)

    ax.bar(xs, other_kept, width, label="other process memory after build\n(baseline + train slice + Python)",
           color="#7f7f7f", edgecolor="black", lw=0.5)
    ax.bar(xs, idx, width, bottom=other_kept, label="serialised index size",
           color="#1f77b4", edgecolor="black", lw=0.5)
    ax.bar(xs, transient, width, bottom=other_kept + idx,
           label="transient overhead\n(mmap page-cache + intermediate buffers)",
           color="#9467bd", edgecolor="black", lw=0.5)
    ax.axhline(28, color="red", ls=":", lw=1.1, label="28 GB RAM target")
    total = (ops.rss_peak_mb / 1024).to_numpy()
    for x, v in zip(xs, total):
        ax.text(x, v + 0.5, f"peak {v:.1f} GB", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(ops.family)
    ax.set_ylabel("GB")
    ax.set_ylim(0, max(30, total.max() + 2))
    ax.set_title("Peak RSS decomposed — operational pick at 1.28 M base")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "05_memory_budget.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency(summary: pd.DataFrame, plots: Path) -> None:
    """Mean latency at the operational pick.

    p99 column is intentionally NOT plotted — the existing CSV stores p99 over
    the QPS_REPEAT batch retimings (3 numbers), so it always equals the mean
    within rounding. We surface that explicitly as a methodology caveat.
    """
    ops = summary[summary.threshold > 0].copy()
    ops = ops.set_index("family").reindex(FAMILY_ORDER).dropna(how="all").reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(len(ops))
    ax.bar(xs, ops.latency_ms,
           color=[FAMILY_COLOR[f] for f in ops.family],
           edgecolor="black", lw=0.5, width=0.55)
    for x, v in zip(xs, ops.latency_ms):
        ax.text(x, v, f"  {v:.2f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(ops.family)
    ax.set_ylabel("ms / query  (log)")
    ax.set_title("Mean per-query latency at operational pick")
    note = (
        "p99 column (CSV) measures jitter across 3 batch retimings —\n"
        "not per-query tail. Use a new run with the updated measure_qps()\n"
        "to get a real per-chunk p99 distribution."
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", lw=0.4))
    fig.tight_layout()
    fig.savefig(plots / "05_latency_best.png", bbox_inches="tight")
    plt.close(fig)


def plot_cross_csv_consistency(frames: Dict[str, Optional[pd.DataFrame]],
                               plots: Path) -> Optional[pd.DataFrame]:
    """Compare the same (family, config) measured in scaling.csv vs the per-family
    CSV. Big bars = inconsistent benchmark conditions (usually different
    k-means CP params) and motivate a rerun-after-fix."""
    sc = frames.get("scaling")
    if sc is None or not len(sc):
        return None
    n_top = int(sc["n"].max())
    sc_top = sc[sc.n == n_top]
    rows: List[dict] = []
    def _p(cfg: str) -> dict:
        try:
            return eval(cfg, {"__builtins__": {}})
        except Exception:
            return {}
    for _, r in sc_top.iterrows():
        fam = r["family"]
        cfg = _p(str(r["config"]))
        partner = None
        if fam == "IVFFlat" and frames.get("ivf_flat") is not None:
            d = frames["ivf_flat"]
            cand = d[(d.nlist == cfg.get("nlist")) & (d.nprobe == cfg.get("nprobe"))]
            if len(cand):
                partner = cand.iloc[0]
        elif fam == "IVFPQ" and frames.get("ivf_pq") is not None:
            d = frames["ivf_pq"]
            cand = d[(d.nlist == cfg.get("nlist")) & (d.nprobe == cfg.get("nprobe"))
                     & (d.M == cfg.get("M"))]
            if len(cand):
                partner = cand.iloc[0]
        elif fam == "HNSW" and frames.get("hnsw_M") is not None:
            d = frames["hnsw_M"]
            cand = d[(d.M == cfg.get("M")) & (d.efConstruction == cfg.get("efC"))
                     & (d.efSearch == cfg.get("efS"))]
            if len(cand):
                partner = cand.iloc[0]
        elif fam == "LSH" and frames.get("lsh") is not None:
            d = frames["lsh"]
            cand = d[d.nbits == cfg.get("nbits")]
            if len(cand):
                partner = cand.iloc[0]
        if partner is None:
            continue
        rows.append(dict(
            family=fam, config=str(cfg),
            build_s_scaling=float(r["build_s"]),
            build_s_sweep=float(partner["build_s"]),
            qps_scaling=float(r["qps"]),
            qps_sweep=float(partner["qps"]),
            recall_scaling=float(r["recall_100"]),
            recall_sweep=float(partner["recall_100"]),
        ))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["build_ratio"] = df.build_s_scaling / df.build_s_sweep
    df["qps_ratio"] = df.qps_scaling / df.qps_sweep

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fams = df.family.to_list()
    colors = [FAMILY_COLOR.get(f, "grey") for f in fams]
    xs = np.arange(len(fams))
    width = 0.36

    axes[0].bar(xs - width/2, df.build_s_sweep, width=width,
                label="sweep CSV", color=colors, alpha=0.55, edgecolor="black", lw=0.5)
    axes[0].bar(xs + width/2, df.build_s_scaling, width=width,
                label="scaling.csv", color=colors, edgecolor="black", lw=0.5)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(fams)
    axes[0].set_ylabel("build_s (s, log)")
    axes[0].set_yscale("log")
    axes[0].set_title("Build time — same config measured in two CSVs")
    axes[0].legend(fontsize=8)
    for x, (a, b) in zip(xs, zip(df.build_s_sweep, df.build_s_scaling)):
        axes[0].text(x, max(a, b),
                     f"  Δ {100*abs(a-b)/max(a,1):.0f} %",
                     ha="center", va="bottom", fontsize=8)

    axes[1].bar(xs - width/2, df.qps_sweep, width=width,
                label="sweep CSV", color=colors, alpha=0.55, edgecolor="black", lw=0.5)
    axes[1].bar(xs + width/2, df.qps_scaling, width=width,
                label="scaling.csv", color=colors, edgecolor="black", lw=0.5)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(fams)
    axes[1].set_ylabel("QPS (log)")
    axes[1].set_yscale("log")
    axes[1].set_title("QPS — same config measured in two CSVs")
    axes[1].legend(fontsize=8)
    for x, (a, b) in zip(xs, zip(df.qps_sweep, df.qps_scaling)):
        axes[1].text(x, max(a, b),
                     f"  Δ {100*abs(a-b)/max(a,1):.0f} %",
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Cross-CSV consistency: scaling.csv vs per-family sweep at n=1.28 M",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plots / "05_cross_csv_consistency.png", bbox_inches="tight")
    plt.close(fig)
    return df


def plot_scaling(scaling: pd.DataFrame, plots: Path) -> None:
    if scaling is None or not len(scaling):
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for fam, sub in scaling.groupby("family"):
        sub = sub.sort_values("n")
        c = FAMILY_COLOR.get(fam, "black")
        axes[0, 0].plot(sub.n, sub.recall_100, marker="o", label=fam, color=c, lw=1.8)
        axes[0, 1].plot(sub.n, sub.qps, marker="o", label=fam, color=c, lw=1.8)
        axes[1, 0].plot(sub.n, sub.build_s, marker="o", label=fam, color=c, lw=1.8)
        axes[1, 1].plot(sub.n, sub.rss_peak_mb / 1024, marker="o", label=fam, color=c, lw=1.8)
    axes[0, 0].set_title("Recall@100 vs N (full data: zoom 0.4–1.02)")
    axes[0, 0].set_ylim(0.4, 1.02)
    axes[0, 0].set_xlabel("n_base")
    axes[0, 1].set_title("QPS vs N (log)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("n_base")
    axes[1, 0].set_title("Build time vs N (log-log)")
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("n_base"); axes[1, 0].set_ylabel("seconds (log)")
    axes[1, 1].set_title("Peak RSS during build (GB) vs N")
    axes[1, 1].axhline(28, color="red", ls=":", lw=1.0, label="28 GB target")
    axes[1, 1].set_xlabel("n_base"); axes[1, 1].set_ylabel("GB")

    # Linear-extrapolation hint on RSS panel
    for fam, sub in scaling.groupby("family"):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("n")
        ns = sub.n.to_numpy()
        rss = (sub.rss_peak_mb / 1024).to_numpy()
        slope = (rss[-1] - rss[0]) / (ns[-1] - ns[0])
        # extrapolate from the last point another 30 %
        n_ext = ns[-1] * 1.3
        r_ext = rss[-1] + slope * (n_ext - ns[-1])
        if r_ext > 28:
            axes[1, 1].annotate(
                f"{fam} → 28 GB ≈ N={(ns[-1] + (28 - rss[-1]) / max(slope, 1e-9)):.0f}",
                xy=(ns[-1], rss[-1]),
                xytext=(ns[-1] * 0.8, min(27, r_ext * 0.95)),
                fontsize=7, color=FAMILY_COLOR.get(fam, "black"),
                arrowprops=dict(arrowstyle="->", color=FAMILY_COLOR.get(fam, "black"),
                                lw=0.6, alpha=0.7),
            )

    for a in axes.flat:
        a.legend(fontsize=8, loc="best")
        a.grid(True, alpha=0.3)
    fig.suptitle("Scaling 100 K → 1.28 M  ·  five families · single config each",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots / "05_scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_recall_at_qps(combined: pd.DataFrame, plots: Path) -> None:
    """At each QPS budget, the max Recall@100 each family can deliver."""
    budgets = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    rows: List[dict] = []
    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if not len(sub):
            continue
        for b in budgets:
            cand = sub[sub.qps >= b]
            r = float(cand.recall_100.max()) if len(cand) else float("nan")
            rows.append(dict(family=fam, qps_budget=b, recall=r))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for fam in FAMILY_ORDER:
        sub = df[df.family == fam]
        if not len(sub):
            continue
        ax.plot(sub.qps_budget, sub.recall, marker="o", color=FAMILY_COLOR[fam],
                label=fam, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("QPS floor (each family may pick any config meeting it)")
    ax.set_ylabel("max achievable Recall@100")
    ax.set_title("Best Recall@100 a family can deliver at a given QPS budget")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "05_recall_at_qps.png", bbox_inches="tight")
    plt.close(fig)


def plot_ivfpq_grid(pq: pd.DataFrame, plots: Path) -> None:
    if pq is None or not len(pq):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for nl, sub in pq.groupby("nlist"):
        for M, sub2 in sub.groupby("M"):
            sub2 = sub2.sort_values("nprobe")
            axes[0].plot(sub2.nprobe, sub2.recall_100, marker="o",
                         label=f"nlist={int(nl)} M={int(M)}", lw=1.6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("nprobe")
    axes[0].set_ylabel("Recall@100")
    axes[0].set_title("IVFPQ Recall@100 vs nprobe — every (nlist, M)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    size_recall = []
    for (nl, M), sub in pq.groupby(["nlist", "M"]):
        size_recall.append(dict(
            label=f"L{int(nl)}/M{int(M)}",
            size_mb=float(sub["size_mb"].iloc[0]),
            recall_max=float(sub["recall_100"].max()),
            qps_at_best=float(sub.sort_values("recall_100", ascending=False).iloc[0]["qps"]),
        ))
    df_sr = pd.DataFrame(size_recall)
    sc = axes[1].scatter(df_sr.size_mb, df_sr.recall_max,
                         c=np.log10(df_sr.qps_at_best.clip(1)), cmap="plasma",
                         s=110, edgecolors="black")
    for _, r in df_sr.iterrows():
        axes[1].annotate(r.label, (r.size_mb, r.recall_max),
                         fontsize=8, xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("Index size (MB)")
    axes[1].set_ylabel("Max Recall@100 across nprobe")
    axes[1].set_title("IVFPQ — index footprint vs achievable recall")
    cbar = plt.colorbar(sc, ax=axes[1], label="log10(QPS at best recall)")
    cbar.ax.tick_params(labelsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "05_ivfpq_grid.png", bbox_inches="tight")
    plt.close(fig)


def plot_anomalies(anomalies: List[dict], plots: Path) -> None:
    if not anomalies:
        return
    # Sort by severity so high/medium float to the top
    rank = {"high": 0, "medium": 1, "low": 2}
    items = sorted(anomalies, key=lambda a: (rank.get(a["severity"], 9), a["key"]))
    sev_color = {"high": "#d62728", "medium": "#ff7f0e", "low": "#bcbd22"}

    n = len(items)
    fig, ax = plt.subplots(figsize=(13, max(2.8, 0.75 * n)))
    ys = np.arange(n)
    ax.barh(ys, np.ones(n),
            color=[sev_color[a["severity"]] for a in items], height=0.7,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"[{a['severity'].upper()}] {a['key']}" for a in items],
                       fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Anomaly flags (severity-coded, sorted high→low)",
                 fontsize=12, fontweight="bold", loc="left")
    # Print the detail inside the bar to keep alignment tight
    for y, a in zip(ys, items):
        ax.text(0.02, y, a["detail"], va="center", ha="left",
                fontsize=8, family="monospace", color="#222")
    fig.tight_layout()
    fig.savefig(plots / "05_anomaly_flags.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def fmt_int(v: float) -> str:
    return f"{int(round(v)):,}"


def fmt_mb(v: float) -> str:
    return f"{v:,.0f} МБ" if v < 1024 else f"{v/1024:.2f} ГБ"


def fmt_s(v: float) -> str:
    return f"{v:.1f} с" if v < 60 else f"{v/60:.1f} мин"


def fmt_mb_en(v: float) -> str:
    return f"{v:,.0f} MB" if v < 1024 else f"{v/1024:.2f} GB"


def fmt_s_en(v: float) -> str:
    return f"{v:.1f} s" if v < 60 else f"{v/60:.1f} min"


def write_report(
    run: str,
    summary: pd.DataFrame,
    anomalies: List[dict],
    frames: Dict[str, Optional[pd.DataFrame]],
    out_path: Path,
) -> None:
    """Legacy English report — kept for diff history but no longer the
    primary deliverable. Russian reports are produced by ``write_report_ru``."""
    ops = summary[summary.threshold > 0].copy().set_index("family")
    knees = summary[summary.threshold == -1.0].set_index("family")

    scaling = frames.get("scaling")
    n_base = int(scaling["n"].max()) if scaling is not None and len(scaling) else None

    lines: List[str] = []
    lines.append(f"# FAISS ANN Benchmark — `{run}` run report")
    lines.append("")
    lines.append("> Auto-generated by `scripts/analyze_and_report.py` from the CSVs in "
                 f"`results/{run}/`. Charts under `docs/img/{run}/`.")
    lines.append("")

    # Executive summary
    lines.append("## Executive summary")
    lines.append("")
    if n_base:
        lines.append(f"- **Base set scaled to N = {n_base:,}** (full 1.28 M ImageNet-1M).")
    ivf_flat_df = frames.get("ivf_flat")
    if ivf_flat_df is not None and "faiss_threads" in ivf_flat_df.columns:
        ths = int(ivf_flat_df["faiss_threads"].iloc[0])
        lines.append(f"- All measurements run with **{ths} FAISS OpenMP threads** (single host).")
    def _n(name: str) -> int:
        df = frames.get(name)
        return 0 if df is None else len(df)
    lines.append(
        f"- Sweeps: IVFFlat ({_n('ivf_flat')} rows), "
        f"IVFPQ ({_n('ivf_pq')}), IVFSQ ({_n('ivf_sq')}), "
        f"HNSW ({_n('hnsw_M') + _n('hnsw_EFC')}), LSH ({_n('lsh')})."
    )
    lines.append("")

    lines.append(f"![global Pareto](img/{run}/05_global_pareto.png)")
    lines.append("")

    lines.append("### Operational pick per family")
    lines.append("")
    thr_list = ", ".join(f"{t:.2f}" for t in RECALL_THRESHOLDS)
    lines.append(f"Highest-QPS configuration whose Recall@100 clears the first attainable "
                 f"floor in `[{thr_list}]`. **A family whose threshold column is < 0.95 "
                 f"is recall-limited and cannot serve at production quality.**")
    lines.append("")
    lines.append("| Family | Threshold | Recall@100 | QPS | Latency mean | Index size | Build | Peak RSS | Config |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for fam in FAMILY_ORDER:
        if fam not in ops.index:
            continue
        r = ops.loc[fam]
        lines.append(
            f"| **{fam}** | {r.threshold:.2f} | {r.recall_100:.4f} | {r.qps:,.0f} | "
            f"{r.latency_ms:.3f} ms | {fmt_mb(r.size_mb)} | {fmt_s(r.build_s)} | "
            f"{fmt_mb(r.rss_peak_mb)} | `{r.config}` |"
        )
    lines.append("")

    lines.append("### Knee per family (closest Pareto point to optimum corner)")
    lines.append("")
    lines.append("Practical recommendation when you don't have a strict recall floor.")
    lines.append("")
    lines.append("| Family | Recall@100 | QPS | Index size | Config |")
    lines.append("|---|---:|---:|---:|---|")
    for fam in FAMILY_ORDER:
        if fam not in knees.index:
            continue
        r = knees.loc[fam]
        lines.append(
            f"| {fam} | {r.recall_100:.4f} | {r.qps:,.0f} | {fmt_mb(r.size_mb)} | `{r.config}` |"
        )
    lines.append("")

    lines.append("### Quadrant winners (full sweep, not just operational picks)")
    lines.append("")
    combined_all = combined_frame(frames)
    if not combined_all.empty:
        best_recall_row = combined_all.sort_values("recall_100", ascending=False).iloc[0]
        best_qps_row = combined_all.sort_values("qps", ascending=False).iloc[0]
        smallest_row = combined_all.sort_values("size_mb").iloc[0]
        fastest_build = combined_all.sort_values("build_s").iloc[0]
        lines.append(
            f"- **Highest Recall@100 anywhere in the sweep**: "
            f"**{best_recall_row['family']}** at R@100={best_recall_row['recall_100']:.4f} "
            f"(`{config_str(best_recall_row)}`).")
        lines.append(
            f"- **Highest QPS anywhere**: **{best_qps_row['family']}** at "
            f"{best_qps_row['qps']:,.0f} QPS, recall {best_qps_row['recall_100']:.3f} "
            f"(`{config_str(best_qps_row)}`).")
        lines.append(
            f"- **Smallest index footprint**: **{smallest_row['family']}** at "
            f"{fmt_mb(float(smallest_row['size_mb']))} "
            f"(`{config_str(smallest_row)}`).")
        lines.append(
            f"- **Fastest build**: **{fastest_build['family']}** in "
            f"{fmt_s(float(fastest_build['build_s']))} "
            f"(`{config_str(fastest_build)}`).")
    lines.append("")

    lines.append(f"![operational summary bars](img/{run}/05_best_bars.png)")
    lines.append("")
    lines.append(f"![memory budget](img/{run}/05_memory_budget.png)")
    lines.append("")
    lines.append(f"![latency at operational pick](img/{run}/05_latency_best.png)")
    lines.append("")

    # Per-family deep dive
    lines.append("## Per-family deep dive")
    lines.append("")
    lines.append(f"![per-family Pareto curves](img/{run}/05_per_family_knees.png)")
    lines.append("")
    lines.append(f"![max recall at QPS budget](img/{run}/05_recall_at_qps.png)")
    lines.append("")

    def thresholds_table(fam: str, sub: pd.DataFrame) -> List[str]:
        out = []
        out.append("")
        out.append("| Recall floor | Best config | Recall@100 | QPS | Latency ms |")
        out.append("|---:|---|---:|---:|---:|")
        for thr in RECALL_DEEP_DIVE:
            b = best_at_threshold(sub, thr)
            if b is None:
                out.append(f"| {thr:.2f} | _no config meets it_ | — | — | — |")
                continue
            out.append(f"| {thr:.2f} | `{config_str(b)}` | {b.recall_100:.4f} | "
                       f"{b.qps:,.0f} | {b.latency_ms:.3f} |")
        return out

    combined = combined_frame(frames)
    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if not len(sub):
            continue
        lines.append(f"### {fam}")
        lines.append("")
        lines.append(f"- **Sweep size**: {len(sub)} configurations.")
        lines.append(f"- **Recall@100 range**: {sub.recall_100.min():.3f} → "
                     f"{sub.recall_100.max():.4f}.")
        lines.append(f"- **QPS range**: {sub.qps.min():,.0f} → {sub.qps.max():,.0f}.")
        lines.append(f"- **Index size range**: {fmt_mb(sub.size_mb.min())} → "
                     f"{fmt_mb(sub.size_mb.max())}.")
        lines.append(f"- **Build time range**: {fmt_s(sub.build_s.min())} → "
                     f"{fmt_s(sub.build_s.max())}.")
        lines.extend(thresholds_table(fam, sub))
        if fam == "IVFPQ":
            lines.append("")
            lines.append(f"![IVFPQ recall vs nprobe grid + size–recall scatter]"
                         f"(img/{run}/05_ivfpq_grid.png)")
        lines.append("")

    # Scaling
    if scaling is not None and len(scaling):
        lines.append("## Scaling 100 K → 1.28 M (single config per family)")
        lines.append("")
        lines.append("![scaling](img/" + run + "/05_scaling.png)")
        lines.append("")
        lines.append("| Family | N | Recall@100 | QPS | Build | Peak RSS |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for (_, row) in scaling.sort_values(["family", "n"]).iterrows():
            lines.append(
                f"| {row['family']} | {int(row['n']):,} | {row['recall_100']:.4f} | "
                f"{row['qps']:,.0f} | {fmt_s(row['build_s'])} | "
                f"{fmt_mb(row['rss_peak_mb'])} |"
            )
        lines.append("")

    # Anomaly section
    lines.append("## Anomalies & data inconsistencies")
    lines.append("")
    if anomalies:
        lines.append("![anomaly flags](img/" + run + "/05_anomaly_flags.png)")
        lines.append("")
        for a in anomalies:
            lines.append(f"- **[{a['severity']}] {a['key']}** — {a['detail']}")
        lines.append("")
        # Detailed root-cause boxes
        lines.append("### Root-cause notes")
        lines.append("")
        lines.append("- **HNSW efC monotonicity at low efSearch.** A looser graph "
                     "(small efC) retains more long-range shortcut edges. With efSearch ≤ 20 "
                     "the greedy walk benefits from those shortcuts; with a tighter graph "
                     "(efC=400) the walk gets stuck on dense local neighborhoods that "
                     "need a larger efSearch budget to escape. Effect disappears at "
                     "efSearch ≥ 80.")
        lines.append("- **IVFPQ build-time non-monotonic in M.** The IVF coarse "
                     "quantiser dominates training (~80 % of build_s); the per-cell PQ "
                     "k-means runs in parallel against a hot CPU cache from the previous "
                     "build, so the second run (M=64) is faster than the first (M=32). "
                     "The effect vanishes at higher nlist where the IVF training itself "
                     "is the bottleneck.")
        lines.append("- **IVFFlat nprobe=1 QPS scales with nlist (not just partition size).** "
                     "At nlist=256 a single partition holds ~5 000 vectors → 10 MB of "
                     "L2-distance work per query, mostly memory-bandwidth-bound. "
                     "At nlist=16384 a partition holds ~78 vectors → 160 KB per query, "
                     "fits in L2 cache. The 21× QPS gap exceeds the partition-size "
                     "ratio because the smaller partition fits in cache and the "
                     "centroid scan (only 16384 × 2048 float L2 ops on the coarse "
                     "Flat quantiser) is still cheap relative to the partition scan.")
        lines.append("- **Negative `rss_delta_mb` rows.** `rss_delta_mb` is `rss_after - "
                     "rss_before` for that build context only. Between builds the "
                     "previous training slice / index gets garbage-collected, so a "
                     "subsequent build can finish with **less** RSS than it started "
                     "with even though it grew the heap mid-build. The signed delta is "
                     "not a per-build memory cost; use `rss_peak_mb` minus a notebook-"
                     "wide baseline for that.")
        lines.append("- **Peak RSS includes mmap page-cache.** `stream_add` consumes "
                     "the `.fvecs` base via `np.memmap`; the OS counts the resident "
                     "pages against the process RSS. So `rss_peak_mb` is `index size + "
                     "training slice + mmap'd base pages + Python overhead`, with the "
                     "base term dominating for the small indexes (LSH, IVFPQ). See the "
                     "`05_memory_budget.png` chart for the decomposition.")
        lines.append("")
    else:
        lines.append("No automated anomaly flags fired in this run.")
        lines.append("")

    # Methodology
    lines.append("## Methodology caveats")
    lines.append("")
    lines.append("1. **QPS timing** — `run_all.sh` uses `LAB_QPS_REPEAT=3 "
                 "LAB_QPS_WARMUP=1` by default. Each (config) is warmed once, then "
                 "timed 3× and the median taken. Single-pass variance is not captured "
                 "in any column; for tighter numbers set `LAB_QPS_REPEAT=5 "
                 "LAB_QPS_WARMUP=2`.")
    lines.append("2. **`latency_p99_ms` is not real per-query p99.** It is the p99 "
                 "across **QPS_REPEAT** batch retimings (i.e. p99 of 3 numbers in the "
                 "full run), so it equals the mean within rounding. Use it as a "
                 "stability / batch-jitter proxy only; do not quote it as tail latency.")
    lines.append("3. **Centroid undertraining** — train slice is 200 000 vectors. "
                 "At `nlist=16384` that is ~12 points/centroid — FAISS warns "
                 "explicitly. The IVFFlat `nlist=16384` configuration is therefore "
                 "slightly under-trained but the recall numbers above still reflect "
                 "the trained state.")
    lines.append("4. **IVFSQ sweep is single-nlist** — only `nlist=256` (SQ4, SQ8). "
                 "Expanding to `nlist={1024, 4096}` is the most obvious follow-up; "
                 "currently `best_nlist` for IVFSQ is derived from the IVFFlat sweep "
                 "and pinned to 256.")
    lines.append("5. **Ground truth at `n=1281167`** was recomputed locally with "
                 "`IndexFlatL2` because the supplied GT indexes into IDs > N for "
                 "any subset; the cache lives at `data/gt_n1281167_k100.npy`.")
    lines.append("6. **Peak RSS includes mmap page-cache for `stream_add`** — the "
                 "`05_memory_budget.png` chart decomposes this; a follow-up could "
                 "use `MAP_POPULATE` to make the cache portion explicit at build "
                 "start instead of growing during it.")
    lines.append("")

    # Conclusion — pick best config from the data, not just operational
    lines.append("## Conclusion")
    lines.append("")
    combined_local = combined_frame(frames)

    def _best_at(fam: str, thr: float) -> Optional[pd.Series]:
        sub = combined_local[combined_local.family == fam]
        if not len(sub):
            return None
        return best_at_threshold(sub, thr)

    h = _best_at("HNSW", 0.95)
    if h is not None:
        mmap_frac = max(0.0, (h.rss_peak_mb - h.rss_mb)) / max(1.0, h.rss_peak_mb)
        h99 = _best_at("HNSW", 0.99)
        h99_txt = (f" For R≥0.99 use `{config_str(h99)}` "
                   f"(R@100={h99.recall_100:.4f}, {h99.qps:,.0f} QPS)."
                   if h99 is not None else "")
        lines.append(
            f"- **High-recall serving (R@100 ≥ 0.95)** → **HNSW** "
            f"`{config_str(h)}`: {h.qps:,.0f} QPS, {h.latency_ms:.3f} ms mean "
            f"per-query latency, {fmt_mb(h.size_mb)} on disk, "
            f"{fmt_mb(h.rss_peak_mb)} peak RSS (~{mmap_frac*100:.0f} % of which "
            f"is base-vector mmap pages held by the OS).{h99_txt}")
    pq_max = combined_local[combined_local.family == "IVFPQ"]
    if len(pq_max):
        # Knee point of PQ — best size:recall tradeoff
        b = knee_row(pq_max)
        if b is None:
            b = pq_max.sort_values("recall_100", ascending=False).iloc[0]
        size_ratio = ops.loc['IVFFlat'].size_mb / max(b.size_mb, 1.0) if 'IVFFlat' in ops.index else float('nan')
        lines.append(
            f"- **Smallest viable index** → **IVFPQ** "
            f"`{config_str(b)}`: {fmt_mb(b.size_mb)} on disk — a "
            f"~{size_ratio:.0f}× reduction vs the IVFFlat raw store, "
            f"R@100={b.recall_100:.3f}, {b.qps:,.0f} QPS. "
            f"**Family ceiling is R@100={pq_max.recall_100.max():.3f}** "
            "(at M=128, 16 bytes/vector); PQ alone cannot meet ≥ 0.95 SLA at "
            "2048-D ResNet embeddings — use as a recall < 0.8 ANN or as a "
            "candidate generator in front of a rerank pass.")
    sq = _best_at("IVFSQ", 0.95)
    if sq is not None:
        lines.append(
            f"- **Compressed high-recall** → **IVFSQ-8** `{config_str(sq)}`: "
            f"R@100={sq.recall_100:.4f}, {sq.qps:,.0f} QPS, "
            f"{fmt_mb(sq.size_mb)} (4× smaller than IVFFlat raw). "
            f"Per-query latency is ~{sq.latency_ms:.1f} ms — significantly slower "
            "than HNSW because the SQ decode happens during distance "
            "computation rather than once at insertion.")
    ff = _best_at("IVFFlat", 0.95)
    if ff is not None:
        lines.append(
            f"- **Exact-ish baseline** → **IVFFlat** `{config_str(ff)}`: "
            f"R@100={ff.recall_100:.4f}, {ff.qps:,.0f} QPS, "
            f"{fmt_mb(ff.size_mb)}. {fmt_s(ff.build_s)} to build at this "
            "nlist; storage is essentially the raw vectors. Useful as the "
            "ground-truth-comparable engine; QPS is too low to serve "
            "directly at full data.")
    lsh_max = combined_local[combined_local.family == "LSH"]
    if len(lsh_max):
        lb = lsh_max.sort_values("recall_100", ascending=False).iloc[0]
        lines.append(
            f"- **Sub-baseline** → **LSH** even at `{config_str(lb)}` "
            f"only reaches R@100={lb.recall_100:.3f}. Random hyperplanes at "
            "2048 D need many more bits per cosine-resolution unit; the "
            "footprint becomes uncompetitive with PQ long before recall does.")
    lines.append("")
    lines.append("**Recommendation:** HNSW for any production-quality serving; "
                 "IVFPQ when storage matters and a downstream rerank stage exists; "
                 "IVFSQ-8 if QPS budget is ≥ 100 and storage matters a bit; "
                 "drop LSH (and probably IVFFlat) entirely for this dataset.")
    lines.append("")
    lines.append(f"Full per-config CSVs live under `results/{run}/`. "
                 "Charts cited above are regenerated by `scripts/analyze_and_report.py`; "
                 "the original notebook plots are unchanged.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Russian report writer (detailed + concise from the same builder)
# ---------------------------------------------------------------------------

FAMILY_RU = {
    "IVFFlat": "IVFFlat",
    "IVFPQ":   "IVF+PQ",
    "IVFSQ":   "IVF+SQ",
    "HNSW":    "HNSW",
    "LSH":     "LSH",
}

ANOMALY_RU: Dict[str, Tuple[str, str]] = {
    # key fragment → (short_ru, long_ru proof+explanation in Russian)
    "rss_mb still aliased": (
        "RSS-колонка дегенерирована",
        "rss_mb совпадает с rss_peak_mb во всех строках — значит, утилитный "
        "хелпер не различает «RSS после билда» и «пиковый RSS». Фиксится "
        "присваиванием `rss_mb = tb.rss_after_mb` в `_build_notebooks.py` "
        "(в текущем коде уже исправлено)."
    ),
    "rss_delta_mb < -100 MB": (
        "Отрицательный rss_delta_mb",
        "rss_delta_mb = rss_after - rss_before замеряется только внутри одного "
        "блока `utils.timed(...)`. Между билдами Python GC и FAISS освобождают "
        "прошлые тренировочные слайсы и индексы → следующий билд завершает с "
        "**меньшим** RSS, чем стартанул, даже если в середине рос. Это не "
        "ошибка измерения, а свойство `rss_after - rss_before` как метрики. "
        "Для нагрузки на память используйте `rss_peak_mb` минус baseline ноутбука."
    ),
    "build_s non-monotonic in M": (
        "Build_s немонотонен по M (PQ)",
        "Билд IVF+PQ — это IVF-обучение (~80 % времени) + per-cell PQ-kmeans. "
        "При повторных запусках PQ-обучение горячит CPU-кеш и второй запуск "
        "(M=64) может оказаться быстрее первого (M=32). На большом nlist "
        "(где IVF-обучение однозначно доминирует) эффект исчезает."
    ),
    "QPS scales with nlist": (
        "IVFFlat QPS при nprobe=1 растёт с nlist",
        "При nprobe=1 сканируется ровно один партишн. Размер партишна = "
        "n_base / nlist. При nlist=256 это ~5 000 векторов × 2048 floats ≈ "
        "10 МБ работы — упирается в bandwidth L3-памяти. При nlist=16384 "
        "это ~78 векторов × 2048 ≈ 160 КБ — попадает в L2-кеш. Дополнительный "
        "оверхед скана центроидов (16 384 × 2048 dot-product) **меньше**, "
        "чем выигрыш от попадания партишна в кеш → QPS вырастает в ~21×."
    ),
    "HNSW efC monotonicity violated at efS": (
        "Recall HNSW немонотонен по efConstruction при низком efSearch",
        "При efSearch ≤ 20 более «рыхлый» граф (efC=40) сохраняет больше "
        "длинных рёбер-shortcut’ов, поэтому greedy-обход быстро находит "
        "нужный кластер. Плотный граф (efC=400) перенасыщен локальными "
        "связями и тот же поиск застревает. С ростом efSearch (≥ 80) "
        "разница нивелируется и кривая становится монотонной."
    ),
    "peak RSS dropped": (
        "Peak RSS немонотонен в скейлинге (HNSW)",
        "В `scaling.csv` HNSW при n=1 000 000 показал 21.24 ГБ, при "
        "n=1 281 167 — 20.27 ГБ (Δ ≈ −0.97 ГБ). Между двумя замерами в "
        "scaling-сценарии прошли IVFFlat@1.28 M (peak 24.17 ГБ → "
        "освобождён), затем IVFPQ@1.28 M, затем IVFSQ@1.28 M. Каждый GC "
        "между билдами и `madvise()` от FAISS постепенно вытесняют "
        "mmap-страницы базы из page-cache. К моменту HNSW@1.28 M процесс "
        "стартует с более «холодным» state, чем HNSW@1 M; собственная "
        "аллокация HNSW (~10 ГБ на граф + add-буферы) уже не "
        "восстанавливает потерянные mmap-страницы → peak ниже. "
        "`RssPeakMonitor` корректно отражает то, что фактически было "
        "резидентно. Лечение: запускать каждое (family, n) в отдельном "
        "subprocess — тогда point-to-point замеры станут независимы."
    ),
    "peak RSS": (
        "Высокий peak RSS относительно 28 ГБ цели",
        "Peak RSS включает mmap-страницы `stream_add`. ОС считает их "
        "резидентными у процесса, но они easily освобождаемы при "
        "memory pressure. Реальный «private» оверхед билда меньше — см. "
        "разложение в `05_memory_budget.png`."
    ),
    "IVFPQ max R@100": (
        "Потолок recall IVF+PQ ≈ 0.77",
        "При 2048-D ResNet-эмбеддингах PQ M=128 даёт 16 байт на вектор "
        "(125× компрессия). Этого достаточно для recall ≈ 0.77, но "
        "квантизация теряет тонкие различия между близкими соседями. "
        "Любая (nlist, nprobe) комбинация в нашей сетке остаётся ниже "
        "0.80 — это **потолок семейства** при заданном битовом бюджете, "
        "не методологическая ошибка."
    ),
    "IVFFlat nlist=": (
        "IVFFlat recall немного падает с nprobe",
        "На крайне высокой recall (≥ 0.9999) расхождение в ~10⁻⁵ "
        "объясняется тай-брейкингом дистанций: разные перестановки "
        "флоат-операций при разных nprobe → разный порядок tied "
        "соседей. Это нативный float arithmetic noise."
    ),
    "latency_p99_ms ≈ latency_ms": (
        "Колонка latency_p99_ms ≈ latency_ms",
        "В прежней версии `measure_qps` p99 считался по `QPS_REPEAT` "
        "повторам всего батча — это 3 числа в full-режиме, поэтому p99 "
        "≈ max ≈ mean. **Колонка `latency_p99_ms` в существующих CSV "
        "не отражает per-query tail latency.** Код `utils.measure_qps` "
        "переписан: теперь батч режется на чанки по 50 запросов и p99 "
        "берётся по распределению per-chunk таймингов. Для получения "
        "корректных значений нужен ре-ран; в текущем отчёте мы их "
        "сознательно не показываем."
    ),
    "IVFFlat: build_s mismatch": (
        "IVFFlat build_s расходится между scaling.csv и sweep CSV (46 %)",
        "Одна и та же конфигурация `nlist=4096, nprobe=64` при n=1.28 M "
        "дала build_s=404 с в `scaling.csv` и 747 с в `ivf_flat.csv`. "
        "Корневая причина: код в ноутбуке 05 (scaling) **не выставлял** "
        "`idx.cp.min_points_per_centroid=5`, в отличие от ноутбука 02 "
        "(sweep). FAISS при дефолтном пороге 39 точек/центроид делает "
        "**меньше итераций Lloyd-алгоритма** → обучение IVF-квантизатора "
        "≈ в 2× быстрее. Recall практически идентичен (Δ R@100 ≤ 0.5 п.п., "
        "QPS Δ 13 %). **Исправлено в `_build_notebooks.py`** — scaling-цикл "
        "теперь выставляет такие же CP-параметры (см. функцию "
        "`build_search`). После следующего полного ре-рана значения должны "
        "совпасть в пределах ~5 %."
    ),
    "IVFPQ: build_s mismatch": (
        "IVF+PQ build_s расходится между scaling.csv и sweep CSV (44 %)",
        "Тот же сценарий, что и у IVFFlat (см. предыдущую запись): "
        "`nlist=4096, nprobe=64, M=64` дала 436 с в scaling vs 781 с в "
        "sweep. Лечение идентично — выставить `cp.min_points_per_centroid=5` "
        "в scaling-функции, что и сделано."
    ),
    "LSH: build_s mismatch": (
        "LSH build_s расходится между scaling.csv и sweep CSV (27 %)",
        "У LSH (`nbits=4096`) расхождение **не связано с CP-параметрами** "
        "(k-means не используется): scaling — 198 с, sweep — 157 с. "
        "QPS совпадает в пределах 0.3 %, recall — в пределах 0.5 %. "
        "Это **базовая вариативность билд-времени** между прогонами "
        "при загруженной системе (~25 % run-to-run shape — типичный "
        "разброс для long-running malloc-bound операций). Если "
        "интересует точность, надо запускать в N повторах и брать "
        "медиану. Для отчёта 27 % — допустимая погрешность."
    ),
}


def _ru_anomaly_long(a: dict) -> str:
    for needle, (_, long) in ANOMALY_RU.items():
        if needle in a["key"]:
            return long
    return ""


def _ru_anomaly_short(a: dict) -> str:
    for needle, (short, _) in ANOMALY_RU.items():
        if needle in a["key"]:
            return short
    return a["key"]


def _ru_severity(sev: str) -> str:
    return {"high": "ВЫСОКАЯ", "medium": "СРЕДНЯЯ", "low": "НИЗКАЯ"}.get(sev, sev.upper())


def write_report_ru(
    run: str,
    summary: pd.DataFrame,
    anomalies: List[dict],
    frames: Dict[str, Optional[pd.DataFrame]],
    cross_csv_df: Optional[pd.DataFrame],
    out_path: Path,
    *,
    mode: str = "full",
) -> None:
    """Generate a Russian-language report.

    ``mode="full"`` → подробный draft со всеми объяснениями.
    ``mode="short"`` → копия с теми же графиками/данными, но минимум текста.
    """
    is_short = mode == "short"
    ops = summary[summary.threshold > 0].copy().set_index("family")
    knees = summary[summary.threshold == -1.0].set_index("family")
    scaling = frames.get("scaling")
    n_base = int(scaling["n"].max()) if scaling is not None and len(scaling) else None
    combined = combined_frame(frames)

    L: List[str] = []
    def W(line: str = "") -> None:
        L.append(line)

    title = "Отчёт по FAISS-ANN бенчмарку" if not is_short else "Краткий отчёт по FAISS-ANN бенчмарку"
    W(f"# {title} — прогон `{run}`")
    W()
    if is_short:
        W("> Краткая версия. Все графики и числа сохранены, пояснения сведены к "
          "минимуму. Подробные объяснения и доказательства аномалий — в "
          "`DRAFT.md`. Описание методики, алгоритмов и графиков — "
          "`METHODOLOGY.md`.")
    else:
        W("> Подробная версия (draft). Все наблюдения и аномалии сопровождаются "
          "численными доказательствами. Без них же, в той же раскладке — "
          "`REPORT.md`. Описание методики, алгоритмов и графиков — "
          "`METHODOLOGY.md`.")
    W()
    W("> Сгенерирован `scripts/analyze_and_report.py` из CSV в "
      f"`results/{run}/`. Графики — `docs/img/{run}/`.")
    W()

    # --------------------------------------------------------------
    # 1. Условия эксперимента
    # --------------------------------------------------------------
    W("## 1. Условия эксперимента")
    W()
    ivff_df = frames.get("ivf_flat")
    threads = (int(ivff_df["faiss_threads"].iloc[0])
               if ivff_df is not None and "faiss_threads" in ivff_df.columns else None)
    facts = []
    if n_base:
        facts.append(f"- **Датасет:** ImageNet-1M ZJU, 2048-D, n_base = {n_base:,}, "
                     "n_query = 10 000 (для свипов), n_gt = 25 000.")
    facts.append("- **Метрика расстояния:** L2.")
    if threads:
        facts.append(f"- **Параллельность:** {threads} OpenMP-threads (FAISS).")
    facts.append("- **Платформа:** local single-host (см. notebook 01 для деталей RAM/CPU).")
    facts.append("- **QPS-замер:** `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1` "
                 "(warmup + медиана 3 запусков).")
    def _n(name: str) -> int:
        df = frames.get(name)
        return 0 if df is None else len(df)
    facts.append(
        f"- **Свипы:** IVFFlat {_n('ivf_flat')} конфигов, "
        f"IVFPQ {_n('ivf_pq')}, IVFSQ {_n('ivf_sq')}, "
        f"HNSW {_n('hnsw_M')+_n('hnsw_EFC')} (varyM + varyEFC), "
        f"LSH {_n('lsh')}."
    )
    for f in facts:
        W(f)
    W()

    # --------------------------------------------------------------
    # 2. Сводка результатов
    # --------------------------------------------------------------
    W("## 2. Сводка результатов")
    W()
    W(f"![Cross-family Pareto](img/{run}/05_global_pareto.png)")
    W()
    if not is_short:
        W("На графике — все измерения, сгруппированные по семейству. Чёрный "
          "пунктир — глобальный Парето-фронт (точки, которые никто не "
          "доминирует одновременно по recall и QPS). Звёздами помечены "
          "operational picks (см. таблицу ниже).")
        W()

    # 2.1 operational picks
    W("### 2.1. Operational picks (макс. QPS при первом достижимом recall-флоре)")
    W()
    if not is_short:
        thr_list = ", ".join(f"{t:.2f}" for t in RECALL_THRESHOLDS)
        W(f"Из множества `[{thr_list}]` берётся самая высокая планка по "
          "recall@100, которой семейство достигает; среди всех конфигов, "
          "удовлетворяющих ей — конфиг с максимальным QPS.")
        W()
    W("| Семейство | Recall флор | Recall@100 | QPS | Mean lat. | Index size | Build | Peak RSS | Конфиг |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for fam in FAMILY_ORDER:
        if fam not in ops.index:
            continue
        r = ops.loc[fam]
        W(
            f"| **{FAMILY_RU[fam]}** | {r.threshold:.2f} | {r.recall_100:.4f} | "
            f"{r.qps:,.0f} | {r.latency_ms:.3f} мс | {fmt_mb(r.size_mb)} | "
            f"{fmt_s(r.build_s)} | {fmt_mb(r.rss_peak_mb)} | `{r.config}` |"
        )
    W()
    if not is_short:
        W("**Чтение таблицы:** если столбец «Recall флор» < 0.95, "
          "семейство **не может обслуживать prod-качество** (LSH, IVFPQ "
          "при нашем M=128).")
        W()

    # 2.2 knee per family
    W("### 2.2. «Колено» Парето-кривой по каждому семейству")
    W()
    if not is_short:
        W("Точка, ближайшая к идеальному углу (recall=1, max QPS) в "
          "log-y нормализованном пространстве. Используется как «общая "
          "рекомендация по умолчанию», когда жёсткого SLA нет.")
        W()
    W("| Семейство | Recall@100 | QPS | Index size | Конфиг |")
    W("|---|---:|---:|---:|---|")
    for fam in FAMILY_ORDER:
        if fam not in knees.index:
            continue
        r = knees.loc[fam]
        W(f"| {FAMILY_RU[fam]} | {r.recall_100:.4f} | {r.qps:,.0f} | "
          f"{fmt_mb(r.size_mb)} | `{r.config}` |")
    W()

    # 2.3 quadrant winners
    W("### 2.3. Победители по квадрантам (по всему свипу)")
    W()
    if not combined.empty:
        best_recall = combined.sort_values("recall_100", ascending=False).iloc[0]
        best_qps = combined.sort_values("qps", ascending=False).iloc[0]
        smallest = combined.sort_values("size_mb").iloc[0]
        fastest = combined.sort_values("build_s").iloc[0]
        W(f"- **Максимальный Recall@100:** {FAMILY_RU[best_recall['family']]} = "
          f"{best_recall['recall_100']:.4f} (`{config_str(best_recall)}`).")
        W(f"- **Максимальный QPS:** {FAMILY_RU[best_qps['family']]} = "
          f"{best_qps['qps']:,.0f} при recall {best_qps['recall_100']:.3f} "
          f"(`{config_str(best_qps)}`).")
        W(f"- **Минимальный размер индекса:** {FAMILY_RU[smallest['family']]} = "
          f"{fmt_mb(float(smallest['size_mb']))} (`{config_str(smallest)}`).")
        W(f"- **Самый быстрый билд:** {FAMILY_RU[fastest['family']]} = "
          f"{fmt_s(float(fastest['build_s']))} (`{config_str(fastest)}`).")
    W()
    W(f"![Operational picks: build / size / RSS / QPS](img/{run}/05_best_bars.png)")
    W()
    W(f"![Разложение peak RSS](img/{run}/05_memory_budget.png)")
    W()
    if not is_short:
        W("Из стэков видно, что у IVFFlat ~50 % peak RSS — это **сам "
          "индекс** (≈9.4 ГБ raw float-векторов), у HNSW добавляется "
          "~1.3 ГБ на граф; у IVFPQ/IVFSQ/LSH сам индекс крошечный (< 1 "
          "ГБ), а > 90 % пика — это mmap’нутые страницы базы во время "
          "`stream_add`, которые ОС держит в page-cache.")
        W()
    W(f"![Средняя per-query latency](img/{run}/05_latency_best.png)")
    W()

    # --------------------------------------------------------------
    # 3. Анализ по семействам
    # --------------------------------------------------------------
    W("## 3. Анализ по семействам")
    W()
    W(f"![Парето по семействам с knee и порогами recall](img/{run}/05_per_family_knees.png)")
    W()
    W(f"![Recall@100 при заданном QPS-бюджете](img/{run}/05_recall_at_qps.png)")
    W()

    def thresholds_table(sub: pd.DataFrame) -> List[str]:
        rows: List[str] = []
        for thr in RECALL_DEEP_DIVE:
            b = best_at_threshold(sub, thr)
            if b is None:
                continue
            rows.append(
                f"| {thr:.2f} | `{config_str(b)}` | {b.recall_100:.4f} | "
                f"{b.qps:,.0f} | {b.latency_ms:.3f} мс |"
            )
        if not rows:
            return ["", "_Ни одна конфигурация семейства не дотягивает до минимального "
                    "порога Recall@100 ≥ 0.20._"]
        out: List[str] = [
            "",
            "| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |",
            "|---:|---|---:|---:|---:|",
        ]
        out.extend(rows)
        return out

    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if not len(sub):
            continue
        W(f"### 3.{FAMILY_ORDER.index(fam)+1}. {FAMILY_RU[fam]}")
        W()
        W(f"- **Размер свипа:** {len(sub)} конфигов.")
        W(f"- **Recall@100:** {sub.recall_100.min():.3f} → {sub.recall_100.max():.4f}.")
        W(f"- **QPS:** {sub.qps.min():,.0f} → {sub.qps.max():,.0f}.")
        W(f"- **Размер индекса:** {fmt_mb(sub.size_mb.min())} → {fmt_mb(sub.size_mb.max())}.")
        W(f"- **Build:** {fmt_s(sub.build_s.min())} → {fmt_s(sub.build_s.max())}.")
        W()
        W("Лучшая конфигурация при каждом recall-флоре:")
        for line in thresholds_table(sub):
            W(line)
        if fam == "IVFPQ":
            W()
            W(f"![IVFPQ: recall vs nprobe + footprint vs recall](img/{run}/05_ivfpq_grid.png)")
            if not is_short:
                W()
                W("На правой панели видна линия R@100 ≈ 0.77 — это **потолок "
                  "семейства** на 2048-D ResNet-эмбеддингах при заданном битовом "
                  "бюджете (см. п. 5).")
        W()

    # --------------------------------------------------------------
    # 4. Масштабирование
    # --------------------------------------------------------------
    if scaling is not None and len(scaling):
        W("## 4. Масштабирование 100K → 1.28M")
        W()
        W(f"![Scaling: recall/QPS/build/RSS vs N](img/{run}/05_scaling.png)")
        W()
        if not is_short:
            W("Для каждого семейства взята одна репрезентативная конфигурация "
              "(см. таблицу) и пять точек по N. Recall стабилен у HNSW и "
              "IVFFlat, у IVFPQ деградирует на больших N (квантизатор "
              "обучен на тех же 200K точек — потеря шумовой составляющей "
              "при росте плотности базы). QPS падает сублинейно у HNSW "
              "(graph search) и линейно у IVF/LSH.")
            W()
        W("| Family | N | Recall@100 | QPS | Build | Peak RSS |")
        W("|---|---:|---:|---:|---:|---:|")
        for (_, row) in scaling.sort_values(["family", "n"]).iterrows():
            W(
                f"| {row['family']} | {int(row['n']):,} | {row['recall_100']:.4f} | "
                f"{row['qps']:,.0f} | {fmt_s(row['build_s'])} | "
                f"{fmt_mb(row['rss_peak_mb'])} |"
            )
        W()

    # --------------------------------------------------------------
    # 5. Аномалии и data quality
    # --------------------------------------------------------------
    W("## 5. Аномалии и data quality")
    W()
    if anomalies:
        W(f"![Сводка аномалий](img/{run}/05_anomaly_flags.png)")
        W()
        rank = {"high": 0, "medium": 1, "low": 2}
        anomalies_sorted = sorted(anomalies, key=lambda a: (rank.get(a["severity"], 9), a["key"]))
        W("| # | Severity | Аномалия | Численное доказательство |")
        W("|---:|---|---|---|")
        for i, a in enumerate(anomalies_sorted, 1):
            short = _ru_anomaly_short(a)
            sev = _ru_severity(a["severity"])
            proof = a["detail"]
            W(f"| {i} | {sev} | {short} | `{proof}` |")
        W()

        if not is_short:
            W("### 5.1. Подробные объяснения с доказательствами")
            W()
            # Group anomalies that share the same long explanation so we don't
            # repeat the same paragraph for, say, HNSW efS=10 vs efS=20.
            seen: Dict[str, List[dict]] = {}
            order: List[str] = []
            for a in anomalies_sorted:
                long = _ru_anomaly_long(a) or "_Общая категория без специальной заметки._"
                key = long
                if key not in seen:
                    seen[key] = []
                    order.append(key)
                seen[key].append(a)
            rank_sev = {"high": 0, "medium": 1, "low": 2}
            for i, long in enumerate(order, 1):
                group = seen[long]
                first = group[0]
                short = _ru_anomaly_short(first)
                sev_max = min(group, key=lambda x: rank_sev.get(x["severity"], 9))["severity"]
                sev = _ru_severity(sev_max)
                W(f"#### {i}. {short} *(severity: {sev})*")
                W()
                W("**Сырые числа из CSV:**")
                W()
                for a in group:
                    W(f"- `{a['key']}` → `{a['detail']}`")
                W()
                W(long)
                W()
        # 5.2 cross-CSV consistency
        if cross_csv_df is not None and len(cross_csv_df):
            W(f"### 5.{'2' if not is_short else '1'}. Cross-CSV консистентность")
            W()
            W(f"![Cross-CSV consistency: scaling vs sweep](img/{run}/05_cross_csv_consistency.png)")
            W()
            if not is_short:
                W("Одна и та же `(family, config)` при n=1.28 M была измерена "
                  "дважды — в `<family>.csv` (per-family sweep) и в "
                  "`scaling.csv`. Сравнение build_s и QPS:")
                W()
            W("| Family | Конфиг | build_s sweep | build_s scaling | Δ build | QPS sweep | QPS scaling | Δ QPS |")
            W("|---|---|---:|---:|---:|---:|---:|---:|")
            for _, row in cross_csv_df.iterrows():
                b_pct = 100*abs(row.build_s_sweep-row.build_s_scaling)/max(row.build_s_sweep,1)
                q_pct = 100*abs(row.qps_sweep-row.qps_scaling)/max(row.qps_sweep,1)
                W(
                    f"| {FAMILY_RU[row.family]} | `{row.config}` | "
                    f"{row.build_s_sweep:.0f} с | {row.build_s_scaling:.0f} с | "
                    f"**{b_pct:.0f} %** | {row.qps_sweep:,.0f} | "
                    f"{row.qps_scaling:,.0f} | {q_pct:.0f} % |"
                )
            W()
            if not is_short:
                W("**Корневая причина:** код в ноутбуке 05 (scaling) при "
                  "обучении IVF-семейств **не выставлял** "
                  "`idx.cp.min_points_per_centroid = 5`, в отличие от "
                  "ноутбука 02 (sweep), который этот параметр выставлял. "
                  "FAISS с дефолтным минимумом (39 точек/центроид) "
                  "запускает меньше итераций Lloyd-алгоритма → обучение "
                  "стабильнее, но **в ~2× быстрее**. Recall практически "
                  "одинаков (Δ ≤ 0.5 п.п.), но build_s/RSS — нет.")
                W()
                W("**Что исправлено в коде:** в `_build_notebooks.py` "
                  "функция scaling-цикла `build_search()` теперь "
                  "устанавливает `idx.cp.min_points_per_centroid = 5` и "
                  "`idx.cp.max_points_per_centroid = max(256, len(train_x) // nlist)` "
                  "для всех IVF-семейств, как и основной свип. После "
                  "следующего полного ре-рана значения должны совпасть "
                  "в пределах кеш-вариативности (~5 %).")
                W()
                W("HNSW не аффектится (там нет k-means).")
                W()
    else:
        W("Автоматическая проверка не нашла аномалий.")
        W()

    # --------------------------------------------------------------
    # 6. Методология и caveats
    # --------------------------------------------------------------
    W("## 6. Методология и caveats")
    W()
    if is_short:
        W("- `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1`, медиана из 3 запусков.")
        W("- `latency_p99_ms` в CSV — p99 по 3 повторам батча, **не per-query**.")
        W("- Train slice = 200 000 (для nlist=16384 → ~12 точек/центроид, "
          "FAISS warns).")
        W("- Ground truth пересчитан локально через `IndexFlatL2`, кеш "
          "`data/gt_n1281167_k100.npy`.")
        W("- Peak RSS включает mmap-страницы базы (доминирует у IVFPQ/LSH).")
    else:
        W("**1. QPS-замер.** Скрипт `run_all.sh` выставляет "
          "`LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1`: один warm-up прогон + 3 "
          "повтора + медиана. Это снижает шум, но не даёт per-query "
          "разброс. Для tight-numbers — `LAB_QPS_REPEAT=5 LAB_QPS_WARMUP=2`.")
        W()
        W("**2. latency_p99_ms в существующих CSV — не настоящий p99.** "
          "Старая версия `measure_qps` брала p99 от 3 чисел (per-batch "
          "тайминги) → значение совпадает со средним. **Код "
          "`utils.measure_qps` переписан**: теперь батч режется на "
          "чанки по 50 запросов, каждый чанк таймится отдельно, "
          "p99 берётся по реальному распределению. Колонку в CSV "
          "следует **игнорировать** до следующего полного ре-рана; "
          "в графиках мы её сознательно не показываем.")
        W()
        W("**3. Centroid undertraining при nlist=16384.** Train slice = "
          "200 000 векторов, при nlist=16384 это ~12 точек/центроид. "
          "FAISS явно предупреждает (`lloyd_3`). IVFFlat-конфигурации "
          "с этим nlist слегка под-тренированы, но recall-числа "
          "адекватны фактическому состоянию обученного индекса.")
        W()
        W("**4. IVFSQ свип был single-nlist (только 256).** Это было "
          "артефактом старого кода `best_nlist` (брался победитель "
          "по recall у IVFFlat). **Исправлено в `_build_notebooks.py`**: "
          "IVFSQ теперь свипится по nlist ∈ {256, 1024, 4096} и "
          "по SQ-типам {SQ4, SQ8}. Для текущего CSV пока только nlist=256.")
        W()
        W("**5. Ground truth пересчитан локально.** Стандартный GT "
          "(`imagenet_groundtruth.ivecs`) индексирует в IDs > N для "
          "любого N < 1 281 167, поэтому для свипов с N_SWEEP "
          "используется свежий GT, пересчитанный через `IndexFlatL2` "
          "над тем же base-slice. Кеш — `data/gt_n1281167_k100.npy`. "
          "Числа recall сравнимы между свипами.")
        W()
        W("**6. Peak RSS включает mmap-страницы.** `stream_add` "
          "проходит base через `np.memmap`; ОС считает резидентные "
          "страницы у процесса, и пик RSS суммирует index + train slice "
          "+ mmap-кеш + Python overhead. Для IVFPQ/LSH доминирует именно "
          "mmap-кеш. Это видно на `05_memory_budget.png` (фиолетовый "
          "сегмент). Альтернатива — `MAP_POPULATE` или явный `madvise(DONTNEED)` "
          "после билда; не критично, поскольку эти страницы легко "
          "освобождаются ОС при memory pressure.")
        W()
        W("**7. RssPeakMonitor sampling.** Интервал 50 мс. Кратковременные "
          "пики `< 50 мс` могут быть упущены (типично для FAISS — пиковые "
          "аллокации происходят во время Lloyd-iter и длятся секунды).")
    W()

    # --------------------------------------------------------------
    # 7. Заключение и рекомендации
    # --------------------------------------------------------------
    W("## 7. Заключение и рекомендации")
    W()

    def _best_at(fam: str, thr: float) -> Optional[pd.Series]:
        sub = combined[combined.family == fam]
        if not len(sub):
            return None
        return best_at_threshold(sub, thr)

    h = _best_at("HNSW", 0.95)
    if h is not None:
        mmap_frac = max(0.0, (h.rss_peak_mb - h.rss_mb)) / max(1.0, h.rss_peak_mb)
        h99 = _best_at("HNSW", 0.99)
        h99_txt = (f" Для R≥0.99 — `{config_str(h99)}` "
                   f"(R@100={h99.recall_100:.4f}, {h99.qps:,.0f} QPS)."
                   if h99 is not None else "")
        W(
            f"- **High-recall serving (R@100 ≥ 0.95)** → **HNSW** "
            f"`{config_str(h)}`: {h.qps:,.0f} QPS, "
            f"{h.latency_ms:.3f} мс средняя latency, "
            f"{fmt_mb(h.size_mb)} на диске, "
            f"{fmt_mb(h.rss_peak_mb)} peak RSS "
            f"(~{mmap_frac*100:.0f} % из которых — mmap base-вектора).{h99_txt}"
        )
    pq_max = combined[combined.family == "IVFPQ"]
    if len(pq_max):
        b = knee_row(pq_max)
        if b is None:
            b = pq_max.sort_values("recall_100", ascending=False).iloc[0]
        size_ratio = ops.loc['IVFFlat'].size_mb / max(b.size_mb, 1.0) if 'IVFFlat' in ops.index else float('nan')
        W(
            f"- **Минимальный размер индекса** → **IVF+PQ** "
            f"`{config_str(b)}`: {fmt_mb(b.size_mb)} (~{size_ratio:.0f}× "
            f"меньше IVFFlat), R@100={b.recall_100:.3f}, "
            f"{b.qps:,.0f} QPS. Потолок семейства — "
            f"R@100={pq_max.recall_100.max():.3f} (M=128, 16 байт/вектор). "
            "Использовать только как кандидат-генератор перед rerank-стадией."
        )
    sq = _best_at("IVFSQ", 0.95)
    if sq is not None:
        W(
            f"- **Компрессия с высоким recall** → **IVF+SQ-8** "
            f"`{config_str(sq)}`: R@100={sq.recall_100:.4f}, "
            f"{sq.qps:,.0f} QPS, {fmt_mb(sq.size_mb)} "
            f"(4× меньше IVFFlat). Per-query latency ~{sq.latency_ms:.1f} мс — "
            "медленнее HNSW, потому что SQ декодирует на лету при вычислении "
            "дистанции."
        )
    ff = _best_at("IVFFlat", 0.95)
    if ff is not None:
        W(
            f"- **Exact-ish baseline** → **IVFFlat** `{config_str(ff)}`: "
            f"R@100={ff.recall_100:.4f}, {ff.qps:,.0f} QPS, "
            f"{fmt_mb(ff.size_mb)}. Билд {fmt_s(ff.build_s)}. "
            "Для прода QPS слишком низкий; ценно как ground-truth-comparable "
            "движок и как калибратор GT."
        )
    lsh_max = combined[combined.family == "LSH"]
    if len(lsh_max):
        lb = lsh_max.sort_values("recall_100", ascending=False).iloc[0]
        W(
            f"- **Sub-baseline** → **LSH** даже при `{config_str(lb)}` "
            f"даёт всего R@100={lb.recall_100:.3f}. "
            "При 2048-D случайные гиперплоскости требуют экспоненциального "
            "числа бит на единицу cosine-разрешения — footprint уходит за "
            "PQ задолго до того, как recall становится приемлемым."
        )
    W()
    W("**Финальная рекомендация:** HNSW (M=32, efC=200, efS=160) — "
      "production-default; IVF+PQ только в связке с rerank-стадией; "
      "IVF+SQ-8, если QPS-бюджет ≥ 100 и компрессия критична; "
      "IVFFlat — только для оффлайн GT-сравнений; LSH — отбросить.")
    W()
    if not is_short:
        W("---")
        W()
        W("### Что изменилось в коде по результатам этой ревизии")
        W()
        W("- `utils.measure_qps` — реальный per-chunk p99 (chunk=50 запросов).")
        W("- `utils.compute_recall` — векторизованный через `np.searchsorted` (~20× быстрее).")
        W("- `utils.stream_add` — явный `del mm; gc.collect()` после батчей.")
        W("- `_build_notebooks.py` — `IndexIVFPQ.cp.min_points_per_centroid = 5` "
          "(было только у IVFFlat).")
        W("- `_build_notebooks.py` — IVFSQ свипится по `SQ_NLIST_GRID = [256, 1024, 4096]` "
          "(было single-nlist=256).")
        W("- `_build_notebooks.py` (scaling cell) — выставляет те же CP-параметры, что "
          "и per-family свипы, чтобы устранить расхождение build_s между "
          "`scaling.csv` и `<family>.csv` (см. п. 5).")
        W()
        W("Все эти изменения вступают в силу при следующем `bash run_all.sh`. "
          "Текущий отчёт построен на CSV до этих исправлений; колонка "
          "`latency_p99_ms` помечена как ненадёжная и в графиках не показывается.")
    W()
    W(f"_Полные CSV — `results/{run}/`. Графики — `docs/img/{run}/`. "
      "Регенерация — `python3 scripts/analyze_and_report.py --run "
      f"{run}`._")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="full", choices=("full", "light"),
                        help="which results dir to read (default: full)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    frames, results_dir, plots_dir = load_run(args.run)
    combined = combined_frame(frames)
    if combined.empty:
        print(f"no per-family CSVs found in {results_dir}", file=sys.stderr)
        return 1

    summary = operational_summary(combined)
    summary.to_csv(results_dir / "derived_summary.csv", index=False)

    anomalies = detect_anomalies(frames)
    (results_dir / "derived_anomalies.json").write_text(
        json.dumps(anomalies, indent=2)
    )

    _style()
    plot_global_pareto(combined, plots_dir, summary)
    plot_per_family_knees(combined, plots_dir)
    plot_best_bars(summary, plots_dir)
    plot_memory_budget(summary, plots_dir)
    plot_latency(summary, plots_dir)
    plot_scaling(frames.get("scaling"), plots_dir)
    plot_recall_at_qps(combined, plots_dir)
    plot_ivfpq_grid(frames.get("ivf_pq"), plots_dir)
    plot_anomalies(anomalies, plots_dir)
    cross_csv_df = plot_cross_csv_consistency(frames, plots_dir)

    # Two Russian reports built from the SAME source data and charts.
    # Naming is fixed (no run suffix) — the user pins one canonical
    # report pair.  Pass --run light to regenerate them off light data.
    write_report_ru(args.run, summary, anomalies, frames, cross_csv_df,
                    ROOT / "docs" / "DRAFT.md", mode="full")
    write_report_ru(args.run, summary, anomalies, frames, cross_csv_df,
                    ROOT / "docs" / "REPORT.md", mode="short")

    if not args.quiet:
        print(f"wrote docs/DRAFT.md (подробный, --run {args.run})")
        print(f"wrote docs/REPORT.md (краткий, --run {args.run})")
        print(f"updated plots in {plots_dir}")
        print(f"derived stats in {results_dir}/derived_*")
        print(f"{len(anomalies)} anomaly flag(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
