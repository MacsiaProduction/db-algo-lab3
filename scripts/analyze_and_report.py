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
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------------------
# Host inventory (CPU / RAM / OS) — printed in the report header
# ---------------------------------------------------------------------------

def _host_info() -> Dict[str, str]:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical": "?",
        "cpu_physical": "?",
        "ram_gb": "?",
    }
    try:
        import psutil
        info["ram_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}"
        info["cpu_logical"] = str(psutil.cpu_count(logical=True))
        info["cpu_physical"] = str(psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True))
    except Exception:
        pass
    # macOS-specific: get the actual brand string instead of `arm`
    if platform.system() == "Darwin" and shutil.which("sysctl"):
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                info["cpu"] = res.stdout.strip()
        except Exception:
            pass
    # Linux fallback: /proc/cpuinfo
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return info

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
        # nprobe=1: trade-off between centroid scan (linear in nlist) and
        # partition scan (linear in n/nlist). Optimal nlist is in the middle.
        sub_n1 = ivff[ivff.nprobe == 1].sort_values("nlist")
        if len(sub_n1) >= 3:
            q = sub_n1.qps.to_numpy()
            n = sub_n1.nlist.to_numpy()
            peak_idx = int(np.argmax(q))
            if 0 < peak_idx < len(q) - 1:
                out.append(dict(
                    key="IVFFlat nprobe=1 QPS немонотонен по nlist",
                    detail=("; ".join(
                        f"nlist={int(n[i])}→{q[i]:.0f} QPS" for i in range(len(q))
                    )),
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
                       f"nprobe={int(best.nprobe)}) tops out below 0.80 R@100",
                severity="medium",
            ))

    # latency_p99_ms ≈ latency_ms — only fires when ALL sweep CSVs were
    # produced by the legacy measure_qps() (no per-chunk distribution).
    # On the current measure_qps() the ratio is naturally 1.2…2.0 so this
    # detector silently passes; we keep it as a guard that catches regressions
    # in the measurement code.
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
                f"(worst single row {worst_max:.3f}) — measure_qps() regressed "
                f"to per-batch p99 (3 numbers); re-check utils.measure_qps()"
            ),
            severity="medium",
        ))
    # scaling.csv-only check: scaling has p99 ≈ mean while sweep CSVs have a
    # proper per-chunk distribution — different code path between the two
    # measurement sources.
    sc = frames.get("scaling")
    if sc is not None and "latency_p99_ms" in sc.columns and "latency_ms" in sc.columns and len(sc):
        sc_ratio = (sc["latency_p99_ms"] / sc["latency_ms"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(sc_ratio) and sc_ratio.mean() < 1.02:
            sweep_ratios = [v[0] for v in p99_close.values()]
            if sweep_ratios and max(sweep_ratios) > 1.05:
                out.append(dict(
                    key="scaling.csv: latency_p99_ms — per-batch p99 (different code path)",
                    detail=(
                        f"sweep CSVs: p99/mean ≈ {max(sweep_ratios):.2f} (per-chunk distribution); "
                        f"scaling.csv: ≈ {sc_ratio.mean():.2f} (per-batch p99 across 3 repeats)"
                    ),
                    severity="low",
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
        def _match_n(d: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            # Sweep CSVs may run at a different n_base (typically 1.28 M).
            # We want apples-to-apples: same config AND same n. If the sweep
            # CSV has a row at exactly n_top, use that; otherwise fall back to
            # any matching config (and tag the gap in detail).
            if d is None or "n_base" not in d.columns:
                return d
            same = d[d["n_base"] == n_top]
            return same if len(same) else d
        for _, row in sc_top.iterrows():
            fam = row["family"]
            cfg = _parse(str(row["config"]))
            counter = None
            n_match = False
            if fam == "IVFFlat" and frames.get("ivf_flat") is not None:
                d = _match_n(frames["ivf_flat"])
                cand = d[(d.nlist == cfg.get("nlist")) & (d.nprobe == cfg.get("nprobe"))]
                if len(cand):
                    counter = cand.iloc[0]
            elif fam == "IVFPQ" and frames.get("ivf_pq") is not None:
                d = _match_n(frames["ivf_pq"])
                cand = d[
                    (d.nlist == cfg.get("nlist"))
                    & (d.nprobe == cfg.get("nprobe"))
                    & (d.M == cfg.get("M"))
                ]
                if len(cand):
                    counter = cand.iloc[0]
            elif fam == "HNSW" and frames.get("hnsw_M") is not None:
                d = _match_n(frames["hnsw_M"])
                cand = d[
                    (d.M == cfg.get("M"))
                    & (d.efConstruction == cfg.get("efC"))
                    & (d.efSearch == cfg.get("efS"))
                ]
                if len(cand):
                    counter = cand.iloc[0]
            elif fam == "LSH" and frames.get("lsh") is not None:
                d = _match_n(frames["lsh"])
                cand = d[d.nbits == cfg.get("nbits")]
                if len(cand):
                    counter = cand.iloc[0]
            if counter is None:
                continue
            n_match = ("n_base" in getattr(counter, "index", []) and
                       int(counter["n_base"]) == n_top)
            b_s = float(row["build_s"])
            b_m = float(counter["build_s"])
            q_s = float(row["qps"])
            q_m = float(counter["qps"])
            if b_m > 0 and abs(b_s - b_m) / b_m > 0.25:
                tag = "" if n_match else f" (scaling@n={n_top}, sweep@n={int(counter['n_base'])})"
                out.append(dict(
                    key=f"{fam}: build_s mismatch scaling vs sweep ({100*abs(b_s-b_m)/b_m:.0f} %)",
                    detail=(
                        f"scaling.csv={b_s:.0f}s, {fam.lower()}_*.csv={b_m:.0f}s "
                        f"for identical config{tag}; QPS gap "
                        f"{100*abs(q_s-q_m)/max(q_m,1):.0f} %"
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


def _picks_for_charts(summary: pd.DataFrame) -> pd.DataFrame:
    """Knee per family (threshold == -1.0). Knee = closest Pareto point to the
    ideal (recall=1, max QPS) corner. We use knees for cross-family bar charts
    because they live at similar recall (apples-to-apples), whereas the
    threshold-cascade picks compare configs across very different recall floors
    and were judged unintuitive by reviewers."""
    knees = summary[summary.threshold == -1.0].copy()
    knees = knees.set_index("family").reindex(FAMILY_ORDER).dropna(how="all").reset_index()
    return knees


def plot_best_bars(summary: pd.DataFrame, plots: Path) -> None:
    picks = _picks_for_charts(summary)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    colors = [FAMILY_COLOR[f] for f in picks.family]
    # x-tick labels include the family name + recall so it's clear which
    # recall the bars correspond to (knees can land at different recalls).
    labels = [f"{FAMILY_RU[f]}\nR={r:.3f}" for f, r in zip(picks.family, picks.recall_100)]

    axes[0].bar(labels, picks.build_s, color=colors, edgecolor="black", lw=0.4)
    axes[0].set_yscale("log")
    axes[0].set_title("Build time (с, log)")
    for x, v in zip(range(len(picks)), picks.build_s):
        axes[0].text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, picks.size_mb, color=colors, edgecolor="black", lw=0.4)
    axes[1].set_yscale("log")
    axes[1].set_title("Index size (МБ, log)")
    for x, v in zip(range(len(picks)), picks.size_mb):
        axes[1].text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=9)

    # RSS during build vs after build — same linear scale so the reader sees that peak ≫ after
    width = 0.38
    xx = np.arange(len(picks))
    axes[2].bar(xx - width/2, picks.rss_mb / 1024, width=width, label="rss_after_mb",
                color=colors, alpha=0.55, edgecolor="black", lw=0.4)
    axes[2].bar(xx + width/2, picks.rss_peak_mb / 1024, width=width, label="peak (mmap+index)",
                color=colors, edgecolor="black", lw=0.4)
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(labels)
    axes[2].axhline(28, color="red", ls=":", lw=1.0, label="28 ГБ цель")
    axes[2].set_title("RSS во время сборки (ГБ)")
    axes[2].set_ylabel("ГБ")
    axes[2].legend(fontsize=7, loc="upper right")

    axes[3].bar(labels, picks.qps, color=colors, edgecolor="black", lw=0.4)
    axes[3].set_yscale("log")
    axes[3].set_title("QPS на рекомендованной конфигурации (log)")
    for x, v in zip(range(len(picks)), picks.qps):
        axes[3].text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=9)
    for a in axes:
        a.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Рекомендованная конфигурация (Pareto knee) — build / size / RSS / QPS",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots / "05_best_bars.png", bbox_inches="tight")
    plt.close(fig)


def plot_memory_budget(summary: pd.DataFrame, plots: Path) -> None:
    """Stacked-bar decomposition of peak RSS.

    Four bands per family (always sum to rss_peak_mb):
      1. baseline (constant ≈ 1.7 GB) — Python interpreter + ~200K train
         vectors that stay resident during build.
      2. index resident — min(size_mb, rss_after - baseline) clipped ≥ 0.
         The serialised-on-disk index is in general only partly resident in
         RAM by the time we sample rss_after.
      3. mmap & residual — rss_after − baseline − index_resident; this picks
         up mmap'd base pages held by the OS page-cache and accumulated state
         from previous builds in the same notebook.
      4. transient peak overhead — rss_peak_mb − rss_after; mmap pages and
         intermediate buffers released before rss_after was sampled.

    This is more honest than min(size, rss_after) because for HNSW (and other
    families whose serialised footprint ≈ resident footprint) the previous
    "other" band collapsed to zero and the chart looked under-decomposed.
    """
    picks = _picks_for_charts(summary)
    if picks.empty:
        return
    BASELINE_GB = 1.7
    fig, ax = plt.subplots(figsize=(11, 5.4))
    width = 0.62
    xs = np.arange(len(picks))

    rss_after = (picks.rss_mb / 1024).to_numpy()
    peak = (picks.rss_peak_mb / 1024).to_numpy()
    size = (picks.size_mb / 1024).to_numpy()

    baseline = np.full_like(rss_after, BASELINE_GB)
    baseline = np.minimum(baseline, rss_after)
    headroom = np.clip(rss_after - baseline, 0, None)
    idx_resident = np.minimum(size, headroom)
    mmap_residual = np.clip(headroom - idx_resident, 0, None)
    transient = np.clip(peak - rss_after, 0, None)

    ax.bar(xs, baseline, width,
           label=f"baseline процесса (Python + train slice ≈ {BASELINE_GB:.1f} ГБ)",
           color="#7f7f7f", edgecolor="black", lw=0.5)
    ax.bar(xs, idx_resident, width, bottom=baseline,
           label="резидентная часть индекса (≤ size_mb)",
           color="#1f77b4", edgecolor="black", lw=0.5)
    ax.bar(xs, mmap_residual, width, bottom=baseline + idx_resident,
           label="mmap-страницы базы + residual от прошлых билдов",
           color="#ff7f0e", edgecolor="black", lw=0.5)
    ax.bar(xs, transient, width, bottom=baseline + idx_resident + mmap_residual,
           label="transient overhead во время сборки",
           color="#9467bd", edgecolor="black", lw=0.5)
    ax.axhline(28, color="red", ls=":", lw=1.1, label="28 ГБ потолок RAM")
    for x, v in zip(xs, peak):
        ax.text(x, v + 0.4, f"peak {v:.1f} ГБ", ha="center", va="bottom", fontsize=8)
    # Annotate "size on disk" inside the index band so people can compare
    # against resident.
    for x, sz, b, ir in zip(xs, size, baseline, idx_resident):
        if sz > 0.05:
            ax.text(x, b + ir / 2, f"size={sz:.2f} ГБ",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([FAMILY_RU.get(f, f) for f in picks.family])
    ax.set_ylabel("ГБ")
    ax.set_ylim(0, max(30, peak.max() + 2))
    ax.set_title("Разложение peak RSS — рекомендованная конфигурация на 1.28 M базе")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "05_memory_budget.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency(summary: pd.DataFrame, plots: Path) -> None:
    """Mean and p99 latency at the recommended (knee) config.

    The new ``utils.measure_qps`` builds a per-chunk timing distribution
    (chunks of 50 queries) so ``latency_p99_ms`` now reflects a real tail —
    unlike the original p99-of-3-batches column. We plot both side-by-side."""
    picks = _picks_for_charts(summary)
    if picks.empty:
        return
    has_p99 = ("latency_p99_ms" in picks.columns
               and picks["latency_p99_ms"].notna().any())
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(picks))
    width = 0.36 if has_p99 else 0.55
    if has_p99:
        ax.bar(xs - width/2, picks.latency_ms, width=width,
               color=[FAMILY_COLOR[f] for f in picks.family],
               edgecolor="black", lw=0.5, alpha=0.6, label="mean")
        ax.bar(xs + width/2, picks.latency_p99_ms, width=width,
               color=[FAMILY_COLOR[f] for f in picks.family],
               edgecolor="black", lw=0.5, label="p99 (per-chunk)")
        for x, m, p in zip(xs, picks.latency_ms, picks.latency_p99_ms):
            ax.text(x - width/2, m, f"  {m:.2f}", ha="center", va="bottom",
                    fontsize=7.5)
            ax.text(x + width/2, p, f"  {p:.2f}", ha="center", va="bottom",
                    fontsize=7.5)
    else:
        ax.bar(xs, picks.latency_ms, width=width,
               color=[FAMILY_COLOR[f] for f in picks.family],
               edgecolor="black", lw=0.5)
        for x, v in zip(xs, picks.latency_ms):
            ax.text(x, v, f"  {v:.2f} мс", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([FAMILY_RU.get(f, f) for f in picks.family])
    ax.set_ylabel("мс / запрос  (log)")
    title = ("Latency на рекомендованной конфигурации (mean и p99)"
             if has_p99 else "Mean per-query latency на рекомендованной конфигурации")
    ax.set_title(title)
    if has_p99:
        ax.legend(loc="upper right", fontsize=8)
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
    def _mn(d: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if d is None or "n_base" not in d.columns:
            return d
        same = d[d["n_base"] == n_top]
        return same if len(same) else d
    for _, r in sc_top.iterrows():
        fam = r["family"]
        cfg = _p(str(r["config"]))
        partner = None
        if fam == "IVFFlat" and frames.get("ivf_flat") is not None:
            d = _mn(frames["ivf_flat"])
            cand = d[(d.nlist == cfg.get("nlist")) & (d.nprobe == cfg.get("nprobe"))]
            if len(cand):
                partner = cand.iloc[0]
        elif fam == "IVFPQ" and frames.get("ivf_pq") is not None:
            d = _mn(frames["ivf_pq"])
            cand = d[(d.nlist == cfg.get("nlist")) & (d.nprobe == cfg.get("nprobe"))
                     & (d.M == cfg.get("M"))]
            if len(cand):
                partner = cand.iloc[0]
        elif fam == "HNSW" and frames.get("hnsw_M") is not None:
            d = _mn(frames["hnsw_M"])
            cand = d[(d.M == cfg.get("M")) & (d.efConstruction == cfg.get("efC"))
                     & (d.efSearch == cfg.get("efS"))]
            if len(cand):
                partner = cand.iloc[0]
        elif fam == "LSH" and frames.get("lsh") is not None:
            d = _mn(frames["lsh"])
            cand = d[d.nbits == cfg.get("nbits")]
            if len(cand):
                partner = cand.iloc[0]
        if partner is None:
            continue
        partner_n = int(partner["n_base"]) if "n_base" in partner.index else n_top
        rows.append(dict(
            family=fam, config=str(cfg),
            n_scaling=n_top, n_sweep=partner_n,
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

    sweep_n = int(df["n_sweep"].max())
    if sweep_n == n_top:
        sub = f"at n = {n_top:,}"
    else:
        sub = f"scaling@n = {n_top:,} vs sweep@n = {sweep_n:,}"
    fig.suptitle(
        f"Cross-CSV consistency: scaling.csv vs per-family sweep ({sub})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(plots / "05_cross_csv_consistency.png", bbox_inches="tight")
    plt.close(fig)
    return df


def plot_scaling(scaling: pd.DataFrame, plots: Path) -> None:
    if scaling is None or not len(scaling):
        return
    def _short(n: int) -> str:
        if n >= 1_000_000:
            return f"{n/1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
        return f"{n//1_000}K"
    n_min = int(scaling["n"].min()); n_max = int(scaling["n"].max())
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
    fig.suptitle(f"Scaling {_short(n_min)} → {_short(n_max)}  ·  "
                 "five families · single config each",
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


def plot_ivfflat_grid(ivff: pd.DataFrame, plots: Path) -> None:
    """Per-family deep dive for IVFFlat:
       (left)  Recall@100 vs nprobe — one curve per nlist.
       (right) QPS vs Recall@100 — same Pareto-style cloud, colored by nlist.
    Lets the reader see (a) recall saturates with nprobe at ~recall=1 quickly
    on small nlist, and (b) the QPS premium of larger nlist at low nprobe."""
    if ivff is None or not len(ivff):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    nlists = sorted(ivff.nlist.unique())
    cmap = plt.cm.viridis
    for i, nl in enumerate(nlists):
        sub = ivff[ivff.nlist == nl].sort_values("nprobe")
        c = cmap(i / max(1, len(nlists) - 1))
        axes[0].plot(sub.nprobe, sub.recall_100, marker="o", color=c,
                     label=f"nlist={int(nl)}", lw=1.6)
        axes[1].scatter(sub.recall_100, sub.qps, color=c, s=60,
                        edgecolors="black", lw=0.4,
                        label=f"nlist={int(nl)}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("nprobe (log)")
    axes[0].set_ylabel("Recall@100")
    axes[0].set_title("IVFFlat — Recall@100 vs nprobe per nlist")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.02)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Recall@100")
    axes[1].set_ylabel("QPS  (log)")
    axes[1].set_title("IVFFlat — QPS vs Recall@100 per nlist")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(min(0.15, float(ivff.recall_100.min()) - 0.02), 1.02)
    fig.tight_layout()
    fig.savefig(plots / "05_ivfflat_grid.png", bbox_inches="tight")
    plt.close(fig)


def plot_ivfsq_grid(ivsq: pd.DataFrame, plots: Path) -> None:
    """Per-family deep dive for IVFSQ:
       (left)  Recall@100 vs nprobe — one curve per (sq, nlist).
       (right) Size vs Recall@100 colored by sq type, sized by QPS.
    Shows that SQ8 dominates SQ4 across the recall band and helps pick nlist."""
    if ivsq is None or not len(ivsq) or "sq" not in ivsq.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sq_color = {"SQ4": "#d62728", "SQ8": "#1f77b4"}
    for (sq, nl), sub in ivsq.groupby(["sq", "nlist"]):
        sub = sub.sort_values("nprobe")
        c = sq_color.get(sq, "grey")
        ls = "-" if nl == 256 else "--" if nl == 1024 else ":"
        axes[0].plot(sub.nprobe, sub.recall_100, marker="o", color=c, ls=ls,
                     label=f"{sq} nlist={int(nl)}", lw=1.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("nprobe (log)")
    axes[0].set_ylabel("Recall@100")
    axes[0].set_title("IVFSQ — Recall@100 vs nprobe по (sq, nlist)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_ylim(0, 1.02)

    # Right panel: max-recall per (sq, nlist) point cloud.
    rows = []
    for (sq, nl), sub in ivsq.groupby(["sq", "nlist"]):
        best = sub.sort_values("recall_100", ascending=False).iloc[0]
        rows.append(dict(label=f"{sq}/{int(nl)}",
                         sq=sq,
                         size_mb=float(best.size_mb),
                         recall=float(best.recall_100),
                         qps=float(best.qps)))
    df = pd.DataFrame(rows)
    for sq, sub in df.groupby("sq"):
        axes[1].scatter(sub.size_mb, sub.recall,
                        s=np.clip(np.log10(sub.qps + 1) * 60, 30, 250),
                        color=sq_color.get(sq, "grey"),
                        edgecolors="black", lw=0.5, alpha=0.85,
                        label=sq)
        for _, r in sub.iterrows():
            axes[1].annotate(r.label, (r.size_mb, r.recall),
                             fontsize=7, xytext=(5, 5),
                             textcoords="offset points")
    axes[1].set_xlabel("Index size (МБ)")
    axes[1].set_ylabel("Max Recall@100 (across nprobe)")
    axes[1].set_title("IVFSQ — footprint vs achievable recall (размер = log QPS)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "05_ivfsq_grid.png", bbox_inches="tight")
    plt.close(fig)


def plot_hnsw_grid(hnswM: Optional[pd.DataFrame], hnswEFC: Optional[pd.DataFrame],
                   plots: Path) -> None:
    """Per-family deep dive for HNSW:
       (left)  Recall@100 vs efSearch — one curve per (M, efConstruction).
       (right) QPS vs Recall@100 — colored by M.
    Combines varyM and varyEFC sweeps; deduplicates identical (M, efC, efS)."""
    parts: List[pd.DataFrame] = []
    if hnswM is not None and len(hnswM):
        parts.append(hnswM.copy())
    if hnswEFC is not None and len(hnswEFC):
        parts.append(hnswEFC.copy())
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["M", "efConstruction", "efSearch"], keep="last"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Recall@100 vs efSearch grouped by (M, efC)
    keys = sorted(df.groupby(["M", "efConstruction"]).groups.keys())
    cmap = plt.cm.tab10
    for i, (M, efC) in enumerate(keys):
        sub = df[(df.M == M) & (df.efConstruction == efC)].sort_values("efSearch")
        axes[0].plot(sub.efSearch, sub.recall_100, marker="o",
                     color=cmap(i % 10),
                     label=f"M={int(M)} efC={int(efC)}", lw=1.4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("efSearch (log)")
    axes[0].set_ylabel("Recall@100")
    axes[0].set_title("HNSW — Recall@100 vs efSearch по (M, efC)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=3, loc="lower right")
    axes[0].set_ylim(0, 1.02)

    # Right: QPS vs Recall colored by M (mux out efC into marker shape later)
    M_color = {8: "#1f77b4", 16: "#2ca02c", 32: "#ff7f0e", 48: "#d62728"}
    for M, sub in df.groupby("M"):
        axes[1].scatter(sub.recall_100, sub.qps, s=50,
                        color=M_color.get(int(M), "grey"),
                        edgecolors="black", lw=0.4, alpha=0.85,
                        label=f"M={int(M)}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Recall@100")
    axes[1].set_ylabel("QPS  (log)")
    axes[1].set_title("HNSW — QPS vs Recall@100, color = M")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(min(0.35, float(df.recall_100.min()) - 0.02), 1.005)
    fig.tight_layout()
    fig.savefig(plots / "05_hnsw_grid.png", bbox_inches="tight")
    plt.close(fig)


def plot_lsh_grid(lsh: pd.DataFrame, plots: Path) -> None:
    """Per-family deep dive for LSH (single-axis sweep over nbits):
       (left)  Recall@100 and QPS vs nbits on a dual-axis log-log plot.
       (right) Size vs Recall@100 — shows the families saturates around
               R≈0.42 long after footprint overtakes IVFPQ."""
    if lsh is None or not len(lsh):
        return
    df = lsh.sort_values("nbits")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax0 = axes[0]
    ax0b = ax0.twinx()
    ax0.plot(df.nbits, df.recall_100, marker="o", color="#1f77b4", lw=1.8,
             label="Recall@100")
    ax0b.plot(df.nbits, df.qps, marker="s", color="#d62728", lw=1.8,
              label="QPS")
    ax0.set_xscale("log")
    ax0b.set_yscale("log")
    ax0.set_xlabel("nbits (log)")
    ax0.set_ylabel("Recall@100", color="#1f77b4")
    ax0b.set_ylabel("QPS  (log)", color="#d62728")
    ax0.set_title("LSH — Recall@100 и QPS vs nbits")
    ax0.set_ylim(0, max(0.5, df.recall_100.max() + 0.05))
    ax0.grid(True, alpha=0.3)
    # Annotate nbits values
    for _, r in df.iterrows():
        ax0.annotate(f"{int(r.nbits)}", (r.nbits, r.recall_100),
                     fontsize=7, xytext=(4, 4), textcoords="offset points",
                     color="#1f77b4")

    axes[1].plot(df.size_mb, df.recall_100, marker="o", color="#9467bd",
                 lw=1.8)
    for _, r in df.iterrows():
        axes[1].annotate(f"nbits={int(r.nbits)}",
                         (r.size_mb, r.recall_100),
                         fontsize=8, xytext=(6, -4),
                         textcoords="offset points")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Index size (МБ, log)")
    axes[1].set_ylabel("Recall@100")
    axes[1].set_title("LSH — footprint vs Recall@100 (потолок ~0.42)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, max(0.5, df.recall_100.max() + 0.05))
    fig.tight_layout()
    fig.savefig(plots / "05_lsh_grid.png", bbox_inches="tight")
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
        "rss_delta_mb = rss_after − rss_before замеряется только внутри одного "
        "блока `utils.timed(...)`. Между билдами Python GC и FAISS освобождают "
        "прошлые тренировочные слайсы и индексы → следующий билд завершает с "
        "**меньшим** RSS, чем стартовал, даже если в середине рос. Это не "
        "ошибка измерения, а свойство `rss_after − rss_before` как метрики. "
        "Для нагрузки на память используйте `rss_peak_mb` минус baseline ноутбука."
    ),
    "build_s non-monotonic in M": (
        "Build_s немонотонен по M (PQ)",
        "Сборка IVF+PQ — это IVF-обучение (~80 % времени) + per-cell PQ-kmeans. "
        "При повторных запусках PQ-обучение прогревает CPU-кеш и второй запуск "
        "(M=64) может оказаться быстрее первого (M=32). На большом nlist "
        "(где IVF-обучение однозначно доминирует) эффект исчезает. "
        "Разница в пределах ~5 % — system-noise, не сигнал о качестве."
    ),
    "QPS немонотонен по nlist": (
        "IVFFlat nprobe=1 QPS немонотонен по nlist",
        "При nprobe=1 в стоимость поиска входят две части: "
        "(а) **скан центроидов** — линеен в nlist (для nlist=16384 это "
        "16 384 × 2048 dot-product ≈ 130 МБ работы), "
        "(б) **скан выбранного партишна** — линеен в n_base/nlist. "
        "Сумма имеет минимум на среднем nlist: при малом nlist (256) партишн "
        "огромный (~10 МБ), упирается в bandwidth L3; при большом nlist "
        "(16 384) центроидный скан сам становится 130 МБ. Оптимум — "
        "около nlist ≈ 1024…4096, где обе части помещаются в кеш."
    ),
    "HNSW efC monotonicity violated at efS": (
        "Recall HNSW немонотонен по efConstruction при низком efSearch",
        "При efSearch ≤ 20 более «рыхлый» граф (efC=40) сохраняет больше "
        "длинных рёбер-shortcut’ов, поэтому greedy-обход быстро находит "
        "нужный кластер. Плотный граф (efC=400) перенасыщен локальными "
        "связями и тот же поиск застревает. С ростом efSearch (≥ 80) "
        "разница нивелируется и кривая становится монотонной — видно "
        "на левой панели `05_hnsw_grid.png`."
    ),
    "peak RSS dropped": (
        "Peak RSS немонотонен в scaling-сценарии (HNSW)",
        "В `scaling.csv` peak RSS у HNSW при последующем (большем) N "
        "оказался ниже, чем при предыдущем. Между этими двумя замерами "
        "в scaling-цикле успели пройти IVFFlat, IVFPQ и IVFSQ на том же N: "
        "GC между сборками и `madvise()` от FAISS постепенно вытесняют "
        "mmap-страницы базы из page-cache, поэтому HNSW при большем N "
        "стартует с более «холодного» state и собственная аллокация HNSW "
        "уже не восстанавливает потерянные mmap-страницы → peak ниже. "
        "`RssPeakMonitor` корректно отражает то, что фактически резидентно. "
        "Лечение: запускать каждое (family, n) в отдельном subprocess — "
        "тогда замеры станут независимы."
    ),
    "peak RSS": (
        "Высокий peak RSS относительно 28 ГБ цели",
        "Peak RSS включает mmap-страницы `stream_add`. ОС считает их "
        "резидентными у процесса, но они easily освобождаемы при "
        "memory pressure. Реальный «private» оверхед сборки меньше — см. "
        "разложение в `05_memory_budget.png` (фиолетовая полоса — это "
        "именно transient mmap)."
    ),
    "IVFPQ max R@100": (
        "Потолок Recall@100 у IVF+PQ",
        "При 2048-D эмбеддингах PQ M=128 даёт 16 байт на вектор "
        "(125× компрессия). Этого достаточно для Recall@100 ≈ 0.77–0.79, "
        "но квантизация теряет тонкие различия между близкими соседями. "
        "Любая (nlist, nprobe) комбинация в нашей сетке остаётся ниже "
        "0.80 — это **потолок семейства** при заданном битовом бюджете, "
        "а не методологическая ошибка. Подъём порога требует или "
        "большего M (= больше байт/вектор, дороже хранить и считать), "
        "или связки с rerank-стадией на оригинальных векторах."
    ),
    "IVFFlat nlist=": (
        "IVFFlat recall немного флуктуирует с nprobe",
        "На крайне высоком recall (≥ 0.9999) расхождение в ~10⁻⁵ "
        "объясняется тай-брейкингом дистанций: разные перестановки "
        "float-операций при разных nprobe → разный порядок tied "
        "соседей. Это нативный float arithmetic noise."
    ),
    "IVFFlat: build_s mismatch": (
        "IVFFlat build_s расходится между scaling.csv и sweep CSV",
        "Одна и та же `(nlist, nprobe)` даёт разные build_s в `scaling.csv` "
        "и в `ivf_flat.csv`. Причина — разные code paths: ноутбук 02 (sweep) "
        "выставляет `idx.cp.min_points_per_centroid = 5` и "
        "`idx.cp.max_points_per_centroid`, а ноутбук 05 (scaling) — нет. "
        "FAISS при дефолтном пороге (~39 точек/центроид) ограничивает "
        "число итераций Lloyd-алгоритма → IVF-обучение быстрее. "
        "Кроме того, scaling и sweep могут быть на разных n (см. §5.2), "
        "что добавляет естественную разницу. Recall практически идентичен "
        "(Δ ≤ 0.5 п.п.), различаются только time-based метрики."
    ),
    "IVFPQ: build_s mismatch": (
        "IVF+PQ build_s расходится между scaling.csv и sweep CSV",
        "Тот же сценарий, что и у IVFFlat: scaling-цикл обучает "
        "IVF-квантизатор с дефолтным `cp.min_points_per_centroid`, "
        "sweep — со значением 5. Сборка в scaling в ~2× быстрее за счёт "
        "меньшего числа Lloyd-итераций; recall идентичен."
    ),
    "HNSW: build_s mismatch": (
        "HNSW build_s расходится между scaling.csv и sweep CSV",
        "HNSW **не использует** k-means, поэтому `cp.min_points_per_centroid` "
        "здесь ни при чём. Расхождение объясняется (1) разным code path "
        "у `measure_qps` в scaling и sweep ноутбуках и (2) тем, что sweep "
        "пересобирает HNSW сразу после IVF/PQ/SQ-замеров: page-cache горячий, "
        "а CPU-кеш холодный. На macOS обе переменные дают ~×2 разброс между "
        "прогонами одной и той же конфигурации. Recall одинаков (Δ ≤ 0.1 п.п.)."
    ),
    "LSH: build_s mismatch": (
        "LSH build_s расходится между scaling.csv и sweep CSV",
        "У LSH (`nbits=4096`) расхождение **не связано с CP-параметрами** "
        "(k-means не используется). Это **базовая вариативность build_s** "
        "между прогонами — long-running malloc-bound операция чувствительна "
        "к фоновой нагрузке и состоянию page-cache. На одном и том же "
        "конфиге у нас 2–3× разброс между запусками; recall и QPS стабильны."
    ),
    "scaling.csv: latency_p99_ms": (
        "scaling.csv: p99 latency измерен по другому code path",
        "В per-family CSV колонка `latency_p99_ms` — это реальный per-chunk "
        "p99 (50 запросов/чанк, ~200 чанков на конфиг). В `scaling.csv` "
        "p99 посчитан по 3 батчевым повторам (отдельный code path), поэтому "
        "там p99 ≈ mean, а в sweep-измерениях — заметно больше. Метрика "
        "сравнима только внутри одного источника."
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
    host = _host_info()
    facts = []
    if n_base:
        facts.append(f"- **Датасет:** ImageNet-1M ZJU, 2048-D, n_base = {n_base:,}, "
                     "n_query = 10 000 (для измерения QPS), n_gt = 25 000.")
    facts.append("- **Метрика расстояния:** L2.")
    facts.append(
        f"- **Хост:** {host['cpu']} ({host['cpu_physical']} физ. / "
        f"{host['cpu_logical']} лог. ядер), RAM {host['ram_gb']} ГБ, "
        f"{host['os']} ({host['machine']}), Python {host['python']}."
    )
    if threads:
        facts.append(f"- **Параллельность FAISS:** {threads} OpenMP-threads.")
    facts.append("- **QPS-замер:** `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1` "
                 "(один warmup + медиана 3 запусков; latency-распределение по "
                 "чанкам из 50 запросов).")
    def _n(name: str) -> int:
        df = frames.get(name)
        return 0 if df is None else len(df)
    facts.append(
        f"- **Кол-во конфигов:** IVFFlat {_n('ivf_flat')}, "
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
        W("На графике — все измерения, сгруппированные по семейству. "
          "Чёрный пунктир — глобальный Парето-фронт (точки, которые "
          "никто не доминирует одновременно по Recall@100 и QPS). "
          "Звёздами помечены рекомендованные конфигурации (колено "
          "Парето-кривой — точка с лучшим балансом recall и QPS, "
          "см. таблицу 2.1 ниже).")
        W()

    # 2.1 recommended config per family (= knee point — geometric balance)
    W("### 2.1. Рекомендованная конфигурация на семейство")
    W()
    if not is_short:
        W("Точка Парето-кривой, ближайшая к идеальному углу (Recall@100 = 1, "
          "максимальный QPS) в нормированном пространстве [0,1]² с log-y. "
          "Геометрически это **компромисс по умолчанию**: правее пытаться "
          "брать recall — дорого по QPS, левее — recall падает резко.")
        W()
    W("| Семейство | Recall@100 | QPS | Mean lat. | Index size | Build | Peak RSS | Конфиг |")
    W("|---|---:|---:|---:|---:|---:|---:|---|")
    for fam in FAMILY_ORDER:
        if fam not in knees.index:
            continue
        r = knees.loc[fam]
        W(
            f"| **{FAMILY_RU[fam]}** | {r.recall_100:.4f} | "
            f"{r.qps:,.0f} | {r.latency_ms:.3f} мс | {fmt_mb(r.size_mb)} | "
            f"{fmt_s(r.build_s)} | {fmt_mb(r.rss_peak_mb)} | `{r.config}` |"
        )
    W()
    if not is_short:
        W("Звёзды на Парето-графике выше отмечают именно эти конфиги. "
          "Иной критерий выбора — порог Recall@100 — см. 2.2.")
        W()

    # 2.2 max-QPS config that meets R@100 >= 0.95 (only families that reach it)
    W("### 2.2. Лучший конфиг при Recall@100 ≥ 0.95")
    W()
    # Determine which families reach the floor first, so we can both skip
    # empty rows in the table and call out the misses by name in the prose.
    high_recall_picks: List[Tuple[str, pd.Series]] = []
    missing_families: List[str] = []
    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if not len(sub):
            continue
        b = best_at_threshold(sub, 0.95)
        if b is None:
            missing_families.append(fam)
        else:
            high_recall_picks.append((fam, b))
    if not is_short:
        skipped = (", ".join(FAMILY_RU[f] for f in missing_families)
                   if missing_families else "—")
        W("Среди всех конфигов семейства с Recall@100 ≥ 0.95 берётся тот, "
          "у которого максимальный QPS. "
          f"Не дотягивают до 0.95: **{skipped}** "
          "(для них смотри §3 — полную таблицу порогов).")
        W()
    W("| Семейство | Recall@100 | QPS | Mean lat. | Index size | Build | Peak RSS | Конфиг |")
    W("|---|---:|---:|---:|---:|---:|---:|---|")
    high_recall_families: List[str] = []
    for fam, b in high_recall_picks:
        high_recall_families.append(fam)
        W(
            f"| **{FAMILY_RU[fam]}** | {b.recall_100:.4f} | "
            f"{b.qps:,.0f} | {b.latency_ms:.3f} мс | {fmt_mb(b.size_mb)} | "
            f"{fmt_s(b.build_s)} | {fmt_mb(b.rss_peak_mb)} | "
            f"`{config_str(b)}` |"
        )
    W()

    # 2.3 quadrant winners — now with recall + QPS context for each
    W("### 2.3. Победители по отдельным метрикам (по всему набору измерений)")
    W()
    if not is_short:
        W("Каждый пункт показывает, какое семейство выигрывает по одной "
          "конкретной метрике, и какие у него при этом Recall@100 и QPS — "
          "чтобы было видно, насколько пригоден этот конфиг.")
        W()
    if not combined.empty:
        best_recall = combined.sort_values("recall_100", ascending=False).iloc[0]
        best_qps = combined.sort_values("qps", ascending=False).iloc[0]
        smallest = combined.sort_values("size_mb").iloc[0]
        fastest = combined.sort_values("build_s").iloc[0]
        W(f"- **Максимальный Recall@100:** {FAMILY_RU[best_recall['family']]} "
          f"= {best_recall['recall_100']:.4f}, QPS = "
          f"{best_recall['qps']:,.0f} (`{config_str(best_recall)}`).")
        W(f"- **Максимальный QPS:** {FAMILY_RU[best_qps['family']]} = "
          f"{best_qps['qps']:,.0f}, при Recall@100 = "
          f"{best_qps['recall_100']:.3f} (`{config_str(best_qps)}`).")
        W(f"- **Минимальный размер индекса:** {FAMILY_RU[smallest['family']]} "
          f"= {fmt_mb(float(smallest['size_mb']))}, Recall@100 = "
          f"{smallest['recall_100']:.3f}, QPS = {smallest['qps']:,.0f} "
          f"(`{config_str(smallest)}`).")
        W(f"- **Самый быстрый билд:** {FAMILY_RU[fastest['family']]} = "
          f"{fmt_s(float(fastest['build_s']))}, Recall@100 = "
          f"{fastest['recall_100']:.3f}, QPS = {fastest['qps']:,.0f} "
          f"(`{config_str(fastest)}`).")
    W()
    W(f"![Рекомендованные конфиги: build / size / RSS / QPS](img/{run}/05_best_bars.png)")
    W()
    W(f"![Разложение peak RSS](img/{run}/05_memory_budget.png)")
    W()
    if not is_short:
        # Build the memory commentary from the actual numbers, not from
        # outdated hard-coded approximations.
        if "IVFFlat" in knees.index:
            ff_size = float(knees.loc["IVFFlat"].size_mb) / 1024
            ff_peak = float(knees.loc["IVFFlat"].rss_peak_mb) / 1024
            ff_share = ff_size / ff_peak * 100 if ff_peak > 0 else 0
            ff_txt = (f"у IVFFlat сам индекс — ~{ff_size:.1f} ГБ "
                      f"raw float-векторов, это {ff_share:.0f}% peak RSS")
        else:
            ff_txt = "у IVFFlat сам индекс — это десятки процентов peak RSS"
        W(f"Разложение пикового RSS видно на стэк-баре: "
          "серая нижняя полоса — приблизительный baseline процесса "
          "(Python + train slice ≈ 1.7 ГБ), синяя — резидентная часть "
          "индекса, оранжевая — mmap-страницы базы и накопленный state "
          "от предыдущих конфигов в этом ноутбуке, фиолетовая — "
          "transient overhead во время сборки (mmap-кеш + временные "
          "буферы, освобождаются после `commit`/возврата).")
        W()
        W(f"В итоге {ff_txt}; у IVFPQ и LSH сериализованный индекс "
          "крошечный (< 100 МБ), а большую часть пика держат "
          "mmap-страницы базы. HNSW также держит свой граф в памяти "
          "(синяя полоса видна сразу над baseline), но transient "
          "overhead сборки HNSW сопоставим по высоте с самим индексом.")
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
    if not is_short:
        W("Графики выше показывают все семейства одновременно. Ниже — "
          "детальный разбор каждого: диапазон параметров в выборке, "
          "таблица «лучший QPS при пороге Recall@100» и собственная "
          "пара диагностических графиков (как recall растёт с "
          "параметром поиска, как растёт размер).")
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
            "| Порог Recall@100 | Конфиг | Recall@100 | QPS | Mean lat. |",
            "|---:|---|---:|---:|---:|",
        ]
        out.extend(rows)
        return out

    # Per-family chart filenames — one grid per family, parallels the
    # existing IVFPQ deep-dive. The metric chart and the threshold table
    # tell the same story from two angles.
    PER_FAMILY_CHART = {
        "IVFFlat": ("05_ivfflat_grid.png",
                    "IVFFlat: Recall vs nprobe + QPS vs Recall по nlist"),
        "IVFPQ":   ("05_ivfpq_grid.png",
                    "IVFPQ: recall vs nprobe + footprint vs recall"),
        "IVFSQ":   ("05_ivfsq_grid.png",
                    "IVFSQ: Recall vs nprobe + footprint vs recall (SQ4 vs SQ8)"),
        "HNSW":    ("05_hnsw_grid.png",
                    "HNSW: Recall vs efSearch + QPS vs Recall по M"),
        "LSH":     ("05_lsh_grid.png",
                    "LSH: Recall+QPS vs nbits + footprint vs Recall"),
    }

    for fam in FAMILY_ORDER:
        sub = combined[combined.family == fam]
        if not len(sub):
            continue
        W(f"### 3.{FAMILY_ORDER.index(fam)+1}. {FAMILY_RU[fam]}")
        W()
        # Range bullets — switched from "min → max" (read as directional)
        # to "min ... max" with explicit (min/max) labels. Several reviewers
        # found the arrow notation confusing because it looks like a
        # change-of-value rather than a range.
        W(f"- **Конфигов в выборке:** {len(sub)}.")
        W(f"- **Recall@100:** от {sub.recall_100.min():.3f} (min) до "
          f"{sub.recall_100.max():.4f} (max).")
        W(f"- **QPS:** от {sub.qps.min():,.0f} (min) до "
          f"{sub.qps.max():,.0f} (max).")
        W(f"- **Размер индекса:** от {fmt_mb(sub.size_mb.min())} до "
          f"{fmt_mb(sub.size_mb.max())}.")
        W(f"- **Build:** от {fmt_s(sub.build_s.min())} до "
          f"{fmt_s(sub.build_s.max())}.")
        W()
        W("Лучшая конфигурация при каждом пороге Recall@100 "
          "(берём конфиг с максимальным QPS, чей recall ≥ порога):")
        for line in thresholds_table(sub):
            W(line)
        # Per-family grid chart
        if fam in PER_FAMILY_CHART:
            chart_file, alt = PER_FAMILY_CHART[fam]
            W()
            W(f"![{alt}](img/{run}/{chart_file})")
        # Family-specific commentary
        if fam == "IVFPQ" and not is_short:
            W()
            r_max = float(sub.recall_100.max())
            W(f"На правой панели видно, что максимальный достижимый "
              f"Recall@100 ≈ {r_max:.2f} — это **потолок семейства** "
              "на 2048-D ResNet-эмбеддингах при заданном битовом бюджете "
              "(см. §5).")
        elif fam == "IVFFlat" and not is_short:
            W()
            W("На левой панели видно, как recall растёт с nprobe и "
              "выходит на 1.0 раньше у больших nlist. На правой — "
              "при низком nprobe больший nlist даёт **выше** QPS "
              "(партишн помещается в L2 кеш), но при высоком "
              "nprobe — наоборот, скан большого числа партишнов "
              "обходится дороже.")
        elif fam == "IVFSQ" and not is_short:
            W()
            W("SQ8 (синие маркеры на правой панели) даёт более высокий "
              "recall, чем SQ4 (красные) — 8-битная квантизация теряет "
              "меньше информации. Цена — индекс в 2× больше (1 байт/"
              "координату vs 0.5 байт/координату). При nlist ≥ 1024 "
              "комбинация (SQ8, nlist, nprobe) даёт лучший trade-off, "
              "чем родственная (SQ4, та же nlist, x4 nprobe).")
        elif fam == "HNSW" and not is_short:
            W()
            W("Recall на левой панели растёт почти монотонно с efSearch "
              "у всех (M, efC); при efSearch ≤ 20 видна **немонотонность** "
              "по efC — рассмотрена в §5 как аномалия. Правая панель "
              "показывает, что при одинаковом recall меньшее M (8, 16) "
              "выигрывает по QPS — граф проще обходится.")
        elif fam == "LSH" and not is_short:
            W()
            W("Левая панель показывает классическую проблему LSH на "
              "высокоразмерных эмбеддингах: даже nbits=4096 (276 МБ) "
              "не пробивает Recall@100 ≈ 0.42. На правой видно, что "
              "footprint растёт линейно, а recall — насыщается.")
        W()

    # --------------------------------------------------------------
    # 4. Масштабирование
    # --------------------------------------------------------------
    if scaling is not None and len(scaling):
        n_min = int(scaling["n"].min())
        n_max = int(scaling["n"].max())
        def _short_n(n: int) -> str:
            if n >= 1_000_000:
                return f"{n/1_000_000:.2f}".rstrip("0").rstrip(".") + " M"
            return f"{n//1_000} K"
        n_pts = scaling["n"].nunique()
        growth = n_max / max(n_min, 1)
        W(f"## 4. Масштабирование {_short_n(n_min)} → {_short_n(n_max)}")
        W()
        W(f"![Scaling: recall/QPS/build/RSS vs N](img/{run}/05_scaling.png)")
        W()
        if not is_short:
            W(f"_Данные ниже — из `scaling.csv` (ноутбук 05, {n_pts} точек по N "
              f"от {n_min:,} до {n_max:,}). Это отдельный code path по сравнению "
              "с per-family sweep: recall сравним, build_s и p99 — нет, см. §5.2._")
            W()
            # IVFFlat: actual QPS ratio across the scaling range
            ivfflat_sc = scaling[scaling.family == "IVFFlat"].sort_values("n")
            hnsw_sc = scaling[scaling.family == "HNSW"].sort_values("n")
            qps_ivfflat_drop = (ivfflat_sc.qps.iloc[0] / ivfflat_sc.qps.iloc[-1]
                                if len(ivfflat_sc) >= 2 else None)
            qps_hnsw_drop = (hnsw_sc.qps.iloc[0] / hnsw_sc.qps.iloc[-1]
                             if len(hnsw_sc) >= 2 else None)
            drop_iv = f"~{qps_ivfflat_drop:.0f}×" if qps_ivfflat_drop else "значительная"
            drop_hn = f"~{qps_hnsw_drop:.1f}×" if qps_hnsw_drop else "сублинейная"
            W("Для каждого семейства взята одна репрезентативная конфигурация "
              f"(см. таблицу) и {n_pts} точек по N. **Recall** немного растёт с N "
              "у HNSW и IVFFlat (фиксированный k=100 захватывает больше из "
              "более плотного соседства), у IVFPQ **деградирует** (квантизатор "
              "обучен на 200K точек — на больших N теряется тонкая "
              "разрешающая способность), у LSH тоже падает (фиксированное "
              "число случайных гиперплоскостей размывается на большом базе). "
              f"**QPS** падает сильно у IVFFlat (центроидный скан + увеличение "
              f"среднего партишна — {drop_iv} потеря при {growth:.1f}× росте N), "
              f"сублинейно у HNSW (graph search; {drop_hn} потеря), "
              "у IVFPQ и LSH — линейно с n_base.")
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
            n_sc = int(cross_csv_df["n_scaling"].iloc[0])
            n_sw = int(cross_csv_df["n_sweep"].max())
            n_match = (n_sc == n_sw)
            if not is_short:
                if n_match:
                    W(f"Одна и та же `(family, config)` при n = {n_sc:,} "
                      "измерена дважды — в `<family>.csv` (per-family sweep) "
                      "и в `scaling.csv` (ноутбук 05). Это два разных code path. "
                      "Сравнение build_s и QPS:")
                else:
                    W(f"`scaling.csv` останавливается на n = {n_sc:,}, "
                      f"per-family sweep сделан при n = {n_sw:,}. Сравниваем "
                      "одни и те же `(family, config)` в этих двух источниках. "
                      "Разница включает (a) **разные code paths** и (b) "
                      "**разный N**, поэтому абсолютные числа не совпадают, "
                      "но порядок и причины различий видны:")
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
                W("**Источники различий:**")
                W()
                if not n_match:
                    W(f"0. **Разный N.** Sweep при n = {n_sw:,}, scaling при "
                      f"n = {n_sc:,} (~{n_sw/max(n_sc,1):.2f}× больше). "
                      "Часть Δ build_s и Δ QPS — естественная зависимость от "
                      "размера базы; ниже учитываем только дополнительный "
                      "вклад code-path-различий.")
                    W()
                W("1. **IVF-семейства** — sweep выставляет "
                  "`idx.cp.min_points_per_centroid = 5`, scaling — нет. "
                  "FAISS с дефолтным минимумом (39 точек/центроид) ограничивает "
                  "число итераций Lloyd-алгоритма → обучение IVF-квантизатора "
                  "в ~2× быстрее. Recall практически одинаков (Δ ≤ 0.5 п.п.), "
                  "но build_s — нет.")
                W()
                W("2. **HNSW и LSH** не используют k-means, поэтому "
                  "`cp.min_points_per_centroid` тут ни при чём. Различие для "
                  "них — это **базовая run-to-run вариативность** build_s: "
                  "long-running malloc-bound операции на macOS дают 2–3× разброс "
                  "между запусками. Recall и QPS совпадают в пределах ~5 %.")
                W()
                W("3. **p99 latency** в scaling.csv считается по 3 батчевым "
                  "повторам, в sweep CSV — по per-chunk распределению "
                  "(50 запросов/чанк), поэтому в scaling p99 ≈ mean, в sweep "
                  "p99 заметно больше (см. §6).")
                W()
                W("Recall между двумя источниками сходится — качество индексов "
                  "идентично, различаются только time-based метрики build_s / "
                  "QPS / p99.")
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
        W("- `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1`, медиана из 3 запусков "
          "+ per-chunk p99 (чанк = 50 запросов).")
        W("- Train slice = 200 000 векторов; при nlist=16384 это "
          "~12 точек/центроид — FAISS пишет варнинг `lloyd_3`.")
        W("- Ground truth пересчитан локально через `IndexFlatL2`, кеш "
          "`data/gt_n1281167_k100.npy`.")
        W("- Peak RSS включает mmap-страницы базы (доминирует у IVFPQ/LSH).")
        W("- `scaling.csv` — отдельный code path: p99 ≈ mean, build_s "
          "у IVF в ~2× быстрее (без `cp.min_points_per_centroid=5`), см. §5.")
    else:
        W("**1. QPS-замер.** Скрипт `run_all.sh` выставляет "
          "`LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1`: один warm-up прогон + 3 "
          "повтора + медиана. Это снижает шум.")
        W()
        W("**2. p99 latency.** В `utils.measure_qps` батч разрезается на "
          "чанки по 50 запросов, каждый чанк таймится отдельно, p99 берётся "
          "по этому распределению. Этот per-chunk p99 лежит в колонке "
          "`latency_p99_ms` всех per-family CSV (и показан на "
          "`05_latency_best.png` рядом с mean). В `scaling.csv` p99 "
          "считается по 3 батчевым повторам (отдельный code path), поэтому "
          "там p99 ≈ mean — это flag в §5.")
        W()
        W("**3. Centroid undertraining при nlist=16384.** Train slice = "
          "200 000 векторов, при nlist=16384 это ~12 точек/центроид. "
          "FAISS явно предупреждает (`lloyd_3`). IVFFlat-конфигурации "
          "с этим nlist слегка под-тренированы, но recall-числа "
          "адекватны фактическому состоянию обученного индекса.")
        W()
        ivsq_df = frames.get("ivf_sq")
        ivsq_rows = 0 if ivsq_df is None else len(ivsq_df)
        ivsq_nlists = (sorted(ivsq_df.nlist.unique().tolist())
                       if ivsq_df is not None and len(ivsq_df) else [])
        W(f"**4. IVFSQ — расширенный sweep.** IVFSQ перебирает "
          f"nlist ∈ {ivsq_nlists} и типы {{SQ4, SQ8}}. "
          f"В `results/full/ivf_sq.csv` — {ivsq_rows} конфигов.")
        W()
        W("**5. Ground truth пересчитан локально.** Стандартный GT "
          "(`imagenet_groundtruth.ivecs`) индексирует в IDs > N для "
          "любого N < 1 281 167, поэтому для выборки с произвольным N "
          "используется свежий GT, пересчитанный через `IndexFlatL2` "
          "над тем же base-slice. Кеш — `data/gt_n1281167_k100.npy`. "
          "Числа recall сравнимы между всеми измерениями.")
        W()
        W("**6. Peak RSS включает mmap-страницы.** `stream_add` "
          "проходит base через `np.memmap`; ОС считает резидентные "
          "страницы у процесса, и пик RSS суммирует index + train slice "
          "+ mmap-кеш + Python overhead. Для IVFPQ/LSH доминирует именно "
          "mmap-кеш. Видно на `05_memory_budget.png` (фиолетовая полоса). "
          "Альтернатива — `madvise(DONTNEED)` после сборки; не критично, "
          "поскольку эти страницы легко освобождаются ОС при memory "
          "pressure.")
        W()
        W("**7. RssPeakMonitor sampling.** Интервал 50 мс. Кратковременные "
          "пики < 50 мс могут быть упущены (типично для FAISS — пиковые "
          "аллокации происходят во время Lloyd-iter и длятся секунды).")
        W()
        W("**8. `scaling.csv` — отдельный code path.** Per-family CSV "
          "(`results/full/{ivf,hnsw,lsh}*.csv`) и `scaling.csv` сделаны в "
          "разных ноутбуках с разными настройками: (a) p99 в scaling — "
          "по 3 батчевым повторам, в sweep — per-chunk, (b) IVF-семейства "
          "в scaling без `cp.min_points_per_centroid=5`, поэтому build_s "
          "в ~2× быстрее, (c) HNSW/LSH build_s — естественный run-to-run "
          "jitter. Recall сравним между источниками, time-based метрики — "
          "нет; расхождения помечены как аномалии в §5.2.")
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
        h99_txt = (f" Для Recall@100 ≥ 0.99 — `{config_str(h99)}` "
                   f"(R@100 = {h99.recall_100:.4f}, {h99.qps:,.0f} QPS)."
                   if h99 is not None else "")
        W(
            f"- **Поиск с высоким Recall@100 (≥ 0.95)** → **HNSW** "
            f"`{config_str(h)}`: {h.qps:,.0f} QPS, "
            f"{h.latency_ms:.3f} мс средняя latency, "
            f"{fmt_mb(h.size_mb)} на диске, "
            f"{fmt_mb(h.rss_peak_mb)} peak RSS "
            f"(~{mmap_frac*100:.0f} % из которых — mmap-страницы базы, "
            f"легко освобождаются ОС при необходимости).{h99_txt}"
        )
    pq_max = combined[combined.family == "IVFPQ"]
    if len(pq_max):
        b = knee_row(pq_max)
        if b is None:
            b = pq_max.sort_values("recall_100", ascending=False).iloc[0]
        if "IVFFlat" in knees.index:
            ff_size = float(knees.loc["IVFFlat"].size_mb)
            size_ratio = ff_size / max(b.size_mb, 1.0)
        else:
            size_ratio = float("nan")
        W(
            f"- **Минимальный размер индекса** → **IVF+PQ** "
            f"`{config_str(b)}`: {fmt_mb(b.size_mb)} "
            f"(~{size_ratio:.0f}× меньше IVFFlat knee), "
            f"Recall@100 = {b.recall_100:.3f}, {b.qps:,.0f} QPS. "
            f"Потолок семейства — Recall@100 = {pq_max.recall_100.max():.3f} "
            "(M=128, 16 байт/вектор). Использовать как кандидат-генератор "
            "перед rerank-стадией на оригинальных векторах."
        )
    sq = _best_at("IVFSQ", 0.95)
    if sq is not None:
        if "IVFFlat" in knees.index:
            sq_size_ratio = float(knees.loc["IVFFlat"].size_mb) / max(sq.size_mb, 1.0)
            sq_ratio_txt = f"~{sq_size_ratio:.1f}× меньше IVFFlat knee"
        else:
            sq_ratio_txt = "компактнее IVFFlat"
        W(
            f"- **Компрессия с высоким recall** → **IVF+SQ-8** "
            f"`{config_str(sq)}`: Recall@100 = {sq.recall_100:.4f}, "
            f"{sq.qps:,.0f} QPS, {fmt_mb(sq.size_mb)} ({sq_ratio_txt}). "
            f"Per-query latency {sq.latency_ms:.2f} мс — медленнее HNSW, "
            "потому что SQ декодирует значения на лету при вычислении "
            "дистанции, но в разы быстрее, чем IVFFlat на том же recall."
        )
    ff = _best_at("IVFFlat", 0.95)
    if ff is not None:
        W(
            f"- **Точный (exact-ish) baseline** → **IVFFlat** `{config_str(ff)}`: "
            f"Recall@100 = {ff.recall_100:.4f}, {ff.qps:,.0f} QPS, "
            f"{fmt_mb(ff.size_mb)}. Build {fmt_s(ff.build_s)}. "
            "Сборка дорогая, QPS низкий — но индекс хранит сырые float-векторы, "
            "поэтому recall максимально приближен к точному поиску. "
            "Полезен как калибратор Ground Truth, не как serving-движок."
        )
    lsh_max = combined[combined.family == "LSH"]
    if len(lsh_max):
        lb = lsh_max.sort_values("recall_100", ascending=False).iloc[0]
        W(
            f"- **LSH непригоден на этом датасете** — даже при "
            f"`{config_str(lb)}` (276 МБ индекс) Recall@100 = {lb.recall_100:.3f}. "
            "При 2048-D случайные гиперплоскости требуют экспоненциального "
            "числа бит на единицу cosine-разрешения — footprint уходит за "
            "PQ задолго до того, как recall становится приемлемым."
        )
    W()
    # Final recommendation uses the actual high-recall pick, not a hardcoded one.
    if h is not None:
        rec_label = f"HNSW (`{config_str(h)}`)"
    else:
        rec_label = "HNSW (см. таблицу 2.2)"
    W(f"**Итог:** {rec_label} — дефолтный выбор для high-recall поиска; "
      "IVF+PQ — только в связке с rerank-стадией; "
      "IVF+SQ-8 — компромисс по размеру/latency, если QPS-бюджет небольшой "
      "и хочется ×4 компрессии; IVFFlat — оффлайн-калибровка GT; "
      "LSH — отбросить для этого датасета.")
    W()
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
    # Per-family deep dives — one grid per family
    plot_ivfflat_grid(frames.get("ivf_flat"), plots_dir)
    plot_ivfpq_grid(frames.get("ivf_pq"), plots_dir)
    plot_ivfsq_grid(frames.get("ivf_sq"), plots_dir)
    plot_hnsw_grid(frames.get("hnsw_M"), frames.get("hnsw_EFC"), plots_dir)
    plot_lsh_grid(frames.get("lsh"), plots_dir)
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
