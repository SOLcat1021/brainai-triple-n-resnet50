"""Compare response-derived temporal metrics across six coarse visual ROIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from plot_raw_su_boundary_time_axes import compute_metrics


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "cross_roi_observed_temporal_metrics_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_LABELS = ["V1", "V2", "V4", "pIT", "mIT", "aIT"]
METRICS = [
    ("evoked_centroid_ms", "Evoked-curve centroid", "Time (ms)"),
    ("mode1_centroid_ms", "Mode-1 temporal centroid", "Time (ms)"),
    ("effective_temporal_rank", "Effective temporal rank", "Rank"),
    ("nonwarp_residual_fraction", "Non-warp residual fraction", "Fraction"),
]
MONKEY_MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P"}


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    center = float(np.mean(values))
    total = float(np.sum((values - center) ** 2))
    between = 0.0
    for label in np.unique(labels):
        q = values[labels == label]
        between += len(q) * float((np.mean(q) - center) ** 2)
    return between / max(total, 1e-12)


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = data.groupby(["roi", "monkey", "session"], as_index=False)[
        [m[0] for m in METRICS]
    ].median()
    rows = []
    for metric, _, _ in METRICS:
        groups = [session.loc[session.roi.eq(roi), metric].to_numpy(float) for roi in ROIS]
        groups = [g for g in groups if len(g)]
        h, p = kruskal(*groups)
        hierarchy = session.roi.map({roi: i for i, roi in enumerate(ROIS)}).to_numpy(float)
        rho, rho_p = spearmanr(hierarchy, session[metric].to_numpy(float))
        rows.append({
            "metric": metric,
            "n_units": len(data),
            "n_session_roi_groups": len(session),
            "unit_level_roi_eta2_descriptive": eta_squared(data[metric].to_numpy(float), data.roi.to_numpy(str)),
            "session_level_roi_eta2": eta_squared(session[metric].to_numpy(float), session.roi.to_numpy(str)),
            "session_kruskal_h": float(h),
            "session_kruskal_p": float(p),
            "session_hierarchy_spearman_rho": float(rho),
            "session_hierarchy_spearman_p": float(rho_p),
        })
    return session, pd.DataFrame(rows)


def violin_with_sessions(data: pd.DataFrame, session: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = ["#4E79A7", "#76A5AF", "#59A14F", "#EDC948", "#F28E2B", "#B65A4A"]
    rng = np.random.default_rng(20260827)
    for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
        arrays = [data.loc[data.roi.eq(roi), metric].to_numpy(float) for roi in ROIS]
        violin = ax.violinplot(arrays, positions=np.arange(6), widths=0.82,
                               showmeans=False, showmedians=True, showextrema=False,
                               points=150)
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        violin["cmedians"].set_color("#222222")
        violin["cmedians"].set_linewidth(2)
        for ri, roi in enumerate(ROIS):
            q = session[session.roi.eq(roi)]
            jitter = rng.uniform(-0.17, 0.17, len(q))
            for monkey, marker in MONKEY_MARKERS.items():
                m = q.monkey.eq(monkey).to_numpy()
                if np.any(m):
                    ax.scatter(np.full(np.sum(m), ri) + jitter[m], q.loc[m, metric],
                               s=34, marker=marker, facecolors=colors[ri], edgecolors="#222222",
                               linewidths=0.6, alpha=0.9, label=monkey if ri == 0 else None)
        ax.set_xticks(np.arange(6), ROI_LABELS)
        ax.set_xlabel("Coarse visual ROI")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0, 0].legend(title="Monkey", frameon=False, ncol=5, fontsize=8, title_fontsize=8)
    fig.suptitle("Observed temporal metrics across the ventral visual hierarchy",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.004,
             "Violin: all mapped units. Symbols: monkey x session medians (statistical replication level). No proxy predictions or quality filtering.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "01_cross_roi_temporal_metric_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def roi_profiles(data: pd.DataFrame, session: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
        medians, lows, highs = [], [], []
        for roi in ROIS:
            q = session.loc[session.roi.eq(roi), metric].to_numpy(float)
            medians.append(float(np.median(q)))
            lows.append(float(np.quantile(q, 0.25)))
            highs.append(float(np.quantile(q, 0.75)))
        y = np.asarray(medians)
        ax.errorbar(np.arange(6), y, yerr=[y - lows, np.asarray(highs) - y],
                    color="#2F6F8F", marker="o", ms=7, lw=2, capsize=4)
        ax.set_xticks(np.arange(6), ROI_LABELS)
        ax.set_xlabel("Coarse visual ROI")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Session-level median temporal profiles across ROIs",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.004, "Point: median across session-level medians. Whisker: session-level interquartile range.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "02_cross_roi_session_level_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "all_unit_observed_temporal_metrics.csv"
    if cache.exists():
        data = pd.read_csv(cache)
    else:
        meta = pd.read_csv(BANK / "unit_metadata.csv")
        meta = meta[meta.roi.isin(ROIS)].sort_values("unit_global").reset_index(drop=True)
        data = compute_metrics(meta)
        data.to_csv(cache, index=False, encoding="utf-8-sig")
    session, tests = summarize(data)
    session.to_csv(OUT / "session_roi_temporal_metric_medians.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "cross_roi_metric_tests.csv", index=False, encoding="utf-8-sig")
    violin_with_sessions(data, session)
    roi_profiles(data, session)
    print(tests.to_string(index=False), flush=True)
    print(data.groupby("roi").size().reindex(ROIS).to_string(), flush=True)


if __name__ == "__main__":
    main()
