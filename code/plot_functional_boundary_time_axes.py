"""Plot response-derived temporal metrics along cross-area penetration axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
OUT = PROJECT / "results" / "functional_boundary_gradients_2026-08-27"

SESSIONS = [24, 29, 30, 60, 61]
SESSION_LABELS = {
    24: "M1 session 24 | site 8 | MB-MF",
    29: "M3 session 29 | site 18 | AF-AB",
    30: "M3 session 30 | site 18 | AF-AB",
    60: "M1 session 60 | site 30 | AF-AB",
    61: "M1 session 61 | site 30 | AF-AB",
}
METRICS = [
    ("evoked_centroid_ms", "Evoked-curve centroid", "Time (ms)"),
    ("mode1_centroid_ms", "Mode-1 temporal centroid", "Time (ms)"),
    ("effective_temporal_rank", "Effective temporal rank", "Rank"),
    ("nonwarp_residual_fraction", "Non-warp residual fraction", "Fraction"),
]


def axis_figure() -> None:
    data = pd.read_csv(OUT / "scalar_temporal_metrics_by_depth_bin.csv")
    data = data[data.cohort.eq("neural_reliable_ge_0.40")].copy()
    fig, axes = plt.subplots(len(METRICS), len(SESSIONS), figsize=(18, 12), sharex="col",
                             constrained_layout=True)
    area_colors = {"MB": "#3B82A0", "MF": "#C7662A", "AF": "#3B82A0", "AB": "#C7662A"}
    for col, session in enumerate(SESSIONS):
        sq = data[data.session.eq(session)]
        pair = str(sq.pair.iloc[0])
        left, right = pair.split("-")
        for row, (metric, title, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            q = sq[sq.metric.eq(metric)].sort_values("relative_position_mm")
            for area in (left, right):
                a = q[q.side.eq(area)]
                ax.errorbar(a.relative_position_mm, a["mean"], yerr=a["sem"],
                            color=area_colors[area], marker="o", ms=4, lw=1.5,
                            capsize=2, label=area)
            ax.axvline(0, color="#222222", lw=1.2, ls="--")
            ax.grid(axis="y", color="#D6D6D6", lw=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(SESSION_LABELS[session], fontsize=10, fontweight="bold")
                ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(METRICS) - 1:
                ax.set_xlabel("Position relative to area boundary (mm)")
            if col == 0:
                ax.text(-0.30, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle("Observed temporal dynamics along cross-area penetration axes",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.002,
             "Dots: 150-um depth-bin means; whiskers: SEM across units. Dashed line: annotated area boundary. No proxy predictions used.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "01_temporal_metrics_along_penetration_axes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def model_figure() -> None:
    data = pd.read_csv(OUT / "boundary_model_comparison.csv")
    data = data[data.cohort.eq("neural_reliable_ge_0.40")]
    views = ["evoked_curve", "stimulus_temporal_covariance",
             "response_dynamic_descriptors", "combined_balanced"]
    labels = ["Evoked curve", "Temporal covariance", "Dynamic descriptors", "Combined"]
    x = np.arange(len(SESSIONS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    colors = ["#3B82A0", "#7A8E45", "#C7662A", "#665C9E"]
    for i, (view, label, color) in enumerate(zip(views, labels, colors)):
        q = data[data.feature_view.eq(view)].set_index("session").reindex(SESSIONS)
        ax.bar(x + (i - 1.5) * width, q.jump_minus_continuous_cv_r2,
               width=width, label=label, color=color)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks(x, ["s24\nMB-MF", "s29\nAF-AB", "s30\nAF-AB", "s60\nAF-AB", "s61\nAF-AB"])
    ax.set_ylabel("Jump model minus continuous model (CV R2)")
    ax.set_xlabel("Cross-area penetration session")
    ax.set_title("Does an annotated area boundary improve prediction beyond continuous position?",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.grid(axis="y", color="#D6D6D6", lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.text(3.5, 0.50, "M1 site 30:\nconsistent jump candidate", ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.01,
             "Positive values favor a boundary step after accounting for quadratic position; leave-one-depth-bin-out cross-validation.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "02_boundary_jump_vs_continuous_cv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    axis_figure()
    model_figure()
    print(OUT / "01_temporal_metrics_along_penetration_axes.png")
    print(OUT / "02_boundary_jump_vs_continuous_cv.png")


if __name__ == "__main__":
    main()
