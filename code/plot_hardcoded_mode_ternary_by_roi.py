"""Pure ternary map: fixed mode vertices, continuous scores, ROI colors."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
RESULT = PROJECT / "results" / "dynamic_phenotype_clustering_heldout_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = {"V1": "V1", "V2": "V2", "V4": "V4", "posterior IT": "pIT",
          "middle IT": "mIT", "anterior IT": "aIT"}
COLORS = {"V1": "#0072B2", "V2": "#56B4E9", "V4": "#009E73",
          "posterior IT": "#E69F00", "middle IT": "#D55E00", "anterior IT": "#CC79A7"}
SEED = 20260827


def xy(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gain, phase, nonwarp = scores.T
    return phase + 0.5 * nonwarp, np.sqrt(3) / 2 * nonwarp


def main() -> None:
    data = pd.read_csv(RESULT / "unit_dynamic_phenotypes.csv")
    columns = ["gain_score", "phase_warp_score", "nonwarp_score"]
    data["gain_score"] = (data.train_gain_fraction + data.heldout_gain_fraction) / 2
    data["phase_warp_score"] = (data.train_warp_fraction + data.heldout_warp_fraction) / 2
    data["nonwarp_score"] = (data.train_nonwarp_fraction + data.heldout_nonwarp_fraction) / 2
    # Equal display density by ROI changes only how many dots are drawn, not
    # their scores, coordinates, or ROI centroids.
    n_display = int(data.groupby("roi").size().reindex(ROIS).min())
    display = pd.concat([
        data[data.roi.eq(roi)].sample(n=n_display, random_state=SEED + i)
        for i, roi in enumerate(ROIS)
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9.5, 8.2), constrained_layout=True)
    height = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, height, 0], color="#303030", lw=1.7)
    # Light 25/50/75% guides make scores readable without changing geometry.
    for value in (0.25, 0.5, 0.75):
        ax.plot([value, 1 - value / 2], [0, value * height], color="#D4D4D4", lw=0.6)
        ax.plot([1 - value, value / 2], [0, value * height], color="#D4D4D4", lw=0.6)
        ax.plot([value / 2, 1 - value / 2], [value * height, value * height], color="#D4D4D4", lw=0.6)

    centroids = []
    for roi in ROIS:
        q = display[display.roi.eq(roi)]
        scores = q[columns].to_numpy(float)
        x, y = xy(scores)
        ax.scatter(x, y, s=8, alpha=0.18, color=COLORS[roi], edgecolors="none",
                   label=f"{LABELS[roi]} (n={len(q):,})")
        full = data[data.roi.eq(roi)][columns].mean().to_numpy(float)[None]
        cx, cy = xy(full)
        ax.scatter(cx, cy, s=125, color=COLORS[roi], edgecolors="#202020", linewidths=1,
                   zorder=6)
        ax.text(float(cx[0]), float(cy[0]) + 0.026, LABELS[roi], ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        centroids.append({"roi": roi, **dict(zip(columns, full[0]))})

    ax.text(-0.025, -0.035, "100% gain-like", ha="right", va="top", fontsize=12)
    ax.text(1.025, -0.035, "100% phase/warp", ha="left", va="top", fontsize=12)
    ax.text(0.5, height + 0.032, "100% nonwarp", ha="center", va="bottom", fontsize=12)
    ax.set_xlim(-0.10, 1.10); ax.set_ylim(-0.08, height + 0.10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    ax.set_title("Fixed-mode ternary map of unit dynamics", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.012,
             "Position is determined only by three variance scores that sum to 1; no clustering or ROI label enters the coordinates. Small dots are equally sampled by ROI; outlined dots are full-ROI centroids.",
             ha="center", fontsize=9)
    fig.savefig(RESULT / "03_fixed_mode_ternary_by_roi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(centroids).to_csv(RESULT / "fixed_mode_ternary_roi_centroids.csv",
                                   index=False, encoding="utf-8-sig")
    print(RESULT / "03_fixed_mode_ternary_by_roi.png")
    print(pd.DataFrame(centroids).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
