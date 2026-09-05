"""Plot raw and population-balanced dynamic compositions colored by coarse ROI."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
IN_DIR = PROJECT / "results" / "dynamic_phenotype_clustering_heldout_2026-08-27"
OUT = IN_DIR / "02_dynamic_composition_triangles_by_roi.png"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = {"V1": "V1", "V2": "V2", "V4": "V4", "posterior IT": "pIT",
          "middle IT": "mIT", "anterior IT": "aIT"}
COLORS = {"V1": "#0072B2", "V2": "#56B4E9", "V4": "#009E73",
          "posterior IT": "#E69F00", "middle IT": "#D55E00", "anterior IT": "#CC79A7"}
SEED = 20260827


def ternary_xy(composition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gain, warp, nonwarp = composition.T
    return warp + 0.5 * nonwarp, np.sqrt(3) / 2 * nonwarp


def draw_triangle(ax: plt.Axes, data: pd.DataFrame, columns: list[str], title: str,
                  vertex_prefix: str = "") -> None:
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(triangle_x, triangle_y, color="#333333", lw=1.5)
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        comp = q[columns].to_numpy(float)
        x, y = ternary_xy(comp)
        ax.scatter(x, y, s=8, alpha=0.20, color=COLORS[roi], edgecolors="none",
                   label=f"{LABELS[roi]} (n={len(q)})")
        centroid = comp.mean(0, keepdims=True)
        cx, cy = ternary_xy(centroid)
        ax.scatter(cx, cy, s=105, color=COLORS[roi], edgecolors="#222222",
                   linewidths=0.9, marker="o", zorder=5)
        ax.text(float(cx[0]), float(cy[0]) + 0.027, LABELS[roi], ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222222")
    ax.text(-0.02, -0.035, f"{vertex_prefix}gain", ha="right", va="top", fontsize=11)
    ax.text(1.02, -0.035, f"{vertex_prefix}phase/warp", ha="left", va="top", fontsize=11)
    ax.text(0.5, np.sqrt(3) / 2 + 0.028, f"{vertex_prefix}nonwarp", ha="center", va="bottom", fontsize=11)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.07, np.sqrt(3) / 2 + 0.09)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    ax.set_title(title, fontsize=13, fontweight="bold")


def main() -> None:
    data = pd.read_csv(IN_DIR / "unit_dynamic_phenotypes.csv")
    raw_cols = ["mean_gain", "mean_warp", "mean_nonwarp"]
    data["mean_gain"] = (data.train_gain_fraction + data.heldout_gain_fraction) / 2
    data["mean_warp"] = (data.train_warp_fraction + data.heldout_warp_fraction) / 2
    data["mean_nonwarp"] = (data.train_nonwarp_fraction + data.heldout_nonwarp_fraction) / 2

    # Equal sampling prevents the 10,369-unit middle-IT population from
    # visually dominating the smaller ROIs. Use the smallest ROI count.
    n_equal = int(data.groupby("roi").size().reindex(ROIS).min())
    sampled = pd.concat([
        data[data.roi.eq(roi)].sample(n=n_equal, random_state=SEED + i)
        for i, roi in enumerate(ROIS)
    ], ignore_index=True)

    baseline = data[raw_cols].median().to_numpy(float)
    balanced = sampled[raw_cols].to_numpy(float) / baseline[None]
    balanced /= balanced.sum(1, keepdims=True)
    bal_cols = ["balanced_gain", "balanced_warp", "balanced_nonwarp"]
    sampled[bal_cols] = balanced

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)
    draw_triangle(axes[0], sampled, raw_cols,
                  "Raw variance fractions\n(equal number of units per ROI)")
    draw_triangle(axes[1], sampled, bal_cols,
                  "Population-balanced relative enrichment\n(component / global median, then renormalized)",
                  vertex_prefix="relative ")
    axes[1].legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.suptitle("Dynamic composition across six visual ROIs", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.012,
             f"Each panel uses {n_equal:,} units per ROI ({n_equal * 6:,} total). Small points: units; large outlined points: ROI centroids. Fractions are averaged across the two held-out image halves.",
             ha="center", fontsize=9)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    centroids = []
    for roi in ROIS:
        q = sampled[sampled.roi.eq(roi)]
        row = {"roi": roi, "n_equal_sample": len(q)}
        row.update({column: float(q[column].mean()) for column in raw_cols + bal_cols})
        centroids.append(row)
    pd.DataFrame(centroids).to_csv(IN_DIR / "dynamic_composition_roi_centroids_balanced_plot.csv",
                                   index=False, encoding="utf-8-sig")
    print(OUT)
    print(pd.DataFrame(centroids).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
