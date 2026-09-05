"""Plot image-wise RF-local head score against OOF-predicted G for MF unit 3447."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
OUT = PROJECT / "results" / "mf_head_wall_rf_gtd_pilot_2026-08-28"
UNIT = 3447


def main() -> None:
    data = np.load(OUT / "head_wall_rf_scores_and_GTD.npz", allow_pickle=True)
    concepts = data["concepts"].astype(str).tolist()
    units = data["units"].astype(int).tolist()
    ci = concepts.index("head")
    ui = units.index(UNIT)
    x_raw = data["image_scores"][ci, ui].astype(float)
    y = data["predicted_gtd"][ui, :, 0].astype(float)
    x = (x_raw - x_raw.mean()) / x_raw.std(ddof=1)

    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    residual = y - fitted
    residual_sd = np.sqrt(np.sum(residual**2) / (len(x) - 2))
    x_grid = np.linspace(x.min(), x.max(), 300)
    y_grid = intercept + slope * x_grid
    mean_x = x.mean()
    sxx = np.sum((x - mean_x) ** 2)
    se_mean = residual_sd * np.sqrt(1 / len(x) + (x_grid - mean_x) ** 2 / sxx)
    r = float(np.corrcoef(x, y)[0, 1])

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    ax.scatter(
        x, y, s=19, color="#2878A9", alpha=.32, edgecolors="none",
        rasterized=True, label="Triple-N image",
    )
    ax.fill_between(
        x_grid, y_grid - 1.96 * se_mean, y_grid + 1.96 * se_mean,
        color="#D5523D", alpha=.18, linewidth=0, label="95% fit CI",
    )
    ax.plot(x_grid, y_grid, color="#C43B2B", lw=2.4, label="Linear fit")
    ax.axhline(0, color="#777777", lw=.75, ls="--")
    ax.set_xlabel("RF-local headlike score (SD)")
    ax.set_ylabel("OOF proxy-predicted relative G")
    ax.set_title(f"MF unit {UNIT}: headlike score vs gain dynamics", fontweight="bold")
    ax.text(
        .035, .955, f"n = {len(x):,} images\nPearson r = {r:+.3f}\nSlope = {slope:+.3f} G / SD",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "square,pad=.45", "facecolor": "white", "edgecolor": "#BDBDBD", "alpha": .94},
    )
    ax.grid(color="#E3E3E3", lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "03_unit3447_headlike_vs_predicted_G_scatter.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    np.savetxt(
        OUT / "unit3447_headlike_vs_predicted_G_points.csv",
        np.column_stack([np.arange(1, len(x) + 1), x_raw, x, y, fitted]),
        delimiter=",", comments="",
        header="image_index,headlike_raw,headlike_z,predicted_G,fitted_G",
    )
    print(f"unit={UNIT} n={len(x)} r={r:.6f} slope={slope:.6f}")


if __name__ == "__main__":
    main()
