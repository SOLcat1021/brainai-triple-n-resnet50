"""Four spatial concept-score controls for MF unit 3447."""
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from train_64d_high_quality_proxy_batch import gaussian_fields

PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
PILOT = PROJECT / "results" / "mf_head_wall_rf_gtd_pilot_2026-08-28"
BANK = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
UNIT = 3447
METHODS = ("full_image", "correct_RF", "shifted_RF", "center_only")
LABELS = {"full_image": "Full image", "correct_RF": "Correct RF",
          "shifted_RF": "Shifted RF", "center_only": "Center only"}

def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)

def main():
    data = np.load(PILOT / "head_wall_rf_gtd_pilot_2026-08-28" / "head_wall_rf_scores_and_GTD.npz", allow_pickle=True) if (PILOT / "head_wall_rf_gtd_pilot_2026-08-28").exists() else np.load(PILOT / "head_wall_rf_scores_and_GTD.npz", allow_pickle=True)
    maps = np.load(PILOT / "head_wall_native_cav_maps_f32.npy").astype(float)
    ui = data["units"].astype(int).tolist().index(UNIT)
    y = data["observed_gtd"][ui, :, 0].astype(float)
    selected = pd.read_csv(BANK / "selected_units.csv")
    middle = selected[selected.roi.eq("middle IT")].reset_index(drop=True)
    local = int(middle.index[middle.unit_global.eq(UNIT)][0])
    with h5py.File(BANK / "middle_it_proxy_bank_64d.h5", "r") as h:
        windows = h["windows_ms"][:].astype(int)
        unit_row = middle.iloc[local]
        wi = int(np.flatnonzero((windows[:, 0] == unit_row.window_start_ms) & (windows[:, 1] == unit_row.window_end_ms))[0])
        spatial_params = np.asarray(h["spatial_parameters"][wi, local], dtype=np.float32)
        field = gaussian_fields(spatial_params[None])[0].reshape(14, 14)
    field = field / field.sum()
    shifted = np.roll(field, shift=(3, 4), axis=(0, 1))
    shifted /= shifted.sum()
    p = np.linspace(-10 + 10 / 14, 10 - 10 / 14, 14)
    xx, yy = np.meshgrid(p, p)
    sigma = float(spatial_params[2])
    center = (xx - float(spatial_params[0])) ** 2 + (yy - float(spatial_params[1])) ** 2 <= (0.75 * sigma) ** 2
    center_field = field * center
    center_field /= center_field.sum()
    scores = np.zeros((2, 4, 1000), float)
    for ci in range(2):
        m = maps[ci]
        scores[ci, 0] = m.mean((1, 2))
        scores[ci, 1] = np.einsum("nhw,hw->n", m, field)
        scores[ci, 2] = np.einsum("nhw,hw->n", m, shifted)
        scores[ci, 3] = np.einsum("nhw,hw->n", m, center_field)
    rows = []
    for ci, concept in enumerate(("head", "wall")):
        base = rankdata(scores[ci, 0], method="average")
        base_pct = (base - 1) / 999
        for mi, method in enumerate(METHODS):
            ranks = rankdata(scores[ci, mi], method="average")
            pct = (ranks - 1) / 999
            top_base = set(np.argsort(scores[ci, 0])[-100:])
            top_now = set(np.argsort(scores[ci, mi])[-100:])
            rows.append({"concept": concept, "method": method,
                         "rank_spearman_vs_full": float(spearmanr(scores[ci, 0], scores[ci, mi]).statistic),
                         "top10pct_overlap": len(top_base & top_now) / 100,
                         "mean_abs_percentile_shift": float(np.mean(np.abs(base_pct - pct))),
                         "median_abs_percentile_shift": float(np.median(np.abs(base_pct - pct)))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(PILOT / "unit3447_head_wall_spatial_order_metrics.csv", index=False)
    np.savez_compressed(PILOT / "unit3447_head_wall_four_spatial_scores.npz",
                        concepts=np.asarray(("head", "wall")), methods=np.asarray(METHODS),
                        scores=scores, observed_G=y, rf_field=field, shifted_field=shifted,
                        center_field=center_field)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True, sharey="row")
    colors = ("#2878A9", "#C43B2B")
    for ci, concept in enumerate(("head", "wall")):
        for mi, method in enumerate(METHODS):
            ax = axes[ci, mi]
            x = z(scores[ci, mi]); X = np.column_stack([np.ones(1000), x]); a, b = np.linalg.lstsq(X, y, rcond=None)[0]
            grid = np.linspace(x.min(), x.max(), 200)
            ax.scatter(x, y, s=9, alpha=.24, color=colors[ci], edgecolors="none", rasterized=True)
            ax.plot(grid, a + b * grid, color="#222222", lw=1.8)
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{LABELS[method]}\nr = {r:+.3f}", fontweight="bold")
            ax.set_xlabel(f"{concept} score (SD)")
            if mi == 0: ax.set_ylabel("Observed relative G")
            ax.grid(color="#E5E5E5", lw=.55); ax.set_axisbelow(True)
    fig.suptitle("MF unit 3447: spatial scoring controls for concept–G relation", fontweight="bold")
    fig.savefig(PILOT / "07_unit3447_head_wall_four_spatial_scatter_observed_G.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ci, concept in enumerate(("head", "wall")):
        q = metrics[metrics.concept.eq(concept)]
        axes[ci].bar(np.arange(4), q.rank_spearman_vs_full, color="#6A8FB3")
        axes[ci].axhline(1, color="#444", lw=.8, ls="--")
        axes[ci].set_title(f"{concept}: order agreement with full image")
        axes[ci].set_xticks(np.arange(4), [LABELS[m] for m in METHODS], rotation=25, ha="right")
        axes[ci].set_ylabel("Spearman rank correlation")
        axes[ci].set_ylim(0, 1.05); axes[ci].grid(axis="y", color="#E5E5E5", lw=.55); axes[ci].set_axisbelow(True)
    fig.savefig(PILOT / "08_unit3447_head_wall_order_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

if __name__ == "__main__":
    main()

