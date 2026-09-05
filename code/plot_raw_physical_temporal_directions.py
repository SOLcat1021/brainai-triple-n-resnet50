"""Plot raw, non-orthogonal gain, time-shift, and duration alignment scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "raw_physical_temporal_directions_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
N_PER_ROI = 1200
SEED = 20260827
BATCH = 128


def unit_vector(x: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(x))
    return x / norm if np.isfinite(norm) and norm > 1e-10 else None


def physical_directions(template: np.ndarray, times: np.ndarray):
    """Return raw unit G, T, D vectors without mutual orthogonalization."""
    smooth = gaussian_filter1d(template.astype(float), sigma=.75, mode="nearest")
    derivative = np.gradient(smooth, times)
    energy = smooth**2
    center = float(np.sum(times * energy) / max(float(np.sum(energy)), 1e-12))
    g = unit_vector(smooth)
    t = unit_vector(-derivative)
    d = unit_vector(-(times - center) * derivative - .5 * smooth)
    if g is None or t is None or d is None:
        return None, center
    return np.column_stack([g, t, d]), center


def compute(meta: pd.DataFrame) -> pd.DataFrame:
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    baseline_idx = np.flatnonzero(windows[:, 1] < 0)
    time_idx = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time_idx].mean(axis=1).astype(float)
    units = meta.unit_global.to_numpy(int)
    coefficients = np.lib.format.open_memmap(
        OUT / "image_raw_direction_coefficients.npy", mode="w+", dtype=np.float32,
        shape=(3, 1000, len(units)))
    coefficients[:] = np.nan
    rows = []

    for start in range(0, len(units), BATCH):
        stop = min(start + BATCH, len(units))
        take = units[start:stop]
        raw = np.asarray(responses[:, :, take], np.float32)
        baseline = np.nanmean(raw[baseline_idx], axis=0)
        curves = (raw[time_idx] - baseline[None]).transpose(2, 1, 0)
        for local, y in enumerate(curves):
            template = np.nanmean(y, axis=0)
            directions, center = physical_directions(template, times)
            row = {"unit_global": int(take[local]), "template_center_ms": center,
                   "identifiable": directions is not None}
            if directions is None:
                row.update({"G_score": np.nan, "T_score": np.nan, "D_score": np.nan,
                            "cos_G_T": np.nan, "cos_G_D": np.nan, "cos_T_D": np.nan})
            else:
                delta = y - template[None]
                total_energy = float(np.sum(delta**2))
                coef = delta @ directions
                score = np.sum(coef**2, axis=0) / max(total_energy, 1e-12)
                gram = directions.T @ directions
                row.update({
                    "G_score": float(score[0]), "T_score": float(score[1]),
                    "D_score": float(score[2]), "total_image_variation_energy": total_energy,
                    "cos_G_T": float(gram[0, 1]), "cos_G_D": float(gram[0, 2]),
                    "cos_T_D": float(gram[1, 2]),
                })
                coefficients[:, :, start + local] = coef.T.astype(np.float32)
            rows.append(row)
        coefficients.flush()
        print(f"raw directions {stop}/{len(units)}", flush=True)
    return meta.merge(pd.DataFrame(rows), on="unit_global", validate="one_to_one")


def pair_plot(data: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, filename: str):
    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        ax.scatter(q[x], q[y], s=8, alpha=.22, color=COLORS[roi], edgecolors="none",
                   rasterized=True, label=f"{LABELS[roi]} (n={len(q):,})")
        ax.scatter(q[x].median(), q[y].median(), s=105, marker="D", color=COLORS[roi],
                   edgecolors="white", linewidths=1.4)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel + " alignment score")
    ax.set_ylabel(ylabel + " alignment score")
    ax.grid(color="#D8D8D8", lw=.6, alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=2, markerscale=2)
    ax.set_title("Raw physical temporal-direction scores", fontweight="bold")
    fig.text(.5, .006,
             "Each ROI is a response-blind random sample of 1,200 stable units. Linear axes; raw G/T/D directions are normalized but not mutually orthogonalized.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        rows.append({
            "roi": roi, "n_units": len(q),
            "median_G_score": float(q.G_score.median()),
            "median_T_score": float(q.T_score.median()),
            "median_D_score": float(q.D_score.median()),
            "median_score_sum_nonexclusive": float((q.G_score + q.T_score + q.D_score).median()),
            "median_abs_cos_G_T": float(q.cos_G_T.abs().median()),
            "median_abs_cos_G_D": float(q.cos_G_D.abs().median()),
            "median_abs_cos_T_D": float(q.cos_T_D.abs().median()),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    eligible = meta[meta.roi.isin(ROIS) & meta.released_independent_consistency.gt(.4)]
    sampled = []
    for ri, roi in enumerate(ROIS):
        pool = eligible[eligible.roi.eq(roi)]
        if len(pool) < N_PER_ROI:
            raise RuntimeError(f"{roi}: only {len(pool)} eligible units")
        sampled.append(pool.sample(N_PER_ROI, random_state=SEED + ri))
    selected = pd.concat(sampled, ignore_index=True).sort_values("unit_global").reset_index(drop=True)
    data = compute(selected)
    data = data[data.identifiable].copy()
    data.to_csv(OUT / "unit_raw_physical_direction_scores.csv", index=False, encoding="utf-8-sig")
    roi_summary = summary(data)
    roi_summary.to_csv(OUT / "roi_raw_direction_summary.csv", index=False, encoding="utf-8-sig")
    pair_plot(data, "G_score", "T_score", "G: gain", "T: time shift", "01_G_vs_T.png")
    pair_plot(data, "G_score", "D_score", "G: gain", "D: duration", "02_G_vs_D.png")
    pair_plot(data, "T_score", "D_score", "T: time shift", "D: duration", "03_T_vs_D.png")
    audit = {
        "selection": "released_independent_consistency > 0.4; response-blind random sample",
        "n_per_roi": N_PER_ROI, "sampling_seed": SEED,
        "n_selected": int(len(selected)), "n_identifiable": int(len(data)),
        "n_images": 1000, "window_ms": [70, 189],
        "G": "unit-normalized mean response template",
        "T": "unit-normalized negative temporal derivative of template",
        "D": "unit-normalized energy-preserving dilation derivative: -(t-tc)s'(t)-0.5s(t)",
        "directions_mutually_orthogonalized": False,
        "score": "sum_i projection_coefficient^2 / sum_i ||image response - unit template||^2",
        "score_sum_constraint": "none because raw directions are not orthogonal",
        "roi_labels_used_to_define_directions": False,
        "proxy_features_used": False,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(roi_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
