"""Symmetrically orthogonalize G/T/D and plot their complete 2-D composition."""

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
OUT = PROJECT / "results" / "orthogonal_temporal_modes_2d_high_consistency_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
CONSISTENCY_THRESHOLD = .8
N_PER_ROI = 200
SEED = 20260827
BATCH = 128


def unit_vector(x: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(x))
    return x / norm if np.isfinite(norm) and norm > 1e-10 else None


def raw_directions(template: np.ndarray, times: np.ndarray):
    smooth = gaussian_filter1d(template.astype(float), sigma=.75, mode="nearest")
    derivative = np.gradient(smooth, times)
    energy = smooth**2
    center = float(np.sum(times * energy) / max(float(np.sum(energy)), 1e-12))
    vectors = [unit_vector(smooth), unit_vector(-derivative),
               unit_vector(-(times - center) * derivative - .5 * smooth)]
    if any(x is None for x in vectors):
        return None, center
    return np.column_stack(vectors), center


def symmetric_orthogonalize(basis: np.ndarray):
    gram = basis.T @ basis
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    if np.min(eigenvalue) <= 1e-8:
        return None, eigenvalue, None
    inverse_sqrt = eigenvector @ np.diag(eigenvalue ** -.5) @ eigenvector.T
    orthogonal = basis @ inverse_sqrt
    similarity = np.diag(basis.T @ orthogonal)
    return orthogonal, eigenvalue, similarity


def compute(meta: pd.DataFrame) -> pd.DataFrame:
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    baseline_idx = np.flatnonzero(windows[:, 1] < 0)
    time_idx = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time_idx].mean(axis=1).astype(float)
    units = meta.unit_global.to_numpy(int)
    coefficients = np.lib.format.open_memmap(
        OUT / "image_orthogonal_direction_coefficients.npy", mode="w+", dtype=np.float32,
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
            physical, center = raw_directions(template, times)
            row = {"unit_global": int(take[local]), "template_center_ms": center}
            if physical is None:
                row["identifiable"] = False
                rows.append(row); continue
            orthogonal, eigenvalue, similarity = symmetric_orthogonalize(physical)
            if orthogonal is None:
                row["identifiable"] = False
                rows.append(row); continue
            delta = y - template[None]
            coef = delta @ orthogonal
            energy = np.sum(coef**2, axis=0)
            weights = energy / max(float(np.sum(energy)), 1e-12)
            mode1 = float((2 * weights[0] - weights[1] - weights[2]) / np.sqrt(6))
            mode2 = float((weights[1] - weights[2]) / np.sqrt(2))
            physical_gram = physical.T @ physical
            row.update({
                "identifiable": True,
                "w_G": float(weights[0]), "w_T": float(weights[1]), "w_D": float(weights[2]),
                "mode1_gain_vs_temporal": mode1, "mode2_shift_vs_duration": mode2,
                "raw_cos_G_T": float(physical_gram[0, 1]),
                "raw_cos_G_D": float(physical_gram[0, 2]),
                "raw_cos_T_D": float(physical_gram[1, 2]),
                "gram_min_eigenvalue": float(eigenvalue.min()),
                "similarity_orthG_rawG": float(similarity[0]),
                "similarity_orthT_rawT": float(similarity[1]),
                "similarity_orthD_rawD": float(similarity[2]),
            })
            coefficients[:, :, start + local] = coef.T.astype(np.float32)
            rows.append(row)
        coefficients.flush()
        print(f"orthogonal modes {stop}/{len(units)}", flush=True)
    return meta.merge(pd.DataFrame(rows), on="unit_global", validate="one_to_one")


def pair_plot(data: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str,
              filename: str):
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        ax.scatter(q[x], q[y],
                   s=19, alpha=.52, color=COLORS[roi], edgecolors="none",
                   label=f"{LABELS[roi]} (n={len(q)})", zorder=2)
        ax.scatter(q[x].median(), q[y].median(),
                   s=120, marker="D", color=COLORS[roi], edgecolors="white",
                   linewidths=1.5, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel + " weight")
    ax.set_ylabel(ylabel + " weight")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E0E0E0", lw=.55, alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    ax.set_title("Orthogonalized temporal-direction weights", fontweight="bold")
    fig.text(.5, .006,
             "200 response-blind sampled units per ROI (released consistency > 0.8). Löwdin-orthogonalized G/T/D projections; diamonds are ROI medians.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(data: pd.DataFrame):
    rows = []
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        rows.append({
            "roi": roi, "n_units": len(q),
            "median_w_G": float(q.w_G.median()), "median_w_T": float(q.w_T.median()),
            "median_w_D": float(q.w_D.median()),
            "median_mode1": float(q.mode1_gain_vs_temporal.median()),
            "median_mode2": float(q.mode2_shift_vs_duration.median()),
            "median_similarity_G": float(q.similarity_orthG_rawG.median()),
            "median_similarity_T": float(q.similarity_orthT_rawT.median()),
            "median_similarity_D": float(q.similarity_orthD_rawD.median()),
            "median_gram_min_eigenvalue": float(q.gram_min_eigenvalue.median()),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    eligible = meta[meta.roi.isin(ROIS) &
                    meta.released_independent_consistency.gt(CONSISTENCY_THRESHOLD)]
    sampled = []
    counts = {}
    for ri, roi in enumerate(ROIS):
        pool = eligible[eligible.roi.eq(roi)]
        counts[roi] = int(len(pool))
        if len(pool) < N_PER_ROI:
            raise RuntimeError(f"{roi}: only {len(pool)} eligible units")
        sampled.append(pool.sample(N_PER_ROI, random_state=SEED + ri))
    selected = pd.concat(sampled, ignore_index=True).sort_values("unit_global").reset_index(drop=True)
    data = compute(selected)
    data = data[data.identifiable].copy()
    data.to_csv(OUT / "unit_orthogonal_GTD_weights_and_modes.csv", index=False, encoding="utf-8-sig")
    roi_summary = summarize(data)
    roi_summary.to_csv(OUT / "roi_orthogonal_GTD_summary.csv", index=False, encoding="utf-8-sig")
    pair_plot(data, "w_G", "w_T", "Orthogonal G", "Orthogonal T", "01_orthogonal_G_vs_T.png")
    pair_plot(data, "w_G", "w_D", "Orthogonal G", "Orthogonal D", "02_orthogonal_G_vs_D.png")
    pair_plot(data, "w_T", "w_D", "Orthogonal T", "Orthogonal D", "03_orthogonal_T_vs_D.png")
    audit = {
        "discovery_selection": f"released_independent_consistency > {CONSISTENCY_THRESHOLD}",
        "eligible_counts_by_roi": counts, "n_per_roi": N_PER_ROI,
        "sampling_seed": SEED, "n_selected": int(len(selected)),
        "n_identifiable": int(len(data)), "window_ms": [70, 189], "n_images": 1000,
        "orthogonalization": "symmetric Lowdin B(B'B)^(-1/2), per unit",
        "weights": "orthogonal coefficient energy normalized across G/T/D",
        "mode1": "(2wG-wT-wD)/sqrt(6)", "mode2": "(wT-wD)/sqrt(2)",
        "roi_labels_used_to_define_directions_or_modes": False,
        "proxy_features_used": False,
        "filter_on_orthogonalization_diagnostics": False,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(roi_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
