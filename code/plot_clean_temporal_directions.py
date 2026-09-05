"""Three clean response-derived temporal directions across visual ROIs."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kruskal, mannwhitneyu


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "clean_temporal_directions_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
METRICS = [
    ("gain_weight", "Gain weight"),
    ("timing_weight", "Time-shift weight"),
    ("duration_weight", "Duration weight"),
]
BATCH = 128
N_PER_ROI = 1200
SEED = 20260827


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def clean_basis(template: np.ndarray, times: np.ndarray, weights: np.ndarray):
    """Weighted orthonormal gain, shift and dilation directions."""
    g = gaussian_filter1d(template.astype(float), 0.75, mode="nearest")
    derivative = np.gradient(g, times)
    energy = weights * g**2
    center = float(np.sum(times * energy) / max(np.sum(energy), 1e-12))
    raw = np.column_stack([g, -derivative, -(times - center) * derivative])
    weighted = raw * np.sqrt(weights)[:, None]
    q, r = np.linalg.qr(weighted)
    strength = np.abs(np.diag(r))
    if np.min(strength) <= 1e-8:
        return None, center, float(np.linalg.cond(r))
    # q is orthonormal in the weighted response space. Fix arbitrary signs so
    # the three directions remain aligned with their physical generators.
    for k in range(3):
        if float(q[:, k] @ weighted[:, k]) < 0:
            q[:, k] *= -1
    return q, center, float(np.linalg.cond(r))


def compute(meta: pd.DataFrame) -> pd.DataFrame:
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    baseline_idx = np.flatnonzero(windows[:, 1] < 0)
    time_idx = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time_idx].mean(axis=1).astype(float)
    weights = np.ones(len(time_idx), float)
    weights[:2] = [0.25, 0.75]
    weights[-2:] = [0.75, 0.25]
    units = meta.unit_global.to_numpy(int)
    rows = []
    coefficients = np.lib.format.open_memmap(
        OUT / "image_direction_coefficients.npy", mode="w+", dtype=np.float32,
        shape=(3, 1000, len(units)))
    coefficients[:] = np.nan

    for start in range(0, len(units), BATCH):
        stop = min(start + BATCH, len(units))
        take = units[start:stop]
        raw = np.asarray(responses[:, :, take], np.float32)
        baseline = np.nanmean(raw[baseline_idx], axis=0)
        curves = (raw[time_idx] - baseline[None]).transpose(2, 1, 0)
        for local, y in enumerate(curves):
            template = np.nanmean(y, axis=0)
            basis, center, condition = clean_basis(template, times, weights)
            row = {"unit_global": int(take[local]), "template_center_ms": center,
                   "basis_condition": condition, "identifiable": basis is not None}
            if basis is None:
                row.update({m: np.nan for m, _ in METRICS})
            else:
                delta = y - template[None]
                coef = (delta * np.sqrt(weights)[None]) @ basis
                # By construction every coefficient has zero image mean. RMS
                # is the direct magnitude of the original projection weights.
                rms = np.sqrt(np.nanmean(coef**2, axis=0))
                row.update(dict(zip([m for m, _ in METRICS], rms.tolist())))
                coefficients[:, :, start + local] = coef.T.astype(np.float32)
            rows.append(row)
        coefficients.flush()
        print(f"directions {stop}/{len(units)}", flush=True)
    return meta.merge(pd.DataFrame(rows), on="unit_global", validate="one_to_one")


def tests(data: pd.DataFrame):
    metrics = [m for m, _ in METRICS]
    session = data.groupby(["roi", "monkey", "session"], as_index=False)[metrics].median()
    omnibus, pairwise = [], []
    for metric in metrics:
        groups = [session.loc[session.roi.eq(roi), metric].to_numpy(float) for roi in ROIS]
        h, p = kruskal(*[x for x in groups if len(x)])
        omnibus.append({"direction": metric, "n_units": len(data),
                        "n_session_roi_groups": len(session),
                        "session_kruskal_h": float(h), "session_kruskal_p": float(p)})
        start = len(pairwise)
        for left, right in combinations(ROIS, 2):
            a = session.loc[session.roi.eq(left), metric].to_numpy(float)
            b = session.loc[session.roi.eq(right), metric].to_numpy(float)
            u, pp = mannwhitneyu(a, b, alternative="two-sided")
            pairwise.append({"direction": metric, "roi_1": left, "roi_2": right,
                             "n_1": len(a), "n_2": len(b), "u": float(u), "p": float(pp)})
        corrected = bh(np.asarray([x["p"] for x in pairwise[start:]]))
        for row, q in zip(pairwise[start:], corrected):
            row["p_bh_within_direction"] = float(q)
    omnibus = pd.DataFrame(omnibus)
    omnibus["p_bh_three_directions"] = bh(omnibus.session_kruskal_p.to_numpy(float))
    return session, omnibus, pd.DataFrame(pairwise)


def pair_plot(data: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, name: str):
    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        ax.scatter(q[x], q[y], s=7, alpha=.18, color=COLORS[roi], edgecolors="none",
                   rasterized=True, label=f"{LABELS[roi]} (n={len(q):,})")
        ax.scatter(q[x].median(), q[y].median(), s=100, marker="D", color=COLORS[roi],
                   edgecolors="white", linewidths=1.4)
    ax.set(xlabel=xlabel + " (RMS projection)", ylabel=ylabel + " (RMS projection)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(color="#D8D8D8", lw=.6, alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncol=2, markerscale=2)
    ax.set_title("Clean image-dependent temporal directions", fontweight="bold")
    fig.text(.5, .006,
             "One point per unit; diamonds are ROI medians. Linear axes, no binning, clustering, logarithm, or discriminant mode.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def distribution_plot(data: pd.DataFrame, session: pd.DataFrame, omnibus: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), constrained_layout=True)
    for ax, (metric, label), stat in zip(axes, METRICS, omnibus.itertuples()):
        arrays = [data.loc[data.roi.eq(r), metric].to_numpy(float) for r in ROIS]
        violin = ax.violinplot(arrays, positions=np.arange(6), widths=.82,
                               showmedians=True, showextrema=False, points=150)
        for body, roi in zip(violin["bodies"], ROIS):
            body.set_facecolor(COLORS[roi]); body.set_alpha(.28); body.set_edgecolor("none")
        violin["cmedians"].set_color("#222222")
        for ri, roi in enumerate(ROIS):
            q = session[session.roi.eq(roi)]
            ax.scatter(np.full(len(q), ri), q[metric], s=22, color=COLORS[roi],
                       edgecolors="#222222", linewidths=.35, alpha=.8)
        ax.set_xticks(np.arange(6), [LABELS[r] for r in ROIS])
        ax.set_ylabel(label + " (RMS projection)")
        ax.set_ylim(bottom=0); ax.grid(axis="y", color="#D8D8D8", lw=.6)
        ax.set_title(f"session Kruskal p(BH)={stat.p_bh_three_directions:.3g}")
    fig.suptitle("ROI differences in three clean temporal directions", fontweight="bold")
    fig.text(.5, .006, "Violin: all units. Dots: monkey x session medians used for inference.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / "04_direction_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    meta = meta[meta.roi.isin(ROIS) & meta.released_independent_consistency.gt(.4)]
    # Balance ROIs before looking at any response curve or direction weight.
    # The fixed random sample is intentionally blind to neural outcomes.
    sampled = []
    for ri, roi in enumerate(ROIS):
        pool = meta[meta.roi.eq(roi)]
        if len(pool) < N_PER_ROI:
            raise RuntimeError(f"{roi}: only {len(pool)} eligible units; need {N_PER_ROI}")
        sampled.append(pool.sample(N_PER_ROI, random_state=SEED + ri))
    meta = pd.concat(sampled, ignore_index=True).sort_values("unit_global").reset_index(drop=True)
    data = compute(meta)
    data = data[data.identifiable].copy()
    data.to_csv(OUT / "unit_clean_direction_weights.csv", index=False, encoding="utf-8-sig")
    session, omnibus, pairwise = tests(data)
    session.to_csv(OUT / "session_roi_direction_medians.csv", index=False, encoding="utf-8-sig")
    omnibus.to_csv(OUT / "direction_omnibus_tests.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT / "direction_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    pair_plot(data, "gain_weight", "timing_weight", "Gain", "Time-shift", "01_gain_vs_timing.png")
    pair_plot(data, "gain_weight", "duration_weight", "Gain", "Duration", "02_gain_vs_duration.png")
    pair_plot(data, "timing_weight", "duration_weight", "Time-shift", "Duration", "03_timing_vs_duration.png")
    distribution_plot(data, session, omnibus)
    audit = {
        "selection": "released_independent_consistency > 0.4; all mapped unit types; response-blind fixed random sample",
        "n_per_roi": N_PER_ROI, "sampling_seed": SEED,
        "n_selected": int(len(meta)), "n_identifiable": int(len(data)),
        "window_ms": [70, 189], "n_images": 1000,
        "directions": ["mean-template gain", "template temporal derivative", "centered dilation derivative"],
        "orthogonalization": "weighted QR in the declared direction order",
        "unit_fixed_effect_removed": "each unit's mean image-response curve",
        "unit_coordinate": "RMS of raw image projection coefficients",
        "transformations_not_used": ["log axes", "clustering", "ROI-discriminant modes", "nuisance regression"],
        "roi_labels_used_to_define_directions": False,
        "proxy_features_used": False,
        "inference": "Kruskal-Wallis on monkey x session x ROI medians; BH across three directions",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(omnibus.to_string(index=False), flush=True)
    print(data.groupby("roi").size().reindex(ROIS).to_string(), flush=True)


if __name__ == "__main__":
    main()
