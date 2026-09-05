"""Estimate image-dependent gain, timing, and duration modulation by ROI.

The three coordinates are defined without ROI labels.  Unit-specific mean
response properties are removed before modulation magnitudes are calculated.
Only the common 70--189 ms physiological response window is used; tapered
edge weights prevent either boundary from dominating a parameter estimate.
"""

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
from scipy.stats import kruskal, mannwhitneyu, spearmanr


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "image_dependent_temporal_parameters_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
COLORS = dict(zip(ROIS, ["#2166AC", "#2CA25F", "#7B3294", "#E6AB02", "#D95F0E", "#C51B7D"]))
SEED = 20260827
N_IMAGES = 1000
BATCH = 128
RIDGE = 0.03
SHIFT_GRID_MS = np.arange(-40.0, 40.1, 10.0)
LOG_DURATION_GRID = np.linspace(-0.5, 0.5, 9)

PARAMS = [
    ("gain_modulation", "Fractional gain modulation (robust SD)"),
    ("timing_modulation_ms", "Relative timing modulation (ms, robust SD)"),
    ("duration_modulation", "Log-duration modulation (robust SD)"),
]


def robust_sd(x: np.ndarray, axis: int = 0) -> np.ndarray:
    q25, q75 = np.nanpercentile(x, [25, 75], axis=axis)
    return (q75 - q25) / 1.349


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def warped_template_bank(template: np.ndarray, times: np.ndarray, weights: np.ndarray):
    """Build a bounded nonlinear shift/duration bank in physical coordinates."""
    smooth = gaussian_filter1d(template.astype(float), sigma=0.75, mode="nearest")
    energy = weights * smooth**2
    center = float(np.sum(times * energy) / max(np.sum(energy), 1e-12))
    bank, labels = [], []
    for shift in SHIFT_GRID_MS:
        for log_duration in LOG_DURATION_GRID:
            duration = float(np.exp(log_duration))
            source_time = center + (times - center - shift) / duration
            warped = np.interp(source_time, times, smooth, left=0.0, right=0.0)
            bank.append(warped)
            labels.append((shift, log_duration))
    bank = np.asarray(bank, np.float32)
    norm = np.sum(weights[None] * bank**2, axis=1) + RIDGE * max(float(np.sum(energy)), 1e-8)
    return bank, np.asarray(labels, float), norm, smooth, center


def fit_parameters(curves: np.ndarray, template_images: np.ndarray,
                   times: np.ndarray, weights: np.ndarray):
    """Fit template gain and model-free temporal moments.

    Timing and duration are defined only for images above the training-half
    median evoked energy. This avoids assigning a time coordinate to images
    for which the unit barely responds.
    """
    template = np.nanmean(curves[template_images], axis=0)
    smooth = gaussian_filter1d(template.astype(float), sigma=0.75, mode="nearest")
    template_energy = float(np.sum(weights * smooth**2))
    template_temporal_energy = weights * smooth**2
    center = float(np.sum(times * template_temporal_energy) /
                   max(np.sum(template_temporal_energy), 1e-12))
    if not np.isfinite(template_energy) or template_energy <= 1e-8:
        return None, {"template_center_ms": center, "template_energy": template_energy}
    smoothed_curves = gaussian_filter1d(curves.astype(float), sigma=0.75, axis=1, mode="nearest")
    gain = ((smoothed_curves * weights[None]) @ smooth) / (
        template_energy + RIDGE * template_energy)
    temporal_energy = weights[None] * smoothed_curves**2
    total = np.sum(temporal_energy, axis=1)
    timing = np.sum(temporal_energy * times[None], axis=1) / np.maximum(total, 1e-12)
    duration = np.sqrt(np.sum(
        temporal_energy * (times[None] - timing[:, None])**2, axis=1
    ) / np.maximum(total, 1e-12))
    energy_threshold = float(np.nanmedian(total[template_images]))
    low_response = total <= max(energy_threshold, 1e-12)
    timing[low_response] = np.nan
    duration[low_response] = np.nan
    beta = np.column_stack([gain, timing, np.log(np.maximum(duration, 1e-6))])
    # The fixed effects are explicitly outside the three image coordinates.
    alpha = np.nanmedian(beta[template_images], axis=0)
    delta = beta - alpha[None]
    info = {
        "alpha_gain": float(alpha[0]),
        "alpha_timing_ms": float(alpha[1]),
        "alpha_log_duration": float(alpha[2]),
        "template_center_ms": center,
        "template_energy": template_energy,
        "timing_duration_support_fraction": float(np.mean(~low_response)),
        "energy_threshold": energy_threshold,
    }
    return delta, info


def compute_unit_parameters(meta: pd.DataFrame):
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    baseline = np.flatnonzero(windows[:, 1] < 0)
    use = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[use].mean(axis=1).astype(float)
    weights = np.ones(len(use), float)
    weights[:2] = [0.25, 0.75]
    weights[-2:] = [0.75, 0.25]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(N_IMAGES)
    half_a, half_b = np.sort(order[:500]), np.sort(order[500:])
    units = meta.unit_global.to_numpy(int)
    rows = []
    full_parameter_file = OUT / "image_dependent_parameters.npy"
    full_parameters = np.lib.format.open_memmap(
        full_parameter_file, mode="w+", dtype=np.float32,
        shape=(3, N_IMAGES, len(units)),
    )
    full_parameters[:] = np.nan

    for start in range(0, len(units), BATCH):
        stop = min(start + BATCH, len(units))
        take = units[start:stop]
        raw = np.asarray(responses[:, :, take], np.float32)
        base = np.nanmean(raw[baseline], axis=0)
        curves = (raw[use] - base[None]).transpose(2, 1, 0)
        for local, y in enumerate(curves):
            global_col = start + local
            delta_full, info = fit_parameters(y, np.arange(N_IMAGES), times, weights)
            delta_a, _ = fit_parameters(y, half_a, times, weights)
            delta_b, _ = fit_parameters(y, half_b, times, weights)
            row = {"unit_global": int(take[local]), **info}
            if delta_full is None or delta_a is None or delta_b is None:
                row.update({
                    "identifiable": False,
                    "gain_modulation": np.nan,
                    "timing_modulation_ms": np.nan,
                    "duration_modulation": np.nan,
                })
            else:
                full_parameters[:, :, global_col] = delta_full.T.astype(np.float32)
                scale_a = robust_sd(delta_a[half_b], axis=0)
                scale_b = robust_sd(delta_b[half_a], axis=0)
                scale = np.sqrt(np.maximum(scale_a, 0) * np.maximum(scale_b, 0))
                row.update({
                    "identifiable": True,
                    "gain_modulation": float(scale[0]),
                    "timing_modulation_ms": float(scale[1]),
                    "duration_modulation": float(scale[2]),
                    "gain_modulation_A_to_B": float(scale_a[0]),
                    "timing_modulation_ms_A_to_B": float(scale_a[1]),
                    "duration_modulation_A_to_B": float(scale_a[2]),
                    "gain_modulation_B_to_A": float(scale_b[0]),
                    "timing_modulation_ms_B_to_A": float(scale_b[1]),
                    "duration_modulation_B_to_A": float(scale_b[2]),
                })
            rows.append(row)
        full_parameters.flush()
        print(f"parameters {stop}/{len(units)}", flush=True)
    data = meta.merge(pd.DataFrame(rows), on="unit_global", how="left", validate="one_to_one")
    return data, windows[use].tolist(), weights.tolist(), half_a, half_b


def residualize_nuisance(data: pd.DataFrame) -> pd.DataFrame:
    """Remove measured quality/unit-type effects without using ROI labels."""
    out = data.copy()
    numeric = []
    for col in ["released_independent_consistency", "snr", "snrmax", "reliability_basic"]:
        if col in out and out[col].notna().sum() > len(out) // 2:
            values = pd.to_numeric(out[col], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            values = values.fillna(values.median())
            transformed = np.log1p(np.maximum(values.to_numpy(float), 0))
            transformed = np.nan_to_num(transformed, nan=0.0,
                                        posinf=np.nanmax(transformed[np.isfinite(transformed)]))
            numeric.append(transformed)
    x_parts = [np.ones(len(out))]
    x_parts.extend(numeric)
    if "unit_type_name" in out:
        dummies = pd.get_dummies(out.unit_type_name.astype(str), drop_first=True, dtype=float)
        x_parts.extend([dummies[c].to_numpy(float) for c in dummies])
    x = np.column_stack(x_parts)
    for metric, _ in PARAMS:
        y = np.log(np.maximum(out[metric].to_numpy(float), 1e-6))
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        out[metric + "_adjusted"] = y - x @ beta
    return out


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    center = float(np.mean(values)); total = float(np.sum((values - center) ** 2))
    between = sum(np.sum(labels == label) * (float(np.mean(values[labels == label])) - center) ** 2
                  for label in np.unique(labels))
    return float(between / max(total, 1e-12))


def statistical_tests(data: pd.DataFrame):
    session = data.groupby(["roi", "monkey", "session"], as_index=False)[
        [p[0] for p in PARAMS] + [p[0] + "_adjusted" for p in PARAMS]
    ].median()
    omnibus, pairwise = [], []
    for metric, _ in PARAMS:
        adjusted = metric + "_adjusted"
        groups = [session.loc[session.roi.eq(r), adjusted].to_numpy(float) for r in ROIS]
        h, p = kruskal(*[g for g in groups if len(g)])
        order = session.roi.map({r: i for i, r in enumerate(ROIS)}).to_numpy(float)
        rho, rho_p = spearmanr(order, session[adjusted].to_numpy(float))
        omnibus.append({
            "parameter": metric,
            "n_units": len(data),
            "n_session_roi_groups": len(session),
            "session_kruskal_h_adjusted": float(h),
            "session_kruskal_p_adjusted": float(p),
            "session_hierarchy_spearman_rho_adjusted": float(rho),
            "session_hierarchy_spearman_p_adjusted": float(rho_p),
            "unit_roi_eta2_raw_descriptive": eta_squared(
                np.log(np.maximum(data[metric].to_numpy(float), 1e-6)), data.roi.to_numpy(str)),
            "unit_roi_eta2_adjusted_descriptive": eta_squared(
                data[adjusted].to_numpy(float), data.roi.to_numpy(str)),
        })
        start = len(pairwise)
        for left, right in combinations(ROIS, 2):
            a = session.loc[session.roi.eq(left), adjusted].to_numpy(float)
            b = session.loc[session.roi.eq(right), adjusted].to_numpy(float)
            stat, pp = mannwhitneyu(a, b, alternative="two-sided")
            pairwise.append({"parameter": metric, "roi_1": left, "roi_2": right,
                             "n_1": len(a), "n_2": len(b), "u": float(stat), "p": float(pp)})
        end = len(pairwise)
        corrected = bh_adjust(np.asarray([x["p"] for x in pairwise[start:end]]))
        for item, q in zip(pairwise[start:end], corrected):
            item["p_bh_within_parameter"] = float(q)
    omni = pd.DataFrame(omnibus)
    omni["session_kruskal_p_bh_three_parameters"] = bh_adjust(
        omni.session_kruskal_p_adjusted.to_numpy(float))
    return session, omni, pd.DataFrame(pairwise)


def pair_plot(data: pd.DataFrame, xmetric: str, ymetric: str, xlabel: str, ylabel: str,
              filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    for roi in ROIS:
        q = data[data.roi.eq(roi)]
        ax.scatter(q[xmetric], q[ymetric], s=8, alpha=0.20, color=COLORS[roi],
                   edgecolors="none", rasterized=True, label=f"{LABELS[roi]} (n={len(q):,})")
        ax.scatter(q[xmetric].median(), q[ymetric].median(), s=95, color=COLORS[roi],
                   edgecolors="white", linewidths=1.3, marker="D")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(color="#D7D7D7", lw=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, markerscale=2.0, fontsize=9, ncol=2)
    ax.set_title("Image-dependent temporal modulation across visual ROIs", fontweight="bold")
    fig.text(0.5, 0.006,
             "Every point is one identifiable unit; diamonds are ROI medians. Coordinates are cross-half robust modulation magnitudes, not discriminant modes.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_figure(data: pd.DataFrame, session: pd.DataFrame, tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for ax, (metric, label), row in zip(axes, PARAMS, tests.itertuples()):
        values = [data.loc[data.roi.eq(r), metric].to_numpy(float) for r in ROIS]
        violin = ax.violinplot(values, positions=np.arange(6), widths=.82, showmedians=True,
                               showextrema=False, points=150)
        for body, roi in zip(violin["bodies"], ROIS):
            body.set_facecolor(COLORS[roi]); body.set_alpha(.28); body.set_edgecolor("none")
        violin["cmedians"].set_color("#222222")
        for ri, roi in enumerate(ROIS):
            q = session[session.roi.eq(roi)]
            ax.scatter(np.full(len(q), ri), q[metric], s=24, color=COLORS[roi],
                       edgecolors="#222222", linewidths=.4, alpha=.8)
        ax.set_yscale("log"); ax.set_xticks(np.arange(6), [LABELS[r] for r in ROIS])
        ax.set_ylabel(label); ax.grid(axis="y", color="#D7D7D7", lw=.6)
        ax.set_title(f"Kruskal p(BH)={row.session_kruskal_p_bh_three_parameters:.3g}\n"
                     f"hierarchy rho={row.session_hierarchy_spearman_rho_adjusted:+.2f}")
    fig.suptitle("ROI differences in image-dependent temporal modulation", fontweight="bold")
    fig.text(0.5, 0.005,
             "Violin: all units. Dots: monkey x session medians. Tests use session medians after removing measured reliability/SNR and unit-type effects without ROI labels.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / "04_roi_parameter_distributions_and_tests.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    meta = meta[meta.roi.isin(ROIS) & meta.released_independent_consistency.gt(0.4)].copy()
    meta = meta.sort_values("unit_global").reset_index(drop=True)
    data, used_windows, weights, half_a, half_b = compute_unit_parameters(meta)
    finite = np.isfinite(data[[p[0] for p in PARAMS]].to_numpy(float)).all(axis=1)
    positive = (data[[p[0] for p in PARAMS]].to_numpy(float) > 0).all(axis=1)
    data = data[data.identifiable & finite & positive].copy()
    data = residualize_nuisance(data)
    data.to_csv(OUT / "unit_image_dependent_temporal_parameters.csv", index=False, encoding="utf-8-sig")
    session, tests, pairwise = statistical_tests(data)
    session.to_csv(OUT / "session_roi_parameter_medians.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "roi_parameter_omnibus_tests.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT / "roi_parameter_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    pair_plot(data, PARAMS[0][0], PARAMS[1][0], PARAMS[0][1], PARAMS[1][1],
              "01_gain_vs_timing.png")
    pair_plot(data, PARAMS[0][0], PARAMS[2][0], PARAMS[0][1], PARAMS[2][1],
              "02_gain_vs_duration.png")
    pair_plot(data, PARAMS[1][0], PARAMS[2][0], PARAMS[1][1], PARAMS[2][1],
              "03_timing_vs_duration.png")
    summary_figure(data, session, tests)
    audit = {
        "selection": "released_independent_consistency > 0.4; all mapped unit types",
        "n_units_selected": int(len(meta)), "n_units_identifiable": int(len(data)),
        "n_images": N_IMAGES, "image_split_seed": SEED,
        "physiological_window_ms": [70, 189], "used_windows": used_windows,
        "edge_reliability_weights": weights,
        "parameterization": "template-projection gain plus model-free response-energy centroid and width",
        "timing_duration_support": "images above the training-half median evoked energy",
        "identifiability_rule": "finite positive modulation estimates replicated across two disjoint 500-image halves",
        "fixed_effects_excluded_from_coordinates": ["unit mean gain", "unit mean timing", "unit mean duration", "pre-stimulus baseline"],
        "roi_labels_used_to_define_parameters": False,
        "proxy_or_semantic_features_used": False,
        "inference_level": "monkey x session x ROI median",
        "nuisance_adjustment": "reliability/SNR fields and unit type; no ROI labels",
        "important_limitation": "released responses are image-condition means, not single trials",
        "image_half_a": half_a.tolist(), "image_half_b": half_b.tolist(),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tests.to_string(index=False), flush=True)
    print(data.groupby("roi").size().reindex(ROIS).to_string(), flush=True)


if __name__ == "__main__":
    main()
