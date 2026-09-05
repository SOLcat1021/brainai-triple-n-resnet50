"""Test continuous versus boundary-discontinuous temporal dynamics along IT penetrations.

The analysis is response-only: temporal endpoints are computed from released
image-condition mean curves. ResNet predictions and semantic labels are not used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
QUALITY = PROJECT / "cohorts" / "high_quality_64d_pilot_2026-08-26" / "all_unit_quality_audit.csv"
OUT = PROJECT / "results" / "functional_boundary_gradients_2026-08-27"
SESSIONS = (24, 29, 30, 60, 61)
PAIR_BY_SESSION = {
    24: ("MB", "MF"),
    29: ("AF", "AB"),
    30: ("AF", "AB"),
    60: ("AF", "AB"),
    61: ("AF", "AB"),
}
BIN_WIDTH_UM = 150.0
MIN_UNITS_PER_BIN = 3
MIN_BINS_PER_SIDE = 3


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-12 else 0.0


def orthonormal_columns(columns: list[np.ndarray]) -> np.ndarray:
    x = np.column_stack(columns)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > max(float(s[0]) * 1e-7, 1e-10)]


def temporal_features(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    post = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    pre = windows[:, 1] < 0
    times = windows[post].mean(1)
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[:, :, units], np.float32).transpose(2, 1, 0)
    baseline = y[:, :, pre].mean((1, 2))
    evoked = y[:, :, post].mean(1) - baseline[:, None]
    evoked_norm = evoked / np.maximum(np.linalg.norm(evoked, axis=1, keepdims=True), 1e-8)

    covariance, descriptors = [], []
    tri = np.triu_indices(len(post))
    for ui, yy in enumerate(y[:, :, post]):
        covariance.append(np.nan_to_num(np.corrcoef(yy, rowvar=False)[tri], nan=0.0))
        centered = yy - yy.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        frac = s**2 / max(float(np.sum(s**2)), 1e-12)
        h1, h2 = vt[0], vt[1]
        e1 = h1**2 / max(float(np.sum(h1**2)), 1e-12)
        e2 = h2**2 / max(float(np.sum(h2**2)), 1e-12)
        nuisance = orthonormal_columns([h1, np.gradient(h1), np.gradient(np.gradient(h1))])
        residual = centered - (centered @ nuisance) @ nuisance.T
        nonwarp_fraction = float(np.sum(residual**2) / max(float(np.sum(centered**2)), 1e-12))
        mean_energy = evoked[ui] ** 2
        evoked_centroid = float(times @ mean_energy / max(float(mean_energy.sum()), 1e-12))
        descriptors.append([
            evoked_centroid,
            float(times @ e1),
            float(times @ e2),
            abs(corr(h2, np.gradient(h1))),
            float(np.exp(-np.sum(frac * np.log(np.maximum(frac, 1e-12))))),
            nonwarp_fraction,
            *frac[:5],
        ])
    return {
        "evoked_curve": evoked_norm,
        "stimulus_temporal_covariance": np.asarray(covariance),
        "response_dynamic_descriptors": np.asarray(descriptors),
    }


def reduce_block(x: np.ndarray, fit_mask: np.ndarray, max_pc: int = 8) -> np.ndarray:
    x = np.asarray(x, float)
    fit = x[fit_mask]
    median = np.nanmedian(fit, axis=0)
    x = np.where(np.isfinite(x), x, median[None])
    mean, sd = x[fit_mask].mean(0), x[fit_mask].std(0)
    keep = sd > 1e-8
    z = (x[:, keep] - mean[keep]) / sd[keep]
    _, s, vt = np.linalg.svd(z[fit_mask], full_matrices=False)
    variance = s**2
    k90 = int(np.searchsorted(np.cumsum(variance) / max(float(variance.sum()), 1e-12), 0.90) + 1)
    k = min(max_pc, max(2, k90), len(s))
    scores = z @ vt[:k].T
    scale = np.sqrt(np.mean(scores[fit_mask] ** 2))
    return scores / max(float(scale), 1e-12)


def design(x: np.ndarray, side: np.ndarray, model: str) -> np.ndarray:
    columns = [np.ones_like(x), x, x**2]
    if model in {"jump", "piecewise"}:
        columns.append(side)
    if model == "piecewise":
        columns.append(side * x)
    return np.column_stack(columns)


def loo_sse(x: np.ndarray, side: np.ndarray, y: np.ndarray, weights: np.ndarray, model: str) -> float:
    X = design(x, side, model)
    total = 0.0
    for test in range(len(x)):
        train = np.arange(len(x)) != test
        sw = np.sqrt(weights[train])[:, None]
        coef = np.linalg.lstsq(X[train] * sw, y[train] * sw, rcond=None)[0]
        residual = y[test] - X[test] @ coef
        total += float(weights[test] * np.sum(residual**2))
    return total


def boundary_midpoint(q: pd.DataFrame, first: str, second: str) -> float:
    left_end = q.loc[q.native_area.eq(first), "position_range_end_um"].astype(float).median()
    right_start = q.loc[q.native_area.eq(second), "position_range_start_um"].astype(float).median()
    return float((left_end + right_start) / 2.0)


def binned_data(q: pd.DataFrame, values: np.ndarray, boundary: float) -> tuple[pd.DataFrame, np.ndarray]:
    position = q.electrode_position_um.to_numpy(float)
    bin_id = np.floor((position - boundary) / BIN_WIDTH_UM).astype(int)
    rows, means = [], []
    for bid in np.unique(bin_id):
        idx = np.flatnonzero(bin_id == bid)
        if len(idx) < MIN_UNITS_PER_BIN:
            continue
        rows.append({
            "bin_id": int(bid),
            "position_um": float(position[idx].mean()),
            "relative_position_mm": float((position[idx].mean() - boundary) / 1000.0),
            "side": int(position[idx].mean() >= boundary),
            "n_units": int(len(idx)),
        })
        means.append(values[idx].mean(0))
    return pd.DataFrame(rows), np.asarray(means)


def model_comparison(bins: pd.DataFrame, y: np.ndarray) -> dict[str, float] | None:
    if bins.empty or bins.side.value_counts().min() < MIN_BINS_PER_SIDE:
        return None
    x = bins.relative_position_mm.to_numpy(float)
    side = bins.side.to_numpy(float)
    weights = bins.n_units.to_numpy(float)
    center = np.average(y, axis=0, weights=weights)
    total = float(np.sum(weights[:, None] * (y - center) ** 2))
    sse = {name: loo_sse(x, side, y, weights, name) for name in ("continuous", "jump", "piecewise")}
    X = design(x, side, "jump")
    sw = np.sqrt(weights)[:, None]
    coef = np.linalg.lstsq(X * sw, y * sw, rcond=None)[0]
    actual_delta = (sse["continuous"] - sse["jump"]) / max(total, 1e-12)

    placebo = []
    for cut in np.sort(x)[:-1]:
        fake_side = (x > cut).astype(float)
        counts = np.bincount(fake_side.astype(int), minlength=2)
        if counts.min() < MIN_BINS_PER_SIDE:
            continue
        fake = loo_sse(x, fake_side, y, weights, "jump")
        placebo.append((sse["continuous"] - fake) / max(total, 1e-12))
    placebo = np.asarray(placebo, float)
    return {
        "n_bins": int(len(bins)),
        "n_bins_first": int(np.sum(side == 0)),
        "n_bins_second": int(np.sum(side == 1)),
        "continuous_cv_r2": 1.0 - sse["continuous"] / max(total, 1e-12),
        "jump_cv_r2": 1.0 - sse["jump"] / max(total, 1e-12),
        "piecewise_cv_r2": 1.0 - sse["piecewise"] / max(total, 1e-12),
        "jump_minus_continuous_cv_r2": actual_delta,
        "piecewise_minus_continuous_cv_r2": (sse["continuous"] - sse["piecewise"]) / max(total, 1e-12),
        "standardized_jump_norm": float(np.linalg.norm(coef[3])),
        "placebo_boundary_p": float((1 + np.sum(placebo >= actual_delta)) / (1 + len(placebo))),
        "n_placebo_boundaries": int(len(placebo)),
    }


def scalar_bin_rows(meta: pd.DataFrame, descriptors: np.ndarray, cohort_masks: dict[str, np.ndarray]) -> pd.DataFrame:
    names = ["evoked_centroid_ms", "mode1_centroid_ms", "mode2_centroid_ms",
             "mode2_derivative_abs_r", "effective_temporal_rank", "nonwarp_residual_fraction"]
    rows = []
    for cohort, mask in cohort_masks.items():
        for session in SESSIONS:
            idx = np.flatnonzero(mask & meta.session.eq(session).to_numpy())
            if not len(idx):
                continue
            q = meta.iloc[idx].reset_index(drop=True)
            first, second = PAIR_BY_SESSION[session]
            boundary = boundary_midpoint(q, first, second)
            bin_id = np.floor((q.electrode_position_um.to_numpy(float) - boundary) / BIN_WIDTH_UM).astype(int)
            for bid in np.unique(bin_id):
                local = np.flatnonzero(bin_id == bid)
                if len(local) < MIN_UNITS_PER_BIN:
                    continue
                position = float(q.electrode_position_um.iloc[local].mean())
                for ci, name in enumerate(names):
                    rows.append({
                        "cohort": cohort, "session": session, "monkey": q.monkey.iloc[0],
                        "site": int(q.recording_site_index.iloc[0]), "pair": f"{first}-{second}",
                        "metric": name, "bin_id": int(bid), "n_units": int(len(local)),
                        "relative_position_mm": (position - boundary) / 1000.0,
                        "side": second if position >= boundary else first,
                        "mean": float(np.mean(descriptors[idx[local], ci])),
                        "sem": float(np.std(descriptors[idx[local], ci], ddof=1) / np.sqrt(len(local))) if len(local) > 1 else 0.0,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    quality = pd.read_csv(QUALITY)[["unit_global", "tier_A_strict"]]
    meta = meta.merge(quality, on="unit_global", how="left", validate="one_to_one")
    keep = meta.session.isin(SESSIONS)
    keep &= np.asarray([
        int(session) in PAIR_BY_SESSION and area in PAIR_BY_SESSION[int(session)]
        for session, area in zip(meta.session, meta.native_area)
    ])
    meta = meta[keep].sort_values(["session", "electrode_position_um", "unit_global"]).reset_index(drop=True)
    raw = temporal_features(meta)
    neural_reliable = meta.released_independent_consistency.ge(0.40).to_numpy()
    reduced = {name: reduce_block(value, neural_reliable) for name, value in raw.items()}
    reduced["combined_balanced"] = np.concatenate(
        [value / np.sqrt(value.shape[1]) for value in reduced.values()], axis=1
    )
    cohorts = {
        "all_mapped": np.ones(len(meta), bool),
        "neural_reliable_ge_0.40": neural_reliable,
        "tier_A_strict": meta.tier_A_strict.fillna(False).to_numpy(bool),
    }

    results, bin_manifest = [], []
    for cohort, cohort_mask in cohorts.items():
        for session in SESSIONS:
            idx = np.flatnonzero(cohort_mask & meta.session.eq(session).to_numpy())
            if not len(idx):
                continue
            q = meta.iloc[idx].reset_index(drop=True)
            first, second = PAIR_BY_SESSION[session]
            boundary = boundary_midpoint(q, first, second)
            for feature_view, full_values in reduced.items():
                bins, y = binned_data(q, full_values[idx], boundary)
                if not bins.empty:
                    for row in bins.itertuples(index=False):
                        bin_manifest.append({"cohort": cohort, "session": session, "feature_view": feature_view,
                                             "boundary_um": boundary, **row._asdict()})
                comparison = model_comparison(bins, y)
                if comparison is None:
                    continue
                results.append({
                    "cohort": cohort, "session": session, "monkey": q.monkey.iloc[0],
                    "site": int(q.recording_site_index.iloc[0]), "pair": f"{first}-{second}",
                    "feature_view": feature_view, "n_units": int(len(q)), "boundary_um": boundary,
                    **comparison,
                })

    result = pd.DataFrame(results)
    scalar = scalar_bin_rows(meta, raw["response_dynamic_descriptors"], cohorts)
    result.to_csv(OUT / "boundary_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bin_manifest).to_csv(OUT / "spatial_bin_manifest.csv", index=False, encoding="utf-8-sig")
    scalar.to_csv(OUT / "scalar_temporal_metrics_by_depth_bin.csv", index=False, encoding="utf-8-sig")
    audit = {
        "response_only": True,
        "proxy_predictions_used": False,
        "semantic_labels_used": False,
        "sessions": list(SESSIONS),
        "pairs": {str(k): list(v) for k, v in PAIR_BY_SESSION.items()},
        "bin_width_um": BIN_WIDTH_UM,
        "minimum_units_per_bin": MIN_UNITS_PER_BIN,
        "minimum_bins_per_side": MIN_BINS_PER_SIDE,
        "models": {
            "continuous": "quadratic position",
            "jump": "quadratic position plus boundary indicator",
            "piecewise": "quadratic position plus boundary indicator and side-specific linear slope",
        },
        "primary_cohort": "neural_reliable_ge_0.40",
        "sensitivity_cohorts": ["all_mapped", "tier_A_strict"],
        "independent_sites": sorted(meta.recording_site_index.unique().astype(int).tolist()),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
