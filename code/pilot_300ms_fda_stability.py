"""Small-sample 0-300 ms robustness check for FDA coordinates.

The existing response cache ends at 189 ms. This pilot reads the released raw
H5 response matrices for five reliable units per ROI, bins them to 10 ms, and
compares the existing 70-189 ms window with a 0-299 ms window. It is a
representation-level stability check; no 300 ms proxy predictions are claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "pilot_300ms_fda_parameters_2026-08-30"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V4": "V4", "posterior IT": "pIT",
             "middle IT": "mIT", "anterior IT": "aIT"}
ROI_COLOR = {"V1": "#2166AC", "V2": "#1B9E77", "V4": "#6A3D9A",
             "posterior IT": "#E6AB02", "middle IT": "#E66101",
             "anterior IT": "#C51B7D"}
N_PER_ROI = 5
N_IMAGES = 1000
TIME_ZERO_INDEX = 49
BASELINE_SLICE = slice(TIME_ZERO_INDEX - 20, TIME_ZERO_INDEX)
BIN_STARTS = np.arange(0, 300, 10)
BIN_TIMES = BIN_STARTS.astype(float) + 4.5
WINDOWS = {"short_70_189": np.flatnonzero((BIN_STARTS >= 70) & (BIN_STARTS <= 180)),
           "long_0_299": np.arange(len(BIN_STARTS))}
FDA_NAMES = ("A", "tau", "lambda", "peak_amplitude", "peak_latency",
                "positive_auc", "positive_width")


def unit_vector(x: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(x))
    return x / norm if np.isfinite(norm) and norm > 1e-10 else None


def fda_basis(template: np.ndarray, times: np.ndarray):
    smooth = gaussian_filter1d(template.astype(float), sigma=.75, mode="nearest")
    derivative = np.gradient(smooth, times)
    energy = smooth ** 2
    center = float(np.sum(times * energy) / max(float(np.sum(energy)), 1e-12))
    vectors = [unit_vector(smooth), unit_vector(-derivative),
               unit_vector(-(times - center) * derivative - .5 * smooth)]
    if any(v is None for v in vectors):
        return None
    return np.column_stack(vectors)


def traditional_scalars(y: np.ndarray, times: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter1d(y.astype(float), .75, mode="nearest")
    positive = np.maximum(smooth, 0.0)
    mass = float(positive.sum())
    peak = int(np.argmax(smooth))
    center = float(np.average(times, weights=positive)) if mass > 1e-9 else float(times.mean())
    width = float(np.sqrt(np.average((times - center) ** 2, weights=positive))) if mass > 1e-9 else np.nan
    return np.array([smooth[peak], times[peak], mass, width], dtype=float)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cv_reconstruction(y: np.ndarray, features: np.ndarray, seed: int = 20260830) -> float:
    """Five-fold held-out curve R2 from a low-dimensional representation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, 5)
    pred = np.empty_like(y, dtype=float)
    for test in folds:
        train = np.setdiff1d(order, test, assume_unique=True)
        mean = features[train].mean(0)
        std = np.maximum(features[train].std(0), 1e-9)
        x_train = np.column_stack([np.ones(len(train)), (features[train] - mean) / std])
        x_test = np.column_stack([np.ones(len(test)), (features[test] - mean) / std])
        beta = np.linalg.lstsq(x_train, y[train], rcond=None)[0]
        pred[test] = x_test @ beta
    sse = float(np.sum((y - pred) ** 2))
    centered = y - y.mean(0, keepdims=True)
    sst = float(np.sum(centered ** 2))
    return 1.0 - sse / sst if sst > 1e-12 else np.nan


def choose_units(meta: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for roi in ROIS:
        q = meta[meta.roi.eq(roi)].sort_values("released_independent_consistency", ascending=False)
        selected.append(q.head(N_PER_ROI))
    out = pd.concat(selected, ignore_index=True)
    # Keep the previously discussed MF example unit in the pilot whenever possible.
    if 3447 not in set(out.unit_global.astype(int)):
        mf = meta[meta.unit_global.eq(3447)]
        if not mf.empty:
            replace = out.index[out.roi.eq("middle IT")][-1]
            out.loc[replace] = mf.iloc[0]
    return out.sort_values(["roi", "unit_global"]).reset_index(drop=True)


def read_binned_curve(row: pd.Series) -> np.ndarray:
    with h5py.File(str(row.source_h5), "r") as handle:
        source = handle["response_matrix_img"]
        baseline = np.asarray(source[BASELINE_SLICE, :N_IMAGES, int(row.unit_index_zero_based)], np.float32).mean(0)
        raw = np.asarray(source[TIME_ZERO_INDEX:TIME_ZERO_INDEX + 300, :N_IMAGES,
                               int(row.unit_index_zero_based)], np.float32)
    return raw.reshape(len(BIN_STARTS), 10, N_IMAGES).mean(1).T - baseline[:, None]


def analyze_unit(row: pd.Series) -> dict[str, object]:
    curves = read_binned_curve(row)
    valid_images = np.isfinite(curves).all(axis=1)
    curves = curves[valid_images]
    result: dict[str, object] = {
        "unit_global": int(row.unit_global), "roi": str(row.roi),
        "session": int(row.session), "native_area": str(row.native_area),
        "released_independent_consistency": float(row.released_independent_consistency),
    }
    raw_metrics = {}
    for name, indices in WINDOWS.items():
        y = curves[:, indices]
        t = BIN_TIMES[indices]
        template = y.mean(0)
        basis = fda_basis(template, t)
        if basis is None:
            continue
        fda = (y - template[None]) @ basis
        scalar = np.vstack([traditional_scalars(image, t) for image in y])
        key = "short" if name.startswith("short") else "long"
        raw_metrics[key] = (fda, scalar, y, t)
    if not all(k in raw_metrics for k in ("short", "long")):
        return result
    common_keep = np.ones(len(curves), dtype=bool)
    for fda, scalar, y, _ in raw_metrics.values():
        common_keep &= np.isfinite(y).all(1) & np.isfinite(fda).all(1) & np.isfinite(scalar).all(1)
    short_metrics = {}; long_metrics = {}
    for key, (fda, scalar, y, t) in raw_metrics.items():
        fda, scalar, y = fda[common_keep], scalar[common_keep], y[common_keep]
        result[f"{key}_n_images"] = int(len(y))
        features_fda = fda
        features_trad = scalar
        short_metrics[key] = (fda, scalar, y, t)
        long_metrics[key] = (fda, scalar, y, t)
        result[f"{key}_reconstruction_r2_traditional"] = cv_reconstruction(y, features_trad, 20260830)
        result[f"{key}_reconstruction_r2_fda"] = cv_reconstruction(y, features_fda, 20260831)
        result[f"{key}_delta_r2_fda"] = (result[f"{key}_reconstruction_r2_fda"] -
                                          result[f"{key}_reconstruction_r2_traditional"])
    if "short" in short_metrics and "long" in long_metrics:
        fs, ss, _, _ = short_metrics["short"]
        fl, sl, _, _ = long_metrics["long"]
        for j, metric in enumerate(("A", "tau", "lambda")):
            result[f"stability_{metric}_r"] = corr(fs[:, j], fl[:, j])
            result[f"stability_{metric}_rank_r"] = corr(np.argsort(np.argsort(fs[:, j])),
                                                          np.argsort(np.argsort(fl[:, j])))
        for j, metric in enumerate(("peak_amplitude", "peak_latency", "positive_auc", "positive_width")):
            result[f"stability_{metric}_r"] = corr(ss[:, j], sl[:, j])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    units = choose_units(meta)
    rows = [analyze_unit(row) for _, row in units.iterrows()]
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "unit_300ms_fda_parameter_stability.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for roi, q in detail.groupby("roi", sort=False):
        row = {"roi": roi, "n_units": len(q)}
        for metric in ("A", "tau", "lambda", "peak_amplitude", "peak_latency", "positive_auc", "positive_width"):
            row[f"median_stability_{metric}_r"] = float(q[f"stability_{metric}_r"].median())
        for key in ("short", "long"):
            row[f"median_{key}_delta_r2_fda"] = float(q[f"{key}_delta_r2_fda"].median())
            row[f"positive_{key}_delta_fraction"] = float(np.mean(q[f"{key}_delta_r2_fda"] > 0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "roi_300ms_fda_parameter_stability_summary.csv", index=False, encoding="utf-8-sig")

    # Figure 1: parameter stability between the two windows.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for ax, metric in zip(axes, ("A", "tau", "lambda")):
        col = f"stability_{metric}_r"
        for roi, q in detail.groupby("roi", sort=False):
            ax.scatter(q[f"stability_{metric}_r"], np.zeros(len(q)), s=0)  # keep color registry deterministic
        vals = detail[col].to_numpy(float)
        ax.hist(vals[np.isfinite(vals)], bins=np.linspace(-1, 1, 21), color="#4C78A8", alpha=.78, edgecolor="white")
        ax.axvline(0, color="#777", lw=.8); ax.axvline(np.nanmedian(vals), color="#D45050", lw=2)
        ax.set(xlim=(-1, 1), xlabel="Pearson r: 70–189 vs 0–299 ms", title=f"FDA {metric} stability\nmedian={np.nanmedian(vals):.3f}")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Number of pilot units")
    fig.suptitle("FDA parameter stability when extending the time window", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "fda_parameter_stability_70_189_vs_0_299.png", dpi=220)
    plt.close(fig)

    # Figure 2: incremental curve reconstruction by ROI and window.
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    x = np.arange(len(ROIS)); width = .36
    short = [summary.loc[summary.roi.eq(r), "median_short_delta_r2_fda"].iloc[0] for r in ROIS]
    long = [summary.loc[summary.roi.eq(r), "median_long_delta_r2_fda"].iloc[0] for r in ROIS]
    ax.bar(x - width / 2, short, width, label="70–189 ms", color="#4C78A8")
    ax.bar(x + width / 2, long, width, label="0–299 ms", color="#E76F51")
    ax.axhline(0, color="#555", lw=.8); ax.set_xticks(x, [ROI_LABEL[r] for r in ROIS])
    ax.set_ylabel("Median FDA reconstruction ΔR²")
    ax.set_title("Does FDA add curve information under the longer window?")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "fda_parameter_incremental_delta_r2_by_window.png", dpi=220)
    plt.close(fig)

    audit = {
        "n_units": int(len(detail)), "units_per_roi": N_PER_ROI,
        "included_mf_unit_3447": bool(3447 in set(detail.unit_global.astype(int))),
        "raw_source": "Triple-N response_matrix_img",
        "raw_sampling": "1 ms source, averaged into 10 ms bins",
        "windows_ms": {"short": [70, 189], "long": [0, 299]},
        "baseline_ms": [-20, -1],
        "proxy_predictions_300ms": False,
        "parameterization": "FDA tangent coordinates: A=amplitude, tau=phase/translation, lambda=time dilation",
        "interpretation": "FDA-parameter representation stability and curve reconstruction only",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
