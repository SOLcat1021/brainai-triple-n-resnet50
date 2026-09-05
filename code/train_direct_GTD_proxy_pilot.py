"""Train a small, genuinely GTD-supervised proxy pilot.

Each unit x G/T/D scalar target independently learns the original proxy's
continuous spatial field, smooth ResNet-depth gate, and signed ridge readout.
Only the neural supervision target changes; no firing-rate proxy parameters
are reused.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
ROOT = PROJECT.parent
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))

import run_trin_closed_loop_validation as foldbase  # noqa: E402
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack  # noqa: E402
from plot_orthogonal_temporal_modes_2d import raw_directions, symmetric_orthogonalize  # noqa: E402
from train_all_units_final_resnet50 import DIMS, N_LAYERS, normalize  # noqa: E402
from train_full_10ms_proxy_bank import (  # noqa: E402
    predict_with_parameters, select_spatial_and_gate,
)


BANK = PROJECT / "proxy_bank_10ms_all_windows"
FROZEN = PROJECT / "results" / "orthogonal_temporal_modes_2d_high_consistency_2026-08-27"
OLD = PROJECT / "results" / "proxy_GTD_oof_recovery_discovery1200_2026-08-27"
NOISE = PROJECT / "support" / "single_trial_nc1_by_window.csv"
MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "block_pca8_14_tvsd_frozen" / "maps_f16.npy"
OUT = PROJECT / "results" / "direct_GTD_proxy_pilot24_independent_2026-08-27"

ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
ROI_COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
DIRECTIONS = ["G", "T", "D"]
PER_ROI = 4
SEED = 20260827
NOISE_VARIANCE_SCALE = 1_000_000.0  # archived spikes/ms variance -> curve spikes/s variance


def corr_columns(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    a = np.asarray(target, np.float64) - np.mean(target, axis=0, keepdims=True)
    b = np.asarray(prediction, np.float64) - np.mean(prediction, axis=0, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    return np.divide(np.sum(a * b, axis=0), den, out=np.full(den.shape, np.nan), where=den > 1e-12)


def select_units() -> pd.DataFrame:
    frozen = pd.read_csv(FROZEN / "unit_orthogonal_GTD_weights_and_modes.csv")
    rng = np.random.default_rng(SEED)
    parts = []
    for roi in ROIS:
        q = frozen[frozen.roi.eq(roi)].sort_values("unit_global")
        take = np.sort(rng.choice(len(q), PER_ROI, replace=False))
        parts.append(q.iloc[take])
    selected = pd.concat(parts, ignore_index=True)
    if len(selected) != PER_ROI * len(ROIS):
        raise RuntimeError("Pilot unit selection changed")
    return selected


def neural_coefficients(neural: np.ndarray, train: np.ndarray, times: np.ndarray):
    n_units, n_images, n_time = neural.shape
    target = np.empty((n_images, n_units, 3), np.float32)
    basis = np.empty((n_units, n_time, 3), np.float32)
    for unit in range(n_units):
        template = neural[unit, train].mean(0)
        raw, _ = raw_directions(template, times)
        orthogonal, _, _ = symmetric_orthogonalize(raw)
        basis[unit] = orthogonal
        target[:, unit] = (neural[unit] - template[None]) @ orthogonal
    return target, basis


def train_fold(raw_features: np.ndarray, train: np.ndarray, test: np.ndarray,
               target: np.ndarray, device: torch.device, fold: int,
               bank: torch.Tensor, xs: np.ndarray, ys: np.ndarray, sigmas: np.ndarray):
    """Exact original scalar-target proxy fit over flattened unit x direction."""
    ftr, fte = normalize(raw_features, train, test, device)
    n_units = target.shape[1]
    flat_target = target.reshape(len(target), n_units * 3)
    y_train = flat_target[train]
    cx, cy, cs, gate = select_spatial_and_gate(
        ftr, y_train, bank, xs, ys, sigmas, seed=SEED + fold)
    prediction, _, _ = predict_with_parameters(
        ftr, fte, y_train, cx, cy, cs, gate, save_weights=False)
    target_index = np.arange(n_units * 3)
    parameters = pd.DataFrame({
        "fold": fold, "target_index": target_index,
        "unit_local": target_index // 3,
        "direction": np.asarray(DIRECTIONS)[target_index % 3],
        "center_x": cx, "center_y": cy, "sigma_space": cs,
        "center_depth": gate @ np.arange(N_LAYERS, dtype=np.float32),
    })
    del ftr, fte
    torch.cuda.empty_cache()
    return prediction.reshape(len(test), n_units, 3).astype(np.float32), gate.reshape(
        n_units, 3, N_LAYERS), parameters


def boundary_complete_noise(noise: pd.DataFrame, unit: int, times: np.ndarray, column: str):
    q = noise[noise.unit_global.eq(unit)].sort_values("window_start_ms")
    lookup = dict(zip(q.window_start_ms.astype(int), q[column].astype(float)))
    available = np.asarray(sorted(lookup), int)
    if not len(available):
        raise RuntimeError(f"No noise rows for unit {unit}")
    first = float(np.max([lookup[x] for x in available[:min(3, len(available))]]))
    last = float(np.max([lookup[x] for x in available[-min(3, len(available)):]]))
    values = []
    for start in times.astype(int):
        if start in lookup:
            values.append(lookup[start])
        elif start < available.min():
            values.append(first)
        elif start > available.max():
            values.append(last)
        else:
            lo = available[available < start].max(); hi = available[available > start].min()
            weight = (start - lo) / (hi - lo)
            values.append((1 - weight) * lookup[lo] + weight * lookup[hi])
    return np.asarray(values, float) * NOISE_VARIANCE_SCALE, first * NOISE_VARIANCE_SCALE


def attach_noise_ceiling(metrics: pd.DataFrame, selected: pd.DataFrame,
                         target_oof: np.ndarray, bases: np.ndarray, time_starts: np.ndarray):
    noise = pd.read_csv(NOISE)
    rows = []
    for ui, unit in enumerate(selected.unit_global.astype(int)):
        repeat_noise, _ = boundary_complete_noise(
            noise, unit, time_starts, "repeat_mean_noise_variance")
        single_noise, _ = boundary_complete_noise(
            noise, unit, time_starts, "single_trial_noise_variance")
        for di, direction in enumerate(DIRECTIONS):
            projected_repeat, projected_single = [], []
            for fold in range(5):
                vector = bases[fold, ui, :, di].astype(float)
                projected_repeat.append(np.sum(vector ** 2 * repeat_noise))
                projected_single.append(np.sum(vector ** 2 * single_noise))
            repeat_variance = float(np.mean(projected_repeat))
            single_variance = float(np.mean(projected_single))
            observed = float(np.var(target_oof[:, ui, di], ddof=1))
            signal = max(observed - repeat_variance, 0.0)
            repeat_ceiling = math.sqrt(signal / (signal + repeat_variance)) if signal > 0 else 0.0
            single_ceiling = math.sqrt(signal / (signal + single_variance)) if signal > 0 else 0.0
            attenuation = math.sqrt((signal + repeat_variance) /
                                    (signal + single_variance)) if signal + single_variance > 0 else 0.0
            rows.append({
                "unit_global": unit, "direction": direction,
                "observed_coefficient_variance": observed,
                "propagated_repeat_mean_noise_variance": repeat_variance,
                "propagated_single_trial_noise_variance": single_variance,
                "estimated_signal_variance": signal,
                "repeat_mean_r_ceiling": repeat_ceiling,
                "single_trial_r_ceiling": single_ceiling,
                "single_trial_attenuation": attenuation,
            })
    ceiling = pd.DataFrame(rows)
    merged = metrics.merge(ceiling, on=["unit_global", "direction"], validate="one_to_one")
    merged["single_trial_equivalent_oof_r"] = merged.oof_r * merged.single_trial_attenuation
    merged["fraction_single_trial_correlation_ceiling"] = np.divide(
        merged.single_trial_equivalent_oof_r, merged.single_trial_r_ceiling,
        out=np.full(len(merged), np.nan), where=merged.single_trial_r_ceiling > 0)
    merged["fraction_repeat_mean_ceiling"] = np.divide(
        merged.oof_r, merged.repeat_mean_r_ceiling,
        out=np.full(len(merged), np.nan), where=merged.repeat_mean_r_ceiling > 0)
    return merged


def scatter_single_trial_ceiling(ax: plt.Axes, frame: pd.DataFrame, direction: str):
    q = frame[frame.direction.eq(direction)]
    x = q.single_trial_r_ceiling.to_numpy(float)
    y = q.single_trial_equivalent_oof_r.to_numpy(float)
    ax.scatter(x, y, s=34, alpha=.38, color="#72ADD4", edgecolors="none")
    ax.plot([0, 1], [0, 1], color="#F06D6A", lw=1.4)
    ax.scatter([np.nanmean(x)], [np.nanmean(y)], marker="+", s=150,
               linewidths=3, color="#D6273B", zorder=5)
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    ax.text(.04, .96,
            f"n={len(q):,}\nmean: ({np.nanmean(x):.3f}, {np.nanmean(y):.3f})\nSpearman ρ={rho:.3f}",
            transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(-.25, 1)
    ax.grid(True, color="#D9D9D9", lw=.55, alpha=.75)
    ax.set_title(f"Relative {direction} coefficient", loc="left", fontweight="bold", fontsize=13)
    ax.set_xlabel("Single-trial correlation ceiling, √NC(1)")
    ax.set_ylabel("Single-trial-equivalent OOF Pearson r")


def make_figures(metrics: pd.DataFrame):
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 1.0,
                         "figure.dpi": 160, "savefig.facecolor": "white"})
    for index, direction in enumerate(DIRECTIONS, 1):
        fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        scatter_single_trial_ceiling(ax, metrics, direction)
        fig.suptitle("Direct G/T/D-supervised unit proxy pilot",
                     fontweight="bold", fontsize=15)
        fig.savefig(OUT / f"0{index}_{direction}_single_trial_noise_ceiling.png",
                    dpi=320, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharey=True, constrained_layout=True)
    for ax, direction in zip(axes, DIRECTIONS):
        q = metrics[metrics.direction.eq(direction)].copy()
        x = np.arange(len(q))
        ax.plot([x, x], [q.old_windowed_r, q.oof_r], color="#B5B5B5", lw=.7)
        ax.scatter(x, q.old_windowed_r, s=24, facecolor="white", edgecolor="#555555",
                   label="Old windowed proxy")
        ax.scatter(x, q.oof_r, s=24, c=[ROI_COLORS[r] for r in q.roi],
                   edgecolor="none", label="New direct GTD proxy")
        ax.axhline(0, color="#555555", lw=.7)
        ax.set_title(direction, fontweight="bold"); ax.set_xlabel("Pilot units")
        ax.set_xticks([]); ax.grid(axis="y", color="#D9D9D9", lw=.45)
    axes[0].set_ylabel("OOF Pearson r")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("New GTD supervision vs old firing-rate supervision", fontweight="bold")
    fig.savefig(OUT / "04_new_direct_vs_old_windowed_oof_r.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    selected = select_units().sort_values(["roi", "unit_global"]).reset_index(drop=True)
    selected.to_csv(OUT / "pilot24_frozen_response_blind_units.csv", index=False, encoding="utf-8-sig")
    units = selected.unit_global.to_numpy(int)
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    baseline_idx = np.flatnonzero(windows[:, 1] < 0)
    time_idx = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time_idx].mean(1).astype(float)
    time_starts = windows[time_idx, 0].astype(int)
    raw_response = np.asarray(responses[:, :, units], np.float32)
    baseline = raw_response[baseline_idx].mean(0)
    neural = (raw_response[time_idx] - baseline[None]).transpose(2, 1, 0)

    flat = np.load(MAPS, mmap_mode="r")
    raw_features = flat.reshape(len(flat), N_LAYERS, DIMS, flat.shape[-2], flat.shape[-1])
    xs, ys, sigmas = candidates()
    bank = torch.as_tensor(
        gaussian_mass_stack(xs, ys, sigmas, flat.shape[-1]).reshape(len(xs), -1), device=device)

    prediction = np.full((1000, len(units), 3), np.nan, np.float32)
    target_oof = np.full_like(prediction, np.nan)
    bases = np.empty((5, len(units), len(times), 3), np.float32)
    gates = np.empty((5, len(units), 3, N_LAYERS), np.float32)
    parameter_parts = []
    for fold, (train, test) in enumerate(foldbase.fixed_folds(1000)):
        target, basis = neural_coefficients(neural, train, times)
        pred, gate, parameters = train_fold(
            raw_features, train, test, target, device, fold, bank, xs, ys, sigmas)
        prediction[test] = pred
        target_oof[test] = target[test]
        bases[fold] = basis
        gates[fold] = gate
        parameters["unit_global"] = units[parameters.unit_local.to_numpy(int)]
        parameter_parts.append(parameters)
        print(f"completed direct GTD outer fold {fold + 1}/5", flush=True)

    rows = []
    old = pd.read_csv(OLD / "unit_GTD_proxy_recovery.csv")[
        ["unit_global", "direction", "direct_r"]].rename(columns={"direct_r": "old_windowed_r"})
    for ui, row in selected.iterrows():
        r = corr_columns(target_oof[:, ui], prediction[:, ui])
        for di, direction in enumerate(DIRECTIONS):
            rows.append({
                "unit_global": int(row.unit_global), "roi": row.roi,
                "monkey": row.monkey, "session": int(row.session),
                "direction": direction, "oof_r": float(r[di]),
            })
    metrics = pd.DataFrame(rows).merge(old, on=["unit_global", "direction"], validate="one_to_one")
    metrics["delta_r_vs_old"] = metrics.oof_r - metrics.old_windowed_r
    metrics = attach_noise_ceiling(metrics, selected, target_oof, bases, time_starts)
    metrics.to_csv(OUT / "unit_direction_oof_r_and_noise_ceiling.csv", index=False, encoding="utf-8-sig")
    pd.concat(parameter_parts).to_csv(OUT / "fold_independent_GTD_spatial_depth_parameters.csv",
                                      index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / "pilot24_oof_predictions_and_targets.npz",
                        unit_global=units, target=target_oof, prediction=prediction,
                        bases=bases, gates=gates, directions=np.asarray(DIRECTIONS))
    summary = metrics.groupby("direction", as_index=False).agg(
        n=("unit_global", "size"), median_oof_r=("oof_r", "median"),
        positive_oof_fraction=("oof_r", lambda x: float(np.mean(x > 0))),
        median_old_windowed_r=("old_windowed_r", "median"),
        median_delta_r_vs_old=("delta_r_vs_old", "median"),
        mean_delta_r_vs_old=("delta_r_vs_old", "mean"),
        fraction_units_better_than_old=("delta_r_vs_old", lambda x: float(np.mean(x > 0))),
        mean_single_trial_r_ceiling=("single_trial_r_ceiling", "mean"),
        median_single_trial_r_ceiling=("single_trial_r_ceiling", "median"),
        mean_single_trial_equivalent_oof_r=("single_trial_equivalent_oof_r", "mean"),
        median_single_trial_equivalent_oof_r=("single_trial_equivalent_oof_r", "median"),
        median_fraction_single_trial_correlation_ceiling=(
            "fraction_single_trial_correlation_ceiling", "median"),
    )
    summary.to_csv(OUT / "pilot_direction_summary.csv", index=False, encoding="utf-8-sig")
    make_figures(metrics)
    audit = {
        "purpose": "small operational test of image-to-unit relative GTD mapping",
        "units": "4 response-blind random frozen units per ROI; 24 total",
        "seed": SEED, "folds": 5, "images": 1000,
        "target": "fold-train-defined relative orthogonal G/T/D coefficients over 70-189 ms",
        "model": "original scalar proxy logic applied independently to each unit x G/T/D target: continuous field + smooth depth gate + signed ridge",
        "only_change_from_firing_rate_proxy": "scalar supervision target replaced by fold-train-defined relative G/T/D coefficient",
        "parameters_reused_from_firing_rate_proxy": False,
        "primary_metric": "Pearson r over concatenated OOF predictions",
        "noise_ceiling": "diagonal propagation of archived marginal response-window variances",
        "noise_variance_unit_conversion": "multiply archived spikes/ms variances by 1000^2 to match spikes/s response curves",
        "noise_boundary_completion": "missing early windows use max of first 3 available; missing late 170-189 ms use max of last 3 available",
        "noise_ceiling_status": "pilot diagnostic; not a replacement for trial-level covariance/split-repeat ceiling",
        "noise_ceiling_limit": "cross-window covariance and shared baseline-subtraction noise are unavailable and omitted",
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
