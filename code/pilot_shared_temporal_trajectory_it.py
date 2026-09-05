"""Small IT pilot: independent-window readouts vs a shared temporal trajectory.

The visual representation, frozen fwRF fields, layer gates, 10-ms windows and
five image folds match the current 64-D TVSD bank.  The only changed part is
the temporal readout: a unit's feature weights are parameterized as
    w_u(t) = sum_k a[u,k] * psi[k,t]
where psi is learned from training-fold neural curves.  This is an exploratory
comparison, not a replacement for the main proxy bank.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from train_64d_high_quality_proxy_batch import gaussian_fields


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_consistency_ge_0p4_all_units_2026-08-30"
OLD_BANK = PROJECT / "proxy_bank_10ms_all_windows"
MAPS = PROJECT / "results" / "rank_sweep_extended_2026-08-26" / "trin_resnet50_pca256_maps_f16.npy"
OUT = PROJECT / "results" / "shared_temporal_trajectory_it_pilot_2026-09-03"

ROIS = ("posterior IT", "middle IT", "anterior IT")
ROI_FILE = {
    "posterior IT": "posterior_it_proxy_bank_64d.h5",
    "middle IT": "middle_it_proxy_bank_64d.h5",
    "anterior IT": "anterior_it_proxy_bank_64d.h5",
}
TIME_START, TIME_END = 50, 169  # 12 matched 10-ms windows, inclusive by endpoint
N_FOLDS = 5
N_IMAGES = 1000
RANK = 64
N_LAYERS = 17
GRID = 14
RIDGE = 3.0
N_UNITS_PER_ROI = 2
TEMPORAL_RANKS = (1, 2, 3, 4)
SEED = 20260818 + 901


def fixed_folds(n: int = N_IMAGES) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.random.default_rng(SEED).permutation(n)
    chunks = np.array_split(order, N_FOLDS)
    return [(np.concatenate([chunks[j] for j in range(N_FOLDS) if j != k]), chunks[k])
            for k in range(N_FOLDS)]


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, np.float64) - np.nanmean(a, axis=0, keepdims=True)
    bb = np.asarray(b, np.float64) - np.nanmean(b, axis=0, keepdims=True)
    den = np.sqrt(np.nansum(aa * aa, axis=0) * np.nansum(bb * bb, axis=0))
    return np.divide(np.nansum(aa * bb, axis=0), den,
                     out=np.full(den.shape, np.nan), where=den > 1e-12)


def curve_corr(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    y -= y.mean(); p -= p.mean()
    d = np.linalg.norm(y) * np.linalg.norm(p)
    return float(y @ p / d) if d > 1e-12 else np.nan


def temporal_basis(y_train: np.ndarray) -> np.ndarray:
    """Learn an ROI-shared, fold-local temporal basis from neural curves."""
    # y_train: images x time x units. Remove unit mean and equalize unit scale
    # so a high-firing unit cannot determine the shared path by itself.
    residual = y_train - y_train.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(residual ** 2, axis=(0, 1))).clip(min=1e-6)
    rows = np.transpose(residual / scale[None, None], (0, 2, 1)).reshape(-1, y_train.shape[1])
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    basis = vt.astype(np.float32).T
    for k in range(basis.shape[1]):
        peak = int(np.argmax(np.abs(basis[:, k])))
        if basis[peak, k] < 0:
            basis[:, k] *= -1
    return basis


def ridge_primal(z: torch.Tensor, y: torch.Tensor, ridge: float = RIDGE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit centered ridge and return coefficients plus centering values."""
    z_mean = z.mean(0)
    y_mean = y.mean()
    zc = z - z_mean
    yc = y - y_mean
    d = z.shape[1]
    lam = ridge * zc.square().sum() / max(float(d), 1.0)
    gram = zc.T @ zc
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    chol, info = torch.linalg.cholesky_ex(gram + eye * lam.clamp_min(1e-6))
    if bool((info != 0).any()):
        beta = torch.linalg.solve(gram + eye * lam.clamp_min(1e-6), zc.T @ yc)
    else:
        beta = torch.cholesky_solve((zc.T @ yc)[:, None], chol)[:, 0]
    return beta, z_mean, y_mean


def load_selection() -> pd.DataFrame:
    quality = pd.read_csv(BANK / "selected_units.csv")
    parts = []
    for roi in ROIS:
        q = quality[quality.roi.eq(roi)].sort_values(
            ["released_independent_consistency", "unit_global"], ascending=[False, True]
        ).head(N_UNITS_PER_ROI).copy()
        parts.append(q)
    selected = pd.concat(parts, ignore_index=True)
    selected.to_csv(OUT / "selected_units.csv", index=False, encoding="utf-8-sig")
    return selected


def load_h5_roi(roi: str, selected: pd.DataFrame, time_idx: np.ndarray):
    path = BANK / ROI_FILE[roi]
    with h5py.File(path, "r") as h:
        all_units = h["unit_global"][:].astype(int)
        lookup = {int(u): i for i, u in enumerate(all_units)}
        units = selected[selected.roi.eq(roi)].unit_global.to_numpy(int)
        local = np.asarray([lookup[int(u)] for u in units], dtype=np.int64)
        # h5py requires increasing fancy indices; read the small unit block via
        # NumPy after loading the selected columns.
        oof = h["oof_prediction"][time_idx, :, :].astype(np.float32)[:, :, local]
        fold_field = h["fold_spatial_parameters"][:, time_idx, :, :].astype(np.float32)[:, :, local, :]
        fold_gate = h["fold_layer_gate"][:, time_idx, :, :].astype(np.float32)[:, :, local, :]
        windows = h["windows_ms"][:].astype(int)[time_idx]
    return units, oof, fold_field, fold_gate, windows


def build_features(raw: torch.Tensor, fields: np.ndarray, gates: np.ndarray,
                   device: torch.device) -> torch.Tensor:
    """Return images x units x 1088 for one time window."""
    field = torch.as_tensor(gaussian_fields(fields), device=device)
    gate = torch.as_tensor(gates, device=device)
    x = torch.einsum("nldp,up->nuld", raw, field)
    x = x * torch.sqrt((gate * N_LAYERS).clamp_min(1e-5))[None, :, :, None]
    return x.flatten(2)


def fit_shared_for_fold(
    x_train: list[torch.Tensor], x_test: list[torch.Tensor],
    y_train: np.ndarray, basis: np.ndarray, rank: int,
    device: torch.device,
) -> np.ndarray:
    """Jointly fit each unit over images x windows with shared temporal modes."""
    n_train, n_time, n_units = y_train.shape
    d = x_train[0].shape[-1]
    psi = torch.as_tensor(basis[:, :rank], device=device)
    pred_test = np.empty((n_time, x_test[0].shape[0], n_units), np.float32)
    # The design has one block of visual features per temporal mode.
    for ui in range(n_units):
        blocks = []
        test_blocks = []
        for wi in range(n_time):
            xtr = x_train[wi][:, ui, :]
            xte = x_test[wi][:, ui, :]
            blocks.append(torch.cat([xtr * psi[wi, k] for k in range(rank)], dim=1))
            test_blocks.append(torch.cat([xte * psi[wi, k] for k in range(rank)], dim=1))
        ztr = torch.cat(blocks, dim=0)
        zte = torch.cat(test_blocks, dim=0)
        target = torch.as_tensor(y_train[:, :, ui].T.reshape(-1), device=device)
        beta, z_mean, y_mean = ridge_primal(ztr, target)
        pte = (zte - z_mean) @ beta + y_mean
        pred_test[:, :, ui] = pte.reshape(n_time, -1).detach().cpu().numpy()
        del ztr, zte, target, beta, z_mean, y_mean, pte
    return pred_test


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows_all = np.load(OLD_BANK / "windows_10ms.npy").astype(int)
    time_idx = np.flatnonzero((windows_all[:, 0] >= TIME_START) & (windows_all[:, 1] <= TIME_END))
    windows = windows_all[time_idx]
    responses = np.load(OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    maps = np.load(MAPS, mmap_mode="r")
    selected = load_selection()
    folds = fixed_folds()
    rows = []
    basis_rows = []

    for roi in ROIS:
        roi_sel = selected[selected.roi.eq(roi)].reset_index(drop=True)
        units, independent_all, fold_field, fold_gate, h5_windows = load_h5_roi(roi, selected, time_idx)
        if not np.array_equal(windows, h5_windows):
            raise RuntimeError(f"window mismatch for {roi}")
        y = np.asarray(responses[time_idx][:, :, units], np.float32).transpose(1, 0, 2)
        # Use one fold-local basis shared by the two pilot units in this ROI;
        # this keeps the pilot focused while testing genuine temporal sharing.
        for fi, (train, test) in enumerate(folds):
            raw_train = torch.as_tensor(np.asarray(maps[train, :, :RANK], np.float32), device=device).flatten(3)
            raw_test = torch.as_tensor(np.asarray(maps[test, :, :RANK], np.float32), device=device).flatten(3)
            mean = raw_train.mean((0, 3), keepdim=True)
            std = raw_train.std((0, 3), keepdim=True).clamp_min(1e-5)
            ftr = (raw_train - mean) / std
            fte = (raw_test - mean) / std
            xtr = [build_features(ftr, fold_field[fi, wi], fold_gate[fi, wi], device) for wi in range(len(time_idx))]
            xte = [build_features(fte, fold_field[fi, wi], fold_gate[fi, wi], device) for wi in range(len(time_idx))]
            basis = temporal_basis(y[train])
            for k in range(min(4, basis.shape[1])):
                for wi, t in enumerate(windows[:, 0]):
                    basis_rows.append({"roi": roi, "fold": fi, "mode": k + 1,
                                       "window_start_ms": int(t), "amplitude": float(basis[wi, k])})
            yte = y[test]
            # Existing independent-window model, restricted to exactly these windows.
            p_ind = independent_all[:, test, :].transpose(1, 0, 2)
            for ui, unit in enumerate(units):
                for wi, (t0, t1) in enumerate(windows):
                    rows.append({"roi": roi, "unit_global": int(unit), "fold": fi,
                                 "window_start_ms": int(t0), "window_end_ms": int(t1),
                                 "method": "independent_windows", "rank": np.nan,
                                 "pearson_r": float(corr_columns(yte[:, wi, ui], p_ind[:, wi, ui])[()]),
                                 "curve_r": curve_corr(yte[:, :, ui], p_ind[:, :, ui])})
            for rank in TEMPORAL_RANKS:
                p_shared = fit_shared_for_fold(xtr, xte, y[train], basis, rank, device)
                for ui, unit in enumerate(units):
                    for wi, (t0, t1) in enumerate(windows):
                        rows.append({"roi": roi, "unit_global": int(unit), "fold": fi,
                                     "window_start_ms": int(t0), "window_end_ms": int(t1),
                                     "method": "shared_temporal_trajectory", "rank": rank,
                                     "pearson_r": float(corr_columns(yte[:, wi, ui], p_shared[wi, :, ui])[()]),
                                     "curve_r": curve_corr(yte[:, :, ui], p_shared[:, :, ui])})
                del p_shared
            del raw_train, raw_test, ftr, fte, xtr, xte
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"{roi}: fold {fi + 1}/{N_FOLDS}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "same_window_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(basis_rows).to_csv(OUT / "fold_temporal_basis.csv", index=False, encoding="utf-8-sig")
    unit_summary = metrics.groupby(["roi", "unit_global", "method", "rank"], dropna=False, as_index=False).agg(
        mean_window_pearson_r=("pearson_r", "mean"), median_window_pearson_r=("pearson_r", "median"),
        curve_r=("curve_r", "mean"), n_folds=("fold", "nunique"))
    unit_summary.to_csv(OUT / "unit_summary.csv", index=False, encoding="utf-8-sig")
    summary = metrics.groupby(["roi", "method", "rank", "window_start_ms"], dropna=False, as_index=False).agg(
        mean_pearson_r=("pearson_r", "mean"), median_pearson_r=("pearson_r", "median"),
        n_unit_fold=("pearson_r", "count"))
    summary.to_csv(OUT / "same_window_summary.csv", index=False, encoding="utf-8-sig")

    # Aggregate at the unit level before comparing methods, avoiding a large
    # high-quality unit dominating a ROI-level claim.
    independent = unit_summary[unit_summary.method.eq("independent_windows")][
        ["roi", "unit_global", "mean_window_pearson_r"]
    ].rename(columns={"mean_window_pearson_r": "independent_r"})
    shared = unit_summary[unit_summary.method.eq("shared_temporal_trajectory")].copy()
    compare = shared.merge(independent, on=["roi", "unit_global"], validate="many_to_one")
    compare["delta_r"] = compare.mean_window_pearson_r - compare.independent_r
    compare.to_csv(OUT / "unit_level_comparison.csv", index=False, encoding="utf-8-sig")

    colors = {"independent_windows": "#4C78A8", 1: "#F58518", 2: "#54A24B", 3: "#E45756", 4: "#B279A2"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
    for ax, roi in zip(axes, ROIS):
        q = summary[summary.roi.eq(roi)]
        b = q[q.method.eq("independent_windows")].sort_values("window_start_ms")
        ax.plot(b.window_start_ms, b.mean_pearson_r, marker="o", ms=3, lw=1.8,
                color=colors["independent_windows"], label="Independent windows")
        for rank in TEMPORAL_RANKS:
            z = q[(q.method.eq("shared_temporal_trajectory")) & q["rank"].eq(rank)].sort_values("window_start_ms")
            ax.plot(z.window_start_ms, z.mean_pearson_r, marker="o", ms=2.5, lw=1.1,
                    color=colors[rank], label=f"Shared K={rank}")
        ax.axhline(0, color="#777", lw=.7); ax.grid(alpha=.2)
        ax.set_title(roi); ax.set_xlabel("Window start (ms)")
    axes[0].set_ylabel("Mean held-out Pearson r")
    axes[-1].legend(fontsize=8, loc="best")
    fig.savefig(OUT / "01_same_window_pearson_by_time.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    q = compare[compare["rank"].eq(3)]
    pos = np.arange(len(ROIS)); vals = [q[q.roi.eq(roi)].delta_r.to_numpy(float) for roi in ROIS]
    ax.boxplot(vals, positions=pos, widths=.55, showmeans=True,
               meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4})
    for i, z in enumerate(vals):
        ax.scatter(np.full(len(z), i), z, s=22, alpha=.8, color="#E45756")
    ax.axhline(0, color="#777", lw=.8); ax.set_xticks(pos, ROIS)
    ax.set_ylabel("K=3 shared minus independent mean-window Pearson r")
    ax.set_title("Unit-level same-window change (exploratory IT pilot)")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "02_unit_level_delta_k3.png", dpi=300)
    plt.close(fig)

    audit = {
        "design": "6 IT units: posterior/middle/anterior IT, 2 each; official released consistency >= 0.4 only",
        "time_windows_ms": windows.tolist(),
        "same_folds": True, "n_folds": N_FOLDS, "n_images": N_IMAGES,
        "independent_baseline": "existing 64D TVSD OOF predictions from the same windows",
        "shared_model": "w_u(t)=sum_k a[u,k] psi_k(t); ROI-specific fold-local SVD basis learned from training-fold neural curves",
        "ranks_tested": list(TEMPORAL_RANKS), "ridge": RIDGE,
        "representation": "same PCA64 ResNet maps, same continuous fwRF fields, same smooth 17-node gates",
        "device": str(device), "elapsed_seconds": time.time() - started,
        "warning": "exploratory; only 2 units per IT ROI and no claim of physiological trajectory without replication",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
