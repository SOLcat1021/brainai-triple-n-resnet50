"""TVSD sample-size sensitivity for unit readout-axis stability.

This pilot fixes the spatial candidate for each target using all training
images, then varies only the number of images used to estimate the 40-D PCA8
readout axis.  It uses official 20-ms bins and the same independent 100-image
test set for every sample size.
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
import torch
import h5py
from scipy.io import loadmat

import sys
CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
FEATURES = ROOT / "cache" / "tvsd" / "original_fwrf_resnet50" / "spatial_pca8_14"
RAW_CACHE = ROOT / "cache" / "tvsd" / "vit_features_targets" / "monkeyN_raw_windows_f16.npy"
RAW_MAT = Path(r"G:\BrainAI_Data\TVSD\monkeyN\THINGS_MUA_trials.mat")
OUT = PROJECT / "results" / "tvsd_axis_sample_size_sensitivity_2026-09-03"

N_TRAIN = 22248
N_TEST = 100
N_CHANNELS = 40
N_WINDOWS = 8
WINDOW_STARTS = np.arange(30, 190, 20)
USE_WINDOWS = np.flatnonzero((WINDOW_STARTS >= 50) & (WINDOW_STARTS <= 150))
SAMPLE_SIZES = (200, 500, 1000, 2000, 5000, 10000, 20000)
N_REPS = 5
N_PER_ROI = 5
TARGET_ROIS = ("V4", "IT")
RIDGE_SCALE = 0.05
SEED = 20260903


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float) - np.mean(a, axis=0, keepdims=True)
    b = np.asarray(b, float) - np.mean(b, axis=0, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    return np.divide(np.sum(a * b, axis=0), den,
                     out=np.full(den.shape, np.nan), where=den > 1e-12)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), den,
                     out=np.full(len(a), np.nan), where=den > 1e-12)


def load_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test targets aligned to THINGS image order and V4 channels."""
    with h5py.File(RAW_MAT, "r") as h:
        mat = np.asarray(h["ALLMAT"])
    raw = np.load(RAW_CACHE, mmap_mode="r")
    train_idx = np.flatnonzero(mat[1] > 0)
    test_idx = np.flatnonzero(mat[2] > 0)
    train_ids = mat[1, train_idx].astype(int) - 1
    test_ids = mat[2, test_idx].astype(int) - 1
    train_order = np.argsort(train_ids)
    # TVSD mapping follows the released 1024-channel layout used by the
    # existing scripts: 512 V1, 256 V4, 256 IT, then remap to recording order.
    mapping_path = Path(r"G:\BrainAI_Data\TVSD_repo_metadata\monkeyN\_logs\1024chns_mapping_20220105.mat")
    mapping = np.asarray(loadmat(mapping_path, squeeze_me=True)["mapping"], int).reshape(-1) - 1
    base = np.asarray(["V1"] * 512 + ["V4"] * 256 + ["IT"] * 256)
    roi = base[mapping]
    chosen_parts = []
    # Response-only, training-image-only selection; no test oracle is used.
    for name in TARGET_ROIS:
        channels = np.flatnonzero(roi == name)
        train_roi = np.asarray(raw[train_idx[train_order]][:, channels, :], np.float32)
        variance = np.var(train_roi[:, :, USE_WINDOWS], axis=(0, 2))
        chosen_parts.append(channels[np.argsort(variance)[::-1][:N_PER_ROI]])
    chosen = np.concatenate(chosen_parts).astype(int)
    y_train = np.asarray(raw[train_idx[train_order]][:, chosen, :], np.float32)[:, :, USE_WINDOWS]
    y_train = np.transpose(y_train, (0, 2, 1))  # image x time x unit
    test_raw = np.asarray(raw[test_idx][:, chosen, :], np.float32)
    y_test = np.full((N_TEST, len(USE_WINDOWS), len(chosen)), np.nan, np.float32)
    for image in range(N_TEST):
        q = np.flatnonzero(test_ids == image)
        if len(q):
            y_test[image] = np.nanmean(test_raw[q][:, :, USE_WINDOWS], axis=0).T
    chosen_roi = np.asarray(["V4"] * N_PER_ROI + ["IT"] * N_PER_ROI)
    return y_train, y_test, chosen, chosen_roi, mapping


def fit_ridge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batch ridge: x=image x target x feature, y=image x target."""
    xm = x.mean(0); ym = y.mean(0)
    xc = x - xm[None]; yc = y - ym[None]
    gram = torch.einsum("ntd,nte->tde", xc, xc)
    rhs = torch.einsum("ntd,nt->td", xc, yc)
    d = x.shape[-1]
    lam = RIDGE_SCALE * torch.diagonal(gram, dim1=1, dim2=2).sum(1) / max(d, 1)
    eye = torch.eye(d, device=x.device, dtype=x.dtype)[None]
    beta = torch.linalg.solve(gram + eye * lam[:, None, None].clamp_min(1e-6), rhs[..., None])[..., 0]
    return beta, xm, ym


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps = np.load(FEATURES / "train_maps_f16.npy", mmap_mode="r")
    test_maps = np.load(FEATURES / "test_maps_f16.npy", mmap_mode="r")
    y_train, y_test, chosen_channels, chosen_roi, mapping = load_targets()
    n, channels, grid, _ = maps.shape
    assert n == N_TRAIN and channels == N_CHANNELS and grid == 14
    n_time = len(USE_WINDOWS); n_units = len(chosen_channels); n_targets = n_units * n_time
    y = y_train.reshape(n, n_targets)
    fields_x, fields_y, fields_sigma = candidates()
    fields = torch.as_tensor(gaussian_mass_stack(fields_x, fields_y, fields_sigma, grid).reshape(-1, grid * grid), device=device)
    f = torch.as_tensor(np.asarray(maps, np.float32), device=device).flatten(2)
    ft = torch.as_tensor(np.asarray(test_maps, np.float32), device=device).flatten(2)
    yt = torch.as_tensor(y, device=device)
    # Full-data candidate selection is frozen before the sample-size sweep.
    mean_f = f.mean(0); mean_y = yt.mean(0)
    fc = f - mean_f[None]; yc = yt - mean_y[None]
    cov_fy = torch.einsum("ndp,nt->dpt", fc, yc)
    cov_ff = torch.einsum("ndp,ndq->dpq", fc, fc)
    var_y = torch.sum(yc * yc, 0).clamp_min(1e-8)
    var_x = torch.einsum("gp,dpq,gq->gd", fields, cov_ff, fields).clamp_min(1e-8)
    numerator = torch.einsum("gp,dpt->gdt", fields, cov_fy)
    energy = (numerator.square() / (var_x[:, :, None] * var_y[None, None])).sum(1)
    selected_field_idx = energy.argmax(0)
    selected_fields = fields[selected_field_idx]
    x_all = torch.einsum("ndp,tp->ntd", f, selected_fields)
    x_test = torch.einsum("ndp,tp->ntd", ft, selected_fields)
    beta_full, xm_full, ym_full = fit_ridge(x_all, yt)
    test_y = torch.as_tensor(y_test.reshape(N_TEST, n_targets), device=device)
    full_pred = (x_test - xm_full[None]) * beta_full[None]
    full_pred = full_pred.sum(-1) + ym_full[None]
    full_test_r = corr_columns(full_pred.cpu().numpy(), y_test.reshape(N_TEST, n_targets))

    rng = np.random.default_rng(SEED)
    rows = []
    beta_rows = []
    target_names = [(int(unit), int(WINDOW_STARTS[wi]), str(chosen_roi[unit]))
                    for unit in range(n_units) for wi in USE_WINDOWS]
    for size in SAMPLE_SIZES:
        for rep in range(N_REPS):
            idx = rng.choice(n, size, replace=False)
            beta, xm, ym = fit_ridge(x_all[idx], yt[idx])
            b = beta.detach().cpu().numpy()
            c_ref = cosine_rows(b, beta_full.detach().cpu().numpy())
            pred = ((x_test - xm[None]) * beta[None]).sum(-1) + ym[None]
            test_r = corr_columns(pred.detach().cpu().numpy(), y_test.reshape(N_TEST, n_targets))
            for ti, (unit_idx, wi, roi_name) in enumerate(target_names):
                rows.append({
                    "sample_size": size, "replicate": rep, "unit_index": unit_idx,
                    "roi": roi_name, "window_start_ms": wi, "axis_cosine_to_full": float(c_ref[ti]),
                    "axis_angle_to_full_degrees": float(np.degrees(np.arccos(np.clip(c_ref[ti], -1, 1)))),
                    "test_pearson_r": float(test_r[ti]), "full_reference_test_r": float(full_test_r[ti]),
                })
            beta_rows.append({"sample_size": size, "replicate": rep, "beta": b})
        # Pairwise replicate stability at this sample size, target by target.
        reps = np.stack([x["beta"] for x in beta_rows if x["sample_size"] == size])
        pair = np.stack([cosine_rows(reps[a], reps[b]) for a, b in combinations(range(N_REPS), 2)])
        pair_median = np.nanmedian(pair, axis=0)
        for ti, (unit_idx, wi, roi_name) in enumerate(target_names):
            rows.append({
                "sample_size": size, "replicate": -1, "unit_index": unit_idx,
                "roi": roi_name, "window_start_ms": wi, "axis_cosine_to_full": np.nan,
                "axis_angle_to_full_degrees": np.nan,
                "test_pearson_r": np.nan, "full_reference_test_r": np.nan,
                "pairwise_replicate_cosine": float(pair_median[ti]),
            })
        print(f"sample size {size} complete", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "sample_size_axis_metrics.csv", index=False, encoding="utf-8-sig")
    regular = detail[detail.replicate >= 0].copy()
    pair_rows = detail[detail.replicate == -1].copy()
    summary = regular.groupby("sample_size", as_index=False).agg(
        median_axis_cosine=("axis_cosine_to_full", "median"),
        p10_axis_cosine=("axis_cosine_to_full", lambda x: float(np.nanpercentile(x, 10))),
        median_axis_angle_degrees=("axis_angle_to_full_degrees", "median"),
        mean_test_pearson_r=("test_pearson_r", "mean"),
        median_test_pearson_r=("test_pearson_r", "median"),
    )
    pair_summary = pair_rows.groupby("sample_size", as_index=False).agg(
        median_pairwise_replicate_cosine=("pairwise_replicate_cosine", "median"),
        p10_pairwise_replicate_cosine=("pairwise_replicate_cosine", lambda x: float(np.nanpercentile(x, 10))),
    )
    summary = summary.merge(pair_summary, on="sample_size")
    summary.to_csv(OUT / "sample_size_summary.csv", index=False, encoding="utf-8-sig")
    summary_roi = regular.groupby(["roi", "sample_size"], as_index=False).agg(
        median_axis_cosine=("axis_cosine_to_full", "median"),
        p10_axis_cosine=("axis_cosine_to_full", lambda x: float(np.nanpercentile(x, 10))),
        median_axis_angle_degrees=("axis_angle_to_full_degrees", "median"),
        mean_test_pearson_r=("test_pearson_r", "mean"),
    )
    pair_roi = pair_rows.groupby(["roi", "sample_size"], as_index=False).agg(
        median_pairwise_replicate_cosine=("pairwise_replicate_cosine", "median"),
        p10_pairwise_replicate_cosine=("pairwise_replicate_cosine", lambda x: float(np.nanpercentile(x, 10))),
    )
    summary_roi.merge(pair_roi, on=["roi", "sample_size"]).to_csv(
        OUT / "sample_size_summary_by_roi.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(summary.sample_size, summary.median_axis_cosine, "-o", label="Median cosine to full-data axis")
    axes[0].fill_between(summary.sample_size, summary.p10_axis_cosine, summary.median_axis_cosine, alpha=.18, label="10th–50th percentile")
    axes[0].plot(summary.sample_size, summary.median_pairwise_replicate_cosine, "-s", label="Median pairwise replicate cosine")
    axes[0].axhline(.9, color="0.45", ls="--", lw=.8, label="cosine = 0.90")
    axes[0].set(xlabel="Training images", ylabel="Axis cosine", title="TVSD unit-axis stability")
    axes[0].set_xscale("symlog", linthresh=200); axes[0].set_ylim(-.1, 1.02); axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    axes[1].plot(summary.sample_size, summary.mean_test_pearson_r, "-o", color="#D95F02")
    axes[1].axhline(0, color="0.45", lw=.8); axes[1].set(xlabel="Training images", ylabel="Mean test Pearson r", title="Same-test-set prediction check")
    axes[1].set_xscale("symlog", linthresh=200); axes[1].grid(alpha=.2)
    fig.savefig(OUT / "01_axis_stability_by_training_images.png", dpi=300)
    plt.close(fig)

    audit = {
        "dataset": "TVSD monkey N",
        "n_train_images": N_TRAIN, "n_test_images": N_TEST, "n_replicates": N_REPS,
        "sample_sizes": list(SAMPLE_SIZES), "official_time_windows_ms": [[int(s), int(s + 19)] for s in WINDOW_STARTS[USE_WINDOWS]],
        "selected_channels": chosen_channels.tolist(), "selected_rois": chosen_roi.tolist(),
        "selection": "top training-only response variance among V4 and IT channels",
        "axis": "40-D PCA8 stage readout after one full-data frozen Gaussian candidate per unit-window",
        "candidate_selection": "frozen using all training images; not re-selected during sample-size sweep",
        "ridge_scale": RIDGE_SCALE, "stability_reference": "full-data fit at each window",
        "test_set": "same 100 independent test images for every subset",
        "device": str(device),
        "note": "Exploratory sample-size sensitivity; not a final claim about all TVSD channels.",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
