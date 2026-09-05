"""Train one independent interpretable ResNet proxy per unit x 10-ms window.

The estimator is identical for every ROI and every time window:
continuous Gaussian fwRF -> smooth Gaussian depth gate over all 17 ResNet-50
nodes -> signed joint ridge over 17 x 8 frozen PCA channels.  Five-fold OOF
predictions provide the single-window r.  A final all-image refit is archived as
the loadable proxy; no parameters are shared across neural time windows.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
CODE = PROJECT / "code"
BANK = PROJECT / "proxy_bank_10ms_all_windows"
RESP_FILE = BANK / "trin_all_units_10ms_responses.npy"
META_FILE = BANK / "unit_metadata.csv"
WINDOW_FILE = BANK / "windows_10ms.npy"
MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "block_pca8_14_tvsd_frozen" / "maps_f16.npy"

sys.path.insert(0, str(CODE))
import run_trin_closed_loop_validation as foldbase  # noqa: E402
import run_tvsd_fwrf_where_pilot as statbase  # noqa: E402
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack  # noqa: E402
from pilot_tvsd_continuous_gaussian import continuous_gaussian  # noqa: E402
from refine_trin_alltypes_continuous_gaussian import refine_from_stats  # noqa: E402
from train_all_units_final_resnet50 import (  # noqa: E402
    DIMS, LAYERS, N_LAYERS, RIDGE, SCREEN_DIMS, TARGET_BATCH,
    batched_ridge, corr_columns, corr_torch, infer_gate, make_stats, normalize,
    select_fields,
)

ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
DEPTH_COORD = np.linspace(0.0, 1.0, N_LAYERS, dtype=np.float32)


def slug(s: str) -> str:
    return s.lower().replace(" ", "_")


def normalize_all(raw: np.ndarray, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    tensors, means, stds = [], [], []
    for layer in range(N_LAYERS):
        x = torch.as_tensor(np.asarray(raw[:, layer], np.float32), device=device).flatten(2)
        mean = x.mean((0, 2), keepdim=True)
        std = x.std((0, 2), keepdim=True).clamp_min(1e-5)
        tensors.append((x - mean) / std)
        means.append(mean.flatten().detach().cpu().numpy())
        stds.append(std.flatten().detach().cpu().numpy())
    return torch.cat(tensors, 1), np.concatenate(means), np.concatenate(stds)


def select_spatial_and_gate(ftr: torch.Tensor, y: np.ndarray, bank: torch.Tensor,
                            xs: np.ndarray, ys: np.ndarray, sigmas: np.ndarray,
                            seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    screen = torch.cat(
        [ftr[:, l * DIMS:l * DIMS + SCREEN_DIMS] for l in range(N_LAYERS)], 1)
    stats = make_stats(screen, y)
    initial = select_fields(stats, bank)
    cx, cy, cs = refine_from_stats(initial, stats, bank, xs, ys, sigmas, ftr.device)
    del screen, stats

    order = np.random.default_rng(seed).permutation(len(y))
    n_val = min(200, max(100, len(order) // 4))
    inner_val = torch.as_tensor(order[:n_val], device=ftr.device)
    inner_train = torch.as_tensor(order[n_val:], device=ftr.device)
    y_full = torch.as_tensor(np.asarray(y, np.float32), device=ftr.device)
    grid = int(round(np.sqrt(ftr.shape[-1])))
    gates = np.empty((y.shape[1], N_LAYERS), np.float32)
    for t0 in range(0, y.shape[1], TARGET_BATCH):
        t1 = min(y.shape[1], t0 + TARGET_BATCH)
        field = continuous_gaussian(
            torch.as_tensor(cx[t0:t1], device=ftr.device),
            torch.as_tensor(cy[t0:t1], device=ftr.device),
            torch.as_tensor(cs[t0:t1], device=ftr.device), grid)
        x = torch.einsum("ndp,bp->nbd", ftr, field)
        yb = y_full[:, t0:t1]
        score = np.empty((t1 - t0, N_LAYERS), np.float32)
        for layer in range(N_LAYERS):
            sl = slice(layer * DIMS, (layer + 1) * DIMS)
            pv, _, _ = batched_ridge(
                x[inner_train, :, sl], yb[inner_train], x[inner_val, :, sl])
            score[:, layer] = corr_torch(pv, yb[inner_val]).detach().cpu().numpy()
        gates[t0:t1] = infer_gate(score)[0]
    return cx.astype(np.float32), cy.astype(np.float32), cs.astype(np.float32), gates


def predict_with_parameters(ftr: torch.Tensor, fte: torch.Tensor, ytr: np.ndarray,
                            cx: np.ndarray, cy: np.ndarray, cs: np.ndarray,
                            gates: np.ndarray, save_weights: bool = False
                            ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    prediction = np.empty((len(fte), ytr.shape[1]), np.float32)
    weights = np.empty((ytr.shape[1], N_LAYERS * DIMS), np.float32) if save_weights else None
    intercept = np.empty(ytr.shape[1], np.float32) if save_weights else None
    y = torch.as_tensor(np.asarray(ytr, np.float32), device=ftr.device)
    grid = int(round(np.sqrt(ftr.shape[-1])))
    for t0 in range(0, ytr.shape[1], TARGET_BATCH):
        t1 = min(ytr.shape[1], t0 + TARGET_BATCH)
        field = continuous_gaussian(
            torch.as_tensor(cx[t0:t1], device=ftr.device),
            torch.as_tensor(cy[t0:t1], device=ftr.device),
            torch.as_tensor(cs[t0:t1], device=ftr.device), grid)
        xtr = torch.einsum("ndp,bp->nbd", ftr, field)
        xte = torch.einsum("ndp,bp->nbd", fte, field)
        scale = torch.as_tensor(
            np.sqrt(np.clip(gates[t0:t1], 1e-5, None) * N_LAYERS),
            device=ftr.device, dtype=xtr.dtype)
        scale = torch.repeat_interleave(scale, DIMS, dim=1)
        pred, beta, mx_scaled = batched_ridge(
            xtr * scale[None], y[:, t0:t1], xte * scale[None], ridge=RIDGE)
        prediction[:, t0:t1] = pred.detach().cpu().numpy()
        if save_weights:
            effective = beta * scale
            weights[t0:t1] = effective.detach().cpu().numpy()
            mean_y = y[:, t0:t1].mean(0)
            intercept[t0:t1] = (mean_y - (mx_scaled * beta).sum(1)).detach().cpu().numpy()
    return prediction, weights, intercept


def create_roi_file(path: Path, n_windows: int, n_units: int,
                    units: np.ndarray, windows: np.ndarray,
                    norm_mean: np.ndarray, norm_std: np.ndarray) -> h5py.File:
    h = h5py.File(path, "a")
    if "unit_global" not in h:
        h.create_dataset("unit_global", data=units.astype(np.int32))
        h.create_dataset("windows_ms", data=windows.astype(np.int16))
        h.create_dataset("normalization_mean", data=norm_mean.astype(np.float32))
        h.create_dataset("normalization_std", data=norm_std.astype(np.float32))
        h.create_dataset("completed", shape=(n_windows,), dtype=np.bool_, fillvalue=False)
        h.create_dataset("oof_prediction", shape=(n_windows, 1000, n_units), dtype=np.float16,
                         chunks=(1, 100, min(256, n_units)), compression="lzf")
        h.create_dataset("oof_r", shape=(n_windows, n_units), dtype=np.float32)
        h.create_dataset("fold_layer_gate", shape=(5, n_windows, n_units, N_LAYERS),
                         dtype=np.float16, chunks=(1, 1, min(256, n_units), N_LAYERS), compression="lzf")
        h.create_dataset("fold_spatial_parameters", shape=(5, n_windows, n_units, 3),
                         dtype=np.float32, chunks=(1, 1, min(512, n_units), 3), compression="lzf")
        h.create_dataset("spatial_parameters", shape=(n_windows, n_units, 3), dtype=np.float32,
                         chunks=(1, min(512, n_units), 3), compression="lzf")
        h.create_dataset("layer_gate", shape=(n_windows, n_units, N_LAYERS), dtype=np.float32,
                         chunks=(1, min(256, n_units), N_LAYERS), compression="lzf")
        h.create_dataset("depth", shape=(n_windows, n_units), dtype=np.float32)
        h.create_dataset("channel_weights", shape=(n_windows, n_units, N_LAYERS * DIMS), dtype=np.float32,
                         chunks=(1, min(128, n_units), N_LAYERS * DIMS), compression="lzf")
        h.create_dataset("intercept", shape=(n_windows, n_units), dtype=np.float32)
        h.create_dataset("fitted_prediction", shape=(n_windows, 1000, n_units), dtype=np.float16,
                         chunks=(1, 100, min(256, n_units)), compression="lzf")
        h.create_dataset("full_fit_r", shape=(n_windows, n_units), dtype=np.float32)
        h.attrs["architecture"] = (
            "ResNet50-IMAGENET1K_V2; continuous Gaussian fwRF; exact smooth Gaussian "
            "gate over all 17 nodes; signed ridge over 17x8 TVSD-frozen PCA channels")
        h.attrs["temporal_parameter_sharing"] = False
        h.attrs["roi_specific_routing"] = False
        h.attrs["depth_definition"] = "sum(layer_gate * linspace(0,1,17))"
        h.attrs["channel_weight_definition"] = (
            "effective signed weights after absorbing sqrt(layer_gate*17) into ridge beta")
    return h


def train_roi(roi: str, units: np.ndarray, targets: np.ndarray, windows: np.ndarray,
              raw: np.ndarray, fall: torch.Tensor, norm_mean: np.ndarray,
              norm_std: np.ndarray, device: torch.device) -> Path:
    path = BANK / f"{slug(roi)}_proxy_bank.h5"
    xs, ys, sigmas = candidates()
    grid = int(raw.shape[-1])
    gaussian_bank = torch.as_tensor(
        gaussian_mass_stack(xs, ys, sigmas, grid).reshape(len(xs), -1), device=device)
    h = create_roi_file(path, len(windows), len(units), units, windows, norm_mean, norm_std)
    try:
        for wi, window in enumerate(windows):
            if bool(h["completed"][wi]):
                print(f"{roi} {window.tolist()}: resumed", flush=True)
                continue
            started = time.time()
            y = np.asarray(targets[wi])[:, units].astype(np.float32, copy=False)
            oof = np.empty_like(y, np.float32)
            fold_gates = np.empty((5, len(units), N_LAYERS), np.float32)
            fold_space = np.empty((5, len(units), 3), np.float32)
            for fi, (train, test) in enumerate(foldbase.fixed_folds(len(y))):
                ftr, fte = normalize(raw, train, test, device)
                cx, cy, cs, gates = select_spatial_and_gate(
                    ftr, y[train], gaussian_bank, xs, ys, sigmas,
                    seed=20260825 + 101 * wi + fi)
                pred, _, _ = predict_with_parameters(
                    ftr, fte, y[train], cx, cy, cs, gates, save_weights=False)
                oof[test] = pred
                fold_gates[fi] = gates
                fold_space[fi, :, 0] = cx
                fold_space[fi, :, 1] = cy
                fold_space[fi, :, 2] = cs
                del ftr, fte
                torch.cuda.empty_cache()

            # The archived proxy uses the cross-validated mean field/gate, then
            # refits only the signed readout on all images.  Thus model selection
            # is stable across folds and the loadable proxy has one unambiguous
            # parameter set.
            final_space = fold_space.mean(0)
            final_gate = fold_gates.mean(0)
            final_gate /= np.maximum(final_gate.sum(1, keepdims=True), 1e-12)
            fitted, weights, intercept = predict_with_parameters(
                fall, fall, y, final_space[:, 0], final_space[:, 1], final_space[:, 2],
                final_gate, save_weights=True)
            oof_r = corr_columns(oof, y)
            full_r = corr_columns(fitted, y)
            depth = final_gate @ DEPTH_COORD

            h["oof_prediction"][wi] = oof.astype(np.float16)
            h["oof_r"][wi] = oof_r
            h["fold_layer_gate"][:, wi] = fold_gates.astype(np.float16)
            h["fold_spatial_parameters"][:, wi] = fold_space
            h["spatial_parameters"][wi] = final_space
            h["layer_gate"][wi] = final_gate
            h["depth"][wi] = depth
            h["channel_weights"][wi] = weights
            h["intercept"][wi] = intercept
            h["fitted_prediction"][wi] = fitted.astype(np.float16)
            h["full_fit_r"][wi] = full_r
            h["completed"][wi] = True
            h.flush()
            print(
                f"{roi} {int(window[0]):+d}..{int(window[1]):+d} ms: "
                f"mean OOF r={np.nanmean(oof_r):.4f}, {time.time()-started:.1f}s", flush=True)
    finally:
        h.close()
    return path


def build_index(meta: pd.DataFrame, windows: np.ndarray, paths: dict[str, Path]) -> None:
    db = BANK / "proxy_index.sqlite"
    if db.exists():
        db.unlink()
    connection = sqlite3.connect(db)
    meta.to_sql("units", connection, index=False, if_exists="replace")
    proxy_id = 0
    core_parts = []
    for roi in ROIS:
        q = meta[meta.roi.eq(roi)].reset_index(drop=True)
        with h5py.File(paths[roi], "r") as h:
            r = h["oof_r"][:]
            depth = h["depth"][:]
        rows = []
        for wi, (start, end) in enumerate(windows):
            n = len(q)
            rows.append(pd.DataFrame({
                "proxy_id": np.arange(proxy_id, proxy_id + n, dtype=np.int64),
                "unit_global": q.unit_global.to_numpy(np.int32),
                "roi": roi,
                "roi_local_unit": np.arange(n, dtype=np.int32),
                "window_index": wi,
                "t": int(start),
                "window_end_ms": int(end),
                "r": r[wi].astype(np.float32),
                "depth": depth[wi].astype(np.float32),
                "weight_file": str(paths[roi]),
            }))
            proxy_id += n
        part = pd.concat(rows, ignore_index=True)
        part.to_sql("proxies", connection, index=False, if_exists="append", chunksize=10000)
        core_parts.append(part.drop(columns="weight_file"))
    connection.execute("CREATE UNIQUE INDEX idx_proxy_id ON proxies(proxy_id)")
    connection.execute("CREATE INDEX idx_proxy_unit_time ON proxies(unit_global, t)")
    connection.execute("CREATE INDEX idx_proxy_roi_time ON proxies(roi, t)")
    connection.execute("CREATE UNIQUE INDEX idx_unit_global ON units(unit_global)")
    connection.commit(); connection.close()
    pd.concat(core_parts, ignore_index=True).to_csv(
        BANK / "proxy_manifest_core.csv.gz", index=False, compression="gzip")


def main() -> None:
    BANK.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    meta = pd.read_csv(META_FILE)
    windows = np.load(WINDOW_FILE)
    targets = np.load(RESP_FILE, mmap_mode="r")
    flat = np.load(MAPS, mmap_mode="r")
    raw = flat.reshape(len(flat), N_LAYERS, DIMS, flat.shape[-2], flat.shape[-1])
    fall, norm_mean, norm_std = normalize_all(raw, device)

    paths: dict[str, Path] = {}
    for roi in ROIS:
        units = meta.loc[meta.roi.eq(roi), "unit_global"].to_numpy(np.int32)
        paths[roi] = train_roi(
            roi, units, targets, windows, raw, fall, norm_mean, norm_std, device)
        torch.cuda.empty_cache()
    build_index(meta, windows, paths)
    audit = {
        "n_units": int(len(meta)),
        "n_windows": int(len(windows)),
        "n_proxies": int(len(meta) * len(windows)),
        "one_independent_proxy_per_unit_window": True,
        "time_windows_ms": windows.tolist(),
        "reliable_onset_filter": False,
        "proxy_parameters": ["continuous_spatial_center_x", "continuous_spatial_center_y",
                             "continuous_spatial_sigma", "17_layer_gate", "depth",
                             "136_signed_channel_weights", "intercept"],
        "performance": "five-fold image OOF Pearson r per unit-window",
        "elapsed_seconds": time.time() - started,
        "roi_files": {k: str(v) for k, v in paths.items()},
    }
    (BANK / "proxy_training_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
