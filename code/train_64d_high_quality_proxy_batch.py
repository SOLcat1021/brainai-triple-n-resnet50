"""Train a balanced high-quality batch of 64-D Triple-N proxy models.

This is a channel-rank upgrade of the archived all-window PCA8 proxy bank.
The spatial field and the smooth 17-node depth gate are frozen fold-by-fold
from that bank, while the signed channel readout is independently refit at
rank 64 for every unit and every 10-ms window.  No reliable-onset cropping or
temporal parameter sharing is used.

The default cohort is the predeclared balanced expansion cohort (coarse ROI
50 each plus functional-IT 20 each, de-duplicated).  The full strict pool and
the smaller pilot remain available as immutable CSV indices.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.special import erf


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
OLD_BANK = PROJECT / "proxy_bank_10ms_all_windows"
COHORT_DIR = PROJECT / "cohorts" / "high_quality_64d_pilot_2026-08-26"
RANK_DIR = PROJECT / "results" / "rank_sweep_extended_2026-08-26"
MAPS_PATH = RANK_DIR / "trin_resnet50_pca256_maps_f16.npy"
BASIS_PATH = RANK_DIR / "response_blind_pca256_basis.npz"
OUT = PROJECT / "proxy_bank_64d_high_quality_all_windows_2026-08-26"

LAYERS = (
    "stem", "res2.1", "res2.2", "res2.3", "res3.1", "res3.2", "res3.3", "res3.4",
    "res4.1", "res4.2", "res4.3", "res4.4", "res4.5", "res4.6",
    "res5.1", "res5.2", "res5.3",
)
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
RANK = 64
N_LAYERS = 17
GRID = 14
N_IMAGES = 1000
N_FOLDS = 5
RIDGE = 3.0  # median optimum at rank 64 in the frozen 120-unit rank sweep
SEED = 20260818 + 901
# Keep the target batch small for the consistency-only cohort: IT regions can
# contain thousands of units, and the 17 x 64 gated design is memory-heavy.
TARGET_BATCH = 32


def slug(x: str) -> str:
    return x.lower().replace(" ", "_")


def fixed_folds(n: int):
    order = np.random.default_rng(SEED).permutation(n)
    chunks = np.array_split(order, N_FOLDS)
    return [(np.concatenate([chunks[j] for j in range(N_FOLDS) if j != k]), chunks[k])
            for k in range(N_FOLDS)]


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, np.float64) - np.nanmean(a, axis=0, keepdims=True)
    bb = np.asarray(b, np.float64) - np.nanmean(b, axis=0, keepdims=True)
    den = np.sqrt(np.nansum(aa * aa, 0) * np.nansum(bb * bb, 0))
    return np.divide(np.nansum(aa * bb, 0), den, out=np.full(den.shape, np.nan),
                     where=den > 1e-12).astype(np.float32)


def gaussian_fields(parameters: np.ndarray) -> np.ndarray:
    """Exact archived 20-degree fwRF geometry, returned as unit x 196."""
    x, y, sigma = parameters.T.astype(np.float32)
    dpix = 20.0 / GRID
    coord = np.arange(GRID, dtype=np.float32) * dpix - 10.0 + dpix / 2
    xm, ym0 = np.meshgrid(coord, coord)
    ym = -ym0
    out = []
    for xx, yy, ss in zip(x, y, sigma):
        gx = .5 * (erf((xm - xx + dpix / 2) / (math.sqrt(2) * ss)) -
                   erf((xm - xx - dpix / 2) / (math.sqrt(2) * ss)))
        gy = .5 * (erf((ym - yy + dpix / 2) / (math.sqrt(2) * ss)) -
                   erf((ym - yy - dpix / 2) / (math.sqrt(2) * ss)))
        exact = gx * gy
        d = 2 * ss ** 2
        approx = dpix ** 2 / (d * np.pi) * np.exp(-((xm - xx) ** 2 + (ym - yy) ** 2) / d)
        out.append(exact if ss < dpix else approx)
    return np.asarray(out, np.float32).reshape(len(out), -1)


def dual_ridge(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor,
               ridge: float = RIDGE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target-specific dual ridge. X is images x targets x features."""
    mx = xtr.mean(0)
    my = ytr.mean(0)
    xc = xtr - mx[None]
    yc = ytr - my[None]
    # target x images x features
    xb = xc.permute(1, 0, 2).contiguous()
    kernel = torch.bmm(xb, xb.transpose(1, 2))
    lam = ridge * xb.square().sum((1, 2)) / xb.shape[-1]
    eye = torch.eye(kernel.shape[-1], device=xtr.device, dtype=xtr.dtype)[None]
    kernel = kernel + eye * lam[:, None, None].clamp_min(1e-6)
    chol, info = torch.linalg.cholesky_ex(kernel)
    if bool((info != 0).any()):
        # Rare numerical fallback; solve is slower but robust.
        dual = torch.linalg.solve(kernel, yc.T.unsqueeze(-1)).squeeze(-1)
    else:
        dual = torch.cholesky_solve(yc.T.unsqueeze(-1), chol).squeeze(-1)
    beta = torch.bmm(xb.transpose(1, 2), dual.unsqueeze(-1)).squeeze(-1)
    pred = torch.einsum("nbd,bd->nb", xte - mx[None], beta) + my[None]
    intercept = my - torch.einsum("bd,bd->b", mx, beta)
    return pred, beta, intercept, lam


def load_cohort(cohort_mode: str) -> pd.DataFrame:
    if cohort_mode == "consistency_only":
        # Exploratory/full cohort: selection is based only on the released
        # TVSD/Triple-N consistency metric; no model-prediction r is used.
        audit_path = PROJECT / "cohorts" / "high_quality_64d_pilot_2026-08-26" / "all_unit_quality_audit.csv"
        cohort = pd.read_csv(audit_path)
        cohort = cohort[cohort.released_independent_consistency >= 0.40].copy()
        cohort = cohort.sort_values(["released_independent_consistency", "unit_global"], ascending=[False, True])
        cohort = cohort.drop_duplicates("unit_global").reset_index(drop=True)
        cohort["selection_tier"] = "released_consistency_ge_0.4"
        cohort_name = "all_units_released_consistency_ge_0.4"
    elif cohort_mode == "pilot":
        coarse_file = "pilot_coarse_roi_12_each.csv"
        functional_file = "pilot_functional_it_8_each.csv"
        cohort_name = "balanced_pilot_A_strict"
        a = pd.read_csv(COHORT_DIR / coarse_file)
        b = pd.read_csv(COHORT_DIR / functional_file)
    elif cohort_mode == "expansion":
        coarse_file = "expansion_coarse_roi_50_each.csv"
        functional_file = "expansion_functional_it_20_each.csv"
        cohort_name = "balanced_expansion_A_strict"
        a = pd.read_csv(COHORT_DIR / coarse_file)
        b = pd.read_csv(COHORT_DIR / functional_file)
    elif cohort_mode == "expansion_strict":
        # Preserve the frozen expansion ordering but never relax the strict
        # response-only quality threshold to fill a small functional area.
        a = pd.read_csv(COHORT_DIR / "expansion_coarse_roi_50_each.csv")
        b = pd.read_csv(COHORT_DIR / "expansion_functional_it_20_each.csv")
        a = a[a.tier_A_strict.astype(bool)].copy()
        b = b[b.tier_A_strict.astype(bool)].copy()
        cohort_name = "balanced_expansion_strict_only"
    elif cohort_mode == "functional_validation":
        # The 20-per-functional-area file contains all 8 pilot units per area.
        # Exclude the entire merged 113-unit pilot before fitting, leaving 12
        # genuinely held-out units per functional area (96 total).
        expansion = pd.read_csv(COHORT_DIR / "expansion_functional_it_20_each.csv")
        pilot = pd.concat([
            pd.read_csv(COHORT_DIR / "pilot_coarse_roi_12_each.csv"),
            pd.read_csv(COHORT_DIR / "pilot_functional_it_8_each.csv"),
        ], ignore_index=True).drop_duplicates("unit_global")
        a = expansion[
            (~expansion.unit_global.isin(pilot.unit_global))
            & expansion.tier_A_strict.astype(bool)
        ].copy()
        b = expansion.iloc[0:0].copy()
        cohort_name = "held_out_functional_it_validation_92_strict"
    else:
        raise ValueError(f"unknown cohort mode: {cohort_mode}")
    if cohort_mode != "consistency_only":
        cohort = pd.concat([a, b], ignore_index=True)
        cohort = cohort.sort_values(
            ["min_window_r", "mean_window_r", "released_independent_consistency"],
            ascending=False,
        ).drop_duplicates("unit_global", keep="first").reset_index(drop=True)
    if cohort_mode != "consistency_only" and not cohort.tier_A_strict.astype(bool).all():
        raise RuntimeError("cohort contains a non-strict unit")
    cohort["proxy_rank"] = RANK
    cohort["cohort_name"] = cohort_name
    OUT.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUT / "selected_units.csv", index=False, encoding="utf-8-sig")
    return cohort


def load_selected_roi(roi: str, cohort: pd.DataFrame):
    selected = cohort[cohort.roi.eq(roi)].copy().reset_index(drop=True)
    old_path = OLD_BANK / f"{slug(roi)}_proxy_bank.h5"
    with h5py.File(old_path, "r") as old:
        old_units = old["unit_global"][:].astype(int)
        lookup = {int(u): i for i, u in enumerate(old_units)}
        local = np.asarray([lookup[int(u)] for u in selected.unit_global], np.int64)
        # h5py fancy indices must be strictly increasing.  The frozen cohort is
        # deliberately ordered by response-blind quality, so index in NumPy
        # after reading each modest parameter array to preserve that order.
        fold_field = old["fold_spatial_parameters"][:][:, :, local, :].astype(np.float32)
        fold_gate = old["fold_layer_gate"][:][:, :, local, :].astype(np.float32)
        final_field = old["spatial_parameters"][:][:, local, :].astype(np.float32)
        final_gate = old["layer_gate"][:][:, local, :].astype(np.float32)
        windows = old["windows_ms"][:].astype(np.int16)
    return selected, local, fold_field, fold_gate, final_field, final_gate, windows


def create_output(path: Path, selected: pd.DataFrame, windows: np.ndarray,
                  fold_field: np.ndarray, fold_gate: np.ndarray,
                  final_field: np.ndarray, final_gate: np.ndarray):
    w, u = len(windows), len(selected)
    h = h5py.File(path, "w")
    h.create_dataset("unit_global", data=selected.unit_global.to_numpy(np.int32))
    h.create_dataset("windows_ms", data=windows)
    h.create_dataset("fold_spatial_parameters", data=fold_field, compression="gzip", compression_opts=4)
    h.create_dataset("fold_layer_gate", data=fold_gate, compression="gzip", compression_opts=4)
    h.create_dataset("spatial_parameters", data=final_field, compression="gzip", compression_opts=4)
    h.create_dataset("layer_gate", data=final_gate, compression="gzip", compression_opts=4)
    depth_grid = np.arange(N_LAYERS, dtype=np.float32)
    depth = np.sum(final_gate * depth_grid[None, None, :], axis=-1)
    h.create_dataset("depth", data=depth, compression="gzip", compression_opts=4)
    h.create_dataset("oof_prediction", shape=(w, N_IMAGES, u), dtype="f4",
                     chunks=(1, N_IMAGES, min(u, 16)), compression="gzip", compression_opts=4)
    h.create_dataset("oof_r", shape=(w, u), dtype="f4")
    h.create_dataset("fold_channel_weights", shape=(N_FOLDS, w, u, N_LAYERS * RANK), dtype="f4",
                     chunks=(1, 1, min(u, 4), N_LAYERS * RANK), compression="gzip", compression_opts=4)
    h.create_dataset("fold_normalization_std", shape=(N_FOLDS, N_LAYERS, RANK), dtype="f4")
    h.create_dataset("channel_weights", shape=(w, u, N_LAYERS * RANK), dtype="f4",
                     chunks=(1, min(u, 8), N_LAYERS * RANK), compression="gzip", compression_opts=4)
    h.create_dataset("intercept", shape=(w, u), dtype="f4")
    h.create_dataset("ridge_alpha", data=np.full((w, u), RIDGE, np.float32))
    h.attrs["architecture"] = "ResNet50-ImageNetV2 / response-blind PCA64 / continuous fwRF / smooth 17-node depth gate / signed ridge"
    h.attrs["temporal_independence"] = "every unit-window refit independently; no shared neural targets or parameters across windows"
    h.attrs["field_and_gate_policy"] = "frozen fold-local values from archived PCA8 bank; 64D signed readout refit"
    h.attrs["selection_policy"] = (
        f"{selected.cohort_name.iloc[0]}; selection indices frozen before rank64 training"
    )
    h.attrs["ridge"] = RIDGE
    return h


def train_roi(roi: str, cohort: pd.DataFrame, maps: np.ndarray,
              responses: np.memmap, full_mean: torch.Tensor, full_std: torch.Tensor,
              device: torch.device) -> dict:
    full_mean = full_mean.to(device)
    full_std = full_std.to(device)
    (selected, _, fold_field, fold_gate, final_field, final_gate,
     windows) = load_selected_roi(roi, cohort)
    if selected.empty:
        return {"roi": roi, "n_units": 0}
    units = selected.unit_global.to_numpy(int)
    target = np.asarray(responses[:, :, units], np.float32)  # windows x images x units
    path = OUT / f"{slug(roi)}_proxy_bank_64d.h5"
    h = create_output(path, selected, windows, fold_field, fold_gate, final_field, final_gate)
    oof = np.full_like(target, np.nan, np.float32)
    folds = fixed_folds(N_IMAGES)

    # One fold-normalized feature tensor is reused by every time window and ROI unit.
    for fi, (train, test) in enumerate(folds):
        raw_train = torch.as_tensor(np.asarray(maps[train, :, :RANK], np.float32), device=device).flatten(3)
        raw_test = torch.as_tensor(np.asarray(maps[test, :, :RANK], np.float32), device=device).flatten(3)
        mean = raw_train.mean((0, 3), keepdim=True)
        std = raw_train.std((0, 3), keepdim=True).clamp_min(1e-5)
        h["fold_normalization_std"][fi] = std.squeeze(0).squeeze(-1).cpu().numpy()
        ftr = (raw_train - mean) / std
        fte = (raw_test - mean) / std
        del raw_train, raw_test
        for wi in range(len(windows)):
            scale = torch.as_tensor(
                np.sqrt(np.clip(fold_gate[fi, wi] * N_LAYERS, 1e-5, None)), device=device)
            ytr = torch.as_tensor(target[wi, train], device=device)
            for u0 in range(0, len(units), TARGET_BATCH):
                u1 = min(len(units), u0 + TARGET_BATCH)
                fields = torch.as_tensor(gaussian_fields(fold_field[fi, wi, u0:u1]), device=device)
                xtr = torch.einsum("nldp,up->nuld", ftr, fields)
                xte = torch.einsum("nldp,up->nuld", fte, fields)
                batch_scale = scale[u0:u1]
                xtr = (xtr * batch_scale[None, :, :, None]).flatten(2)
                xte = (xte * batch_scale[None, :, :, None]).flatten(2)
                pred, beta, _, _ = dual_ridge(xtr, ytr[:, u0:u1], xte)
                oof[wi, test, u0:u1] = pred.detach().cpu().numpy()
                effective = beta * torch.repeat_interleave(batch_scale, RANK, dim=1)
                h["fold_channel_weights"][fi, wi, u0:u1] = effective.detach().cpu().numpy()
                del xtr, xte, fields
        del ftr, fte
        torch.cuda.empty_cache()
        print(f"{roi}: OOF fold {fi + 1}/5", flush=True)

    h["oof_prediction"][:] = oof
    h["oof_r"][:] = np.stack([corr_columns(oof[wi], target[wi]) for wi in range(len(windows))])

    # Final deployable weights use all 1000 images and the archived final field/gate.
    raw = torch.as_tensor(np.asarray(maps[:, :, :RANK], np.float32), device=device).flatten(3)
    f = (raw - full_mean) / full_std
    del raw
    for wi in range(len(windows)):
        scale = torch.as_tensor(
            np.sqrt(np.clip(final_gate[wi] * N_LAYERS, 1e-5, None)), device=device)
        y = torch.as_tensor(target[wi], device=device)
        for u0 in range(0, len(units), TARGET_BATCH):
            u1 = min(len(units), u0 + TARGET_BATCH)
            fields = torch.as_tensor(gaussian_fields(final_field[wi, u0:u1]), device=device)
            x = torch.einsum("nldp,up->nuld", f, fields)
            batch_scale = scale[u0:u1]
            x = (x * batch_scale[None, :, :, None]).flatten(2)
            _, beta, intercept, _ = dual_ridge(
                x, y[:, u0:u1], x[:1])
            # Effective weights act directly on normalized PCA features.
            effective = beta * torch.repeat_interleave(batch_scale, RANK, dim=1)
            h["channel_weights"][wi, u0:u1] = effective.detach().cpu().numpy()
            h["intercept"][wi, u0:u1] = intercept.detach().cpu().numpy()
            del x, fields
    del f
    h.close()
    rr = np.stack([corr_columns(oof[wi], target[wi]) for wi in range(len(windows))])
    print(f"{roi}: saved {len(units)} x {len(windows)} proxies; mean r={np.nanmean(rr):.4f}", flush=True)
    return {"roi": roi, "n_units": int(len(units)), "n_proxies": int(len(units) * len(windows)),
            "mean_oof_r_all_windows": float(np.nanmean(rr)), "file": str(path)}


def build_index(cohort: pd.DataFrame, roi_results: list[dict], windows: np.ndarray):
    db = OUT / "proxy_index.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    rows = []
    for item in roi_results:
        if not item.get("n_units"):
            continue
        roi = item["roi"]
        sel = cohort[cohort.roi.eq(roi)].copy().reset_index(drop=True)
        with h5py.File(item["file"], "r") as h:
            rr = h["oof_r"][:]
            depth = h["depth"][:]
        for ui, row in sel.iterrows():
            for wi, (a, b) in enumerate(windows):
                rows.append({
                    "unit_global": int(row.unit_global), "roi": roi,
                    "native_area": str(row.native_area), "monkey": str(row.monkey),
                    "session": int(row.session), "unit_type": int(row.unit_type),
                    "unit_type_name": str(row.unit_type_name), "window_index": wi,
                    "window_start_ms": int(a), "window_end_ms": int(b),
                    "depth": float(depth[wi, ui]), "oof_r": float(rr[wi, ui]),
                    "released_independent_consistency": float(row.released_independent_consistency),
                    "proxy_rank": RANK, "h5_file": Path(item["file"]).name,
                    "h5_unit_index": ui,
                })
    table = pd.DataFrame(rows)
    table.to_sql("proxies", con, index=False)
    con.execute("CREATE INDEX idx_unit_window ON proxies(unit_global, window_index)")
    con.execute("CREATE INDEX idx_roi ON proxies(roi)")
    con.execute("CREATE INDEX idx_area ON proxies(native_area)")
    con.execute("CREATE INDEX idx_r ON proxies(oof_r)")
    con.commit(); con.close()
    table.to_csv(OUT / "proxy_manifest.csv.gz", index=False, compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort", choices=("pilot", "expansion", "expansion_strict", "functional_validation", "consistency_only"), default="expansion",
        help="Frozen response-blind cohort to train.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory. A cohort-specific project directory is used by default.",
    )
    parser.add_argument("--only-roi", choices=ROIS, default=None,
                        help="Train only one ROI, preserving completed ROI files.")
    return parser.parse_args()


def main():
    global OUT
    args = parse_args()
    if args.output is not None:
        OUT = args.output.resolve()
    elif args.cohort == "pilot":
        OUT = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
    elif args.cohort == "functional_validation":
        OUT = PROJECT / "proxy_bank_64d_functional_it_validation_92_2026-08-26"
    elif args.cohort == "expansion_strict":
        OUT = PROJECT / "proxy_bank_64d_expansion_strict_2026-08-27"
    elif args.cohort == "consistency_only":
        OUT = PROJECT / "proxy_bank_64d_consistency_ge_0p4_all_units_2026-08-30"
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    cohort = load_cohort(args.cohort)
    maps = np.load(MAPS_PATH, mmap_mode="r")
    if maps.shape != (N_IMAGES, N_LAYERS, 256, GRID, GRID):
        raise RuntimeError(f"unexpected map shape {maps.shape}")
    responses = np.load(OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(OLD_BANK / "windows_10ms.npy")

    # Final normalization shared by every deployable proxy; keep this temporary
    # full tensor on CPU to avoid reserving the GPU before fold training.
    raw = torch.as_tensor(np.asarray(maps[:, :, :RANK], np.float32)).flatten(3)
    full_mean = raw.mean((0, 3), keepdim=True)
    full_std = raw.std((0, 3), keepdim=True).clamp_min(1e-5)
    np.savez_compressed(
        OUT / "feature_normalization_full.npz",
        mean=full_mean.squeeze(0).squeeze(-1).cpu().numpy(),
        std=full_std.squeeze(0).squeeze(-1).cpu().numpy(),
        layers=np.asarray(LAYERS), rank=RANK,
    )
    del raw
    results = []
    roi_list = [args.only_roi] if args.only_roi else ROIS
    for roi in roi_list:
        results.append(train_roi(roi, cohort, maps, responses, full_mean, full_std, device))
    if args.only_roi is None:
        build_index(cohort, results, windows)
    audit = {
        "cohort": str(cohort.cohort_name.iloc[0]),
        "cohort_mode": args.cohort,
        "strict_pool_units": int(pd.read_csv(COHORT_DIR / "tier_A_strict_pool.csv").unit_global.nunique()),
        "trained_unique_units": int(cohort.unit_global.nunique()),
        "n_windows": int(len(windows)), "n_proxies": int(len(cohort) * len(windows)),
        "all_windows_retained": True, "reliable_onset_used_for_training": False,
        "temporal_sharing": False, "rank_per_node": RANK,
        "pca_basis": str(BASIS_PATH), "feature_maps": str(MAPS_PATH),
        "spatial_and_depth": "fold-local/final continuous fwRF and 17-node Gaussian gate frozen from archived PCA8 proxies",
        "selection_rule": "released_independent_consistency >= 0.4 only; model prediction r not used",
        "readout": f"signed dual ridge, fixed alpha={RIDGE} selected from rank64 sweep median",
        "roi_results": results, "elapsed_seconds": time.time() - started,
    }
    (OUT / "training_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
