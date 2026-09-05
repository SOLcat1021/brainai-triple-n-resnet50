"""Train two PCA64 neural readouts on mutually exclusive image halves.

This audit keeps the response-blind 113-unit cohort, PCA basis, ridge value,
spatial field, and depth gate frozen.  Only the signed channel readout is
refit: one model sees images in half A and the other sees half B.  Each model
is evaluated on the opposite half.  The resulting weights provide a stricter
semantic-profile replication check than the overlapping 80/20 OOF folds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from train_64d_high_quality_proxy_batch import (
    GRID,
    LAYERS,
    N_IMAGES,
    N_LAYERS,
    OLD_BANK,
    PROJECT,
    RANK,
    RANK_DIR,
    RIDGE,
    ROIS,
    TARGET_BATCH,
    corr_columns,
    dual_ridge,
    gaussian_fields,
    slug,
)


BANK = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
OUT = BANK / "disjoint_split_half"
MAPS_PATH = RANK_DIR / "trin_resnet50_pca256_maps_f16.npy"
SEED = 20260826


def disjoint_halves() -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(SEED).permutation(N_IMAGES)
    return np.sort(order[: N_IMAGES // 2]), np.sort(order[N_IMAGES // 2 :])


def train_roi(
    roi: str,
    maps: np.ndarray,
    responses: np.memmap,
    halves: tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> dict:
    source_path = BANK / f"{slug(roi)}_proxy_bank_64d.h5"
    with h5py.File(source_path, "r") as source:
        units = source["unit_global"][:].astype(np.int32)
        windows = source["windows_ms"][:].astype(np.int16)
        fields_all = source["spatial_parameters"][:].astype(np.float32)
        gates_all = source["layer_gate"][:].astype(np.float32)

    target = np.asarray(responses[:, :, units], np.float32)
    out_path = OUT / f"{slug(roi)}_split_half_64d.h5"
    with h5py.File(out_path, "w") as h:
        h.create_dataset("unit_global", data=units)
        h.create_dataset("windows_ms", data=windows)
        h.create_dataset("train_indices", data=np.stack(halves).astype(np.int16))
        h.create_dataset("test_indices", data=np.stack((halves[1], halves[0])).astype(np.int16))
        h.create_dataset("layer_gate", data=gates_all, compression="gzip", compression_opts=4)
        h.create_dataset(
            "split_normalization_std", shape=(2, N_LAYERS, RANK), dtype="f4",
        )
        h.create_dataset(
            "split_channel_weights", shape=(2, len(windows), len(units), N_LAYERS * RANK),
            dtype="f4", chunks=(1, 1, min(len(units), 4), N_LAYERS * RANK),
            compression="gzip", compression_opts=4,
        )
        h.create_dataset("cross_half_r", shape=(2, len(windows), len(units)), dtype="f4")
        h.attrs["split_seed"] = SEED
        h.attrs["readout_training"] = "two mutually exclusive fixed 500-image halves"
        h.attrs["evaluation"] = "each readout evaluated on the opposite 500-image half"
        h.attrs["frozen_components"] = (
            "response-blind cohort, PCA64 basis, ridge alpha, final spatial field, final depth gate"
        )
        h.attrs["important_limit"] = (
            "spatial field and depth gate were estimated previously and are shared; only channel readouts are disjoint"
        )

        split_summaries = []
        for split, train in enumerate(halves):
            test = halves[1 - split]
            raw_train = torch.as_tensor(
                np.asarray(maps[train, :, :RANK], np.float32), device=device
            ).flatten(3)
            raw_test = torch.as_tensor(
                np.asarray(maps[test, :, :RANK], np.float32), device=device
            ).flatten(3)
            mean = raw_train.mean((0, 3), keepdim=True)
            std = raw_train.std((0, 3), keepdim=True).clamp_min(1e-5)
            h["split_normalization_std"][split] = std.squeeze(0).squeeze(-1).cpu().numpy()
            ftr = (raw_train - mean) / std
            fte = (raw_test - mean) / std
            del raw_train, raw_test

            r_by_window = []
            for wi in range(len(windows)):
                fields = torch.as_tensor(gaussian_fields(fields_all[wi]), device=device)
                xtr = torch.einsum("nldp,up->nuld", ftr, fields)
                xte = torch.einsum("nldp,up->nuld", fte, fields)
                scale = torch.as_tensor(
                    np.sqrt(np.clip(gates_all[wi] * N_LAYERS, 1e-5, None)), device=device
                )
                xtr = (xtr * scale[None, :, :, None]).flatten(2)
                xte = (xte * scale[None, :, :, None]).flatten(2)
                ytr = torch.as_tensor(target[wi, train], device=device)
                predictions = np.empty((len(test), len(units)), np.float32)
                for u0 in range(0, len(units), TARGET_BATCH):
                    u1 = min(len(units), u0 + TARGET_BATCH)
                    pred, beta, _, _ = dual_ridge(
                        xtr[:, u0:u1], ytr[:, u0:u1], xte[:, u0:u1]
                    )
                    predictions[:, u0:u1] = pred.detach().cpu().numpy()
                    effective = beta * torch.repeat_interleave(scale[u0:u1], RANK, dim=1)
                    h["split_channel_weights"][split, wi, u0:u1] = (
                        effective.detach().cpu().numpy()
                    )
                rr = corr_columns(predictions, target[wi, test])
                h["cross_half_r"][split, wi] = rr
                r_by_window.append(rr)
                del xtr, xte, fields
            del ftr, fte
            torch.cuda.empty_cache()
            split_summaries.append({
                "split": split,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mean_cross_half_r_all_windows": float(np.nanmean(np.stack(r_by_window))),
            })
            print(f"{roi}: disjoint half {split + 1}/2 complete", flush=True)

    return {
        "roi": roi,
        "n_units": int(len(units)),
        "file": str(out_path),
        "splits": split_summaries,
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    maps = np.load(MAPS_PATH, mmap_mode="r")
    if maps.shape != (N_IMAGES, N_LAYERS, 256, GRID, GRID):
        raise RuntimeError(f"unexpected map shape {maps.shape}")
    responses = np.load(OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    halves = disjoint_halves()
    if len(np.intersect1d(*halves)) != 0 or len(np.union1d(*halves)) != N_IMAGES:
        raise RuntimeError("split halves are not mutually exclusive and exhaustive")

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    results = [train_roi(roi, maps, responses, halves, device) for roi in ROIS]
    audit = {
        "seed": SEED,
        "n_images": N_IMAGES,
        "half_sizes": [int(len(x)) for x in halves],
        "intersection_size": int(len(np.intersect1d(*halves))),
        "union_size": int(len(np.union1d(*halves))),
        "ridge": RIDGE,
        "rank": RANK,
        "frozen_components": [
            "113-unit response-blind cohort",
            "response-blind PCA64 basis",
            "ridge alpha",
            "final spatial field",
            "final depth gate",
        ],
        "independent_component": "signed channel readout neural-response training images",
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "training_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
