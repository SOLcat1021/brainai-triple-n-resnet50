"""Compute reusable similarities for the complete unit-window proxy bank.

Exact same-unit temporal similarities are stored for every pair of windows.
For cross-unit work, all OOF prediction profiles are archived as a 64-D
Johnson-Lindenstrauss fingerprint and an all-unit top-32 graph is built at each
matched time window.  This avoids materializing an intractable 503k x 503k
dense matrix while retaining every proxy for later exact queries.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
BANK = PROJECT / "proxy_bank_10ms_all_windows"
META = BANK / "unit_metadata.csv"
WINDOWS = BANK / "windows_10ms.npy"
OUT = BANK / "similarities"
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
EMBED_DIM = 64
TOP_K = 32


def slug(s: str) -> str:
    return s.lower().replace(" ", "_")


def standardize_prediction(x: np.ndarray) -> np.ndarray:
    # input [windows, images, units]
    x = np.asarray(x, np.float32)
    x -= x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True))
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-8)


def cosine_temporal(x: np.ndarray) -> np.ndarray:
    # input [windows, units, features] -> [units, windows, windows]
    x = np.asarray(x, np.float32)
    norm = np.linalg.norm(x, axis=2, keepdims=True)
    x = np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-8)
    return np.einsum("wuf,vuf->uwv", x, x, optimize=True).astype(np.float32)


def gaussian_overlap(space: np.ndarray) -> np.ndarray:
    # space [windows, units, (x,y,sigma)] -> [units, windows, windows]
    x, y, s = (space[..., i].T.astype(np.float32) for i in range(3))  # [unit,window]
    s2 = s * s
    denom = s2[:, :, None] + s2[:, None, :]
    distance2 = ((x[:, :, None] - x[:, None, :]) ** 2 +
                 (y[:, :, None] - y[:, None, :]) ** 2)
    prefactor = 2 * s[:, :, None] * s[:, None, :] / np.maximum(denom, 1e-8)
    return (prefactor * np.exp(-distance2 / np.maximum(4 * denom, 1e-8))).astype(np.float32)


def build_fingerprints(meta: pd.DataFrame, windows: np.ndarray,
                       projection: np.ndarray) -> Path:
    path = OUT / "all_proxy_functional_fingerprints.h5"
    n_units = len(meta); n_windows = len(windows)
    with h5py.File(path, "w") as out:
        out.create_dataset("unit_global", data=np.arange(n_units, dtype=np.int32))
        out.create_dataset("windows_ms", data=windows.astype(np.int16))
        out.create_dataset("random_projection", data=projection.astype(np.float32), compression="lzf")
        ds = out.create_dataset("oof_prediction_fingerprint", shape=(n_windows, n_units, EMBED_DIM),
                                dtype=np.float16, chunks=(1, min(512, n_units), EMBED_DIM),
                                compression="lzf")
        rds = out.create_dataset("oof_r", shape=(n_windows, n_units), dtype=np.float32)
        dds = out.create_dataset("depth", shape=(n_windows, n_units), dtype=np.float32)
        for roi in ROIS:
            units = meta.loc[meta.roi.eq(roi), "unit_global"].to_numpy(np.int32)
            with h5py.File(BANK / f"{slug(roi)}_proxy_bank.h5", "r") as source:
                rds[:, units] = source["oof_r"][:]
                dds[:, units] = source["depth"][:]
                for wi in range(n_windows):
                    for u0 in range(0, len(units), 512):
                        u1 = min(len(units), u0 + 512)
                        pred = np.asarray(source["oof_prediction"][wi, :, u0:u1], np.float32)
                        pred -= pred.mean(0, keepdims=True)
                        pred /= np.maximum(np.linalg.norm(pred, axis=0, keepdims=True), 1e-8)
                        emb = pred.T @ projection
                        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
                        ds[wi, units[u0:u1]] = emb.astype(np.float16)
            print(f"fingerprints: {roi}", flush=True)
        out.attrs["definition"] = (
            "64-D fixed random projection of standardized 1000-image OOF prediction profile; "
            "cosine approximates functional prediction correlation")
    return path


def same_unit_similarity(meta: pd.DataFrame, windows: np.ndarray) -> None:
    adjacent_rows = []
    for roi in ROIS:
        units = meta.loc[meta.roi.eq(roi), "unit_global"].to_numpy(np.int32)
        source_path = BANK / f"{slug(roi)}_proxy_bank.h5"
        out_path = OUT / f"{slug(roi)}_same_unit_temporal_similarity.h5"
        with h5py.File(source_path, "r") as source, h5py.File(out_path, "w") as out:
            n = len(units); w = len(windows)
            out.create_dataset("unit_global", data=units)
            out.create_dataset("windows_ms", data=windows)
            functional = out.create_dataset("functional_oof_similarity", shape=(n, w, w),
                                            dtype=np.float16, chunks=(min(128, n), w, w), compression="lzf")
            gate_ds = out.create_dataset("layer_gate_cosine", shape=(n, w, w), dtype=np.float16,
                                         chunks=(min(128, n), w, w), compression="lzf")
            weight_ds = out.create_dataset("signed_weight_cosine", shape=(n, w, w), dtype=np.float16,
                                           chunks=(min(128, n), w, w), compression="lzf")
            space_ds = out.create_dataset("spatial_gaussian_overlap", shape=(n, w, w), dtype=np.float16,
                                          chunks=(min(128, n), w, w), compression="lzf")
            depth_ds = out.create_dataset("absolute_depth_difference", shape=(n, w, w), dtype=np.float16,
                                          chunks=(min(128, n), w, w), compression="lzf")
            for u0 in range(0, n, 128):
                u1 = min(n, u0 + 128)
                pred = standardize_prediction(source["oof_prediction"][:, :, u0:u1])
                f_sim = np.einsum("wib,vib->bwv", pred, pred, optimize=True)
                gates = np.asarray(source["layer_gate"][:, u0:u1], np.float32)
                weights = np.asarray(source["channel_weights"][:, u0:u1], np.float32)
                space = np.asarray(source["spatial_parameters"][:, u0:u1], np.float32)
                depth = np.asarray(source["depth"][:, u0:u1], np.float32).T
                g_sim = cosine_temporal(gates)
                w_sim = cosine_temporal(weights)
                s_sim = gaussian_overlap(space)
                d_diff = np.abs(depth[:, :, None] - depth[:, None, :])
                functional[u0:u1] = f_sim.astype(np.float16)
                gate_ds[u0:u1] = g_sim.astype(np.float16)
                weight_ds[u0:u1] = w_sim.astype(np.float16)
                space_ds[u0:u1] = s_sim.astype(np.float16)
                depth_ds[u0:u1] = d_diff.astype(np.float16)
                rr = np.asarray(source["oof_r"][:, u0:u1], np.float32)
                for j, g in enumerate(units[u0:u1]):
                    for wi in range(w - 1):
                        adjacent_rows.append({
                            "unit_global": int(g), "roi": roi,
                            "t": int(windows[wi, 0]), "next_t": int(windows[wi + 1, 0]),
                            "r": float(rr[wi, j]), "next_r": float(rr[wi + 1, j]),
                            "functional_oof_similarity": float(f_sim[j, wi, wi + 1]),
                            "layer_gate_cosine": float(g_sim[j, wi, wi + 1]),
                            "signed_weight_cosine": float(w_sim[j, wi, wi + 1]),
                            "spatial_gaussian_overlap": float(s_sim[j, wi, wi + 1]),
                            "absolute_depth_difference": float(d_diff[j, wi, wi + 1]),
                        })
        print(f"same-unit temporal matrices: {roi}", flush=True)
    pd.DataFrame(adjacent_rows).to_csv(
        OUT / "same_unit_adjacent_window_similarity.csv.gz", index=False, compression="gzip")


def cross_unit_graph(meta: pd.DataFrame, windows: np.ndarray, fingerprint_path: Path,
                     device: torch.device) -> None:
    n = len(meta); w = len(windows)
    graph_path = OUT / "time_matched_cross_unit_top32.h5"
    roi_codes, roi_names = pd.factorize(meta.roi, sort=False)
    rng = np.random.default_rng(20260825)
    sample_rows = []
    with h5py.File(fingerprint_path, "r") as source, h5py.File(graph_path, "w") as out:
        out.create_dataset("unit_global", data=np.arange(n, dtype=np.int32))
        out.create_dataset("windows_ms", data=windows)
        ids_ds = out.create_dataset("neighbor_unit_global", shape=(w, n, TOP_K), dtype=np.int32,
                                    chunks=(1, min(512, n), TOP_K), compression="lzf")
        sim_ds = out.create_dataset("neighbor_similarity", shape=(w, n, TOP_K), dtype=np.float16,
                                    chunks=(1, min(512, n), TOP_K), compression="lzf")
        centroid = np.empty((w, len(roi_names), len(roi_names)), np.float32)
        for wi in range(w):
            emb_np = np.asarray(source["oof_prediction_fingerprint"][wi], np.float32)
            emb_np /= np.maximum(np.linalg.norm(emb_np, axis=1, keepdims=True), 1e-8)
            emb = torch.as_tensor(emb_np, device=device)
            for q0 in range(0, n, 512):
                q1 = min(n, q0 + 512)
                score = emb[q0:q1] @ emb.T
                rows = torch.arange(q1 - q0, device=device)
                score[rows, torch.arange(q0, q1, device=device)] = -torch.inf
                values, indices = torch.topk(score, k=TOP_K, dim=1)
                ids_ds[wi, q0:q1] = indices.cpu().numpy().astype(np.int32)
                sim_ds[wi, q0:q1] = values.cpu().numpy().astype(np.float16)
            centers = []
            for code in range(len(roi_names)):
                c = emb_np[roi_codes == code].mean(0)
                c /= max(np.linalg.norm(c), 1e-8)
                centers.append(c)
            centers = np.asarray(centers)
            centroid[wi] = centers @ centers.T

            # Stable pair samples characterize the distribution, not only its mean.
            for a in range(len(roi_names)):
                ia = np.flatnonzero(roi_codes == a)
                for b in range(a, len(roi_names)):
                    ib = np.flatnonzero(roi_codes == b)
                    count = 10000
                    left = rng.choice(ia, count, replace=True)
                    right = rng.choice(ib, count, replace=True)
                    if a == b:
                        same = left == right
                        while same.any():
                            right[same] = rng.choice(ib, same.sum(), replace=True)
                            same = left == right
                    values = np.sum(emb_np[left] * emb_np[right], axis=1)
                    sample_rows.append({
                        "t": int(windows[wi, 0]), "roi_a": str(roi_names[a]),
                        "roi_b": str(roi_names[b]), "n_pairs": count,
                        "mean": float(values.mean()), "std": float(values.std()),
                        "q05": float(np.quantile(values, .05)),
                        "median": float(np.median(values)),
                        "q95": float(np.quantile(values, .95)),
                    })
            print(f"cross-unit graph: {windows[wi].tolist()}", flush=True)
        out.create_dataset("roi_centroid_similarity", data=centroid)
        out.create_dataset("roi_names", data=np.asarray(roi_names.astype(str), dtype="S32"))
        out.attrs["similarity"] = "cosine of 64-D OOF functional fingerprints"
        out.attrs["top_k"] = TOP_K
    pd.DataFrame(sample_rows).to_csv(OUT / "roi_pair_similarity_distributions.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    meta = pd.read_csv(META).sort_values("unit_global").reset_index(drop=True)
    windows = np.load(WINDOWS)
    rng = np.random.default_rng(20260825)
    projection = rng.normal(size=(1000, EMBED_DIM)).astype(np.float32)
    projection /= np.maximum(np.linalg.norm(projection, axis=0, keepdims=True), 1e-8)
    fingerprint_path = build_fingerprints(meta, windows, projection)
    same_unit_similarity(meta, windows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_unit_graph(meta, windows, fingerprint_path, device)
    schema = {
        "same_unit_exact": "six ROI HDF5 files; complete 21x21 matrices per unit",
        "different_units": "top-32 time-matched neighbor graph for every unit-window",
        "different_rois": "21 complete ROI-centroid matrices plus deterministic pair distributions",
        "complete_query_basis": "all OOF responses remain in ROI proxy banks; all 64-D fingerprints retained",
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "similarity_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(schema, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
