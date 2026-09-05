"""Map nested ResNet depth features to held-out neural G/T/D coefficients."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import h5py
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
from pilot_tvsd_continuous_gaussian import continuous_gaussian  # noqa: E402
from plot_orthogonal_temporal_modes_2d import raw_directions, symmetric_orthogonalize  # noqa: E402
from run_trin_closed_loop_validation import fixed_folds  # noqa: E402

BANK = PROJECT / "proxy_bank_10ms_all_windows"
FROZEN = PROJECT / "results" / "orthogonal_temporal_modes_2d_high_consistency_2026-08-27"
OUT = PROJECT / "results" / "resnet_depth_to_GTD_discovery1200_2026-08-27"
MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "block_pca8_14_tvsd_frozen" / "maps_f16.npy"
SEMANTIC = PROJECT / "SOM_语义地图_原方法复现" / "triple_n_1000_caption_embeddings_minilm.npy"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
DIRECTIONS = ["G", "T", "D"]
MODELS = ["early", "early_mid", "all_depths", "semantic32",
          "early_mid_semantic", "all_depths_semantic"]
RIDGE = .10
BATCH = 32


def slug(roi: str) -> str:
    return roi.lower().replace(" ", "_")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a - np.mean(a); y = b - np.mean(b)
    den = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / den) if den > 1e-12 else np.nan


def r2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.sum((a - np.mean(a))**2))
    return 1 - float(np.sum((a - b)**2)) / den if den > 1e-12 else np.nan


def ridge_predict(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor) -> torch.Tensor:
    """Target-specific batched ridge: images x units x features/targets."""
    mx = xtr.mean(0); sx = xtr.std(0).clamp_min(1e-5)
    xx = (xtr - mx[None]) / sx[None]
    xt = (xte - mx[None]) / sx[None]
    my = ytr.mean(0); yy = ytr - my[None]
    gram = torch.einsum("nbd,nbe->bde", xx, xx)
    alpha = RIDGE * gram.diagonal(dim1=1, dim2=2).mean(1)
    gram.diagonal(dim1=1, dim2=2).add_(alpha[:, None].clamp_min(1e-6))
    rhs = torch.einsum("nbd,nbk->bdk", xx, yy)
    beta = torch.linalg.solve(gram, rhs)
    return torch.einsum("nbd,bdk->nbk", xt, beta) + my[None]


def fold_semantic_pca(embedding: np.ndarray, train: np.ndarray, rank: int = 32) -> np.ndarray:
    mean = embedding[train].mean(0, keepdims=True)
    centered = embedding[train] - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return ((embedding - mean) @ vt[:rank].T).astype(np.float32)


def neural_coefficients(neural: np.ndarray, train: np.ndarray, times: np.ndarray):
    """Return train-defined orthogonal G/T/D coefficients for all images."""
    n_units, n_images, _ = neural.shape
    answer = np.empty((n_images, n_units, 3), np.float32)
    for ui in range(n_units):
        template = np.mean(neural[ui, train], axis=0)
        raw, _ = raw_directions(template, times)
        basis, _, _ = symmetric_orthogonalize(raw)
        answer[:, ui] = ((neural[ui] - template[None]) @ basis).astype(np.float32)
    return answer


def load_roi(roi: str, frozen: pd.DataFrame, responses: np.memmap,
             baseline_idx: np.ndarray, time_idx: np.ndarray):
    q = frozen[frozen.roi.eq(roi)].sort_values("unit_global").reset_index(drop=True)
    units = q.unit_global.to_numpy(int)
    raw = np.asarray(responses[:, :, units], np.float32)
    baseline = np.mean(raw[baseline_idx], axis=0)
    neural = (raw[time_idx] - baseline[None]).transpose(2, 1, 0)
    with h5py.File(BANK / f"{slug(roi)}_proxy_bank.h5", "r") as h:
        bank_units = h["unit_global"][:].astype(int)
        lookup = {int(unit): i for i, unit in enumerate(bank_units)}
        local = np.asarray([lookup[int(unit)] for unit in units], int)
        # Fold-specific fields were selected using only that fold's training images.
        spatial_all = np.asarray(h["fold_spatial_parameters"][:, time_idx], np.float32)
        spatial = spatial_all[:, :, local].mean(axis=1)
    return q, neural, spatial


def evaluate_roi(roi: str, frozen: pd.DataFrame, responses: np.memmap,
                 maps: np.memmap, baseline_idx: np.ndarray, time_idx: np.ndarray,
                 times: np.ndarray, semantic: np.ndarray, device: torch.device):
    q, neural, spatial = load_roi(roi, frozen, responses, baseline_idx, time_idx)
    rows = []
    pooled_prediction = {name: np.full((1000, len(q), 3), np.nan, np.float32) for name in MODELS}
    fmap = torch.as_tensor(np.asarray(maps, np.float32), device=device).flatten(2)
    for fold, (train, test) in enumerate(fixed_folds(1000)):
        target = neural_coefficients(neural, train, times)
        semantic_fold = torch.as_tensor(fold_semantic_pca(semantic, train), device=device)
        for start in range(0, len(q), BATCH):
            stop = min(start + BATCH, len(q))
            params = torch.as_tensor(spatial[fold, start:stop], device=device)
            fields = continuous_gaussian(params[:, 0], params[:, 1], params[:, 2], 14)
            features = torch.einsum("ncp,bp->nbc", fmap, fields)
            semantic_features = semantic_fold[:, None, :].expand(-1, stop - start, -1)
            y = torch.as_tensor(target[:, start:stop], device=device)
            feature_sets = {
                "early": features[:, :, :32],
                "early_mid": features[:, :, :112],
                "all_depths": features,
                "semantic32": semantic_features,
                "early_mid_semantic": torch.cat([features[:, :, :112], semantic_features], dim=2),
                "all_depths_semantic": torch.cat([features, semantic_features], dim=2),
            }
            for name, feature_set in feature_sets.items():
                prediction = ridge_predict(feature_set[train], y[train],
                                           feature_set[test]).detach().cpu().numpy()
                pooled_prediction[name][test, start:stop] = prediction
            del fields, features, semantic_features, feature_sets, y
        print(f"{roi} fold {fold + 1}/5", flush=True)
    for ui, unit in q.iterrows():
        # Recreate each fold's target only for evaluation so test coefficients
        # are always paired with their train-defined basis.
        target_oof = np.empty((1000, 3), np.float32)
        for train, test in fixed_folds(1000):
            target_oof[test] = neural_coefficients(neural[ui:ui + 1], train, times)[test, 0]
        model_metrics = {}
        for name in MODELS:
            for di, direction in enumerate(DIRECTIONS):
                pred = pooled_prediction[name][:, ui, di]
                model_metrics[(name, direction)] = r2(target_oof[:, di], pred)
                rows.append({
                    "unit_global": int(unit.unit_global), "roi": roi,
                    "monkey": str(unit.monkey), "session": int(unit.session),
                    "direction": direction, "model": name,
                    "oof_r": corr(target_oof[:, di], pred),
                    "oof_r2": model_metrics[(name, direction)],
                })
        for direction in DIRECTIONS:
            early = model_metrics[("early", direction)]
            mid = model_metrics[("early_mid", direction)]
            full = model_metrics[("all_depths", direction)]
            semantic_only = model_metrics[("semantic32", direction)]
            mid_semantic = model_metrics[("early_mid_semantic", direction)]
            full_semantic = model_metrics[("all_depths_semantic", direction)]
            rows.append({
                "unit_global": int(unit.unit_global), "roi": roi,
                "monkey": str(unit.monkey), "session": int(unit.session),
                "direction": direction, "model": "increments",
                "oof_r": np.nan, "oof_r2": full,
                "delta_mid_r2": mid - early, "delta_deep_r2": full - mid,
                "semantic_only_r2": semantic_only,
                "delta_semantic_beyond_mid_r2": mid_semantic - mid,
                "delta_semantic_beyond_all_r2": full_semantic - full,
            })
    del fmap
    if device.type == "cuda": torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summaries(data: pd.DataFrame):
    models = data[data.model.ne("increments")]
    increments = data[data.model.eq("increments")]
    session_models = models.groupby(
        ["roi", "monkey", "session", "direction", "model"], as_index=False
    )[["oof_r", "oof_r2"]].median()
    session_inc = increments.groupby(
        ["roi", "monkey", "session", "direction"], as_index=False
    )[["delta_mid_r2", "delta_deep_r2", "semantic_only_r2",
       "delta_semantic_beyond_mid_r2", "delta_semantic_beyond_all_r2"]].median()
    roi_models = session_models.groupby(["roi", "direction", "model"], as_index=False)[
        ["oof_r", "oof_r2"]].median()
    roi_inc = session_inc.groupby(["roi", "direction"], as_index=False)[
        ["delta_mid_r2", "delta_deep_r2", "semantic_only_r2",
         "delta_semantic_beyond_mid_r2", "delta_semantic_beyond_all_r2"]].median()
    return session_models, session_inc, roi_models, roi_inc


def signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    observed = abs(float(values.mean()))
    if len(values) <= 18:
        null = np.asarray([abs(float(np.mean(values * np.asarray(signs))))
                           for signs in product((-1, 1), repeat=len(values))])
    else:
        rng = np.random.default_rng(20260827 + len(values))
        null = np.abs(np.mean(
            rng.choice((-1, 1), size=(200_000, len(values))) * values[None], axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float); order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked); out[order] = np.minimum(ranked, 1)
    return out


def increment_tests(session_inc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["delta_mid_r2", "delta_deep_r2", "semantic_only_r2",
               "delta_semantic_beyond_mid_r2", "delta_semantic_beyond_all_r2"]
    for metric in metrics:
        start = len(rows)
        for direction in DIRECTIONS:
            values = session_inc.loc[session_inc.direction.eq(direction), metric].to_numpy(float)
            rows.append({"metric": metric, "direction": direction,
                         "n_session_roi_groups": len(values),
                         "mean_session_value": float(np.mean(values)),
                         "median_session_value": float(np.median(values)),
                         "positive_session_fraction": float(np.mean(values > 0)),
                         "signflip_p": signflip_p(values)})
        corrected = bh(np.asarray([row["signflip_p"] for row in rows[start:]]))
        for row, q in zip(rows[start:], corrected):
            row["signflip_p_bh_three_directions"] = float(q)
    return pd.DataFrame(rows)


def heatmaps(roi_models: pd.DataFrame, roi_inc: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), constrained_layout=True)
    specifications = [
        ("early", "Early ResNet OOF R²", -.1, .25),
        ("early_mid", "Early + middle OOF R²", -.1, .25),
        ("all_depths", "All depths OOF R²", -.1, .25),
    ]
    for ax, (model, title, lo, hi) in zip(axes, specifications):
        matrix = roi_models[roi_models.model.eq(model)].pivot(
            index="roi", columns="direction", values="oof_r2").reindex(
                index=ROIS, columns=DIRECTIONS).to_numpy(float)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto")
        for i in range(6):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), DIRECTIONS); ax.set_yticks(range(6), [LABELS[r] for r in ROIS])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Nested ResNet feature depth predicts held-out neural G/T/D", fontweight="bold")
    fig.savefig(OUT / "01_nested_depth_GTD_oof_R2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ["delta_mid_r2", "delta_deep_r2"],
                                 ["Middle-layer incremental R²", "Deep-layer incremental R²"]):
        matrix = roi_inc.pivot(index="roi", columns="direction", values=metric).reindex(
            index=ROIS, columns=DIRECTIONS).to_numpy(float)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-.08, vmax=.08, aspect="auto")
        for i in range(6):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), DIRECTIONS); ax.set_yticks(range(6), [LABELS[r] for r in ROIS])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("What additional network depth contributes to G/T/D", fontweight="bold")
    fig.savefig(OUT / "02_mid_and_deep_incremental_R2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["semantic_only_r2", "delta_semantic_beyond_mid_r2", "delta_semantic_beyond_all_r2"],
        ["Caption semantics alone R²", "Semantic ΔR² beyond early+middle",
         "Semantic ΔR² beyond all visual depths"]):
        matrix = roi_inc.pivot(index="roi", columns="direction", values=metric).reindex(
            index=ROIS, columns=DIRECTIONS).to_numpy(float)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-.06, vmax=.06, aspect="auto")
        for i in range(6):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), DIRECTIONS); ax.set_yticks(range(6), [LABELS[r] for r in ROIS])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Caption-semantic contribution to held-out neural G/T/D", fontweight="bold")
    fig.savefig(OUT / "03_caption_semantic_incremental_R2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(FROZEN / "unit_orthogonal_GTD_weights_and_modes.csv")
    if len(frozen) != 1200:
        raise RuntimeError("Frozen discovery cohort changed")
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    baseline_idx = np.flatnonzero(windows[:, 1] < 0)
    time_idx = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time_idx].mean(axis=1).astype(float)
    maps = np.load(MAPS, mmap_mode="r")
    semantic = np.load(SEMANTIC).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = [evaluate_roi(roi, frozen, responses, maps, baseline_idx, time_idx, times, semantic, device)
             for roi in ROIS]
    data = pd.concat(parts, ignore_index=True)
    data.to_csv(OUT / "unit_nested_depth_GTD_mapping.csv", index=False, encoding="utf-8-sig")
    session_models, session_inc, roi_models, roi_inc = summaries(data)
    session_models.to_csv(OUT / "session_nested_depth_GTD_mapping.csv", index=False, encoding="utf-8-sig")
    session_inc.to_csv(OUT / "session_depth_incremental_GTD.csv", index=False, encoding="utf-8-sig")
    roi_models.to_csv(OUT / "roi_nested_depth_GTD_mapping.csv", index=False, encoding="utf-8-sig")
    roi_inc.to_csv(OUT / "roi_depth_incremental_GTD.csv", index=False, encoding="utf-8-sig")
    increment_tests(session_inc).to_csv(
        OUT / "session_increment_signflip_tests.csv", index=False, encoding="utf-8-sig")
    heatmaps(roi_models, roi_inc)
    audit = {
        "frozen_discovery_units": 1200, "n_images": 1000, "folds": 5,
        "target": "fold-train-defined orthogonal neural G/T/D coefficients",
        "feature_backbone": "frozen ImageNet-V2 ResNet50, neural-blind PCA8 per 17 nodes",
        "spatial_pooling": "fold-train proxy fwRF parameters averaged across 70-189 ms",
        "nested_models": {"early": "nodes 0-3 (32D)",
                          "early_mid": "nodes 0-13 (112D)",
                          "all_depths": "nodes 0-16 (136D)",
                          "semantic32": "fold-train PCA32 of BLIP-caption MiniLM embeddings",
                          "semantic_combinations": "semantic32 appended to early_mid and all_depths"},
        "readout": "fixed trace-scaled ridge, fitted jointly to G/T/D on fold training images",
        "roi_labels_used_for_fitting": False, "backbone_retrained": False,
        "proxy_curve_predictions_used_as_features": False,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(roi_models.to_string(index=False), flush=True)
    print(roi_inc.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
