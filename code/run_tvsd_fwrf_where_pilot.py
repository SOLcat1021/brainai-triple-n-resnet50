"""TVSD V4 dynamic-Where pilot with the original shared-Gaussian fwRF form.

Model form retained:
* one 2-D isotropic Gaussian shared by every ResNet feature channel;
* the original 15 x 15 centers x 10 radii = 2,250 candidate bank;
* signed feature weights and independent 20-ms neural fits.

Scalable solver substitution:
* response-blind PCA8 per ResNet stage;
* fold-wise marginal signed regression energy selects the Gaussian;
* marginal signed weights predict the independent repeated 100-image test set.

This is an explicit scale test, not labeled as a bitwise reproduction of the
original 20-epoch exhaustive SGD optimizer.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "cache" / "tvsd" / "original_fwrf_resnet50" / "spatial_pca8_14"
TARGETS = ROOT / "cache" / "tvsd" / "vit_features_targets"
OUT = ROOT / "ResNet50_原版fwRF_20ms时序网络剖析_2026-08-18" / "TVSD_V4_50channel"
MONKEYS = ("N", "F")
WINDOWS = np.asarray([(30 + 20 * i, 49 + 20 * i) for i in range(8)], int)
FOLDS = 5
SEED = 20260818


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 0)


def load_targets() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    train, test, meta = [], [], []
    for monkey in MONKEYS:
        channel = np.load(TARGETS / f"monkey{monkey}_time20_mua_pilot_channels.npz")
        target = np.load(TARGETS / f"monkey{monkey}_targets.npz")
        roi = channel["roi"].astype(str)
        mapped = channel["selected_mapped_channel"].astype(int)
        raw = channel["selected_raw_channel"].astype(int)
        local = np.flatnonzero(roi == "V4")
        oracle = target["oracle"][mapped[local]]
        local = local[np.argsort(oracle)[::-1][:25]]
        ytr = np.asarray(np.load(TARGETS / f"monkey{monkey}_time20_mua_pilot_f16.npy", mmap_mode="r")[:, local], np.float32)
        yte = np.asarray(np.load(TARGETS / f"monkey{monkey}_test_time20_mua_pilot_f16.npy", mmap_mode="r")[:, local], np.float32)
        train.append(ytr); test.append(yte)
        for j in local:
            meta.append({"monkey": monkey, "pilot_index": int(j), "mapped_channel": int(mapped[j]),
                         "raw_channel": int(raw[j]), "oracle": float(target["oracle"][mapped[j]])})
    # source shape: image x channel x window; concatenate channels.
    return np.concatenate(train, 1), np.concatenate(test, 1), meta


def stats_for_indices(f: torch.Tensor, y: torch.Tensor, idx: torch.Tensor):
    x = f[idx]; z = y[idx]; n = int(len(idx))
    return {
        "n": n,
        "sum_f": x.sum(0), "sum_y": z.sum(0), "sum_y2": (z * z).sum(0),
        "cross": torch.einsum("ndp,nv->dpv", x, z),
        "ff": torch.einsum("ndp,ndq->dpq", x, x),
    }


def subtract(a, b):
    return {key: (a[key] - b[key]) for key in a}


def add_many(parts):
    out = {}
    for key in parts[0]:
        out[key] = sum(part[key] for part in parts)
    return out


def centered(s):
    n = float(s["n"])
    mean_f = s["sum_f"] / n; mean_y = s["sum_y"] / n
    cov_fy = s["cross"] - mean_f[:, :, None] * s["sum_y"][None, None, :]
    cov_ff = s["ff"] - torch.einsum("dp,dq->dpq", s["sum_f"], s["sum_f"]) / n
    var_y = (s["sum_y2"] - s["sum_y"] ** 2 / n).clamp_min(1e-8)
    return mean_f, mean_y, cov_fy, cov_ff, var_y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    maps_train = torch.as_tensor(np.asarray(np.load(FEATURES / "train_maps_f16.npy", mmap_mode="r"), np.float32), device=device)
    maps_test = torch.as_tensor(np.asarray(np.load(FEATURES / "test_maps_f16.npy", mmap_mode="r"), np.float32), device=device)
    n, d, h, w = maps_train.shape
    f = maps_train.flatten(2)
    ft = maps_test.flatten(2)
    ytr_np, yte_np, channel_meta = load_targets()
    nu, nw = ytr_np.shape[1], ytr_np.shape[2]
    y = torch.as_tensor(ytr_np.reshape(n, nu * nw), device=device)

    xs, ys, sigmas = candidates()
    fields_np = gaussian_mass_stack(xs, ys, sigmas, h).reshape(len(xs), -1)
    fields = torch.as_tensor(fields_np, device=device)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    fold_id = np.empty(n, int)
    for k, idx in enumerate(np.array_split(order, FOLDS)):
        fold_id[idx] = k
    parts = []
    for k in range(FOLDS):
        idx = torch.as_tensor(np.flatnonzero(fold_id == k), device=device)
        parts.append(stats_for_indices(f, y, idx))
        print(f"sufficient statistics partition {k + 1}/{FOLDS}", flush=True)
    total = add_many(parts)

    fold_candidates = np.empty((FOLDS, nw, nu), int)
    fold_predictions = np.empty((FOLDS, 100, nw, nu), np.float32)
    fold_scores = np.empty((FOLDS, nw, nu), np.float32)
    for k in range(FOLDS):
        dev = subtract(total, parts[k])
        mean_f, mean_y, cov_fy, cov_ff, var_y = centered(dev)
        # Exact variance of every Gaussian-pooled projected feature.
        var_x = torch.einsum("gp,dpq,gq->gd", fields, cov_ff, fields).clamp_min(1e-8)
        numerator = torch.einsum("gp,dpv->gdv", fields, cov_fy)
        energy = (numerator.square() / (var_x[:, :, None] * var_y[None, None, :])).sum(1)
        chosen = energy.argmax(0)
        fold_candidates[k] = chosen.reshape(nu, nw).T.cpu().numpy()

        # Signed marginal weights. PCA channels are orthogonal before spatial
        # pooling; a small shrinkage guards residual cross-stage correlation.
        pick = torch.arange(nu * nw, device=device)
        num = numerator[chosen, :, pick]
        vx = var_x[chosen]
        shrink = 0.05 * vx.mean(1, keepdim=True)
        weight = num / (vx + shrink)
        selected_fields = fields[chosen]
        xt = torch.einsum("ndp,vp->nvd", ft, selected_fields)
        mean_x = torch.einsum("dp,vp->vd", mean_f, selected_fields)
        prediction = torch.einsum("nvd,vd->nv", xt - mean_x[None], weight) + mean_y[None]
        pred_np = prediction.cpu().numpy().reshape(100, nu, nw).transpose(0, 2, 1)
        fold_predictions[k] = pred_np
        fold_scores[k] = np.stack([corr_columns(pred_np[:, wi], yte_np[:, :, wi]) for wi in range(nw)])
        print(f"fold {k + 1}/{FOLDS} selected and predicted", flush=True)

    cx, cy, sg = xs[fold_candidates], ys[fold_candidates], sigmas[fold_candidates]
    pairs = list(combinations(range(FOLDS), 2))
    pair_distance = np.empty((nw, nu), np.float32)
    pair_logsigma = np.empty((nw, nu), np.float32)
    for wi in range(nw):
        for u in range(nu):
            pair_distance[wi, u] = np.mean([
                np.hypot(cx[a, wi, u] - cx[b, wi, u], cy[a, wi, u] - cy[b, wi, u]) for a, b in pairs])
            pair_logsigma[wi, u] = np.mean([
                abs(np.log(sg[a, wi, u]) - np.log(sg[b, wi, u])) for a, b in pairs])
    temporal = np.mean(np.hypot(np.diff(cx, axis=1), np.diff(cy, axis=1)), axis=0)
    jitter = 0.5 * (pair_distance[:-1] + pair_distance[1:])
    score = np.nanmean(fold_scores, 0)
    time_center = WINDOWS.mean(1); transition_center = 0.5 * (time_center[:-1] + time_center[1:])

    np.savez_compressed(
        OUT / "tvsd_v4_where_pilot.npz", candidates=fold_candidates, predictions=fold_predictions,
        scores=fold_scores, responses=yte_np.transpose(0, 2, 1), windows=WINDOWS,
        pair_distance=pair_distance, pair_logsigma=pair_logsigma,
        temporal=temporal, jitter=jitter, centers_x=xs, centers_y=ys, sigmas=sigmas,
    )

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(time_center, np.nanmean(score, 1), "-o", lw=2, color="#2878B5")
    axes[0].fill_between(time_center, np.nanmean(score, 1) - np.nanstd(score, 1) / np.sqrt(nu),
                        np.nanmean(score, 1) + np.nanstd(score, 1) / np.sqrt(nu), color="#2878B5", alpha=.18)
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set(title="A  独立100图测试预测力", xlabel="20 ms窗口中心（ms）", ylabel="Pearson r")
    axes[1].plot(time_center, pair_distance.mean(1), "-o", lw=2, color="#D95319")
    axes[1].fill_between(time_center, np.percentile(pair_distance, 25, axis=1),
                        np.percentile(pair_distance, 75, axis=1), color="#D95319", alpha=.18)
    axes[1].set(title="B  22,248图五折空间误差", xlabel="20 ms窗口中心（ms）",
                ylabel="折间高斯中心距离（视觉角度°）")
    axes[2].plot(transition_center, temporal.mean(1), "-o", lw=2, color="#2E8B57", label="相邻窗口表观位移")
    axes[2].plot(transition_center, jitter.mean(1), "-s", lw=2, color="#7E57C2", label="两窗折间拟合抖动")
    axes[2].set(title="C  位移是否超过拟合不确定性", xlabel="相邻窗口交界（ms）",
                ylabel="中心距离（视觉角度°）")
    axes[2].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.22)
    fig.suptitle("TVSD V4：共享高斯fwRF模型形式（50通道，22,248训练图，20 ms独立拟合）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "TVSD_V4_动态Where_质量与空间稳定性.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    summary = {
        "n_train_images": n, "n_independent_test_images": 100, "n_channels": nu,
        "n_folds": FOLDS, "windows_ms": WINDOWS.tolist(),
        "mean_test_r_by_window": np.nanmean(score, 1).tolist(),
        "mean_crossfold_center_distance_deg": pair_distance.mean(1).tolist(),
        "mean_adjacent_apparent_shift_deg": temporal.mean(1).tolist(),
        "mean_adjacent_fit_jitter_deg": jitter.mean(1).tolist(),
        "fraction_channel_transitions_shift_gt_jitter": float(np.mean(temporal > jitter)),
        "model_form": "one shared isotropic Gaussian; 15x15 centers x10 radii; signed channel weights; independent 20-ms fits",
        "scalable_solver": "response-blind PCA8/stage plus fold-wise marginal signed regression energy",
        "guardrail": "not a bitwise reproduction of the original exhaustive 20-epoch SGD optimizer",
        "test_set": "same independent repeated 100 images for all fold-fitted models",
        "channel_selection": "top 25 oracle V4 channels per monkey inside the existing response cache; never uses model prediction",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "channel_metadata.json").write_text(json.dumps(channel_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
