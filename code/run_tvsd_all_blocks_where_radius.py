"""Final-spec TVSD shared-Gaussian analysis of center and radius dynamics.

Scope: every channel in the 12 representative physical blocks already cached
for TVSD (two monkeys x V1/V4/IT x two blocks).  No channel is selected using
model prediction.  All eight 20-ms windows are fitted independently.

The model form and scalable solver are identical to the preceding 50-channel
pilot.  The inferential/replication unit is the physical recording block or
animal, never the individual MUA channel.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_tvsd_fwrf_where_pilot as base
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ResNet50_原版fwRF_20ms时序网络剖析_2026-08-18" / "TVSD_全部代表性物理块_Where与半径"
ROIS = ("V1", "V4", "IT")
COLORS = {"N": "#2878B5", "F": "#D95319"}


def load_all_targets():
    train, test, rows = [], [], []
    offset = 0
    for monkey in base.MONKEYS:
        c = np.load(base.TARGETS / f"monkey{monkey}_time20_mua_pilot_channels.npz")
        t = np.load(base.TARGETS / f"monkey{monkey}_targets.npz")
        raw = c["selected_raw_channel"].astype(int)
        mapped = c["selected_mapped_channel"].astype(int)
        roi = c["roi"].astype(str)
        ytr = np.asarray(np.load(base.TARGETS / f"monkey{monkey}_time20_mua_pilot_f16.npy", mmap_mode="r"), np.float32)
        yte = np.asarray(np.load(base.TARGETS / f"monkey{monkey}_test_time20_mua_pilot_f16.npy", mmap_mode="r"), np.float32)
        train.append(ytr); test.append(yte)
        for j in range(len(raw)):
            rows.append({
                "channel_index": offset + j, "monkey": monkey, "roi": roi[j],
                "pilot_index": j, "mapped_channel": int(mapped[j]), "raw_channel": int(raw[j]),
                "physical_block": int(raw[j] // 26), "oracle": float(t["oracle"][mapped[j]]),
            })
        offset += len(raw)
    return np.concatenate(train, 1), np.concatenate(test, 1), pd.DataFrame(rows)


def pairwise_fold_metrics(candidates_fold, xs, ys, sigmas):
    folds, nw, nu = candidates_fold.shape
    cx, cy, sg = xs[candidates_fold], ys[candidates_fold], sigmas[candidates_fold]
    logsg = np.log(sg)
    pairs = list(combinations(range(folds), 2))
    center_error = np.empty((nw, nu), np.float32)
    radius_error = np.empty((nw, nu), np.float32)
    for wi in range(nw):
        for u in range(nu):
            center_error[wi, u] = np.mean([
                np.hypot(cx[a, wi, u] - cx[b, wi, u], cy[a, wi, u] - cy[b, wi, u]) for a, b in pairs])
            radius_error[wi, u] = np.mean([abs(logsg[a, wi, u] - logsg[b, wi, u]) for a, b in pairs])
    center_shift = np.mean(np.hypot(np.diff(cx, axis=1), np.diff(cy, axis=1)), axis=0)
    radius_abs_change = np.mean(np.abs(np.diff(logsg, axis=1)), axis=0)
    radius_signed_change = np.mean(np.diff(logsg, axis=1), axis=0)
    center_jitter = 0.5 * (center_error[:-1] + center_error[1:])
    radius_jitter = 0.5 * (radius_error[:-1] + radius_error[1:])
    return {
        "cx": cx, "cy": cy, "sg": sg,
        "center_error": center_error, "radius_error": radius_error,
        "center_shift": center_shift, "radius_abs_change": radius_abs_change,
        "radius_signed_change": radius_signed_change,
        "center_jitter": center_jitter, "radius_jitter": radius_jitter,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    maps_train = torch.as_tensor(np.asarray(np.load(base.FEATURES / "train_maps_f16.npy", mmap_mode="r"), np.float32), device=device)
    maps_test = torch.as_tensor(np.asarray(np.load(base.FEATURES / "test_maps_f16.npy", mmap_mode="r"), np.float32), device=device)
    n, _, h, _ = maps_train.shape
    f, ft = maps_train.flatten(2), maps_test.flatten(2)
    ytr_np, yte_np, meta = load_all_targets()
    nu, nw = ytr_np.shape[1], ytr_np.shape[2]
    assert nu == len(meta) == 270 and nw == 8
    y = torch.as_tensor(ytr_np.reshape(n, nu * nw), device=device)
    xs, ys, sigmas = candidates()
    fields = torch.as_tensor(gaussian_mass_stack(xs, ys, sigmas, h).reshape(len(xs), -1), device=device)

    rng = np.random.default_rng(base.SEED + 500)
    order = rng.permutation(n)
    fold_id = np.empty(n, int)
    for k, idx in enumerate(np.array_split(order, base.FOLDS)):
        fold_id[idx] = k
    parts = []
    for k in range(base.FOLDS):
        idx = torch.as_tensor(np.flatnonzero(fold_id == k), device=device)
        parts.append(base.stats_for_indices(f, y, idx))
        print(f"sufficient statistics partition {k + 1}/{base.FOLDS}", flush=True)
    total = base.add_many(parts)

    fold_candidates = np.empty((base.FOLDS, nw, nu), int)
    fold_predictions = np.empty((base.FOLDS, 100, nw, nu), np.float32)
    fold_scores = np.empty((base.FOLDS, nw, nu), np.float32)
    for k in range(base.FOLDS):
        mean_f, mean_y, cov_fy, cov_ff, var_y = base.centered(base.subtract(total, parts[k]))
        var_x = torch.einsum("gp,dpq,gq->gd", fields, cov_ff, fields).clamp_min(1e-8)
        numerator = torch.einsum("gp,dpv->gdv", fields, cov_fy)
        energy = (numerator.square() / (var_x[:, :, None] * var_y[None, None, :])).sum(1)
        selected = energy.argmax(0)
        fold_candidates[k] = selected.reshape(nu, nw).T.cpu().numpy()
        pick = torch.arange(nu * nw, device=device)
        vx = var_x[selected]; num = numerator[selected, :, pick]
        weight = num / (vx + 0.05 * vx.mean(1, keepdim=True))
        selected_fields = fields[selected]
        xt = torch.einsum("ndp,vp->nvd", ft, selected_fields)
        mean_x = torch.einsum("dp,vp->vd", mean_f, selected_fields)
        prediction = torch.einsum("nvd,vd->nv", xt - mean_x[None], weight) + mean_y[None]
        pred_np = prediction.cpu().numpy().reshape(100, nu, nw).transpose(0, 2, 1)
        fold_predictions[k] = pred_np
        fold_scores[k] = np.stack([base.corr_columns(pred_np[:, wi], yte_np[:, :, wi]) for wi in range(nw)])
        print(f"fold {k + 1}/{base.FOLDS} selected and predicted", flush=True)
        del numerator, energy, prediction, xt
        torch.cuda.empty_cache()

    metric = pairwise_fold_metrics(fold_candidates, xs, ys, sigmas)
    score = np.nanmean(fold_scores, axis=0)
    time_center = base.WINDOWS.mean(1)
    transition_center = 0.5 * (time_center[:-1] + time_center[1:])

    # Unit-window and unit-transition tables.
    window_rows, transition_rows = [], []
    for u, row in meta.iterrows():
        for wi, (start, end) in enumerate(base.WINDOWS):
            window_rows.append({
                **row.to_dict(), "window_index": wi, "window_start_ms": int(start), "window_end_ms": int(end),
                "test_r": float(score[wi, u]),
                "center_x_deg": float(metric["cx"][:, wi, u].mean()),
                "center_y_deg": float(metric["cy"][:, wi, u].mean()),
                "radius_geom_mean_deg": float(np.exp(np.log(metric["sg"][:, wi, u]).mean())),
                "center_fold_error_deg": float(metric["center_error"][wi, u]),
                "radius_fold_error_abs_log": float(metric["radius_error"][wi, u]),
            })
        for ti, boundary in enumerate(transition_center):
            center_shift = float(metric["center_shift"][ti, u]); center_jitter = float(metric["center_jitter"][ti, u])
            radius_change = float(metric["radius_abs_change"][ti, u]); radius_jitter = float(metric["radius_jitter"][ti, u])
            signed = float(metric["radius_signed_change"][ti, u])
            transition_rows.append({
                **row.to_dict(), "transition_index": ti, "boundary_ms": float(boundary),
                "center_shift_deg": center_shift, "center_jitter_deg": center_jitter,
                "center_shift_gt_jitter": center_shift > center_jitter,
                "radius_abs_log_change": radius_change, "radius_jitter_abs_log": radius_jitter,
                "radius_change_gt_jitter": radius_change > radius_jitter,
                "radius_signed_log_change": signed, "radius_ratio": float(np.exp(signed)),
            })
    window_df, transition_df = pd.DataFrame(window_rows), pd.DataFrame(transition_rows)
    window_df.to_csv(OUT / "逐通道逐窗_预测与圆心半径.csv", index=False, encoding="utf-8-sig")
    transition_df.to_csv(OUT / "逐通道相邻窗_圆心与半径变化.csv", index=False, encoding="utf-8-sig")

    # Physical block is the replication unit.
    block_window = window_df.groupby(["monkey", "roi", "physical_block", "window_index"], as_index=False).agg(
        n_channels=("channel_index", "size"), mean_r=("test_r", "mean"),
        center_error_deg=("center_fold_error_deg", "mean"),
        radius_deg=("radius_geom_mean_deg", "median"),
        radius_error_log=("radius_fold_error_abs_log", "mean"),
    )
    block_transition = transition_df.groupby(["monkey", "roi", "physical_block", "transition_index"], as_index=False).agg(
        n_channels=("channel_index", "size"), center_shift_deg=("center_shift_deg", "mean"),
        center_jitter_deg=("center_jitter_deg", "mean"),
        center_pass=("center_shift_gt_jitter", "mean"),
        radius_abs_change=("radius_abs_log_change", "mean"),
        radius_jitter=("radius_jitter_abs_log", "mean"),
        radius_pass=("radius_change_gt_jitter", "mean"),
        radius_signed_change=("radius_signed_log_change", "mean"),
    )
    block_window.to_csv(OUT / "物理块逐窗汇总.csv", index=False, encoding="utf-8-sig")
    block_transition.to_csv(OUT / "物理块相邻窗汇总.csv", index=False, encoding="utf-8-sig")

    # Figure 1: prediction and center identifiability, separately by monkey.
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), sharex=True)
    for ci, roi in enumerate(ROIS):
        for monkey in base.MONKEYS:
            sub = block_window[(block_window.roi == roi) & (block_window.monkey == monkey)]
            grouped = sub.groupby("window_index")
            for ax, col in ((axes[0, ci], "mean_r"), (axes[1, ci], "center_error_deg")):
                mean = grouped[col].mean().reindex(range(nw)).to_numpy()
                lo = grouped[col].min().reindex(range(nw)).to_numpy()
                hi = grouped[col].max().reindex(range(nw)).to_numpy()
                ax.plot(time_center, mean, "-o", color=COLORS[monkey], lw=2, label=f"猴{monkey}")
                ax.fill_between(time_center, lo, hi, color=COLORS[monkey], alpha=.12)
        axes[0, ci].set_title(f"{roi}：独立100图预测")
        axes[1, ci].set_title(f"{roi}：五折圆心误差")
        axes[0, ci].grid(alpha=.2); axes[1, ci].grid(alpha=.2)
        axes[1, ci].set_xlabel("20 ms窗口中心（ms）")
    axes[0, 0].set_ylabel("平均Pearson r"); axes[1, 0].set_ylabel("视觉角度°")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("TVSD全部代表性物理块：预测力与感受野圆心可辨识度", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "图1_分猴分脑区_预测与圆心稳定性.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Figure 2: center movement fractions with every block visible.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    for ax, roi in zip(axes, ROIS):
        for monkey in base.MONKEYS:
            sub = block_transition[(block_transition.roi == roi) & (block_transition.monkey == monkey)]
            for _, block in sub.groupby("physical_block"):
                ax.plot(transition_center, block.sort_values("transition_index").center_pass,
                        color=COLORS[monkey], alpha=.25, lw=1.3)
            mean = sub.groupby("transition_index").center_pass.mean().reindex(range(nw - 1))
            ax.plot(transition_center, mean, "-o", color=COLORS[monkey], lw=2.5, label=f"猴{monkey}")
        ax.set(title=roi, xlabel="相邻窗口交界（ms）", ylim=(-.03, 1.03))
        ax.axhline(.5, color="0.55", ls="--", lw=1); ax.grid(alpha=.2)
    axes[0].set_ylabel("圆心位移超过折间抖动的通道比例")
    axes[-1].legend(frameon=False)
    fig.suptitle("动态Where：粗线为猴内两块均值，细线为每个物理块", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "图2_圆心动态_分猴分物理块复现.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Figure 3: radius preference and changes.
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), sharex="col")
    for ci, roi in enumerate(ROIS):
        for monkey in base.MONKEYS:
            sw = block_window[(block_window.roi == roi) & (block_window.monkey == monkey)]
            st = block_transition[(block_transition.roi == roi) & (block_transition.monkey == monkey)]
            for _, block in sw.groupby("physical_block"):
                axes[0, ci].plot(time_center, block.sort_values("window_index").radius_deg,
                                 color=COLORS[monkey], alpha=.25, lw=1.3)
            axes[0, ci].plot(time_center, sw.groupby("window_index").radius_deg.mean().reindex(range(nw)),
                             "-o", color=COLORS[monkey], lw=2.5, label=f"猴{monkey}")
            for _, block in st.groupby("physical_block"):
                axes[1, ci].plot(transition_center, block.sort_values("transition_index").radius_pass,
                                 color=COLORS[monkey], alpha=.25, lw=1.3)
            axes[1, ci].plot(transition_center, st.groupby("transition_index").radius_pass.mean().reindex(range(nw - 1)),
                             "-o", color=COLORS[monkey], lw=2.5)
        axes[0, ci].set_title(f"{roi}：高斯半径偏好")
        axes[1, ci].set_title(f"{roi}：半径变化超过抖动")
        axes[1, ci].set_xlabel("时间（上：窗口中心；下：窗口交界，ms）")
        axes[0, ci].grid(alpha=.2); axes[1, ci].grid(alpha=.2); axes[1, ci].set_ylim(-.03, 1.03)
    axes[0, 0].set_ylabel("物理块通道中位半径（°）")
    axes[1, 0].set_ylabel("通道比例")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("感受野半径：偏好大小及其时间变化可信度", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "图3_半径偏好与动态_分猴分物理块.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Compact block-level audit table.
    block_rows = []
    for (monkey, roi, physical_block), sw in block_window.groupby(["monkey", "roi", "physical_block"]):
        st = block_transition[(block_transition.monkey == monkey) & (block_transition.roi == roi) &
                              (block_transition.physical_block == physical_block)]
        early = st[st.transition_index <= 1]; late = st[st.transition_index >= 2]
        block_rows.append({
            "monkey": monkey, "roi": roi, "physical_block": int(physical_block),
            "n_channels": int(sw.n_channels.iloc[0]), "mean_test_r": float(sw.mean_r.mean()),
            "mean_center_error_deg": float(sw.center_error_deg.mean()),
            "early_center_pass": float(early.center_pass.mean()), "late_center_pass": float(late.center_pass.mean()),
            "median_radius_deg": float(sw.radius_deg.median()), "mean_radius_error_log": float(sw.radius_error_log.mean()),
            "radius_pass": float(st.radius_pass.mean()), "signed_radius_change_log": float(st.radius_signed_change.mean()),
        })
    block_df = pd.DataFrame(block_rows).sort_values(["roi", "monkey", "physical_block"])
    block_df.to_csv(OUT / "12物理块总表.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        OUT / "all_blocks_where_radius.npz", candidates=fold_candidates, predictions=fold_predictions,
        scores=fold_scores, responses=yte_np.transpose(0, 2, 1), windows=base.WINDOWS,
        centers_x=xs, centers_y=ys, sigmas=sigmas,
        center_error=metric["center_error"], radius_error=metric["radius_error"],
        center_shift=metric["center_shift"], center_jitter=metric["center_jitter"],
        radius_abs_change=metric["radius_abs_change"], radius_signed_change=metric["radius_signed_change"],
        radius_jitter=metric["radius_jitter"],
    )
    audit = {
        "n_train_images": n, "n_test_images": 100, "n_channels": nu,
        "n_physical_blocks": int(block_df.shape[0]),
        "channel_counts_by_monkey": meta.groupby("monkey").size().to_dict(),
        "channel_counts_by_roi": meta.groupby("roi").size().to_dict(),
        "block_counts_by_monkey_roi": meta.groupby(["monkey", "roi"]).physical_block.nunique().to_dict(),
        "selection": "all channels in the pre-existing two representative physical blocks per monkey and ROI",
        "inference_unit": "physical block / monkey; channel summaries are descriptive",
        "model_form": "one shared isotropic Gaussian; 2250 candidates; signed weights; independent 20-ms windows",
        "solver_guardrail": "response-blind PCA8 per ResNet stage plus marginal signed regression; not bitwise original SGD",
    }
    # JSON cannot serialize tuple dict keys.
    audit["block_counts_by_monkey_roi"] = {f"{k[0]}-{k[1]}": int(v) for k, v in audit["block_counts_by_monkey_roi"].items()}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
