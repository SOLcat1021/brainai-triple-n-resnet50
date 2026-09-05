"""Frozen Triple-N validation of the TVSD dynamic-Where discovery.

The ResNet PCA basis, 2,250 Gaussian bank, 20-ms windows, reliability rule and
solver are inherited from TVSD.  V2 is added and IT is split into posterior,
middle and anterior subdivisions.  Every time window is fitted independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_tvsd_fwrf_where_pilot as base
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack


ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "spatial_pca8_14_tvsd_frozen" / "maps_f16.npy"
POP = ROOT / "cache" / "trin" / "population_time20" / "population_20ms.npz"
OUT = ROOT / "ResNet50_原版fwRF_20ms时序网络剖析_2026-08-18" / "闭环科学发现_TripleN冻结验证"
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
ROI_CN = {"V1": "V1", "V2": "V2", "V4": "V4", "posterior IT": "后IT",
          "middle IT": "中IT", "anterior IT": "前IT"}
WINDOWS = np.asarray([(30 + 20 * i, 49 + 20 * i) for i in range(8)], int)
FOLDS = 5
SEED = 20260818 + 901


def corr_columns(a, b):
    a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 0)


def fixed_folds(n):
    order = np.random.default_rng(SEED).permutation(n)
    chunks = np.array_split(order, FOLDS)
    return [(np.concatenate([chunks[j] for j in range(FOLDS) if j != k]), chunks[k]) for k in range(FOLDS)]


def select_batch(fields, ftest, stats, unit_start, unit_stop, nw, static):
    mean_f, mean_y_all, cov_fy_all, cov_ff, var_y_all = base.centered(stats)
    o0, o1 = unit_start * nw, unit_stop * nw
    mean_y = mean_y_all[o0:o1]; cov_fy = cov_fy_all[:, :, o0:o1]; var_y = var_y_all[o0:o1]
    var_x = torch.einsum("gp,dpq,gq->gd", fields, cov_ff, fields).clamp_min(1e-8)
    numerator = torch.einsum("gp,dpv->gdv", fields, cov_fy)
    energy = (numerator.square() / (var_x[:, :, None] * var_y[None, None, :])).sum(1)
    ub = unit_stop - unit_start
    if static:
        uc = energy.reshape(len(fields), ub, nw).sum(2).argmax(0)
        chosen = uc[:, None].expand(-1, nw).reshape(-1)
    else:
        chosen = energy.argmax(0)
    pick = torch.arange(ub * nw, device=ftest.device)
    vx = var_x[chosen]; num = numerator[chosen, :, pick]
    weight = num / (vx + .05 * vx.mean(1, keepdim=True))
    chosen_fields = fields[chosen]
    xt = torch.einsum("ndp,vp->nvd", ftest, chosen_fields)
    mean_x = torch.einsum("dp,vp->vd", mean_f, chosen_fields)
    pred = torch.einsum("nvd,vd->nv", xt - mean_x[None], weight) + mean_y[None]
    return chosen.reshape(ub, nw).T.cpu().numpy(), pred.reshape(len(ftest), ub, nw).permute(0, 2, 1).cpu().numpy()


def run_roi(roi, f, fields, pop, xs, ys, sigmas):
    unit_global = np.flatnonzero(pop["area"].astype(str) == roi)
    y_np = pop["responses"][1:9, :, unit_global].astype(np.float32)  # window,image,unit
    nw, n, nu = y_np.shape
    y_flat = y_np.transpose(1, 2, 0).reshape(n, nu * nw)
    y = torch.as_tensor(y_flat, device=f.device)
    folds = fixed_folds(n)
    parts = []
    for _, test in folds:
        parts.append(base.stats_for_indices(f, y, torch.as_tensor(test, device=f.device)))
    total = base.add_many(parts)
    pred_dynamic = np.empty((n, nw, nu), np.float32)
    pred_static = np.empty_like(pred_dynamic)
    fold_candidates = np.empty((FOLDS, nw, nu), np.int32)
    for k, (_, test) in enumerate(folds):
        stats = base.subtract(total, parts[k])
        ftest = f[torch.as_tensor(test, device=f.device)]
        for u0 in range(0, nu, 128):
            u1 = min(nu, u0 + 128)
            dc, dp = select_batch(fields, ftest, stats, u0, u1, nw, static=False)
            _, sp = select_batch(fields, ftest, stats, u0, u1, nw, static=True)
            fold_candidates[k, :, u0:u1] = dc
            pred_dynamic[test, :, u0:u1] = dp
            pred_static[test, :, u0:u1] = sp
        print(f"{roi}: outer fold {k + 1}/{FOLDS}", flush=True)

    full_candidates = np.empty((nw, nu), np.int32)
    for u0 in range(0, nu, 128):
        u1 = min(nu, u0 + 128)
        dc, _ = select_batch(fields, f[:1], total, u0, u1, nw, static=False)
        full_candidates[:, u0:u1] = dc
    r_dynamic = np.stack([corr_columns(pred_dynamic[:, wi], y_np[wi]) for wi in range(nw)])
    r_static = np.stack([corr_columns(pred_static[:, wi], y_np[wi]) for wi in range(nw)])

    cx, cy, sg = xs[full_candidates], ys[full_candidates], sigmas[full_candidates]
    fcx, fcy = xs[fold_candidates], ys[fold_candidates]
    fold_disp = np.mean(np.hypot(fcx - cx[None], fcy - cy[None]), axis=0)
    shift = np.hypot(np.diff(cx, axis=0), np.diff(cy, axis=0))
    jitter = .5 * (fold_disp[:-1] + fold_disp[1:])
    radius_change = np.abs(np.diff(np.log(sg), axis=0))
    return {
        "unit_global": unit_global, "responses": y_np, "pred_dynamic": pred_dynamic,
        "pred_static": pred_static, "r_dynamic": r_dynamic, "r_static": r_static,
        "fold_candidates": fold_candidates, "full_candidates": full_candidates,
        "cx": cx, "cy": cy, "sg": sg, "fold_disp": fold_disp,
        "shift": shift, "jitter": jitter, "radius_change": radius_change,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    maps = torch.as_tensor(np.asarray(np.load(MAPS, mmap_mode="r"), np.float32), device=device)
    f = maps.flatten(2)
    pop = np.load(POP, allow_pickle=True)
    xs, ys, sigmas = candidates()
    fields = torch.as_tensor(gaussian_mass_stack(xs, ys, sigmas, maps.shape[-1]).reshape(len(xs), -1), device=device)
    results, window_rows, transition_rows = {}, [], []
    nw = len(WINDOWS)
    for roi in ROIS:
        result = run_roi(roi, f, fields, pop, xs, ys, sigmas)
        results[roi] = result
        for u, global_index in enumerate(result["unit_global"]):
            base_row = {"roi": roi, "unit_global": int(global_index),
                        "session": int(pop["session"][global_index]), "monkey": str(pop["monkey"][global_index]),
                        "unit_index_zero_based": int(pop["unit_index_zero_based"][global_index]),
                        "release_independent_reliability": float(pop["reliability"][global_index])}
            for wi, (start, end) in enumerate(WINDOWS):
                window_rows.append({**base_row, "window_index": wi, "window_start_ms": int(start),
                                    "window_end_ms": int(end), "dynamic_oof_r": float(result["r_dynamic"][wi, u]),
                                    "static_oof_r": float(result["r_static"][wi, u]),
                                    "delta_r": float(result["r_dynamic"][wi, u] - result["r_static"][wi, u]),
                                    "center_x_deg": float(result["cx"][wi, u]), "center_y_deg": float(result["cy"][wi, u]),
                                    "radius_deg": float(result["sg"][wi, u]),
                                    "fold_center_dispersion_deg": float(result["fold_disp"][wi, u]),
                                    "reliable_window": bool(result["r_dynamic"][wi, u] >= .30)})
            for ti in range(nw - 1):
                reliable_pair = bool(result["r_dynamic"][ti, u] >= .30 and result["r_dynamic"][ti + 1, u] >= .30)
                transition_rows.append({**base_row, "transition_index": ti,
                                        "boundary_ms": float((WINDOWS[ti].mean() + WINDOWS[ti + 1].mean()) / 2),
                                        "center_shift_deg": float(result["shift"][ti, u]),
                                        "fit_jitter_deg": float(result["jitter"][ti, u]),
                                        "shift_gt_jitter": bool(result["shift"][ti, u] > result["jitter"][ti, u]),
                                        "radius_abs_log_change": float(result["radius_change"][ti, u]),
                                        "reliable_pair": reliable_pair})
        np.savez_compressed(OUT / f"{roi.replace(' ', '_')}_核心数组.npz", **result,
                            centers_x=xs, centers_y=ys, sigmas=sigmas, windows=WINDOWS)

    window = pd.DataFrame(window_rows); transition = pd.DataFrame(transition_rows)
    window.to_csv(OUT / "TripleN逐SU逐窗_预测与感受野.csv", index=False, encoding="utf-8-sig")
    transition.to_csv(OUT / "TripleN逐SU相邻窗_感受野变化.csv", index=False, encoding="utf-8-sig")
    session = transition[transition.reliable_pair].groupby(["roi", "monkey", "session"], as_index=False).agg(
        n_transitions=("center_shift_deg", "size"), median_shift=("center_shift_deg", "median"),
        median_jitter=("fit_jitter_deg", "median"), pass_fraction=("shift_gt_jitter", "mean"))
    session.to_csv(OUT / "TripleN会话级验证汇总.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    positions = np.arange(len(ROIS))
    for i, roi in enumerate(ROIS):
        sub = transition[(transition.roi == roi) & transition.reliable_pair]
        vals = sub.center_shift_deg.to_numpy(); jit = sub.fit_jitter_deg.to_numpy()
        axes[0].boxplot(vals, positions=[i], widths=.55, showfliers=False,
                        patch_artist=True, boxprops={"facecolor": "#4C78A8", "alpha": .55})
        axes[0].scatter(i, np.nanmedian(jit), marker="_", s=220, linewidths=3, color="#D45050")
        sess = session[session.roi == roi]
        axes[1].scatter(np.full(len(sess), i) + np.linspace(-.16, .16, max(len(sess), 1)), sess.pass_fraction,
                        s=25, alpha=.55, color="#4C78A8")
        axes[1].scatter(i, sess.pass_fraction.mean(), marker="D", s=75, color="#D45050")
    for ax in axes:
        ax.set_xticks(positions, [ROI_CN[x] for x in ROIS]); ax.grid(alpha=.2)
    axes[0].set(ylabel="相邻20 ms窗圆心位移（°）", title="A  SU位移分布（红横线=折间抖动中位数）")
    axes[1].set(ylabel="位移超过折间抖动的SU转移比例", title="B  每个session均独立显示", ylim=(-.03, 1.03))
    fig.suptitle("Triple-N冻结验证：空间池化场的时间重构是否沿腹侧层级增强", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "图1_TripleN六脑区_动态Where冻结验证.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4.2), sharey=True)
    for ax, roi in zip(axes, ROIS):
        sub = window[window.roi == roi]
        unit = sub.groupby("unit_global").agg(dynamic=("dynamic_oof_r", "mean"), static=("static_oof_r", "mean"))
        ax.scatter(unit.static, unit.dynamic, s=8, alpha=.22, color="#4C78A8", edgecolors="none")
        ax.plot([-.1, .8], [-.1, .8], color="#E76F51", lw=1.2)
        ax.set(xlim=(-.1, .8), ylim=(-.1, .8), title=ROI_CN[roi], xlabel="静态场 OOF r")
        ax.text(.04, .94, f"平均Δr={(unit.dynamic-unit.static).mean():+.3f}\nn={len(unit)}",
                transform=ax.transAxes, va="top", fontsize=8.5)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("独立时间窗动态场 OOF r")
    fig.suptitle("Triple-N控制：动态空间场是否增加未见图像预测力", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "图2_TripleN动态场相对静态场_OOF预测.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    audit = {
        "discovery_to_validation_freeze": "ResNet50, TVSD-frozen response-blind PCA8/stage, 2250 Gaussian bank, eight independent 20-ms windows",
        "validation_data": "Triple-N SU, released independent reliability >=0.4, V1/V2/V4/posterior/middle/anterior IT",
        "prediction": "five-fold out-of-fold over 1000 images",
        "temporary_quality_gate": "OOF dynamic r >=0.30 in both adjacent windows; exact per-window six-repeat ceilings are audited separately",
        "uncertainty": "distance from each full-data center to five fold-wise centers",
        "inference_guardrail": "session and animal summaries; unit rows are descriptive",
    }
    (OUT / "冻结验证审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
