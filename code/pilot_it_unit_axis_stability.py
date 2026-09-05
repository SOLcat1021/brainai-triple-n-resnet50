"""Small IT audit of unit-axis stability within the reliable time range.

Axes are the signed 64-D channel readouts at a fixed representative ResNet
layer.  Temporal displacement is compared with same-window OOF-fold variation
so independent-fit instability is not mistaken for physiological rotation.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_consistency_ge_0p4_all_units_2026-08-30"
PILOT = PROJECT / "results" / "shared_temporal_trajectory_it_pilot_2026-09-03"
OUT = PROJECT / "results" / "it_unit_axis_stability_pilot_2026-09-03"
ROIS = ("posterior IT", "middle IT", "anterior IT")
ROI_FILE = {
    "posterior IT": "posterior_it_proxy_bank_64d.h5",
    "middle IT": "middle_it_proxy_bank_64d.h5",
    "anterior IT": "anterior_it_proxy_bank_64d.h5",
}
WINDOW_START, WINDOW_END = 50, 169
REFERENCE_START = 150
RANK = 64


def unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.full(v.shape, np.nan)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = unit_vector(a); b = unit_vector(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    return float(np.clip(a @ b, -1.0, 1.0))


def angle(c: float) -> float:
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))) if np.isfinite(c) else np.nan


def read_roi(roi: str, selected: pd.DataFrame):
    with h5py.File(BANK / ROI_FILE[roi], "r") as h:
        all_units = h["unit_global"][:].astype(int)
        wanted = selected[selected.roi.eq(roi)].unit_global.to_numpy(int)
        local = np.asarray([np.flatnonzero(all_units == u)[0] for u in wanted], dtype=int)
        order = np.argsort(local)
        inv = np.argsort(order)
        cw = h["channel_weights"][:, local[order], :][:, inv, :].astype(np.float32)
        fw = h["fold_channel_weights"][:, :, local[order], :][:, :, inv, :].astype(np.float32)
        gate = h["layer_gate"][:, local[order], :][:, inv, :].astype(np.float32)
        windows = h["windows_ms"][:].astype(int)
        fold_std = h["fold_normalization_std"][:].astype(np.float32)
    use = np.flatnonzero((windows[:, 0] >= WINDOW_START) & (windows[:, 1] <= WINDOW_END))
    ref = int(np.flatnonzero(windows[:, 0] == REFERENCE_START)[0])
    return wanted, cw, fw, gate, fold_std, windows, use, ref


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(PILOT / "selected_units.csv")
    full_std = np.load(BANK / "feature_normalization_full.npz")["std"].astype(np.float32)
    rows = []
    window_rows = []
    fold_rows = []

    for roi in ROIS:
        wanted, cw, fw, gate, fold_std, windows, use, ref = read_roi(roi, selected)
        for ui, unit in enumerate(wanted):
            # Fix the representative layer at the reference window. This
            # isolates channel-axis changes from layer-gate switching.
            layer = int(np.argmax(gate[ref, ui]))
            final_axes = []
            for wi in use:
                vec = cw[wi, ui, layer * RANK:(layer + 1) * RANK] / full_std[layer]
                final_axes.append(unit_vector(vec))
                c = cosine(final_axes[-1], cw[ref, ui, layer * RANK:(layer + 1) * RANK] / full_std[layer])
                window_rows.append({
                    "roi": roi, "unit_global": int(unit), "layer_index": layer,
                    "window_start_ms": int(windows[wi, 0]), "window_end_ms": int(windows[wi, 1]),
                    "cosine_to_reference": c, "angle_to_reference_degrees": angle(c),
                })
            final_axes = np.asarray(final_axes)
            pair_cos = [cosine(final_axes[a], final_axes[b]) for a, b in combinations(range(len(use)), 2)]
            temporal_angles = [angle(cosine(final_axes[a], final_axes[b])) for a, b in combinations(range(len(use)), 2)]

            # Fold vectors are corrected back to raw PCA coordinates because
            # each fold used its own feature standard deviation.
            fold_axes = np.empty((len(use), fw.shape[0], RANK), np.float32)
            for ti, wi in enumerate(use):
                for fi in range(fw.shape[0]):
                    v = fw[fi, wi, ui, layer * RANK:(layer + 1) * RANK] / fold_std[fi, layer]
                    fold_axes[ti, fi] = unit_vector(v)
                    fold_rows.append({
                        "roi": roi, "unit_global": int(unit), "layer_index": layer,
                        "window_start_ms": int(windows[wi, 0]), "fold": fi,
                    })
            fold_angles = []
            fold_cos = []
            for ti in range(len(use)):
                for a, b in combinations(range(fold_axes.shape[1]), 2):
                    c = cosine(fold_axes[ti, a], fold_axes[ti, b])
                    fold_cos.append(c); fold_angles.append(angle(c))
            rows.append({
                "roi": roi, "unit_global": int(unit), "fixed_layer_index": layer,
                "fixed_layer_reference": "argmax final layer_gate at 150-159 ms",
                "n_windows": len(use), "temporal_pairwise_cosine_median": float(np.nanmedian(pair_cos)),
                "temporal_pairwise_angle_median_degrees": float(np.nanmedian(temporal_angles)),
                "temporal_reference_angle_median_degrees": float(np.nanmedian([
                    angle(cosine(v, final_axes[int(np.flatnonzero(use == ref)[0])])) for v in final_axes
                ])),
                "fold_same_window_cosine_median": float(np.nanmedian(fold_cos)),
                "fold_same_window_angle_median_degrees": float(np.nanmedian(fold_angles)),
                "excess_temporal_angle_over_fold_baseline_degrees": float(
                    np.nanmedian(temporal_angles) - np.nanmedian(fold_angles)),
            })

    unit_table = pd.DataFrame(rows)
    unit_table.to_csv(OUT / "unit_axis_stability.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(window_rows).to_csv(OUT / "unit_axis_by_window.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_axis_manifest.csv", index=False, encoding="utf-8-sig")
    roi = unit_table.groupby("roi", as_index=False).agg(
        n_units=("unit_global", "nunique"),
        temporal_angle_median=("temporal_pairwise_angle_median_degrees", "median"),
        fold_angle_median=("fold_same_window_angle_median_degrees", "median"),
        excess_angle_median=("excess_temporal_angle_over_fold_baseline_degrees", "median"),
        temporal_cosine_median=("temporal_pairwise_cosine_median", "median"),
        fold_cosine_median=("fold_same_window_cosine_median", "median"),
    )
    roi.to_csv(OUT / "roi_summary.csv", index=False, encoding="utf-8-sig")

    # Plot temporal cosine to reference for each unit and compare angle to the
    # same-window fold baseline at the unit level.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    q = pd.DataFrame(window_rows)
    for _, z in q.groupby(["roi", "unit_global"]):
        label = f"{z.roi.iloc[0]} {int(z.unit_global.iloc[0])}"
        axes[0].plot(z.window_start_ms, z.cosine_to_reference, marker="o", ms=3, lw=1.2, label=label)
    axes[0].axhline(0, color="0.5", lw=.7); axes[0].set(
        title="Unit-axis cosine to 150–159 ms reference",
        xlabel="Window start (ms)", ylabel="Cosine similarity (raw PCA64 coordinates)")
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=7, ncol=2)
    pos = np.arange(len(unit_table)); colors = ["#4C78A8" if r == "posterior IT" else "#54A24B" if r == "middle IT" else "#E45756" for r in unit_table.roi]
    axes[1].bar(pos, unit_table.temporal_pairwise_angle_median_degrees, color=colors, alpha=.85, label="Across windows")
    axes[1].scatter(pos, unit_table.fold_same_window_angle_median_degrees, color="black", s=28, label="Across OOF folds")
    axes[1].set_xticks(pos, [f"{r}\n{u}" for r, u in zip(unit_table.roi, unit_table.unit_global)], rotation=20, ha="right")
    axes[1].set(title="Observed temporal angle vs fit-instability baseline", xlabel="Unit", ylabel="Median angle (degrees)")
    axes[1].grid(axis="y", alpha=.2); axes[1].legend(fontsize=8)
    fig.savefig(OUT / "01_axis_stability_vs_fold_baseline.png", dpi=300)
    plt.close(fig)

    report = [
        "# IT unit 轴稳定性小规模探索", "",
        "- 样本：后/中/前 IT 各 2 个 unit，均来自官方 released independent consistency ≥ 0.4 池。",
        "- 时间：50–169 ms 的 12 个共同 10 ms 窗口；参考轴固定为 150–159 ms。",
        "- 层：每个 unit 固定参考窗口的最大 layer-gate，避免把层切换误当作通道轴旋转。",
        "- 轴：64D signed channel readout，转换回未标准化 PCA 坐标后归一化。",
        "- 基线：同一窗口 5 个 OOF fold 的轴夹角，衡量独立拟合造成的方向不确定性。",
        "",
        "## 结论边界", "",
        "如果跨窗口夹角明显高于跨折夹角，才有理由进一步检验真实时间轴变化；若两者接近，当前证据只能说明独立窗读出不稳定。",
        "本结果只做探索，不把轴的几何位移直接解释为生理表征旋转。",
    ]
    (OUT / "REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")
    audit = {
        "n_units": int(len(unit_table)), "windows_ms": windows[use].tolist(),
        "reference_window_ms": [150, 159], "fixed_layer": "argmax final layer_gate at reference",
        "axis_coordinate": "raw PCA64 coordinates (divide effective weights by fold/full feature std)",
        "fold_baseline": "same-window pairwise cosine across 5 OOF fits",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(roi.to_string(index=False), flush=True)
    print(unit_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
