"""Test whether image-dependent latency is reliable and visually predictable."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_low_rank_temporal_dynamics as pilot
import validate_low_rank_temporal_dynamics as validation


OUT = pilot.PROJECT / "results" / "stimulus_dependent_latency_2026-08-27"
ROIS = ("posterior IT", "middle IT", "anterior IT")
SHIFTS = np.arange(-5, 6)
SEED = 20260827 + 313


def shift_bank(template: np.ndarray) -> np.ndarray:
    return np.stack([validation.shift_vector(template, int(s)) for s in SHIFTS], axis=1)


def infer_shift(curves: np.ndarray, mean: np.ndarray, template: np.ndarray) -> np.ndarray:
    score = (curves - mean[None]) @ shift_bank(template)
    return SHIFTS[np.argmax(np.abs(score), axis=1)]


def derivative_coefficients(curves: np.ndarray, mean: np.ndarray, template: np.ndarray) -> np.ndarray:
    basis = np.column_stack([template, np.gradient(template)])
    q, _ = np.linalg.qr(basis)
    return (curves - mean[None]) @ q[:, :2]


def analyze() -> pd.DataFrame:
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    tidx = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    configs = validation.modal_configs().set_index("roi")
    order = np.random.default_rng(validation.SEED).permutation(pilot.N_IMAGES)
    halves = (np.sort(order[:500]), np.sort(order[500:]))
    thirds = tuple(np.sort(x) for x in np.array_split(order, 3))
    rows = []
    for roi in ROIS:
        lam = float(configs.loc[roi, "lambda"])
        meta, y, pred = pilot.load_roi(roi, responses, tidx)
        for ui, unit in meta.iterrows():
            bases, means = [], []
            for train in halves:
                mean = y[train, :, ui].mean(0)
                template = pilot.svd_basis(y[train, :, ui] - mean[None], lam, 1)[:, 0]
                bases.append(template); means.append(mean)
            # Reliability without circular reuse: learn two templates on two
            # different thirds and compare their shifts on the untouched third.
            third_bases, third_means = [], []
            for train in thirds:
                mu = y[train, :, ui].mean(0)
                third_means.append(mu)
                third_bases.append(pilot.svd_basis(y[train, :, ui] - mu[None], lam, 1)[:, 0])
            reliability_values = []
            for test_third in range(3):
                train_thirds = [j for j in range(3) if j != test_third]
                a, b = train_thirds
                test = thirds[test_third]
                sa = infer_shift(y[test, :, ui], third_means[a], third_bases[a])
                sb = infer_shift(y[test, :, ui], third_means[b], third_bases[b])
                reliability_values.append(pilot.corr(sa, sb))
            reliability = float(np.nanmean(reliability_values))
            for split, train in enumerate(halves):
                test = halves[1 - split]
                mean, template = means[split], bases[split]
                target, raw = y[test, :, ui], pred[test, :, ui]
                target_shift = infer_shift(target, mean, template)
                predicted_shift = infer_shift(raw, mean, template)
                target_coef = derivative_coefficients(target, mean, template)
                pred_coef = derivative_coefficients(raw, mean, template)
                rows.append({
                    "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                    "monkey": str(unit.monkey), "session": int(unit.session), "split": split,
                    "shift_r": pilot.corr(target_shift, predicted_shift),
                    "shift_mae_ms": float(np.mean(np.abs(target_shift - predicted_shift)) * 10),
                    "shift_within_10ms": float(np.mean(np.abs(target_shift - predicted_shift) <= 1)),
                    "target_nonzero_shift_fraction": float(np.mean(target_shift != 0)),
                    "target_shift_sd_ms": float(np.std(target_shift) * 10),
                    "main_amplitude_r": pilot.corr(target_coef[:, 0], pred_coef[:, 0]),
                    "derivative_coefficient_r": pilot.corr(target_coef[:, 1], pred_coef[:, 1]),
                    "cross_basis_target_shift_reliability": reliability,
                })
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    unit = rows.groupby(["roi", "unit_global", "native_area", "monkey", "session"], as_index=False).mean(numeric_only=True)
    out = []
    for roi in ROIS:
        q = unit[unit.roi.eq(roi)]
        cluster = q.groupby(["monkey", "session"]).derivative_coefficient_r.mean().to_numpy(float)
        out.append({
            "roi": roi, "n_units": len(q),
            "median_shift_r": float(q.shift_r.median()),
            "median_shift_mae_ms": float(q.shift_mae_ms.median()),
            "median_shift_within_10ms": float(q.shift_within_10ms.median()),
            "median_target_nonzero_shift_fraction": float(q.target_nonzero_shift_fraction.median()),
            "median_target_shift_sd_ms": float(q.target_shift_sd_ms.median()),
            "median_cross_basis_shift_reliability": float(q.cross_basis_target_shift_reliability.median()),
            "median_main_amplitude_r": float(q.main_amplitude_r.median()),
            "median_derivative_coefficient_r": float(q.derivative_coefficient_r.median()),
            "derivative_session_signflip_p": validation.signflip_p(cluster, SEED + len(out)),
        })
    return pd.DataFrame(out)


def make_figure(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].bar(x - width / 2, summary.median_shift_r, width, label="discrete shift")
    axes[0].bar(x + width / 2, summary.median_derivative_coefficient_r, width, label="derivative coefficient")
    axes[0].axhline(0, color="#777", lw=1)
    axes[0].set_xticks(x, summary.roi, rotation=25)
    axes[0].set(ylabel="Held-out image prediction r", title="Image-dependent timing is visually predictable")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(x, summary.median_cross_basis_shift_reliability, color="#7A5195")
    axes[1].set_xticks(x, summary.roi, rotation=25)
    axes[1].set(ylabel="Correlation", title="Latency estimates across disjoint basis halves")
    axes[1].grid(axis="y", alpha=.2)
    fig.savefig(OUT / "01_stimulus_dependent_latency.png", dpi=300)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# IT刺激依赖潜伏期：互斥图像验证", "",
        "主时间模板仅由500张训练图像学习。对另外500张图像，分别从真实神经响应和OOF视觉模型预测中估计±50 ms离散时移；"
        "另用主模板的一阶导数系数作为连续的小幅时移指标。", "",
        "| ROI | unit | 时移预测r | MAE(ms) | ±10ms命中 | 非零时移比例 | 时移SD(ms) | 跨basis可靠度 | 主幅度r | 导数系数r | session p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary.itertuples():
        lines.append(f"| {x.roi} | {x.n_units} | {x.median_shift_r:.3f} | {x.median_shift_mae_ms:.1f} | "
                     f"{x.median_shift_within_10ms:.1%} | {x.median_target_nonzero_shift_fraction:.1%} | "
                     f"{x.median_target_shift_sd_ms:.1f} | {x.median_cross_basis_shift_reliability:.3f} | "
                     f"{x.median_main_amplitude_r:.3f} | {x.median_derivative_coefficient_r:.3f} | "
                     f"{x.derivative_session_signflip_p:.5f} |")
    lines += [
        "", "离散时移相关要求模型预测正确的时间bin，较苛刻；导数系数相关检验视觉模型能否预测连续潜伏期方向。",
        "如果导数系数可预测且跨互斥basis的真实时移可靠，说明潜伏期不是纯噪声，而是由图像特征系统性驱动的编码变量。",
        "本分析使用图像条件平均响应，不能判断该时移来自单试次放电潜伏期还是不同试次响应概率的混合。",
    ]
    (OUT / "STIMULUS_DEPENDENT_LATENCY_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = analyze()
    summary = summarize(rows)
    rows.to_csv(OUT / "unit_split_latency_validation.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "latency_summary_by_roi.csv", index=False, encoding="utf-8-sig")
    make_figure(summary)
    write_report(summary)
    (OUT / "audit.json").write_text(json.dumps({"shifts_ms": (SHIFTS * 10).tolist(), "seed": SEED,
                                                   "rois": list(ROIS)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
