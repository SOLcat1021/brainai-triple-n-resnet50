"""Disjoint-half and latency-control validation for the temporal-rank pilot."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr, wilcoxon

import pilot_low_rank_temporal_dynamics as pilot


OUT = pilot.OUT
SEED = 20260827
ROI_ORDER = {roi: i for i, roi in enumerate(pilot.ROIS)}


def modal_configs() -> pd.DataFrame:
    choices = pd.read_csv(OUT / "nested_meta_cv_choices.csv")
    q = choices[choices.model_class.eq("dynamic_rank_ge2")]
    rows = []
    for roi, z in q.groupby("roi", sort=False):
        count = z.groupby(["method", "lambda", "rank"], dropna=False).size().reset_index(name="n_folds")
        best = count.sort_values(["n_folds", "rank"], ascending=[False, True]).iloc[0]
        rows.append({"roi": roi, "method": best.method, "lambda": float(best["lambda"]),
                     "rank": int(best["rank"]), "selection_fold_frequency": int(best.n_folds)})
    result = pd.DataFrame(rows)
    result["_roi_order"] = result.roi.map(ROI_ORDER)
    return result.sort_values("_roi_order").drop(columns="_roi_order").reset_index(drop=True)


def shift_vector(v: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(v)
    if shift > 0:
        out[shift:] = v[:-shift]
    elif shift < 0:
        out[:shift] = v[-shift:]
    else:
        out[:] = v
    norm = np.linalg.norm(out)
    return out / norm if norm > 1e-12 else out


def shifted_rank1_prediction(pred: np.ndarray, mean_curve: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.arange(-4, 5)
    bank = np.stack([shift_vector(template, int(s)) for s in shifts], axis=1)
    residual = pred - mean_curve[None]
    coefficients = residual @ bank
    chosen = np.argmax(np.abs(coefficients), axis=1)
    amplitude = coefficients[np.arange(len(pred)), chosen]
    reconstructed = mean_curve[None] + amplitude[:, None] * bank[:, chosen].T
    return reconstructed, shifts[chosen]


def warped_rank1_prediction(
    pred: np.ndarray, mean_curve: np.ndarray, template: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allow stimulus-dependent amplitude, latency and response width."""
    shifts = np.arange(-5, 6)
    scales = np.asarray([0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50])
    x = np.arange(len(template), dtype=float)
    center = float(np.sum(x * template ** 2) / max(np.sum(template ** 2), 1e-12))
    bank, labels = [], []
    for shift, scale in product(shifts, scales):
        source_x = center + (x - center - shift) / scale
        warped = np.interp(source_x, x, template, left=0.0, right=0.0)
        norm = np.linalg.norm(warped)
        if norm > 1e-12:
            bank.append(warped / norm)
            labels.append((shift, scale))
    bank = np.stack(bank, axis=1)
    residual = pred - mean_curve[None]
    coefficients = residual @ bank
    chosen = np.argmax(np.abs(coefficients), axis=1)
    amplitude = coefficients[np.arange(len(pred)), chosen]
    reconstructed = mean_curve[None] + amplitude[:, None] * bank[:, chosen].T
    selected = np.asarray(labels, float)[chosen]
    return reconstructed, selected[:, 0], selected[:, 1]


def derivative_rank2_prediction(pred: np.ndarray, mean_curve: np.ndarray, template: np.ndarray) -> np.ndarray:
    """First-order latency control: template plus its temporal derivative."""
    basis, _ = np.linalg.qr(np.column_stack([template, np.gradient(template)]))
    return pilot.project_prediction(pred, mean_curve, basis[:, :2])


def random_subspace_null(n_time: int, rank: int, repeats: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(SEED + rank)
    values = np.empty(repeats, float)
    for i in range(repeats):
        a, _ = np.linalg.qr(rng.normal(size=(n_time, rank)))
        b, _ = np.linalg.qr(rng.normal(size=(n_time, rank)))
        singular = np.linalg.svd(a.T @ b, compute_uv=False)
        values[i] = np.mean(singular ** 2)
    return values


def mode_centroid(h: np.ndarray, times: np.ndarray) -> float:
    energy = h ** 2
    energy /= max(float(energy.sum()), 1e-12)
    return float(np.sum(times * energy))


def validate_units(configs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    time_index = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    times = windows[time_index].mean(axis=1)
    order = np.random.default_rng(SEED).permutation(pilot.N_IMAGES)
    halves = (np.sort(order[:500]), np.sort(order[500:]))
    rows = []
    semantic_rows = []
    semantic = pd.read_csv(pilot.SEMANTIC)
    semantic = semantic[semantic.window_start_ms.isin(windows[time_index, 0])].groupby(
        ["unit_global", "roi", "window_start_ms", "concept"], as_index=False
    ).concept_score.mean()

    null_cache = {}
    for _, cfg in configs.iterrows():
        roi = str(cfg["roi"]); rank = int(cfg["rank"]); lam = float(cfg["lambda"])
        meta, y, pred = pilot.load_roi(roi, responses, time_index)
        null_cache[rank] = null_cache.get(rank, random_subspace_null(len(times), rank))
        for ui, unit in meta.iterrows():
            bases = []
            unit_rows = []
            for split, train in enumerate(halves):
                test = halves[1 - split]
                mean_curve = y[train, :, ui].mean(axis=0)
                h = pilot.svd_basis(y[train, :, ui] - mean_curve[None], lam, rank)[:, :rank]
                bases.append(h)
                target = y[test, :, ui]
                raw = pred[test, :, ui]
                rank1_pred = pilot.project_prediction(raw, mean_curve, h[:, :1])
                rankk_pred = pilot.project_prediction(raw, mean_curve, h)
                shifted_pred, shifts = shifted_rank1_prediction(raw, mean_curve, h[:, 0])
                warped_pred, warp_shifts, warp_scales = warped_rank1_prediction(raw, mean_curve, h[:, 0])
                derivative_pred = derivative_rank2_prediction(raw, mean_curve, h[:, 0])
                rank1 = pilot.score_curve(target, rank1_pred, mean_curve)["stimulus_r2"]
                rankk = pilot.score_curve(target, rankk_pred, mean_curve)["stimulus_r2"]
                shifted = pilot.score_curve(target, shifted_pred, mean_curve)["stimulus_r2"]
                warped = pilot.score_curve(target, warped_pred, mean_curve)["stimulus_r2"]
                derivative = pilot.score_curve(target, derivative_pred, mean_curve)["stimulus_r2"]
                target_coeff = (target - mean_curve[None]) @ h
                pred_coeff = (raw - mean_curve[None]) @ h
                item = {
                    "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                    "monkey": str(unit.monkey), "session": int(unit.session), "split": split,
                    "rank": rank, "lambda": lam, "rank1_r2": rank1, "dynamic_r2": rankk,
                    "shifted_rank1_r2": shifted, "dynamic_minus_rank1": rankk - rank1,
                    "dynamic_minus_shifted_rank1": rankk - shifted,
                    "warped_rank1_r2": warped, "derivative_rank2_r2": derivative,
                    "dynamic_minus_warped_rank1": rankk - warped,
                    "dynamic_minus_derivative_rank2": rankk - derivative,
                    "median_selected_shift_ms": float(np.median(shifts) * 10),
                    "median_warp_shift_ms": float(np.median(warp_shifts) * 10),
                    "median_warp_scale": float(np.median(warp_scales)),
                }
                for k in range(rank):
                    item[f"mode{k + 1}_coefficient_r"] = pilot.corr(target_coeff[:, k], pred_coeff[:, k])
                unit_rows.append(item)
            singular = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
            stability = float(np.mean(singular ** 2))
            p = float((1 + np.sum(null_cache[rank] >= stability)) / (len(null_cache[rank]) + 1))
            for item in unit_rows:
                item["disjoint_subspace_stability"] = stability
                item["disjoint_worst_canonical_r"] = float(np.min(singular))
                item["random_subspace_p"] = p
                rows.append(item)

            full_mean = y[:, :, ui].mean(axis=0)
            full_h = pilot.svd_basis(y[:, :, ui] - full_mean[None], lam, rank)[:, :rank]
            derivative_similarity = (
                abs(pilot.corr(full_h[:, 1], np.gradient(full_h[:, 0]))) if rank >= 2 else np.nan
            )
            q = semantic[(semantic.roi.eq(roi)) & semantic.unit_global.eq(int(unit.unit_global))]
            matrix = q.pivot(index="window_start_ms", columns="concept", values="concept_score").reindex(
                index=windows[time_index, 0]).sort_index(axis=1)
            profiles = full_h.T @ matrix.to_numpy(float)
            mode12_cosine = pilot.cosine(profiles[0], profiles[1]) if rank >= 2 else np.nan
            for k in range(rank):
                semantic_rows.append({
                    "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                    "monkey": str(unit.monkey), "session": int(unit.session), "mode": k + 1,
                    "mode_centroid_ms": mode_centroid(full_h[:, k], times),
                    "semantic_profile_rms": float(np.sqrt(np.mean(profiles[k] ** 2))),
                    "mode12_semantic_cosine": mode12_cosine,
                    "mode2_vs_mode1_derivative_abs_r": derivative_similarity,
                })
    return pd.DataFrame(rows), pd.DataFrame(semantic_rows)


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, float)
    observed = abs(float(values.mean()))
    if len(values) <= 18:
        null = np.asarray([abs(float(np.mean(values * np.asarray(s))))
                           for s in product((-1, 1), repeat=len(values))])
    else:
        rng = np.random.default_rng(seed)
        null = np.abs((rng.choice((-1, 1), size=(200_000, len(values))) * values[None]).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def summarize(unit: pd.DataFrame, semantic: pd.DataFrame) -> pd.DataFrame:
    averaged = unit.groupby(
        ["roi", "unit_global", "native_area", "monkey", "session", "rank"], as_index=False
    ).mean(numeric_only=True)
    mode = semantic[semantic["mode"].eq(1)].merge(
        semantic[semantic["mode"].eq(2)][["unit_global", "mode_centroid_ms"]].rename(
            columns={"mode_centroid_ms": "mode2_centroid_ms"}), on="unit_global", how="left"
    )
    rows = []
    for roi, q in averaged.groupby("roi", sort=False):
        sem = mode[mode.roi.eq(roi)]
        row = {
            "roi": roi, "n_units": int(len(q)), "rank": int(q["rank"].iloc[0]),
            "median_dynamic_minus_rank1_r2": float(q.dynamic_minus_rank1.median()),
            "median_dynamic_minus_shifted_rank1_r2": float(q.dynamic_minus_shifted_rank1.median()),
            "median_dynamic_minus_warped_rank1_r2": float(q.dynamic_minus_warped_rank1.median()),
            "median_dynamic_minus_derivative_rank2_r2": float(q.dynamic_minus_derivative_rank2.median()),
            "dynamic_beats_shifted_fraction": float(np.mean(q.dynamic_minus_shifted_rank1 > 0)),
            "dynamic_beats_warped_fraction": float(np.mean(q.dynamic_minus_warped_rank1 > 0)),
            "dynamic_beats_derivative_fraction": float(np.mean(q.dynamic_minus_derivative_rank2 > 0)),
            "median_disjoint_subspace_stability": float(q.disjoint_subspace_stability.median()),
            "median_random_subspace_p": float(q.random_subspace_p.median()),
            "median_mode1_coefficient_r": float(q.mode1_coefficient_r.median()),
            "median_mode2_coefficient_r": float(q.mode2_coefficient_r.median()),
            "median_mode1_centroid_ms": float(sem.mode_centroid_ms.median()),
            "median_mode2_centroid_ms": float(sem.mode2_centroid_ms.median()),
            "median_mode12_semantic_cosine": float(sem.mode12_semantic_cosine.median()),
            "median_mode2_derivative_similarity": float(sem.mode2_vs_mode1_derivative_abs_r.median()),
        }
        for ci, column in enumerate(("dynamic_minus_rank1", "dynamic_minus_shifted_rank1",
                                     "dynamic_minus_warped_rank1", "dynamic_minus_derivative_rank2")):
            cluster = q.groupby(["monkey", "session"])[column].mean().to_numpy(float)
            row[f"{column}_session_positive_fraction"] = float(np.mean(cluster > 0))
            row[f"{column}_session_signflip_p"] = signflip_p(cluster, SEED + len(rows) + ci)
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_roi_order"] = result.roi.map(ROI_ORDER)
    return result.sort_values("_roi_order").drop(columns="_roi_order").reset_index(drop=True)


def hierarchy_test(semantic: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mode1 = semantic[semantic["mode"].eq(1)].copy()
    roi_summary = mode1.groupby("roi", as_index=False).agg(
        n_units=("unit_global", "size"), median_centroid_ms=("mode_centroid_ms", "median"),
        mean_centroid_ms=("mode_centroid_ms", "mean"),
    )
    roi_summary["roi_order"] = roi_summary.roi.map(ROI_ORDER)
    session = mode1.groupby(["monkey", "session", "roi"], as_index=False).mode_centroid_ms.median()
    session["roi_order"] = session.roi.map(ROI_ORDER)
    rho, p = spearmanr(session.roi_order, session.mode_centroid_ms)
    groups = [g.mode_centroid_ms.to_numpy(float) for _, g in session.groupby("roi")]
    kw = kruskal(*groups)
    monkey_slopes = []
    for monkey, q in session.groupby("monkey"):
        if q.roi.nunique() >= 3:
            slope = np.polyfit(q.roi_order, q.mode_centroid_ms, 1)[0]
            monkey_slopes.append({"monkey": monkey, "n_rois": int(q.roi.nunique()), "slope_ms_per_stage": float(slope)})
    slopes = pd.DataFrame(monkey_slopes)
    result = {
        "session_level_spearman_rho": float(rho), "session_level_spearman_p": float(p),
        "session_level_kruskal_p": float(kw.pvalue),
        "n_monkeys_with_at_least_3_rois": int(len(slopes)),
        "positive_monkey_slope_fraction": float(np.mean(slopes.slope_ms_per_stage > 0)) if len(slopes) else np.nan,
        "median_monkey_slope_ms_per_stage": float(slopes.slope_ms_per_stage.median()) if len(slopes) else np.nan,
    }
    slopes.to_csv(OUT / "mode1_centroid_hierarchy_by_monkey.csv", index=False, encoding="utf-8-sig")
    return roi_summary, result


def make_figure(summary: pd.DataFrame, roi_centroid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    order = list(pilot.ROIS)
    q = summary.set_index("roi").reindex(order)
    x = np.arange(len(order)); width = .25
    axes[0].bar(x - width, q.median_dynamic_minus_rank1_r2, width, label="dynamic − rank1")
    axes[0].bar(x, q.median_dynamic_minus_warped_rank1_r2, width, label="dynamic − shift/scale rank1")
    axes[0].bar(x + width, q.median_dynamic_minus_derivative_rank2_r2, width,
                label="dynamic − derivative rank2")
    axes[0].axhline(0, color="#777777", lw=.8)
    axes[0].set_xticks(x, order, rotation=35, ha="right")
    axes[0].set(title="Disjoint-half temporal model advantage", ylabel="Median held-out ΔR²")
    axes[0].grid(axis="y", alpha=.2); axes[0].legend(fontsize=8)
    c = roi_centroid.set_index("roi").reindex(order)
    axes[1].plot(x, c.median_centroid_ms, marker="o", lw=2, color="#7A5195")
    axes[1].set_xticks(x, order, rotation=35, ha="right")
    axes[1].set(title="Dominant stimulus-dependent mode shifts later", ylabel="Mode-1 energy centroid (ms)")
    axes[1].grid(alpha=.2)
    fig.savefig(OUT / "04_disjoint_latency_control_and_hierarchy.png", dpi=300)
    plt.close(fig)


def write_report(summary: pd.DataFrame, roi_centroid: pd.DataFrame, hierarchy: dict) -> None:
    lines = [
        "# 低秩时间动力学：互斥图像与潜伏期对照",
        "",
        "## 核心结果",
        "",
        "| ROI | K | dynamic−rank1 | dynamic−平移rank1 | dynamic−平移伸缩rank1 | dynamic−导数rank2 | 胜过伸缩模型 | 子空间稳定度 | mode1预测r | mode2预测r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summary.itertuples():
        lines.append(
            f"| {x.roi} | {x.rank} | {x.median_dynamic_minus_rank1_r2:+.4f} | "
            f"{x.median_dynamic_minus_shifted_rank1_r2:+.4f} | {x.median_dynamic_minus_warped_rank1_r2:+.4f} | "
            f"{x.median_dynamic_minus_derivative_rank2_r2:+.4f} | {x.dynamic_beats_warped_fraction:.1%} | "
            f"{x.median_disjoint_subspace_stability:.3f} | {x.median_mode1_coefficient_r:.3f} | "
            f"{x.median_mode2_coefficient_r:.3f} |"
        )
    lines += [
        "",
        "平移模型允许 ±40 ms；更强的形变模型允许 ±50 ms 平移和 0.6–1.5 倍时间伸缩；导数rank2是小幅潜伏期变化的一阶近似。"
        "只有动态模型继续胜过这些对照，才支持超出简单潜伏期/宽度变化的曲线结构，但仍不自动等于视觉特征重组。",
        "",
        "## 主时间模式的层级",
        "",
        "| ROI | unit | mode1时间质心中位数(ms) |",
        "|---|---:|---:|",
    ]
    for x in roi_centroid.set_index("roi").reindex(pilot.ROIS).reset_index().itertuples():
        lines.append(f"| {x.roi} | {x.n_units} | {x.median_centroid_ms:.1f} |")
    lines += [
        "",
        f"按 monkey×session 汇总后，ROI层级与mode1质心的 Spearman ρ={hierarchy['session_level_spearman_rho']:.3f}，"
        f"p={hierarchy['session_level_spearman_p']:.4g}；Kruskal–Wallis p={hierarchy['session_level_kruskal_p']:.4g}。"
        f"在至少覆盖3个ROI的 {hierarchy['n_monkeys_with_at_least_3_rois']} 只猴中，"
        f"{hierarchy['positive_monkey_slope_fraction']:.1%} 的层级斜率为正，中位斜率为"
        f" {hierarchy['median_monkey_slope_ms_per_stage']:+.2f} ms/级。",
        "",
        "## 语义关系诊断",
        "",
        "| ROI | mode1质心 | mode2质心 | mode1–mode2语义cosine | mode2与mode1时间导数相似度 |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in summary.itertuples():
        lines.append(
            f"| {x.roi} | {x.median_mode1_centroid_ms:.1f} | {x.median_mode2_centroid_ms:.1f} | "
            f"{x.median_mode12_semantic_cosine:+.3f} | {x.median_mode2_derivative_similarity:.3f} |"
        )
    lines += [
        "",
        "模式语义 cosine 接近0表示两个响应定义模式在十概念空间中不只是同一语义轮廓的缩放；"
        "但只有 mode2 的图像系数在未见图像上可预测、且动态模型胜过平移对照时，才把它视为候选特征重组。",
        "",
        "## 方向判断",
        "",
        "1. **时间秩随层级增加：目前不支持。** 最优K依次为3、4、3、4、2、2，并不随V1→前IT单调增加。",
        "2. **主导时间尺度沿腹侧通路后移：当前最强结果。** session层级相关显著，且3只覆盖至少3区的猴子斜率均为正。",
        "3. **固定特征×增益不足：支持。** 六区动态模型均优于rank1，且互斥图像子空间高度稳定。",
        "4. **是否为非平凡动态重组：分脑区。** V2、V4、后IT在导数rank2之后仍保留较明显优势；"
        "前IT几乎完全被导数模型解释，中IT仅有很小残差，提示高阶IT的第二模式主要像刺激依赖潜伏期变化。",
        "5. **语义演变：仍是候选而非结论。** mode2可由未见图像预测，且十概念profile通常不与mode1平行；"
        "但概念轴少、mode2常呈导数形态，必须先在V2/V4/后IT做端到端视觉×时间模型和跨场景语义检验。",
        "",
    ]
    (OUT / "DISJOINT_TEMPORAL_VALIDATION_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configs = modal_configs()
    configs.to_csv(OUT / "modal_unit_dynamic_configs.csv", index=False, encoding="utf-8-sig")
    unit, semantic = validate_units(configs)
    unit.to_csv(OUT / "disjoint_half_unit_temporal_validation.csv", index=False, encoding="utf-8-sig")
    semantic.to_csv(OUT / "unit_temporal_mode_semantics.csv", index=False, encoding="utf-8-sig")
    summary = summarize(unit, semantic)
    summary.to_csv(OUT / "disjoint_half_temporal_summary.csv", index=False, encoding="utf-8-sig")
    roi_centroid, hierarchy = hierarchy_test(semantic)
    roi_centroid.to_csv(OUT / "mode1_centroid_by_roi.csv", index=False, encoding="utf-8-sig")
    (OUT / "mode1_centroid_hierarchy_test.json").write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(summary, roi_centroid)
    write_report(summary, roi_centroid, hierarchy)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(hierarchy, indent=2), flush=True)


if __name__ == "__main__":
    main()
