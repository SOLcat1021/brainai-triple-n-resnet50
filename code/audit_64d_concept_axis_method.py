"""Audit whether PCA64 readouts can express and align with the frozen CAV axes.

The audit compares the historical full-native-space cosine with a minimally
corrected score: cosine to the CAV after projection into the PCA64 subspace.
It also uses the saved permuted-label CAVs as a classifier-matched null.  No
unit is selected or trained with concept information in this script.
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
from scipy.stats import wilcoxon

from analyze_64d_high_quality_pilot import (
    BASIS_FILE, CAV_NODES, LAYERS, ONSET, RANK, ROI_FILE, TCAV, choose_concepts, cosine,
)


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
OUT = PROJECT / "results" / "concept_axis_method_audit_2026-08-26"
BANKS = {
    "pilot_113": (
        PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26",
        ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"),
    ),
    "heldout_functional_92": (
        PROJECT / "proxy_bank_64d_functional_it_validation_92_2026-08-26",
        ("middle IT", "anterior IT"),
    ),
}
CONCEPT_SETS = {
    "all_ten": ("striped", "cat", "sky", "tree", "dog", "car", "torso", "building", "head", "food"),
    "decorrelated_six": ("striped", "cat", "sky", "car", "head", "food"),
}
N_FOLDS = 5
N_REPEATS = 20


def audit_capture(concepts: list[str], layer_quality: pd.DataFrame) -> pd.DataFrame:
    basis = np.load(BASIS_FILE, allow_pickle=True)
    cav = np.load(TCAV / "native_channel_cav_bank.npz", allow_pickle=True)
    valid = layer_quality.set_index(["concept", "node"]).valid.to_dict()
    rows = []
    for li, node in enumerate(CAV_NODES):
        full_basis = basis[f"basis_{li}"].astype(np.float32)
        explained = basis[f"explained_{li}"].astype(np.float32)
        for concept in concepts:
            axes = cav[f"repeated_{concept}_{node}"].astype(np.float32)
            row = {
                "concept": concept, "node": node, "native_dim": int(axes.shape[1]),
                "valid_layer": bool(valid.get((concept, node), False)),
            }
            for rank in (64, 128, 256):
                p = full_basis[:, :min(rank, full_basis.shape[1])]
                capture = np.sum((axes @ p) ** 2, axis=1) / np.maximum(np.sum(axes ** 2, axis=1), 1e-12)
                row[f"cav_energy_capture_rank{rank}_mean"] = float(np.mean(capture))
                row[f"cav_cosine_ceiling_rank{rank}_mean"] = float(np.mean(np.sqrt(capture)))
                row[f"activation_variance_rank{rank}"] = float(explained[:rank].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def precompute_axes(concepts: list[str]) -> dict:
    basis = np.load(BASIS_FILE, allow_pickle=True)
    cav = np.load(TCAV / "native_channel_cav_bank.npz", allow_pickle=True)
    out = {}
    for li, node in enumerate(CAV_NODES):
        p = basis[f"basis_{li}"][:, :RANK].astype(np.float32)
        for concept in concepts:
            actual = cav[f"repeated_{concept}_{node}"].astype(np.float32)
            null = cav[f"null_repeated_{concept}_{node}"].astype(np.float32)
            out[(li, concept)] = {
                "actual_coeff": actual @ p,
                "null_coeff": null @ p,
                "actual_full_norm": np.linalg.norm(actual, axis=1),
                "null_full_norm": np.linalg.norm(null, axis=1),
            }
    return out


def map_bank(
    cohort: str, bank: Path, rois: tuple[str, ...], concepts: list[str],
    layer_quality: pd.DataFrame, axes: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(bank / "selected_units.csv")
    valid = layer_quality.set_index(["concept", "node"]).valid.to_dict()
    score_rows = []
    absorption_rows = []
    for roi in rois:
        meta = selected[selected.roi.eq(roi)].reset_index(drop=True)
        with h5py.File(bank / ROI_FILE[roi], "r") as h:
            units = h["unit_global"][:].astype(int)
            windows = h["windows_ms"][:].astype(int)
            weights = h["fold_channel_weights"][:].astype(np.float32)
            stds = h["fold_normalization_std"][:].astype(np.float32)
            gates = h["fold_layer_gate"][:].astype(np.float32)
            oof = h["oof_r"][:].astype(np.float32)
        if not np.array_equal(units, meta.unit_global.to_numpy(int)):
            raise RuntimeError(f"unit mismatch: {cohort} {roi}")

        for fi in range(N_FOLDS):
            for wi, (t0, t1) in enumerate(windows):
                eligible = bool(t0 >= ONSET[roi] and t1 <= 169)
                for ui, unit in meta.iterrows():
                    actual_raw = np.zeros((len(concepts), N_REPEATS), np.float32)
                    actual_sub = np.zeros_like(actual_raw)
                    null_raw = np.zeros_like(actual_raw)
                    null_sub = np.zeros_like(actual_raw)
                    weight_sum = np.zeros(len(concepts), np.float32)
                    for li, node in enumerate(CAV_NODES):
                        coeff = weights[fi, wi, ui, li * RANK:(li + 1) * RANK]
                        coeff = coeff / np.maximum(stds[fi, li], 1e-6)
                        wn = np.linalg.norm(coeff)
                        if wn < 1e-10:
                            continue
                        layer_weight = float(gates[fi, wi, ui, li])
                        for ci, concept in enumerate(concepts):
                            if not valid.get((concept, node), False):
                                continue
                            item = axes[(li, concept)]
                            num_actual = item["actual_coeff"] @ coeff
                            num_null = item["null_coeff"] @ coeff
                            projected_actual_norm = np.linalg.norm(item["actual_coeff"], axis=1)
                            projected_null_norm = np.linalg.norm(item["null_coeff"], axis=1)
                            actual_raw[ci] += layer_weight * num_actual / np.maximum(
                                item["actual_full_norm"] * wn, 1e-10
                            )
                            null_raw[ci] += layer_weight * num_null / np.maximum(
                                item["null_full_norm"] * wn, 1e-10
                            )
                            actual_sub[ci] += layer_weight * num_actual / np.maximum(
                                projected_actual_norm * wn, 1e-10
                            )
                            null_sub[ci] += layer_weight * num_null / np.maximum(
                                projected_null_norm * wn, 1e-10
                            )
                            weight_sum[ci] += layer_weight
                    for values in (actual_raw, actual_sub, null_raw, null_sub):
                        values /= np.maximum(weight_sum[:, None], 1e-8)
                    base = {
                        "cohort": cohort, "unit_global": int(unit.unit_global), "roi": roi,
                        "native_area": str(unit.native_area), "monkey": str(unit.monkey),
                        "session": int(unit.session), "fold": fi, "window_index": wi,
                        "window_start_ms": int(t0), "window_end_ms": int(t1),
                        "eligible": eligible, "oof_r": float(oof[wi, ui]),
                    }
                    for variant, actual in (("full_native_cosine", actual_raw),
                                            ("pca64_subspace_cosine", actual_sub)):
                        for ci, concept in enumerate(concepts):
                            score_rows.append({
                                **base, "variant": variant, "concept": concept,
                                "concept_score": float(actual[ci].mean()),
                                "cav_repeat_sd": float(actual[ci].std(ddof=1)),
                            })
                    for set_name, subset in CONCEPT_SETS.items():
                        ii = np.asarray([concepts.index(c) for c in subset], int)
                        for variant, actual, null in (
                            ("full_native_cosine", actual_raw, null_raw),
                            ("pca64_subspace_cosine", actual_sub, null_sub),
                        ):
                            observed_profile = actual[ii].mean(axis=1)
                            observed = float(np.sqrt(np.mean(observed_profile ** 2)))
                            null_rms = np.sqrt(np.mean(null[ii] ** 2, axis=0))
                            absorption_rows.append({
                                **base, "variant": variant, "concept_set": set_name,
                                "observed_profile_rms": observed,
                                "null_profile_rms_mean": float(null_rms.mean()),
                                "null_profile_rms_median": float(np.median(null_rms)),
                                "null_profile_rms_sd": float(null_rms.std(ddof=1)),
                                "absorption_excess": float(observed - np.median(null_rms)),
                                "absorption_ratio": float(observed / max(float(np.median(null_rms)), 1e-8)),
                                "permuted_axis_p": float((1 + np.sum(null_rms >= observed)) / (N_REPEATS + 1)),
                            })
    return pd.DataFrame(score_rows), pd.DataFrame(absorption_rows)


def summarize_absorption(absorption: pd.DataFrame) -> pd.DataFrame:
    q = absorption[absorption.eligible].copy()
    return q.groupby(["cohort", "roi", "variant", "concept_set"], as_index=False).agg(
        n_units=("unit_global", "nunique"),
        n_unit_window_folds=("unit_global", "size"),
        median_observed_rms=("observed_profile_rms", "median"),
        median_null_rms=("null_profile_rms_median", "median"),
        median_absorption_excess=("absorption_excess", "median"),
        median_absorption_ratio=("absorption_ratio", "median"),
        fraction_above_all_20_nulls=("permuted_axis_p", lambda x: float(np.mean(x < .05))),
    )


def endpoint_comparison(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = scores[scores.eligible]
    for (cohort, roi, variant), z in q.groupby(["cohort", "roi", "variant"], sort=False):
        fold_mean = z.groupby(
            ["unit_global", "window_start_ms", "concept"], as_index=False
        ).concept_score.mean()
        wide = fold_mean.pivot(
            index=["unit_global", "window_start_ms"], columns="concept", values="concept_score"
        ).reindex(columns=CONCEPT_SETS["all_ten"])
        table = wide.reset_index()[["unit_global", "window_start_ms"]]
        table["profile_rms"] = np.sqrt(np.mean(wide.to_numpy(float) ** 2, axis=1))
        delta = []
        for unit, unit_rows in table.groupby("unit_global"):
            unit_rows = unit_rows.sort_values("window_start_ms")
            delta.append({
                "unit_global": int(unit),
                "early_profile_rms": float(unit_rows.head(2).profile_rms.mean()),
                "late_profile_rms": float(unit_rows.tail(3).profile_rms.mean()),
            })
        unit = pd.DataFrame(delta)
        change = unit.late_profile_rms - unit.early_profile_rms
        rows.append({
            "cohort": cohort, "roi": roi, "variant": variant, "n_units": int(len(unit)),
            "early_median": float(unit.early_profile_rms.median()),
            "late_median": float(unit.late_profile_rms.median()),
            "median_change": float(change.median()),
            "positive_fraction": float(np.mean(change > 0)),
            "wilcoxon_p": float(wilcoxon(change).pvalue),
        })
    return pd.DataFrame(rows)


def make_figures(capture: pd.DataFrame, summary: pd.DataFrame) -> None:
    valid = capture[capture.valid_layer]
    node_order = list(CAV_NODES)
    by_node = valid.groupby("node").agg(
        rank64=("cav_energy_capture_rank64_mean", "median"),
        rank256=("cav_energy_capture_rank256_mean", "median"),
        activation64=("activation_variance_rank64", "median"),
    ).reindex(node_order)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    x = np.arange(len(by_node))
    axes[0].plot(x, by_node.activation64, marker="o", label="activation variance: PCA64")
    axes[0].plot(x, by_node.rank64, marker="o", label="CAV energy: PCA64")
    axes[0].plot(x, by_node.rank256, marker="o", label="CAV energy: PCA256")
    axes[0].set_xticks(x, by_node.index, rotation=55, ha="right")
    axes[0].set(ylabel="Fraction retained", title="PCA preservation of activation variance vs concept axes")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)

    s = summary[(summary.concept_set.eq("all_ten")) & summary.cohort.eq("pilot_113")]
    labels = s.roi.drop_duplicates().tolist()
    width = .36
    for vi, (variant, color) in enumerate((
        ("full_native_cosine", "#4C78A8"), ("pca64_subspace_cosine", "#E45756")
    )):
        z = s[s.variant.eq(variant)].set_index("roi").reindex(labels)
        axes[1].bar(np.arange(len(labels)) + (vi - .5) * width, z.median_absorption_excess,
                    width=width, color=color, label=variant)
    axes[1].axhline(0, color="#777777", lw=.8)
    axes[1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[1].set(ylabel="Observed RMS − permuted-CAV null median",
                title="Concept absorption above classifier-matched null")
    axes[1].grid(axis="y", alpha=.2); axes[1].legend(fontsize=8)
    fig.savefig(OUT / "01_pca64_cav_capacity_and_absorption.png", dpi=300)
    plt.close(fig)


def write_report(capture: pd.DataFrame, summary: pd.DataFrame, endpoints: pd.DataFrame) -> None:
    valid = capture[capture.valid_layer]
    late = valid[valid.node.str.startswith(("res4", "res5"))]
    rank64_median = float(valid.cav_energy_capture_rank64_mean.median())
    res5_last = float(valid[valid.node.eq("res5_b3")].cav_energy_capture_rank64_mean.median())
    late256 = float(late.cav_energy_capture_rank256_mean.median())
    lines = [
        "# PCA64 → CAV 概念轴方法审计",
        "",
        "## 结论",
        "",
        "当前方法可以检测稳定的概念对齐，但不能据现有分数判断 unit 已经“充分吸附”到完整 CAV 轴。主要限制来自表示子空间与零分布，而不是 unit 数量。",
        "",
        "## 发现的两个实质性缺陷",
        "",
        f"1. **PCA64 系统性截断概念方向。** 在通过质量门的 concept×layer 中，PCA64 对 CAV 能量的中位保留率仅 {rank64_median:.1%}；"
        f"res5_b3 仅 {res5_last:.1%}。即使使用现有 PCA256，res4/res5 的中位保留率也只有 {late256:.1%}。"
        "原始 cosine 用完整 CAV 范数作分母，但 neural weight 被限制在 PCA64 内，因此分数受到不可达 CAV 分量的机械性压低。",
        "2. **原零分布没有检验概念吸附。** 把十个真实概念在五折内随机换名，只能检验概念轮廓身份是否跨折稳定；"
        "它不能回答真实 CAV 对齐是否强于随机方向。此前报告的‘显著窗比例’应改名为‘跨折轮廓身份检验’，不能作为语义吸附证据。",
        "",
        "## 本次最小修改",
        "",
        "- 保留原始 `full_native_cosine`，保证历史结果可追溯。",
        "- 新增 `pca64_subspace_cosine`：先把 CAV 投影到 PCA64，再在双方都可表达的同一子空间内归一化。它修正表示上限，但不改变代理训练，也不会强迫神经权重沿概念轴。",
        "- CAV 训练现在保存 permuted-label 方向；以相同分类器、相同概念样本和相同跨层聚合得到 classifier-matched null。",
        "- 十概念与去相关六概念并列报告；不再把相关概念轴当作十个独立自由度。",
        "",
        "## 吸附审计汇总",
        "",
        "`absorption_excess` 是真实概念轮廓 RMS 减去 permuted-CAV 零分布中位数；正值才表示真实轴优于伪概念轴。",
        "",
        "| 队列 | ROI | 分数 | observed RMS | null RMS | excess | ratio | 超过全部20个零轴的比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in summary[summary.concept_set.eq("all_ten")].itertuples():
        lines.append(
            f"| {x.cohort} | {x.roi} | {x.variant} | {x.median_observed_rms:.3f} | "
            f"{x.median_null_rms:.3f} | {x.median_absorption_excess:+.3f} | "
            f"{x.median_absorption_ratio:.2f} | {x.fraction_above_all_20_nulls:.1%} |"
        )
    lines += [
        "",
        "## 对既有时间结论的影响（仅 RMS，不涉及正负极）",
        "",
        "| 队列 | ROI | 分数 | 早期 | 晚期 | 变化 | 正向 unit | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in endpoints.itertuples():
        lines.append(
            f"| {x.cohort} | {x.roi} | {x.variant} | {x.early_median:.3f} | "
            f"{x.late_median:.3f} | {x.median_change:+.3f} | {x.positive_fraction:.1%} | {x.wilcoxon_p:.6f} |"
        )
    lines += [
        "",
        "## 解释限制与下一步",
        "",
        "- 子空间归一化回答‘在 PCA64 能表达的部分里是否对齐’，不是‘完整 native CAV 是否被重建’。两种分数必须并列保留。",
        "- 20 个 permuted CAV 使单窗经验 p 的最小值约为 0.048，只适合当前诊断。正式推断应把 null repeats 增至至少 200，并预注册 ROI×时间聚合统计量。",
        "- 若子空间分数仍不能明显超过 permuted null，增加 unit 数不会解决测量问题；应先改善 CAV 定义或表示基底。",
        "- 若需要让模型覆盖更完整的概念方向，下一步可在现有 PCA64 后加入由独立 Broden CAV 构造的少量正交残差维度，并使用完全未参与构造的 CAV/图像评估；不能直接用同一 CAV 构造基底又检验自身。",
        "",
    ]
    (OUT / "PCA64_CAV_METHOD_AUDIT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concepts, _, layer_quality = choose_concepts()
    if tuple(concepts) != CONCEPT_SETS["all_ten"]:
        raise RuntimeError(f"unexpected concept set: {concepts}")
    capture = audit_capture(concepts, layer_quality)
    capture.to_csv(OUT / "pca_cav_capture_by_concept_layer.csv", index=False, encoding="utf-8-sig")
    axes = precompute_axes(concepts)
    score_parts, absorption_parts = [], []
    for cohort, (bank, rois) in BANKS.items():
        score, absorption = map_bank(cohort, bank, rois, concepts, layer_quality, axes)
        score_parts.append(score); absorption_parts.append(absorption)
        print(f"mapped {cohort}: {score.unit_global.nunique()} units", flush=True)
    scores = pd.concat(score_parts, ignore_index=True)
    absorption = pd.concat(absorption_parts, ignore_index=True)
    summary = summarize_absorption(absorption)
    endpoints = endpoint_comparison(scores)
    scores.to_csv(OUT / "raw_and_subspace_concept_scores.csv.gz", index=False, compression="gzip")
    absorption.to_csv(OUT / "permuted_cav_absorption_by_unit_window_fold.csv.gz", index=False, compression="gzip")
    summary.to_csv(OUT / "absorption_summary_by_cohort_roi.csv", index=False, encoding="utf-8-sig")
    endpoints.to_csv(OUT / "profile_rms_endpoint_raw_vs_subspace.csv", index=False, encoding="utf-8-sig")
    make_figures(capture, summary)
    write_report(capture, summary, endpoints)
    audit = {
        "unit_selection_uses_concepts": False,
        "proxy_retrained": False,
        "historical_score_preserved": "full_native_cosine",
        "new_score": "pca64_subspace_cosine",
        "semantic_null": "20 repeated permuted-label CAV classifiers per concept and layer",
        "warning": "minimum per-window empirical p is 1/21; diagnostic only",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(endpoints.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
