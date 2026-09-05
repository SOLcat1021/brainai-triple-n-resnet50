"""Audit concept-axis dependence and disjoint split-half semantic replication."""

from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from analyze_64d_high_quality_pilot import (
    BASIS_FILE,
    CAV_NODES,
    LAYERS,
    N_CAV_REPEATS,
    ONSET,
    OUT,
    RANK,
    ROI_FILE,
    ROIS,
    TCAV,
    choose_concepts,
    cosine,
)


BANK = OUT.parent.parent / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
SPLIT_BANK = BANK / "disjoint_split_half"
SEED = 20260826
DECORRELATED_SIX = ("striped", "cat", "sky", "car", "head", "food")


def slug(x: str) -> str:
    return x.lower().replace(" ", "_")


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    observed = abs(float(np.mean(values)))
    if len(values) <= 18:
        null = np.asarray([
            abs(float(np.mean(values * np.asarray(signs))))
            for signs in product((-1, 1), repeat=len(values))
        ])
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1, 1), size=(200_000, len(values)))
        null = np.abs((signs * values[None]).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def audit_axis_correlations(concepts: list[str]) -> pd.DataFrame:
    cav = np.load(TCAV / "native_channel_cav_bank.npz", allow_pickle=True)
    late_nodes = [n for n in CAV_NODES if n.startswith(("res4", "res5"))]
    rows = []
    for a, b in combinations(concepts, 2):
        for node in late_nodes:
            aa = cav[f"repeated_{a}_{node}"].astype(np.float32).mean(axis=0)
            bb = cav[f"repeated_{b}_{node}"].astype(np.float32).mean(axis=0)
            rows.append({"concept_a": a, "concept_b": b, "node": node, "axis_cosine": cosine(aa, bb)})
    layer = pd.DataFrame(rows)
    pair = layer.groupby(["concept_a", "concept_b"], as_index=False).agg(
        median_axis_cosine=("axis_cosine", "median"),
        mean_axis_cosine=("axis_cosine", "mean"),
        max_abs_axis_cosine=("axis_cosine", lambda x: float(np.max(np.abs(x)))),
        n_late_nodes=("axis_cosine", "size"),
    ).sort_values("median_axis_cosine", ascending=False)
    layer.to_csv(OUT / "concept_axis_correlation_by_late_layer.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(OUT / "concept_axis_correlation_summary.csv", index=False, encoding="utf-8-sig")

    matrix = pd.DataFrame(np.eye(len(concepts)), index=concepts, columns=concepts)
    for row in pair.itertuples():
        matrix.loc[row.concept_a, row.concept_b] = row.median_axis_cosine
        matrix.loc[row.concept_b, row.concept_a] = row.median_axis_cosine
    matrix.to_csv(OUT / "concept_axis_median_cosine_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(concepts)), concepts, rotation=42, ha="right")
    ax.set_yticks(range(len(concepts)), concepts)
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            color = "white" if abs(matrix.iloc[i, j]) > .55 else "black"
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Median cosine between mean CAV axes across res4/res5 nodes")
    fig.colorbar(im, ax=ax, label="CAV-axis cosine")
    fig.savefig(OUT / "06_concept_axis_correlation.png", dpi=300)
    plt.close(fig)
    return pair


def map_split_scores(concepts: list[str], layer_quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    basis = np.load(BASIS_FILE, allow_pickle=True)
    cav = np.load(TCAV / "native_channel_cav_bank.npz", allow_pickle=True)
    basis_arrays = [basis[f"basis_{li}"][:, :RANK].astype(np.float32) for li in range(len(LAYERS))]
    cav_axes = {
        (concept, node): cav[f"repeated_{concept}_{node}"].astype(np.float32)
        for concept in concepts for node in CAV_NODES
    }
    quality_lookup = layer_quality.set_index(["concept", "node"]).valid.to_dict()
    selected = pd.read_csv(BANK / "selected_units.csv")
    rows = []
    window_rows = []
    for roi in ROIS:
        meta = selected[selected.roi.eq(roi)].copy().reset_index(drop=True)
        path = SPLIT_BANK / f"{slug(roi)}_split_half_64d.h5"
        with h5py.File(path, "r") as h:
            units = h["unit_global"][:].astype(int)
            windows = h["windows_ms"][:].astype(int)
            weights = h["split_channel_weights"][:].astype(np.float32)
            stds = h["split_normalization_std"][:].astype(np.float32)
            gates = h["layer_gate"][:].astype(np.float32)
            cross_r = h["cross_half_r"][:].astype(np.float32)
        if not np.array_equal(units, meta.unit_global.to_numpy(int)):
            raise RuntimeError(f"unit order mismatch for {roi}")
        for ui, unit in meta.iterrows():
            for wi, (t0, t1) in enumerate(windows):
                eligible = bool(t0 >= ONSET[roi] and t1 <= 169)
                for split in range(2):
                    window_rows.append({
                        "unit_global": int(unit.unit_global), "roi": roi,
                        "native_area": str(unit.native_area), "monkey": str(unit.monkey),
                        "session": int(unit.session), "unit_type_name": str(unit.unit_type_name),
                        "split": split, "window_index": wi,
                        "window_start_ms": int(t0), "window_end_ms": int(t1),
                        "eligible": eligible, "cross_half_r": float(cross_r[split, wi, ui]),
                    })
                    values: dict[str, list[np.ndarray]] = {c: [] for c in concepts}
                    layer_weights: dict[str, list[float]] = {c: [] for c in concepts}
                    for li, node in enumerate(CAV_NODES):
                        w = weights[split, wi, ui, li * RANK:(li + 1) * RANK]
                        native_w = basis_arrays[li] @ (w / np.maximum(stds[split, li], 1e-6))
                        wn = np.linalg.norm(native_w)
                        if wn < 1e-10:
                            continue
                        for concept in concepts:
                            if not quality_lookup.get((concept, node), False):
                                continue
                            axes = cav_axes[(concept, node)]
                            den = np.maximum(np.linalg.norm(axes, axis=1) * wn, 1e-10)
                            values[concept].append((axes @ native_w) / den)
                            layer_weights[concept].append(float(gates[wi, ui, li]))
                    for concept in concepts:
                        if values[concept]:
                            lw = np.asarray(layer_weights[concept], np.float32)
                            lw /= max(float(lw.sum()), 1e-8)
                            repeated = np.sum(np.stack(values[concept]) * lw[:, None], axis=0)
                        else:
                            repeated = np.full(N_CAV_REPEATS, np.nan, np.float32)
                        rows.append({
                            "unit_global": int(unit.unit_global), "roi": roi,
                            "native_area": str(unit.native_area), "monkey": str(unit.monkey),
                            "session": int(unit.session), "split": split,
                            "window_index": wi, "window_start_ms": int(t0),
                            "window_end_ms": int(t1), "eligible": eligible,
                            "concept": concept, "concept_score": float(np.nanmean(repeated)),
                            "cav_repeat_sd": float(np.nanstd(repeated, ddof=1)),
                        })
    return pd.DataFrame(rows), pd.DataFrame(window_rows)


def semantic_metrics(
    scores: pd.DataFrame, windows: pd.DataFrame, concepts: list[str], concept_set: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = scores[scores.concept.isin(concepts)]
    index = ["unit_global", "split", "window_index"]
    wide = q.pivot(index=index, columns="concept", values="concept_score").reindex(columns=concepts)
    profile = wide.to_numpy(float)
    positive = np.nanmax(profile, axis=1)
    negative = -np.nanmin(profile, axis=1)
    metric = wide.reset_index()[index]
    metric["concept_set"] = concept_set
    metric["profile_rms"] = np.sqrt(np.nanmean(profile ** 2, axis=1))
    metric["positive_strength"] = positive
    metric["negative_strength"] = negative
    metric["secondary_pole_strength"] = np.where(
        (positive > 0) & (negative > 0), np.minimum(positive, negative), 0.0
    )
    metric["bidirectional_balance"] = np.where(
        (positive > 0) & (negative > 0),
        np.minimum(positive, negative) / np.maximum(np.maximum(positive, negative), 1e-8),
        0.0,
    )
    metric = windows.merge(metric, on=index, validate="one_to_one")

    summaries = []
    deltas = []
    measures = ("secondary_pole_strength", "bidirectional_balance", "profile_rms")
    for (roi, split), group in metric[metric.eligible].groupby(["roi", "split"], sort=False):
        unit_rows = []
        for unit, z in group.groupby("unit_global"):
            z = z.sort_values("window_start_ms")
            row = {
                "concept_set": concept_set, "roi": roi, "split": int(split),
                "unit_global": int(unit), "monkey": z.monkey.iloc[0],
                "session": int(z.session.iloc[0]),
            }
            for measure in measures:
                row[f"early_{measure}"] = float(z.head(2)[measure].mean())
                row[f"late_{measure}"] = float(z.tail(3)[measure].mean())
                row[f"delta_{measure}"] = row[f"late_{measure}"] - row[f"early_{measure}"]
            unit_rows.append(row)
        unit = pd.DataFrame(unit_rows)
        deltas.append(unit)
        for mi, measure in enumerate(measures):
            delta = unit[f"delta_{measure}"].to_numpy(float)
            cluster = unit.groupby(["monkey", "session"])[f"delta_{measure}"].mean().to_numpy(float)
            summaries.append({
                "concept_set": concept_set, "roi": roi, "split": int(split), "metric": measure,
                "n_units": int(len(unit)),
                "early_median": float(unit[f"early_{measure}"].median()),
                "late_median": float(unit[f"late_{measure}"].median()),
                "median_change": float(np.median(delta)),
                "positive_unit_fraction": float(np.mean(delta > 0)),
                "wilcoxon_p": float(wilcoxon(delta).pvalue),
                "n_monkey_sessions": int(len(cluster)),
                "session_positive_fraction": float(np.mean(cluster > 0)),
                "session_signflip_p": signflip_p(cluster, SEED + 100 * split + mi),
                "median_cross_half_r": float(group.cross_half_r.median()),
            })
    summary = pd.DataFrame(summaries)
    summary["fdr_bh_within_split_metric"] = np.nan
    for (_, split, measure), idx in summary.groupby(["concept_set", "split", "metric"]).groups.items():
        ii = np.asarray(list(idx), int)
        p = summary.loc[ii, "wilcoxon_p"].to_numpy(float)
        order = np.argsort(p)
        adjusted_sorted = np.minimum.accumulate(
            (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
        )[::-1].clip(max=1)
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        summary.loc[ii, "fdr_bh_within_split_metric"] = adjusted
    return metric, pd.concat(deltas, ignore_index=True), summary


def replication_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measures = ("secondary_pole_strength", "bidirectional_balance", "profile_rms")
    for (concept_set, roi), q in deltas.groupby(["concept_set", "roi"], sort=False):
        for measure in measures:
            wide = q.pivot(index="unit_global", columns="split", values=f"delta_{measure}")
            rho, p = spearmanr(wide[0], wide[1])
            rows.append({
                "concept_set": concept_set, "roi": roi, "metric": measure,
                "n_units": int(len(wide)),
                "split_delta_spearman_rho": float(rho),
                "split_delta_spearman_p": float(p),
                "same_sign_fraction": float(np.mean(np.sign(wide[0]) == np.sign(wide[1]))),
                "both_positive_fraction": float(np.mean((wide[0] > 0) & (wide[1] > 0))),
                "split0_median_change": float(wide[0].median()),
                "split1_median_change": float(wide[1].median()),
            })
    return pd.DataFrame(rows)


def add_profile_replication(scores: pd.DataFrame, metrics: pd.DataFrame, concepts: list[str]) -> pd.DataFrame:
    q = scores[scores.concept.isin(concepts)]
    wide = q.pivot(index=["unit_global", "window_index", "split"], columns="concept", values="concept_score")
    rows = []
    for (unit, window), z in wide.groupby(level=[0, 1]):
        a = z.xs(0, level="split").to_numpy(float).ravel()
        b = z.xs(1, level="split").to_numpy(float).ravel()
        rows.append({"unit_global": int(unit), "window_index": int(window), "split_profile_cosine": cosine(a, b)})
    rep = pd.DataFrame(rows)
    base = metrics[metrics.split.eq(0)].merge(rep, on=["unit_global", "window_index"])
    out = []
    for roi, group in base[base.eligible].groupby("roi", sort=False):
        unit = group.groupby("unit_global").apply(
            lambda z: pd.Series({
                "early_split_profile_cosine": z.sort_values("window_start_ms").head(2).split_profile_cosine.mean(),
                "late_split_profile_cosine": z.sort_values("window_start_ms").tail(3).split_profile_cosine.mean(),
            }), include_groups=False
        ).reset_index()
        out.append({
            "roi": roi, "n_units": int(len(unit)),
            "early_median_split_profile_cosine": float(unit.early_split_profile_cosine.median()),
            "late_median_split_profile_cosine": float(unit.late_split_profile_cosine.median()),
            "median_change": float((unit.late_split_profile_cosine - unit.early_split_profile_cosine).median()),
        })
    return pd.DataFrame(out)


def make_split_figure(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    q = metrics[(metrics.concept_set.eq("all")) & metrics.roi.eq("middle IT") & metrics.eligible]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = ("#4C78A8", "#E45756")
    for split in range(2):
        z = q[q.split.eq(split)].pivot(
            index="unit_global", columns="window_start_ms", values="secondary_pole_strength"
        ).sort_index(axis=1)
        axes[0].plot(z.columns + 4.5, z.median(axis=0), marker="o", ms=3,
                     color=colors[split], label=f"disjoint half {split + 1}")
    axes[0].set(title="Middle IT weaker semantic pole", xlabel="Time (ms)", ylabel="Secondary-pole strength")
    axes[0].grid(alpha=.2); axes[0].legend()

    d = deltas[(deltas.concept_set.eq("all")) & deltas.roi.eq("middle IT")].pivot(
        index="unit_global", columns="split", values="delta_secondary_pole_strength"
    )
    axes[1].scatter(d[0], d[1], s=28, alpha=.75, color="#7A5195")
    lim = float(np.max(np.abs(d.to_numpy()))) * 1.08
    axes[1].plot([-lim, lim], [-lim, lim], color="#777777", lw=.8, ls="--")
    axes[1].axhline(0, color="#999999", lw=.7); axes[1].axvline(0, color="#999999", lw=.7)
    axes[1].set(xlim=(-lim, lim), ylim=(-lim, lim),
                title="Unit-level early→late change", xlabel="Half 1 change", ylabel="Half 2 change")
    axes[1].grid(alpha=.15)
    fig.savefig(OUT / "07_disjoint_split_half_replication.png", dpi=300)
    plt.close(fig)


def write_report(
    axis: pd.DataFrame,
    summary: pd.DataFrame,
    replication: pd.DataFrame,
    profile_replication: pd.DataFrame,
) -> None:
    def row(roi: str, metric: str, split: int, concept_set: str = "all") -> pd.Series:
        return summary[(summary.concept_set.eq(concept_set)) & summary.roi.eq(roi)
                       & summary.metric.eq(metric) & summary.split.eq(split)].iloc[0]

    def rep(roi: str, metric: str, concept_set: str = "all") -> pd.Series:
        return replication[(replication.concept_set.eq(concept_set)) & replication.roi.eq(roi)
                           & replication.metric.eq(metric)].iloc[0]

    m0 = row("middle IT", "secondary_pole_strength", 0)
    m1 = row("middle IT", "secondary_pole_strength", 1)
    mr = rep("middle IT", "secondary_pole_strength")
    md0 = row("middle IT", "secondary_pole_strength", 0, "decorrelated_six")
    md1 = row("middle IT", "secondary_pole_strength", 1, "decorrelated_six")
    a0 = row("anterior IT", "profile_rms", 0)
    a1 = row("anterior IT", "profile_rms", 1)
    ar = rep("anterior IT", "profile_rms")
    middle_profile = profile_replication[profile_replication.roi.eq("middle IT")].iloc[0]
    top = axis.head(5)
    lines = [
        "# 113-unit PCA64：概念轴与不重叠 split-half 审计",
        "",
        "## 复核设计",
        "",
        "- 113 个 unit、PCA64 基底、ridge、空间场、深度门、概念集合和早晚时间窗全部冻结。",
        "- 1,000 张神经刺激图像按固定随机种子分成两个互斥的 500 张集合；两套 signed channel readout 分别只用一半神经响应训练，并在另一半上评估。",
        "- 因为空间场和深度门来自此前全数据代理，这不是全流程独立拟合；它严格检验的是最直接决定 CAV 对齐的 channel readout 能否在不共享神经训练图像时复现。",
        "",
        "## 中 IT 主终点：较弱语义极的早晚变化",
        "",
        "| 读出 | 早期中位数 | 晚期中位数 | 变化中位数 | 正向 unit | Wilcoxon p | session 正向 | session sign-flip p | 六区 FDR q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| half 1 | {m0.early_median:.3f} | {m0.late_median:.3f} | {m0.median_change:+.3f} | {m0.positive_unit_fraction:.1%} | {m0.wilcoxon_p:.6f} | {m0.session_positive_fraction:.1%} | {m0.session_signflip_p:.6f} | {m0.fdr_bh_within_split_metric:.6f} |",
        f"| half 2 | {m1.early_median:.3f} | {m1.late_median:.3f} | {m1.median_change:+.3f} | {m1.positive_unit_fraction:.1%} | {m1.wilcoxon_p:.6f} | {m1.session_positive_fraction:.1%} | {m1.session_signflip_p:.6f} | {m1.fdr_bh_within_split_metric:.6f} |",
        "",
        f"两半 unit-level 变化的 Spearman ρ={mr.split_delta_spearman_rho:.3f}（p={mr.split_delta_spearman_p:.4g}）；"
        f"{mr.same_sign_fraction:.1%} 的 unit 两半变化同号，{mr.both_positive_fraction:.1%} 在两半都增强。"
        f"中 IT 概念轮廓的 split-half cosine 在早期中位数为 {middle_profile.early_median_split_profile_cosine:.3f}，晚期为 {middle_profile.late_median_split_profile_cosine:.3f}。",
        "",
        "使用预先写入的去相关六概念集合（striped、cat、sky、car、head、food）时："
        f"half 1 变化 {md0.median_change:+.3f}、p={md0.wilcoxon_p:.6f}；"
        f"half 2 变化 {md1.median_change:+.3f}、p={md1.wilcoxon_p:.6f}。",
        "",
        "## 前 IT 次级终点：整体概念轮廓 RMS",
        "",
        f"half 1 的早/晚中位数为 {a0.early_median:.3f}/{a0.late_median:.3f}（变化 {a0.median_change:+.3f}，p={a0.wilcoxon_p:.6f}）；"
        f"half 2 为 {a1.early_median:.3f}/{a1.late_median:.3f}（变化 {a1.median_change:+.3f}，p={a1.wilcoxon_p:.6f}）。"
        f"两半 unit-level 变化相关 ρ={ar.split_delta_spearman_rho:.3f}（p={ar.split_delta_spearman_p:.4g}），"
        f"两半都增强的 unit 占 {ar.both_positive_fraction:.1%}。",
        "",
        "## 概念轴相关性",
        "",
        "下列数值是每个概念的 20 个 CAV 重复先取平均方向，再在 res4/res5 九个节点上计算两概念方向 cosine，最后取跨节点中位数：",
        "",
        "| 概念对 | 中位 cosine |",
        "|---|---:|",
    ]
    for x in top.itertuples():
        lines.append(f"| {x.concept_a}–{x.concept_b} | {x.median_axis_cosine:.3f} |")
    lines += [
        "",
        "这说明十个轴并非十个正交自由度；因此全十概念结果必须与去相关六概念敏感性结果并列解释。",
        "",
        "## 决策规则",
        "",
        "- 若中 IT 主终点在两个互斥 half 中均为正、unit 检验均显著，并在去相关六概念下保持方向，则允许进入冻结的功能 IT 扩种队列（每区 20 个）。",
        "- 扩种仍只用于复现主终点；不得据扩种结果重新挑概念、时间窗或统计量。",
        "- 只有功能 IT 扩种复现后，才考虑粗 ROI 的更大规模扩种与总体发生率估计。",
        "",
    ]
    (OUT / "DISJOINT_SPLIT_HALF_AUDIT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concepts, _, layer_quality = choose_concepts()
    axis = audit_axis_correlations(concepts)
    scores, windows = map_split_scores(concepts, layer_quality)
    scores.to_csv(OUT / "split_half_concept_scores.csv.gz", index=False, compression="gzip")
    windows.to_csv(OUT / "split_half_window_quality.csv", index=False, encoding="utf-8-sig")

    all_metrics, all_deltas, all_summary = semantic_metrics(scores, windows, concepts, "all")
    dec_metrics, dec_deltas, dec_summary = semantic_metrics(
        scores, windows, [c for c in DECORRELATED_SIX if c in concepts], "decorrelated_six"
    )
    metrics = pd.concat([all_metrics, dec_metrics], ignore_index=True)
    deltas = pd.concat([all_deltas, dec_deltas], ignore_index=True)
    summary = pd.concat([all_summary, dec_summary], ignore_index=True)
    replication = replication_summary(deltas)
    profile_replication = add_profile_replication(scores, all_metrics, concepts)
    metrics.to_csv(OUT / "split_half_unit_window_metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "split_half_unit_early_late_changes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "split_half_early_late_summary.csv", index=False, encoding="utf-8-sig")
    replication.to_csv(OUT / "split_half_replication_summary.csv", index=False, encoding="utf-8-sig")
    profile_replication.to_csv(OUT / "split_half_profile_replication_by_roi.csv", index=False, encoding="utf-8-sig")
    make_split_figure(all_metrics, all_deltas)
    write_report(axis, summary, replication, profile_replication)

    audit = {
        "unit_selection_uses_concepts": False,
        "neural_training_image_intersection": 0,
        "concepts": concepts,
        "decorrelated_six": [c for c in DECORRELATED_SIX if c in concepts],
        "early_windows": "first two eligible 10-ms windows for each ROI",
        "late_windows": "last three eligible windows, 140-169 ms",
        "primary_endpoint": "middle IT secondary_pole_strength late minus early",
        "secondary_endpoint": "anterior IT profile_rms late minus early",
        "shared_components_warning": "final spatial field and depth gate are shared and were estimated previously",
    }
    (OUT / "split_half_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary[(summary.concept_set.eq("all")) & summary.roi.isin(["middle IT", "anterior IT"])].to_string(index=False))
    print(replication[(replication.concept_set.eq("all")) & replication.roi.isin(["middle IT", "anterior IT"])].to_string(index=False))


if __name__ == "__main__":
    main()
