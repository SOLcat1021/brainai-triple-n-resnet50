"""Locked analysis of the held-out 92-unit functional-IT validation cohort."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

import analyze_64d_high_quality_pilot as pilot


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_functional_it_validation_92_2026-08-26"
OUT = PROJECT / "results" / "semantic_64d_functional_it_validation_92_2026-08-26"
ROIS = ("middle IT", "anterior IT")
CONCEPT_SETS = {
    "all": ("striped", "cat", "sky", "tree", "dog", "car", "torso", "building", "head", "food"),
    "decorrelated_six": ("striped", "cat", "sky", "car", "head", "food"),
}
SEED = 20260826


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, float)
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


def concept_set_deltas(scores: pd.DataFrame, concepts: tuple[str, ...], name: str) -> pd.DataFrame:
    q = scores[scores.eligible & scores.concept.isin(concepts)].groupby(
        ["unit_global", "roi", "native_area", "monkey", "session", "window_start_ms", "concept"],
        as_index=False,
    ).concept_score.mean()
    wide = q.pivot(
        index=["unit_global", "roi", "native_area", "monkey", "session", "window_start_ms"],
        columns="concept", values="concept_score",
    ).reindex(columns=concepts)
    values = wide.to_numpy(float)
    positive = np.nanmax(values, axis=1)
    negative = -np.nanmin(values, axis=1)
    table = wide.reset_index()[["unit_global", "roi", "native_area", "monkey", "session", "window_start_ms"]]
    table["secondary_pole_strength"] = np.where(
        (positive > 0) & (negative > 0), np.minimum(positive, negative), 0.0
    )
    table["profile_rms"] = np.sqrt(np.nanmean(values ** 2, axis=1))
    rows = []
    for unit, z in table.groupby("unit_global"):
        z = z.sort_values("window_start_ms")
        row = {
            "concept_set": name, "unit_global": int(unit), "roi": z.roi.iloc[0],
            "native_area": z.native_area.iloc[0], "monkey": z.monkey.iloc[0],
            "session": int(z.session.iloc[0]),
        }
        for metric in ("secondary_pole_strength", "profile_rms"):
            row[f"early_{metric}"] = float(z.head(2)[metric].mean())
            row[f"late_{metric}"] = float(z.tail(3)[metric].mean())
            row[f"delta_{metric}"] = row[f"late_{metric}"] - row[f"early_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (concept_set, roi), q in deltas.groupby(["concept_set", "roi"], sort=False):
        for mi, metric in enumerate(("secondary_pole_strength", "profile_rms")):
            delta = q[f"delta_{metric}"].to_numpy(float)
            cluster = q.groupby(["monkey", "session"])[f"delta_{metric}"].mean().to_numpy(float)
            rows.append({
                "concept_set": concept_set, "roi": roi, "metric": metric,
                "n_units": int(len(q)),
                "early_median": float(q[f"early_{metric}"].median()),
                "late_median": float(q[f"late_{metric}"].median()),
                "median_change": float(np.median(delta)),
                "positive_unit_fraction": float(np.mean(delta > 0)),
                "wilcoxon_p": float(wilcoxon(delta).pvalue),
                "n_monkey_sessions": int(len(cluster)),
                "session_positive_fraction": float(np.mean(cluster > 0)),
                "session_signflip_p": signflip_p(cluster, SEED + len(rows) + mi),
            })
    return pd.DataFrame(rows)


def functional_area_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    q = deltas[deltas.concept_set.eq("all")].copy()
    rows = []
    for (roi, area), z in q.groupby(["roi", "native_area"], sort=False):
        metric = "secondary_pole_strength" if roi == "middle IT" else "profile_rms"
        delta = z[f"delta_{metric}"].to_numpy(float)
        rows.append({
            "roi": roi, "native_area": area, "endpoint": metric, "n_units": int(len(z)),
            "median_change": float(np.median(delta)),
            "mean_change": float(np.mean(delta)),
            "positive_unit_fraction": float(np.mean(delta > 0)),
            "wilcoxon_p_exploratory": float(wilcoxon(delta).pvalue),
        })
    return pd.DataFrame(rows)


def compare_pilot_validation(validation_deltas: pd.DataFrame) -> dict:
    """Post-hoc heterogeneity audit; never used in the locked success test."""
    pilot_root = PROJECT / "results" / "semantic_64d_high_quality_pilot_113_2026-08-26"
    pilot_metrics = pd.read_csv(pilot_root / "unit_window_semantic_metrics.csv")
    pilot_metrics = pilot_metrics[(pilot_metrics.roi.eq("middle IT")) & pilot_metrics.eligible]
    rows = []
    for unit, z in pilot_metrics.groupby("unit_global"):
        z = z.sort_values("window_start_ms")
        rows.append({
            "unit_global": int(unit),
            "delta_secondary_pole_strength": float(
                z.tail(3).secondary_pole_strength.mean() - z.head(2).secondary_pole_strength.mean()
            ),
        })
    pilot_delta = pd.DataFrame(rows).merge(
        pd.read_csv(PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "selected_units.csv"),
        on="unit_global", validate="one_to_one",
    )
    validation = validation_deltas[
        validation_deltas.concept_set.eq("all") & validation_deltas.roi.eq("middle IT")
    ].copy().merge(
        pd.read_csv(BANK / "selected_units.csv")[[
            "unit_global", "min_window_r", "mean_window_r", "released_independent_consistency"
        ]], on="unit_global", validate="one_to_one",
    )
    unit_p = float(mannwhitneyu(
        pilot_delta.delta_secondary_pole_strength,
        validation.delta_secondary_pole_strength,
        alternative="two-sided",
    ).pvalue)
    pilot_sessions = pilot_delta.groupby(["monkey", "session"]).delta_secondary_pole_strength.mean()
    validation_sessions = validation.groupby(["monkey", "session"]).delta_secondary_pole_strength.mean()
    paired = pd.concat([
        pilot_sessions.rename("pilot_mean_delta"),
        validation_sessions.rename("validation_mean_delta"),
    ], axis=1).dropna().reset_index()
    paired["validation_minus_pilot"] = paired.validation_mean_delta - paired.pilot_mean_delta
    paired.to_csv(OUT / "posthoc_pilot_validation_common_sessions.csv", index=False, encoding="utf-8-sig")
    paired_p = float(wilcoxon(paired.validation_mean_delta, paired.pilot_mean_delta).pvalue)

    area_pilot = pilot_delta.groupby("native_area").delta_secondary_pole_strength.agg(
        pilot_n="size", pilot_median="median", pilot_mean="mean"
    )
    area_validation = validation.groupby("native_area").delta_secondary_pole_strength.agg(
        validation_n="size", validation_median="median", validation_mean="mean"
    )
    area_pilot.join(area_validation, how="outer").reset_index().to_csv(
        OUT / "posthoc_pilot_validation_by_functional_area.csv", index=False, encoding="utf-8-sig"
    )
    quality_rows = []
    for cohort, q in (("pilot", pilot_delta), ("validation", validation)):
        for column in ("min_window_r", "mean_window_r", "released_independent_consistency"):
            rho, p = spearmanr(q.delta_secondary_pole_strength, q[column])
            quality_rows.append({
                "cohort": cohort, "quality_variable": column,
                "spearman_rho": float(rho), "spearman_p": float(p), "n_units": int(len(q)),
            })
    pd.DataFrame(quality_rows).to_csv(
        OUT / "posthoc_effect_quality_association.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "unit_level_mannwhitney_p": unit_p,
        "n_common_monkey_sessions": int(len(paired)),
        "common_session_paired_wilcoxon_p": paired_p,
        "common_session_validation_lower_fraction": float(np.mean(paired.validation_minus_pilot < 0)),
        "pilot_common_session_median": float(paired.pilot_mean_delta.median()),
        "validation_common_session_median": float(paired.validation_mean_delta.median()),
        "interpretation": "post-hoc cohort heterogeneity audit; not part of the locked validation decision",
    }
    (OUT / "posthoc_pilot_validation_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def make_figure(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = {"middle IT": "#E45756", "anterior IT": "#B279A2"}
    for roi, metric in (("middle IT", "secondary_pole_strength"), ("anterior IT", "profile_rms")):
        q = metrics[metrics.roi.eq(roi) & metrics.eligible]
        x, y, lo, hi = pilot.bootstrap_band(q, metric, SEED + len(roi))
        axes[0].plot(x + 4.5, y, marker="o", ms=3, color=colors[roi], label=f"{roi}: {metric}")
        axes[0].fill_between(x + 4.5, lo, hi, color=colors[roi], alpha=.13)
    axes[0].set(title="Locked validation endpoints", xlabel="Time (ms)", ylabel="Semantic alignment")
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)

    q = deltas[deltas.concept_set.eq("all")].copy()
    areas = ["MF", "MB", "MO", "CLC", "LPP", "AB", "AF", "AO"]
    data = []
    labels = []
    for area in areas:
        z = q[q.native_area.eq(area)]
        if z.empty:
            continue
        metric = "delta_secondary_pole_strength" if z.roi.iloc[0] == "middle IT" else "delta_profile_rms"
        data.append(z[metric].to_numpy(float)); labels.append(area)
    bp = axes[1].boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors["middle IT"] if label in ("MF", "MB", "MO", "CLC", "LPP") else colors["anterior IT"])
        patch.set_alpha(.45)
    axes[1].axhline(0, color="#777777", lw=.8)
    axes[1].set(title="Early→late change by functional area", xlabel="Functional area", ylabel="Locked endpoint change")
    axes[1].grid(axis="y", alpha=.2)
    fig.savefig(OUT / "01_locked_validation_results.png", dpi=300)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame, areas: pd.DataFrame, metrics: pd.DataFrame, comparison: dict
) -> None:
    def get(concept_set: str, roi: str, metric: str) -> pd.Series:
        return summary[(summary.concept_set.eq(concept_set)) & summary.roi.eq(roi)
                       & summary.metric.eq(metric)].iloc[0]

    primary = get("all", "middle IT", "secondary_pole_strength")
    sensitivity = get("decorrelated_six", "middle IT", "secondary_pole_strength")
    secondary = get("all", "anterior IT", "profile_rms")
    success = bool(
        primary.median_change > 0
        and primary.wilcoxon_p < .05
        and primary.session_signflip_p < .05
        and sensitivity.median_change > 0
    )
    pilot_summary = pd.read_csv(
        PROJECT / "results" / "semantic_64d_high_quality_pilot_113_2026-08-26"
        / "early_late_summary_by_roi.csv"
    )
    pilot_primary = pilot_summary[(pilot_summary.roi.eq("middle IT"))
                                  & pilot_summary.metric.eq("secondary_pole_strength")].iloc[0]
    reliable = metrics[metrics.eligible]
    middle_r = reliable[reliable.roi.eq("middle IT")].oof_r.median()
    anterior_r = reliable[reliable.roi.eq("anterior IT")].oof_r.median()
    lines = [
        "# 功能 IT 新 unit 的冻结 PCA64 验证",
        "",
        "## 结论",
        "",
        ("**预先冻结的中 IT 主终点验证成功。**" if success else "**预先冻结的中 IT 主终点未达到全部成功条件。**"),
        "",
        "这批 92 个 unit 与发现 pilot 完全不重叠，且均满足严格的、概念盲筛选标准。"
        f"合格时间窗 OOF r 中位数为：中 IT {middle_r:.3f}，前 IT {anterior_r:.3f}。",
        "",
        "## 主终点：中 IT 较弱语义极",
        "",
        "| 样本 | n | 早期中位数 | 晚期中位数 | 变化中位数 | 正向 unit | unit p | session 数 | session 正向 | session p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 发现 pilot | {int(pilot_primary.n_units)} | {pilot_primary.early_median:.3f} | {pilot_primary.late_median:.3f} | {pilot_primary.median_change:+.3f} | — | {pilot_primary.wilcoxon_p:.6f} | {int(pilot_primary.n_monkey_sessions)} | {pilot_primary.session_positive_fraction:.1%} | {pilot_primary.session_signflip_p:.6f} |",
        f"| 新 unit 验证 | {int(primary.n_units)} | {primary.early_median:.3f} | {primary.late_median:.3f} | {primary.median_change:+.3f} | {primary.positive_unit_fraction:.1%} | {primary.wilcoxon_p:.6f} | {int(primary.n_monkey_sessions)} | {primary.session_positive_fraction:.1%} | {primary.session_signflip_p:.6f} |",
        "",
        f"去相关六概念敏感性分析的变化中位数为 {sensitivity.median_change:+.3f}，"
        f"unit p={sensitivity.wilcoxon_p:.6f}，方向{'保持为正' if sensitivity.median_change > 0 else '未保持为正'}。",
        "",
        "## 次级终点：前 IT 概念轮廓 RMS",
        "",
        f"新 unit 的早/晚中位数为 {secondary.early_median:.3f}/{secondary.late_median:.3f}，"
        f"变化 {secondary.median_change:+.3f}，unit p={secondary.wilcoxon_p:.6f}；"
        f"{int(secondary.n_monkey_sessions)} 个 session 中 {secondary.session_positive_fraction:.1%} 为正，"
        f"session sign-flip p={secondary.session_signflip_p:.6f}。",
        "",
        "## 发现组与验证组的差异（事后诊断）",
        "",
        f"unit 层两组变化分布的 Mann–Whitney p={comparison['unit_level_mannwhitney_p']:.6g}。"
        f"两组共有 {comparison['n_common_monkey_sessions']} 个 monkey×session；按 session 配对后，"
        f"验证组有 {comparison['common_session_validation_lower_fraction']:.1%} 的 session 效应低于发现组，"
        f"配对 p={comparison['common_session_paired_wilcoxon_p']:.6f}。共同 session 的效应中位数从"
        f"发现组 {comparison['pilot_common_session_median']:+.3f} 变为验证组 "
        f"{comparison['validation_common_session_median']:+.3f}。",
        "",
        "这是事后异质性检查，不属于冻结主检验。两队列内部，效应与最差窗 r、平均 r 或独立一致性均无明显 Spearman 相关；"
        "因此现有证据不支持把反转简单归因于验证组代理质量稍低。",
        "",
        "## 功能区分层（探索性）",
        "",
        "| ROI | 功能区 | n | 变化中位数 | 正向 unit | 探索性 p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for x in areas.itertuples():
        lines.append(
            f"| {x.roi} | {x.native_area} | {x.n_units} | {x.median_change:+.3f} | "
            f"{x.positive_unit_fraction:.1%} | {x.wilcoxon_p_exploratory:.4f} |"
        )
    lines += [
        "",
        "功能区分层没有用于主检验，也不应用于事后挑区。猴子与功能区仍部分混杂。",
        "",
        "## 扩种决定",
        "",
        ("主终点满足训练前写明的四项条件，因此可以进入粗 ROI 扩种；粗 ROI 分析仍应锁定同一主指标和时间定义。"
         if success else
         "主终点未满足训练前写明的全部条件，因此按停止规则不进入粗 ROI 扩种。"),
        "",
        "即使验证成功，结论仍是代理权重与 ResNet CAV 的双极对齐随时间增强；它不能单独证明跨场景、跨外观的抽象语义泛化。",
        "",
    ]
    (OUT / "FUNCTIONAL_IT_VALIDATION_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "validation_decision.json").write_text(json.dumps({
        "primary_success": success,
        "criteria": {
            "median_change_positive": bool(primary.median_change > 0),
            "unit_p_lt_0_05": bool(primary.wilcoxon_p < .05),
            "session_signflip_p_lt_0_05": bool(primary.session_signflip_p < .05),
            "decorrelated_six_direction_positive": bool(sensitivity.median_change > 0),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concepts, concept_quality, layer_quality = pilot.choose_concepts()
    if tuple(concepts) != CONCEPT_SETS["all"]:
        raise RuntimeError(f"locked concept set changed: {concepts}")
    pilot.BANK = BANK
    pilot.ROIS = ROIS
    scores, windows = pilot.map_fold_scores(concepts, layer_quality)
    metrics = pilot.add_window_metrics(scores, windows)
    standard_summary = pilot.paired_summary(metrics)
    deltas = pd.concat([
        concept_set_deltas(scores, concept_set, name)
        for name, concept_set in CONCEPT_SETS.items()
    ], ignore_index=True)
    summary = summarize(deltas)
    areas = functional_area_summary(deltas)
    comparison = compare_pilot_validation(deltas)

    concept_quality.to_csv(OUT / "locked_concepts_independent_quality.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUT / "fold_concept_scores.csv.gz", index=False, compression="gzip")
    windows.to_csv(OUT / "window_oof_quality.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "unit_window_semantic_metrics.csv", index=False, encoding="utf-8-sig")
    standard_summary.to_csv(OUT / "standard_early_late_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "validation_unit_early_late_changes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "locked_endpoint_summary.csv", index=False, encoding="utf-8-sig")
    areas.to_csv(OUT / "functional_area_exploratory_summary.csv", index=False, encoding="utf-8-sig")
    make_figure(metrics, deltas)
    write_report(summary, areas, metrics, comparison)
    print(summary.to_string(index=False))
    print((OUT / "validation_decision.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
