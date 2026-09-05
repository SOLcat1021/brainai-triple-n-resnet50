"""Response-blind semantic audit for the frozen 113-unit PCA64 pilot.

The unit cohort is never selected or ranked with concept information.  Primary
concepts are selected only from independent CAV validation metrics.  Semantic
scores are recomputed for each of the five image folds so that an effect must
survive refitting the neural readout, not merely look clean in the final fit.
"""

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
from scipy.stats import wilcoxon


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
BANK = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
TCAV = PROJECT / "tcav_native"
BASIS_FILE = PROJECT / "results" / "rank_sweep_extended_2026-08-26" / "response_blind_pca256_basis.npz"
OUT = PROJECT / "results" / "semantic_64d_high_quality_pilot_113_2026-08-26"

LAYERS = (
    "stem", "res2.1", "res2.2", "res2.3", "res3.1", "res3.2", "res3.3", "res3.4",
    "res4.1", "res4.2", "res4.3", "res4.4", "res4.5", "res4.6", "res5.1", "res5.2", "res5.3",
)
CAV_NODES = (
    "stem", "res2_b1", "res2_b2", "res2_b3", "res3_b1", "res3_b2", "res3_b3", "res3_b4",
    "res4_b1", "res4_b2", "res4_b3", "res4_b4", "res4_b5", "res4_b6", "res5_b1", "res5_b2", "res5_b3",
)
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
ROI_FILE = {
    "V1": "v1_proxy_bank_64d.h5", "V2": "v2_proxy_bank_64d.h5",
    "V4": "v4_proxy_bank_64d.h5", "posterior IT": "posterior_it_proxy_bank_64d.h5",
    "middle IT": "middle_it_proxy_bank_64d.h5", "anterior IT": "anterior_it_proxy_bank_64d.h5",
}
ONSET = {"V1": 50, "V2": 60, "V4": 70, "posterior IT": 80, "middle IT": 90, "anterior IT": 100}
RANK = 64
N_FOLDS = 5
N_CAV_REPEATS = 20
SEED = 20260826


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else np.nan


def choose_concepts() -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Choose concepts without looking at neural scores or identities."""
    q = pd.read_csv(TCAV / "native_cav_quality_by_concept_and_layer.csv")
    late = q[q.node.str.startswith(("res4", "res5"))]
    metrics = late.groupby("concept", as_index=False).agg(
        balanced_accuracy_median=("balanced_accuracy_mean", "median"),
        retrieval_auc_median=("retrieval_auc_mean", "median"),
        retrieval_precision20_median=("retrieval_precision_at_20", "median"),
        repeat_axis_cosine_median=("repeat_axis_cosine", "median"),
        permuted_accuracy_median=("permuted_accuracy_mean", "median"),
    )
    keep = metrics[
        (metrics.balanced_accuracy_median >= .85)
        & (metrics.retrieval_auc_median >= .94)
        & (metrics.retrieval_precision20_median >= .90)
        & (metrics.repeat_axis_cosine_median >= .68)
        & (metrics.permuted_accuracy_median.between(.45, .55))
    ].sort_values(["retrieval_auc_median", "balanced_accuracy_median"], ascending=False)
    concepts = keep.concept.tolist()
    if len(concepts) < 6:
        raise RuntimeError(f"quality rule retained only {len(concepts)} concepts")
    # Layer-level gate: a weak concept-layer direction cannot contribute even
    # when the neural model gives that layer a large depth weight.
    layer_q = q.assign(
        valid=lambda x: (
            (x.balanced_accuracy_mean >= .65)
            & (x.retrieval_auc_mean >= .75)
            & (x.repeat_axis_cosine >= .60)
            & (x.permuted_accuracy_mean.between(.45, .55))
        )
    )
    return concepts, keep, layer_q


def map_fold_scores(concepts: list[str], layer_q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    basis = np.load(BASIS_FILE, allow_pickle=True)
    cav = np.load(TCAV / "native_channel_cav_bank.npz", allow_pickle=True)
    # NPZ members are compressed.  Loading them inside the unit/window loop
    # repeatedly would spend minutes decompressing the same small arrays.
    basis_arrays = [basis[f"basis_{li}"][:, :RANK].astype(np.float32) for li in range(len(LAYERS))]
    cav_axes = {
        (concept, node): cav[f"repeated_{concept}_{node}"].astype(np.float32)
        for concept in concepts for node in CAV_NODES
    }
    selected = pd.read_csv(BANK / "selected_units.csv")
    quality_lookup = layer_q.set_index(["concept", "node"]).valid.to_dict()
    rows: list[dict] = []
    window_rows: list[dict] = []

    for roi in ROIS:
        meta = selected[selected.roi.eq(roi)].copy().reset_index(drop=True)
        with h5py.File(BANK / ROI_FILE[roi], "r") as h:
            units = h["unit_global"][:].astype(int)
            if not np.array_equal(units, meta.unit_global.to_numpy(int)):
                raise RuntimeError(f"unit order mismatch for {roi}")
            windows = h["windows_ms"][:].astype(int)
            oof_r = h["oof_r"][:].astype(np.float32)
            weights = h["fold_channel_weights"][:].astype(np.float32)
            stds = h["fold_normalization_std"][:].astype(np.float32)
            gates = h["fold_layer_gate"][:].astype(np.float32)

        for ui, unit in meta.iterrows():
            for wi, (t0, t1) in enumerate(windows):
                window_rows.append({
                    "unit_global": int(unit.unit_global), "roi": roi,
                    "native_area": str(unit.native_area), "monkey": str(unit.monkey),
                    "session": int(unit.session), "unit_type_name": str(unit.unit_type_name),
                    "window_index": wi, "window_start_ms": int(t0), "window_end_ms": int(t1),
                    "oof_r": float(oof_r[wi, ui]), "eligible": bool(t0 >= ONSET[roi] and t1 <= 169),
                })
                for fi in range(N_FOLDS):
                    layer_values: dict[str, list[np.ndarray]] = {c: [] for c in concepts}
                    layer_weights: dict[str, list[float]] = {c: [] for c in concepts}
                    for li, node in enumerate(CAV_NODES):
                        w = weights[fi, wi, ui, li * RANK:(li + 1) * RANK]
                        native_w = basis_arrays[li] @ (w / np.maximum(stds[fi, li], 1e-6))
                        wn = np.linalg.norm(native_w)
                        if wn < 1e-10:
                            continue
                        for concept in concepts:
                            if not quality_lookup.get((concept, node), False):
                                continue
                            axes = cav_axes[(concept, node)]
                            den = np.maximum(np.linalg.norm(axes, axis=1) * wn, 1e-10)
                            layer_values[concept].append((axes @ native_w) / den)
                            layer_weights[concept].append(float(gates[fi, wi, ui, li]))
                    for concept in concepts:
                        if not layer_values[concept]:
                            score_repeats = np.full(N_CAV_REPEATS, np.nan, np.float32)
                        else:
                            lw = np.asarray(layer_weights[concept], np.float32)
                            lw /= max(float(lw.sum()), 1e-8)
                            score_repeats = np.sum(np.stack(layer_values[concept]) * lw[:, None], axis=0)
                        rows.append({
                            "unit_global": int(unit.unit_global), "roi": roi,
                            "native_area": str(unit.native_area), "monkey": str(unit.monkey),
                            "session": int(unit.session), "unit_type_name": str(unit.unit_type_name),
                            "window_index": wi, "window_start_ms": int(t0), "window_end_ms": int(t1),
                            "eligible": bool(t0 >= ONSET[roi] and t1 <= 169), "fold": fi,
                            "concept": concept, "concept_score": float(np.nanmean(score_repeats)),
                            "cav_repeat_sd": float(np.nanstd(score_repeats, ddof=1)),
                        })
    return pd.DataFrame(rows), pd.DataFrame(window_rows)


def add_window_metrics(scores: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    concepts = scores.concept.drop_duplicates().tolist()
    rng = np.random.default_rng(SEED)
    out = []
    for key, group in scores.groupby(["unit_global", "window_index"], sort=False):
        matrix = group.pivot(index="fold", columns="concept", values="concept_score").loc[
            range(N_FOLDS), concepts].to_numpy(float)
        profile = np.nanmean(matrix, axis=0)
        pairs = list(combinations(range(N_FOLDS), 2))
        rel = np.nanmean([cosine(matrix[a], matrix[b]) for a, b in pairs])
        null = []
        for _ in range(200):
            shuffled = np.stack([rng.permutation(row) for row in matrix])
            null.append(np.nanmean([cosine(shuffled[a], shuffled[b]) for a, b in pairs]))
        null = np.asarray(null)
        pos_i, neg_i = int(np.nanargmax(profile)), int(np.nanargmin(profile))
        fold_pos = np.nanargmax(matrix, axis=1)
        fold_neg = np.nanargmin(matrix, axis=1)
        positive = float(profile[pos_i]); negative = float(-profile[neg_i])
        out.append({
            "unit_global": int(key[0]), "window_index": int(key[1]),
            "fold_profile_reliability": float(rel),
            "fold_profile_null_mean": float(np.nanmean(null)),
            # This permutation tests whether the *identity/order* of the ten
            # real concepts is stable across neural folds.  It is not a null
            # for semantic absorption; that requires permuted-label CAV axes.
            "fold_identity_permutation_p": float((1 + np.sum(null >= rel)) / (len(null) + 1)),
            "fold_profile_null_p": float((1 + np.sum(null >= rel)) / (len(null) + 1)),
            "profile_rms": float(np.sqrt(np.nanmean(profile ** 2))),
            "profile_range": float(np.nanmax(profile) - np.nanmin(profile)),
            "positive_strength": positive, "negative_strength": negative,
            "secondary_pole_strength": float(min(positive, negative))
                if positive > 0 and negative > 0 else 0.0,
            "bidirectional_balance": float(min(positive, negative) / max(positive, negative, 1e-8))
                if positive > 0 and negative > 0 else 0.0,
            "top_positive": concepts[pos_i], "top_negative": concepts[neg_i],
            "top_positive_fold_fraction": float(np.mean(fold_pos == pos_i)),
            "top_negative_fold_fraction": float(np.mean(fold_neg == neg_i)),
        })
    metrics = windows.merge(pd.DataFrame(out), on=["unit_global", "window_index"], validate="one_to_one")
    metrics = metrics.sort_values(["unit_global", "window_start_ms"]).reset_index(drop=True)
    metrics["adjacent_profile_similarity"] = np.nan
    metrics["late_reference_similarity"] = np.nan
    for unit, idx in metrics.groupby("unit_global").groups.items():
        ii = np.asarray(list(idx), int)
        unit_scores = scores[scores.unit_global.eq(unit)].groupby(
            ["window_index", "concept"], sort=False).concept_score.mean().unstack("concept").reindex(
                columns=concepts)
        vectors = {int(k): v.to_numpy(float) for k, v in unit_scores.iterrows()}
        eligible = metrics.loc[ii].eligible.to_numpy(bool)
        eligible_rows = ii[eligible]
        late_rows = eligible_rows[-3:]
        late_ref = np.nanmean([vectors[int(metrics.loc[j, "window_index"])] for j in late_rows], axis=0)
        previous = None
        for j in eligible_rows:
            vec = vectors[int(metrics.loc[j, "window_index"])]
            metrics.loc[j, "late_reference_similarity"] = cosine(vec, late_ref)
            if previous is not None:
                metrics.loc[j, "adjacent_profile_similarity"] = cosine(previous, vec)
            previous = vec
    return metrics


def paired_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = metrics[metrics.eligible].copy()
    for roi, group in eligible.groupby("roi", sort=False):
        per_unit = []
        for unit, q in group.groupby("unit_global"):
            q = q.sort_values("window_start_ms")
            early, late = q.head(2), q.tail(3)
            row = {
                "unit_global": unit,
                "monkey": q.monkey.iloc[0], "session": int(q.session.iloc[0]),
                "early_fold_reliability": early.fold_profile_reliability.mean(),
                "late_fold_reliability": late.fold_profile_reliability.mean(),
                "early_profile_rms": early.profile_rms.mean(),
                "late_profile_rms": late.profile_rms.mean(),
                "early_adjacent_similarity": early.adjacent_profile_similarity.mean(),
                "late_adjacent_similarity": late.adjacent_profile_similarity.mean(),
                "median_oof_r": q.oof_r.median(),
                "identity_stable_window_fraction": np.mean(q.fold_identity_permutation_p < .05),
                "late_positive_top_stability": q.tail(3).top_positive.value_counts(normalize=True).iloc[0],
                "late_negative_top_stability": q.tail(3).top_negative.value_counts(normalize=True).iloc[0],
            }
            for metric in ("secondary_pole_strength", "bidirectional_balance",
                           "positive_strength", "negative_strength"):
                row[f"early_{metric}"] = early[metric].mean()
                row[f"late_{metric}"] = late[metric].mean()
            per_unit.append(row)
        u = pd.DataFrame(per_unit)
        for metric in ("fold_reliability", "profile_rms", "adjacent_similarity",
                       "secondary_pole_strength", "bidirectional_balance",
                       "positive_strength", "negative_strength"):
            a, b = u[f"early_{metric}"].to_numpy(float), u[f"late_{metric}"].to_numpy(float)
            valid = np.isfinite(a) & np.isfinite(b)
            try:
                p = float(wilcoxon(b[valid], a[valid], alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
            cluster = u.assign(delta=b - a).groupby(["monkey", "session"]).delta.mean().to_numpy(float)
            # Session-level sign-flip test limits pseudo-replication.  Enumerate
            # small designs exactly and use a fixed Monte Carlo null otherwise.
            observed = abs(float(np.mean(cluster)))
            if len(cluster) <= 18:
                null = np.asarray([
                    abs(float(np.mean(cluster * np.asarray(signs))))
                    for signs in product((-1, 1), repeat=len(cluster))
                ])
            else:
                rng = np.random.default_rng(SEED + len(rows))
                signs = rng.choice((-1, 1), size=(200_000, len(cluster)))
                null = np.abs((signs * cluster[None]).mean(axis=1))
            rows.append({
                "roi": roi, "metric": metric, "n_units": int(valid.sum()),
                "early_median": float(np.nanmedian(a)), "late_median": float(np.nanmedian(b)),
                "median_change": float(np.nanmedian(b - a)), "wilcoxon_p": p,
                "n_monkey_sessions": int(len(cluster)),
                "session_positive_fraction": float(np.mean(cluster > 0)),
                "session_signflip_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
                "median_oof_r": float(u.median_oof_r.median()),
                "median_identity_stable_window_fraction": float(u.identity_stable_window_fraction.median()),
                "median_late_positive_top_stability": float(u.late_positive_top_stability.median()),
                "median_late_negative_top_stability": float(u.late_negative_top_stability.median()),
            })
    out = pd.DataFrame(rows)
    out["fdr_bh_within_metric"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        ii = np.asarray(list(idx), int)
        p = out.loc[ii, "wilcoxon_p"].to_numpy(float)
        order = np.argsort(p)
        ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
        adjusted = np.empty_like(ranked); adjusted[order] = ranked
        out.loc[ii, "fdr_bh_within_metric"] = adjusted
    return out


def bootstrap_band(q: pd.DataFrame, column: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = q.pivot(index="unit_global", columns="window_start_ms", values=column).sort_index(axis=1)
    values = pivot.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.stack([
        np.nanmedian(values[rng.integers(0, len(values), len(values))], axis=0) for _ in range(1000)
    ])
    return pivot.columns.to_numpy(), np.nanmedian(values, axis=0), np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def make_figures(scores: pd.DataFrame, metrics: pd.DataFrame, selected: pd.DataFrame, concepts: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(ROIS, ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756", "#B279A2"]))
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    for ri, roi in enumerate(ROIS):
        q = metrics[(metrics.roi.eq(roi)) & metrics.eligible]
        x, y, lo, hi = bootstrap_band(q, "fold_profile_reliability", SEED + ri)
        axes[0].plot(x + 4.5, y, marker="o", ms=3, label=roi, color=colors[roi])
        axes[0].fill_between(x + 4.5, lo, hi, alpha=.12, color=colors[roi])
        x, y, lo, hi = bootstrap_band(q, "profile_rms", SEED + 20 + ri)
        axes[1].plot(x + 4.5, y, marker="o", ms=3, label=roi, color=colors[roi])
        axes[1].fill_between(x + 4.5, lo, hi, alpha=.12, color=colors[roi])
    axes[0].set(title="Across-fold semantic profile reliability", xlabel="Time from stimulus onset (ms)", ylabel="Mean pairwise cosine")
    axes[1].set(title="Signed concept-profile strength", xlabel="Time from stimulus onset (ms)", ylabel="RMS alignment")
    for ax in axes:
        ax.axhline(0, color="#999999", lw=.7); ax.grid(alpha=.2)
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(OUT / "01_semantic_formation_and_reliability.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    for ri, roi in enumerate(ROIS):
        q = metrics[(metrics.roi.eq(roi)) & metrics.eligible]
        for ax, col in zip(axes, ("positive_strength", "negative_strength")):
            x, y, lo, hi = bootstrap_band(q, col, SEED + 40 + ri)
            ax.plot(x + 4.5, y, marker="o", ms=3, label=roi, color=colors[roi])
            ax.fill_between(x + 4.5, lo, hi, alpha=.10, color=colors[roi])
    axes[0].set(title="Strongest positive concept alignment", xlabel="Time (ms)", ylabel="Cosine alignment")
    axes[1].set(title="Strongest negative concept alignment", xlabel="Time (ms)", ylabel="Absolute cosine alignment")
    for ax in axes: ax.grid(alpha=.2)
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(OUT / "02_positive_negative_strength.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for ri, roi in enumerate(ROIS):
        q = metrics[(metrics.roi.eq(roi)) & metrics.eligible]
        x, y, lo, hi = bootstrap_band(q, "secondary_pole_strength", SEED + 60 + ri)
        ax.plot(x + 4.5, y, marker="o", ms=3, label=roi, color=colors[roi])
        ax.fill_between(x + 4.5, lo, hi, alpha=.11, color=colors[roi])
    ax.set(title="Emergence of the weaker signed semantic pole",
           xlabel="Time from stimulus onset (ms)",
           ylabel="min(strongest positive, strongest negative alignment)")
    ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8)
    fig.savefig(OUT / "05_secondary_semantic_pole_by_time.png", dpi=300)
    plt.close(fig)

    eligible_scores = scores[scores.eligible].groupby(
        ["unit_global", "roi", "native_area", "window_start_ms", "concept"], as_index=False).concept_score.mean()
    late = eligible_scores[eligible_scores.window_start_ms >= 140]
    functional = ["MF", "MB", "MO", "CLC", "LPP", "AB", "AF", "AO"]
    area_profile = late[late.native_area.isin(functional)].groupby(
        ["native_area", "concept"]).concept_score.mean().unstack("concept").reindex(functional)[concepts]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    lim = float(np.nanquantile(np.abs(area_profile.to_numpy()), .98))
    im = ax.imshow(area_profile, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(concepts)), concepts, rotation=40, ha="right")
    ax.set_yticks(range(len(functional)), functional)
    ax.set(title="Late (140–169 ms) signed semantic profiles in functional IT areas", xlabel="Predeclared high-quality concept axis", ylabel="Functional area")
    fig.colorbar(im, ax=ax, label="Mean signed CAV alignment")
    fig.savefig(OUT / "03_functional_it_late_profiles.png", dpi=300)
    plt.close(fig)

    # One response-blind representative per coarse ROI: highest predeclared
    # min-window-r, never selected by semantic appearance.
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for ax, roi in zip(axes.flat, ROIS):
        unit = int(selected[selected.roi.eq(roi)].sort_values("min_window_r", ascending=False).iloc[0].unit_global)
        q = eligible_scores[eligible_scores.unit_global.eq(unit)]
        for concept in concepts:
            z = q[q.concept.eq(concept)].sort_values("window_start_ms")
            ax.plot(z.window_start_ms + 4.5, z.concept_score, marker="o", ms=2.5, lw=1.2, label=concept)
        ax.axhline(0, color="#777777", lw=.7); ax.grid(alpha=.2)
        ax.set(title=f"{roi} — unit {unit}", xlabel="Time (ms)", ylabel="Signed alignment")
    axes[-1, -1].legend(ncol=2, fontsize=7)
    fig.suptitle("Response-blind representative units: PCA64 semantic trajectories", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "04_response_blind_representative_trajectories.png", dpi=300)
    plt.close(fig)


def write_report(concepts: list[str], summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    eligible = metrics[metrics.eligible]
    rel = summary[summary.metric.eq("fold_reliability")].copy()
    lines = [
        "# 113 个高质量 unit 的 PCA64 语义 pilot",
        "",
        "## 设计约束",
        "",
        "- 113 个 unit 由预先冻结的可靠性与 PCA8 OOF 指标选出；概念结果未参与筛选。",
        "- 64 维代理逐 unit、逐 10 ms 窗口独立拟合；分析仅使用各脑区预注册可靠起点至 169 ms。",
        "- 主概念集合仅依据独立 CAV 质量选择，不依据神经结果。",
        "- 每个语义分数在五个图像折上分别重估；跨折一致性是主要稳健性指标。",
        "",
        "## 主概念轴",
        "",
        ", ".join(concepts),
        "",
        "## 核心数值",
        "",
        "| ROI | unit 数 | 可靠窗 OOF r 中位数 | 早期跨折一致性 | 晚期跨折一致性 | 晚-早变化 | Wilcoxon p | 跨折概念身份置换 p<.05 的窗比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rel.iterrows():
        n = eligible[eligible.roi.eq(row.roi)].unit_global.nunique()
        lines.append(
            f"| {row.roi} | {n} | {row.median_oof_r:.3f} | {row.early_median:.3f} | "
            f"{row.late_median:.3f} | {row.median_change:+.3f} | {row.wilcoxon_p:.4f} | "
            f"{row.median_identity_stable_window_fraction:.2f} |"
        )
    middle_secondary = summary[(summary.roi.eq("middle IT")) &
                               (summary.metric.eq("secondary_pole_strength"))].iloc[0]
    anterior_strength = summary[(summary.roi.eq("anterior IT")) &
                                (summary.metric.eq("profile_rms"))].iloc[0]
    lines += [
        "",
        "## 当前最明显的候选现象",
        "",
        f"1. **中 IT 的第二语义极随时间增强。** 正、负概念端中较弱一端的强度中位数由 "
        f"{middle_secondary.early_median:.3f} 增至 {middle_secondary.late_median:.3f}，"
        f"unit 配对检验 p={middle_secondary.wilcoxon_p:.6f}，同指标六区内 FDR q="
        f"{middle_secondary.fdr_bh_within_metric:.6f}；按猴子×session 聚类后，"
        f"{middle_secondary.session_positive_fraction:.1%} 的 session 为正向，sign-flip p="
        f"{middle_secondary.session_signflip_p:.6f}。这更像后期出现较完整的双极语义轮廓，而不是整体权重单纯变大。",
        f"2. **前 IT 的整体概念轮廓增强。** RMS 对齐由 {anterior_strength.early_median:.3f} 增至 "
        f"{anterior_strength.late_median:.3f}，unit 配对 p={anterior_strength.wilcoxon_p:.6f}，"
        f"六区内 FDR q={anterior_strength.fdr_bh_within_metric:.6f}；session-level sign-flip p="
        f"{anterior_strength.session_signflip_p:.4f}。目前应视为次级候选，因为只有 8 个独立 session。",
        "3. **未发现普遍的“从不稳定到稳定”。** 各区在预注册可靠起点处跨折概念轮廓已很稳定，之后大多平台化；"
        "因此 pilot 更支持‘概念身份较早稳定、正负两极的相对强度继续演化’，而非所有区域统一的晚期语义诞生。",
    ]
    lines += [
        "",
        "## 解释边界",
        "",
        "- 跨折训练集两两仍有约 75% 图像重叠，因此跨折一致性不是完全独立复现，数值可能偏乐观。",
        "- CAV 表示 ResNet-50 中可线性分离的概念方向；与神经读出对齐不等于神经元具备抽象语义。",
        "- 该 pilot 为高预测力富集样本，只适合机制发现和决定是否扩种，不能估计全体 unit 的现象比例。",
        "- 后续若扩种，应锁定当前概念集合、统计量、时间范围和判断阈值，并回到未经预测力筛选的可靠 unit 总体进行普遍性估计。",
        "",
    ]
    (OUT / "PILOT_113_PCA64_SEMANTIC_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concepts, concept_quality, layer_quality = choose_concepts()
    concept_quality.to_csv(OUT / "selected_concepts_independent_quality.csv", index=False, encoding="utf-8-sig")
    layer_quality[layer_quality.concept.isin(concepts)].to_csv(
        OUT / "selected_concept_layer_quality.csv", index=False, encoding="utf-8-sig")
    scores, windows = map_fold_scores(concepts, layer_quality)
    scores.to_csv(OUT / "fold_concept_scores.csv.gz", index=False, compression="gzip")
    windows.to_csv(OUT / "window_oof_quality.csv", index=False, encoding="utf-8-sig")
    metrics = add_window_metrics(scores, windows)
    metrics.to_csv(OUT / "unit_window_semantic_metrics.csv", index=False, encoding="utf-8-sig")
    summary = paired_summary(metrics)
    summary.to_csv(OUT / "early_late_summary_by_roi.csv", index=False, encoding="utf-8-sig")
    selected = pd.read_csv(BANK / "selected_units.csv")
    # Sensitivity of the main middle-IT result to concept choice.  This is
    # reported transparently as post-hoc robustness, not used to choose units.
    sensitivity_rows = []
    fold_mean = scores[scores.eligible].groupby(
        ["unit_global", "roi", "window_start_ms", "concept"], as_index=False).concept_score.mean()
    concept_sets = {"all": concepts,
                    "decorrelated_six": [c for c in ("striped", "cat", "sky", "car", "head", "food") if c in concepts]}
    concept_sets.update({f"leave_out_{c}": [x for x in concepts if x != c] for c in concepts})
    for name, subset in concept_sets.items():
        q = fold_mean[(fold_mean.roi.eq("middle IT")) & fold_mean.concept.isin(subset)]
        wide = q.pivot(index=["unit_global", "window_start_ms"], columns="concept", values="concept_score")
        secondary = np.minimum(wide.max(axis=1), -wide.min(axis=1)).rename("secondary").reset_index()
        delta = []
        for _, z in secondary.groupby("unit_global"):
            z = z.sort_values("window_start_ms")
            delta.append(z.tail(3).secondary.mean() - z.head(2).secondary.mean())
        delta = np.asarray(delta)
        sensitivity_rows.append({
            "concept_set": name, "n_concepts": len(subset),
            "median_change": float(np.median(delta)), "positive_unit_fraction": float(np.mean(delta > 0)),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
        })
    pd.DataFrame(sensitivity_rows).to_csv(
        OUT / "middle_it_secondary_pole_sensitivity.csv", index=False, encoding="utf-8-sig")
    make_figures(scores, metrics, selected, concepts)
    write_report(concepts, summary, metrics)
    audit = {
        "selection_uses_concepts": False,
        "trained_units": int(selected.unit_global.nunique()),
        "concepts": concepts,
        "concept_selection": {
            "source": "independent native CAV validation metrics on res4/res5 nodes",
            "balanced_accuracy_median_min": .85,
            "retrieval_auc_median_min": .94,
            "retrieval_precision_at_20_median_min": .90,
            "repeat_axis_cosine_median_min": .68,
            "permuted_accuracy_range": [.45, .55],
        },
        "layer_quality_gate": {
            "balanced_accuracy_min": .65, "retrieval_auc_min": .75,
            "repeat_axis_cosine_min": .60, "permuted_accuracy_range": [.45, .55],
        },
        "reliable_onsets_ms": ONSET,
        "analysis_end_ms": 169,
        "folds": N_FOLDS,
        "cav_repeats": N_CAV_REPEATS,
        "fold_overlap_warning": "five 80/20 folds have overlapping training sets; reliability is not independent replication",
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("concepts:", concepts)
    print(summary.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
