"""Functional-IT temporal heterogeneity and two follow-up dynamic analyses.

All temporal modes are learned without concept labels.  Semantics and network
depth are attached only after response-defined modes have been frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_low_rank_temporal_dynamics as pilot
import validate_low_rank_temporal_dynamics as validation


PROJECT = pilot.PROJECT
OUT = PROJECT / "results" / "functional_it_and_temporal_directions_2026-08-27"
COHORT = PROJECT / "cohorts" / "high_quality_64d_pilot_2026-08-26" / "pilot_functional_it_8_each.csv"
VALIDATION = pilot.OUT / "disjoint_half_unit_temporal_validation.csv"
MODE_SEMANTICS = pilot.OUT / "unit_temporal_mode_semantics.csv"
AREAS = ("MF", "MB", "MO", "CLC", "LPP", "AB", "AF", "AO")
TARGET_ROIS = ("V2", "V4", "posterior IT")
N_PERM = 50_000
SEED = 20260827 + 171
N_LAYERS = 17
RANK = 64


def reduce_block(x: np.ndarray, max_pc: int = 12) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med[None])
    sd = x.std(axis=0)
    keep = sd > 1e-8
    x = (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    variance = s ** 2
    k90 = int(np.searchsorted(np.cumsum(variance) / max(variance.sum(), 1e-12), .90) + 1)
    k = min(max_pc, max(2, k90), len(s))
    scores = u[:, :k] * s[:k]
    return scores / max(np.sqrt(np.mean(scores ** 2)), 1e-12)


def group_r2(x: np.ndarray, labels: np.ndarray) -> float:
    center = x.mean(axis=0)
    total = float(np.sum((x - center) ** 2))
    between = 0.0
    for label in np.unique(labels):
        q = x[labels == label]
        between += len(q) * float(np.sum((q.mean(axis=0) - center) ** 2))
    return between / max(total, 1e-12)


def permute_labels(labels: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    out = labels.copy()
    if strata is None:
        rng.shuffle(out)
        return out
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        out[idx] = rng.permutation(out[idx])
    return out


def permutation_test(
    x: np.ndarray, labels: np.ndarray, strata: np.ndarray | None, seed: int
) -> dict[str, float]:
    observed = group_r2(x, labels)
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM, float)
    changed = np.empty(N_PERM, float)
    for i in range(N_PERM):
        shuffled = permute_labels(labels, strata, rng)
        null[i] = group_r2(x, shuffled)
        changed[i] = np.mean(shuffled != labels)
    return {
        "observed_r2": observed,
        "null_mean_r2": float(null.mean()),
        "observed_over_null": observed / max(float(null.mean()), 1e-12),
        "permutation_p": float((1 + np.sum(null >= observed)) / (N_PERM + 1)),
        "median_fraction_labels_changed": float(np.median(changed)),
    }


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def functional_features() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cohort = pd.read_csv(COHORT).sort_values("analysis_group").reset_index(drop=True)
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    units = cohort.unit_global.to_numpy(int)
    y_all = np.asarray(responses[:, :, units], np.float32).transpose(2, 1, 0)  # unit,image,time
    baseline = y_all[:, :, windows[:, 1] < 0].mean(axis=(1, 2))
    evoked = y_all.mean(axis=1) - baseline[:, None]
    evoked /= np.maximum(np.linalg.norm(evoked, axis=1, keepdims=True), 1e-8)

    post = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    covariance = []
    tri = np.triu_indices(len(post))
    for y in y_all[:, :, post]:
        c = np.corrcoef(y, rowvar=False)
        covariance.append(np.nan_to_num(c[tri], nan=0.0))
    covariance = np.asarray(covariance)

    val = pd.read_csv(VALIDATION).groupby("unit_global", as_index=False).mean(numeric_only=True)
    sem = pd.read_csv(MODE_SEMANTICS)
    sem1 = sem[sem["mode"].eq(1)].copy()
    sem2 = sem[sem["mode"].eq(2)][["unit_global", "mode_centroid_ms", "semantic_profile_rms"]].rename(
        columns={"mode_centroid_ms": "mode2_centroid_ms", "semantic_profile_rms": "mode2_semantic_rms"}
    )
    model = cohort[["unit_global"]].merge(val, on="unit_global", how="left").merge(
        sem1[["unit_global", "mode_centroid_ms", "semantic_profile_rms",
              "mode12_semantic_cosine", "mode2_vs_mode1_derivative_abs_r"]], on="unit_global", how="left"
    ).merge(sem2, on="unit_global", how="left")
    model_cols = [
        "dynamic_minus_rank1", "dynamic_minus_warped_rank1", "dynamic_minus_derivative_rank2",
        "disjoint_subspace_stability", "mode1_coefficient_r", "mode2_coefficient_r",
        "mode_centroid_ms", "mode2_centroid_ms", "mode2_vs_mode1_derivative_abs_r",
    ]
    blocks = {
        "evoked_curve": reduce_block(evoked),
        "stimulus_temporal_covariance": reduce_block(covariance),
        "model_descriptors": reduce_block(model[model_cols].to_numpy(float)),
    }
    blocks["combined_balanced"] = np.concatenate(
        [v / np.sqrt(v.shape[1]) for v in blocks.values()], axis=1
    )
    return cohort, blocks


def functional_permutations(cohort: pd.DataFrame, blocks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for scope, mask in (
        ("all_8_areas", np.ones(len(cohort), bool)),
        ("middle_IT_5_areas", cohort.roi.eq("middle IT").to_numpy()),
        ("anterior_IT_3_areas", cohort.roi.eq("anterior IT").to_numpy()),
    ):
        q = cohort.loc[mask].reset_index(drop=True)
        labels = q.analysis_group.to_numpy(str)
        for view, full_x in blocks.items():
            x = full_x[mask]
            for null_name, strata in (
                ("unrestricted", None),
                ("monkey_conditioned", q.monkey.to_numpy(str)),
                ("monkey_unit_type_conditioned", (q.monkey.astype(str) + "|" + q.unit_type_name.astype(str)).to_numpy()),
            ):
                result = permutation_test(x, labels, strata, SEED + len(rows))
                rows.append({"scope": scope, "feature_view": view, "null": null_name,
                             "n_units": len(q), "n_areas": q.analysis_group.nunique(), **result})
    return pd.DataFrame(rows)


def pairwise_functional(cohort: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 909)
    for roi in ("middle IT", "anterior IT"):
        qidx = np.flatnonzero(cohort.roi.eq(roi).to_numpy())
        labels = cohort.analysis_group.to_numpy(str)[qidx]
        monkeys = cohort.monkey.to_numpy(str)[qidx]
        xx = x[qidx]
        area_list = sorted(np.unique(labels))
        for ai, a in enumerate(area_list):
            for b in area_list[ai + 1:]:
                idx = np.flatnonzero(np.isin(labels, [a, b]))
                yy = labels[idx]
                observed = float(np.sum((xx[idx][yy == a].mean(0) - xx[idx][yy == b].mean(0)) ** 2))
                null = np.empty(N_PERM)
                for pi in range(N_PERM):
                    shuffled = yy.copy()
                    for monkey in np.unique(monkeys[idx]):
                        jj = np.flatnonzero(monkeys[idx] == monkey)
                        shuffled[jj] = rng.permutation(shuffled[jj])
                    null[pi] = np.sum((xx[idx][shuffled == a].mean(0) - xx[idx][shuffled == b].mean(0)) ** 2)
                rows.append({"roi": roi, "area_a": a, "area_b": b, "distance2": observed,
                             "monkey_conditioned_p": float((1 + np.sum(null >= observed)) / (N_PERM + 1)),
                             "null_mean_distance2": float(np.mean(null))})
    result = pd.DataFrame(rows)
    result["fdr_q"] = bh_fdr(result.monkey_conditioned_p.to_numpy())
    return result.sort_values("monkey_conditioned_p").reset_index(drop=True)


def session_confound_audit(cohort: pd.DataFrame, blocks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for monkey in ("M1", "M3"):
        mask = (cohort.roi.eq("middle IT") & cohort.monkey.eq(monkey)).to_numpy()
        q = cohort.loc[mask].reset_index(drop=True)
        if q.analysis_group.nunique() < 2:
            continue
        for view, full_x in blocks.items():
            x, labels = full_x[mask], q.analysis_group.to_numpy(str)
            for null_name, strata in (("within_monkey", None),
                                      ("session_conditioned", q.session.astype(str).to_numpy())):
                result = permutation_test(x, labels, strata, SEED + 3000 + len(rows))
                rows.append({"monkey": monkey, "feature_view": view, "null": null_name,
                             "n_units": len(q), "n_areas": q.analysis_group.nunique(), **result})
    return pd.DataFrame(rows)


def orthonormal_columns(columns: list[np.ndarray]) -> np.ndarray:
    x = np.column_stack(columns)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > (s[0] * 1e-7)]


def nonwarp_mode(curves: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    h1 = pilot.svd_basis(curves, lam, 1)[:, 0]
    nuisance = orthonormal_columns([h1, np.gradient(h1), np.gradient(np.gradient(h1))])
    residual = curves - (curves @ nuisance) @ nuisance.T
    g = pilot.svd_basis(residual, lam, 1)[:, 0]
    g = g - nuisance @ (nuisance.T @ g)
    g /= max(np.linalg.norm(g), 1e-12)
    return nuisance, g


def nonwarp_and_depth() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    tidx = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    configs = validation.modal_configs().set_index("roi")
    order = np.random.default_rng(validation.SEED).permutation(pilot.N_IMAGES)
    halves = (np.sort(order[:500]), np.sort(order[500:]))
    semantic = pd.read_csv(pilot.SEMANTIC)
    semantic = semantic[semantic.window_start_ms.isin(windows[tidx, 0])].groupby(
        ["unit_global", "roi", "window_start_ms", "concept"], as_index=False
    ).concept_score.mean()
    eval_rows, depth_rows, sem_rows = [], [], []

    for roi in TARGET_ROIS:
        lam = float(configs.loc[roi, "lambda"])
        meta, y, pred = pilot.load_roi(roi, responses, tidx)
        full_modes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ui, unit in meta.iterrows():
            half_modes = []
            for split, train in enumerate(halves):
                test = halves[1 - split]
                mean = y[train, :, ui].mean(0)
                centered = y[train, :, ui] - mean[None]
                nuisance, g = nonwarp_mode(centered, lam)
                half_modes.append(g)
                target, raw = y[test, :, ui], pred[test, :, ui]
                nuisance_pred = pilot.project_prediction(raw, mean, nuisance)
                extended = orthonormal_columns([*nuisance.T, g])
                extended_pred = pilot.project_prediction(raw, mean, extended)
                base_r2 = pilot.score_curve(target, nuisance_pred, mean)["stimulus_r2"]
                ext_r2 = pilot.score_curve(target, extended_pred, mean)["stimulus_r2"]
                target_c = (target - mean[None]) @ g
                pred_c = (raw - mean[None]) @ g
                eval_rows.append({
                    "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                    "monkey": str(unit.monkey), "session": int(unit.session), "split": split,
                    "nuisance_rank": nuisance.shape[1], "nuisance_r2": base_r2,
                    "nuisance_plus_nonwarp_r2": ext_r2, "nonwarp_delta_r2": ext_r2 - base_r2,
                    "nonwarp_coefficient_r": pilot.corr(target_c, pred_c),
                })
            stability = abs(pilot.corr(half_modes[0], half_modes[1]))
            for row in eval_rows[-2:]:
                row["nonwarp_mode_half_stability"] = stability

            mean = y[:, :, ui].mean(0)
            nuisance, g = nonwarp_mode(y[:, :, ui] - mean[None], lam)
            h1 = pilot.svd_basis(y[:, :, ui] - mean[None], lam, 1)[:, 0]
            full_modes[int(unit.unit_global)] = (h1, g)
            q = semantic[(semantic.roi.eq(roi)) & semantic.unit_global.eq(int(unit.unit_global))]
            matrix = q.pivot(index="window_start_ms", columns="concept", values="concept_score").reindex(
                index=windows[tidx, 0]).sort_index(axis=1).to_numpy(float)
            p1, pg = h1 @ matrix, g @ matrix
            sem_rows.append({
                "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                "monkey": str(unit.monkey), "session": int(unit.session),
                "main_nonwarp_semantic_cosine": pilot.cosine(p1, pg),
                "nonwarp_semantic_rms": float(np.sqrt(np.mean(pg ** 2))),
            })

        with h5py.File(pilot.BANK / pilot.ROI_FILE[roi], "r") as h:
            weights = h["fold_channel_weights"][:, tidx].astype(np.float32)
            ids = h["unit_global"][:].astype(int)
        for ui, unit in meta.iterrows():
            h1, g = full_modes[int(unit.unit_global)]
            w = weights[:, :, ui].reshape(weights.shape[0], len(tidx), N_LAYERS, RANK)
            for fold in range(w.shape[0]):
                profiles = []
                centroids = []
                for mode in (h1, g):
                    projected = np.einsum("t,tlr->lr", mode, w[fold])
                    energy = np.sum(projected ** 2, axis=1)
                    energy /= max(float(energy.sum()), 1e-12)
                    profiles.append(energy)
                    centroids.append(float(energy @ np.arange(N_LAYERS)))
                depth_rows.append({
                    "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                    "monkey": str(unit.monkey), "session": int(unit.session), "fold": fold,
                    "main_depth_centroid": centroids[0], "nonwarp_depth_centroid": centroids[1],
                    "nonwarp_minus_main_depth": centroids[1] - centroids[0],
                    "layer_profile_cosine": pilot.cosine(profiles[0], profiles[1]),
                })
    return pd.DataFrame(eval_rows), pd.DataFrame(depth_rows), pd.DataFrame(sem_rows)


def direction_summary(evaluation: pd.DataFrame, depth: pd.DataFrame, sem: pd.DataFrame) -> pd.DataFrame:
    ev = evaluation.groupby(["roi", "unit_global", "monkey", "session"], as_index=False).mean(numeric_only=True)
    de = depth.groupby(["roi", "unit_global", "monkey", "session"], as_index=False).mean(numeric_only=True)
    rows = []
    for roi in TARGET_ROIS:
        q, d, s = ev[ev.roi.eq(roi)], de[de.roi.eq(roi)], sem[sem.roi.eq(roi)]
        cluster = q.groupby(["monkey", "session"]).nonwarp_delta_r2.mean().to_numpy(float)
        depth_cluster = d.groupby(["monkey", "session"]).nonwarp_minus_main_depth.mean().to_numpy(float)
        rows.append({
            "roi": roi, "n_units": len(q),
            "median_nonwarp_delta_r2": float(q.nonwarp_delta_r2.median()),
            "nonwarp_positive_fraction": float(np.mean(q.nonwarp_delta_r2 > 0)),
            "nonwarp_session_signflip_p": validation.signflip_p(cluster, SEED + len(rows)),
            "median_nonwarp_coefficient_r": float(q.nonwarp_coefficient_r.median()),
            "median_nonwarp_half_stability": float(q.nonwarp_mode_half_stability.median()),
            "median_nonwarp_minus_main_depth": float(d.nonwarp_minus_main_depth.median()),
            "depth_session_signflip_p": validation.signflip_p(depth_cluster, SEED + 100 + len(rows)),
            "median_layer_profile_cosine": float(d.layer_profile_cosine.median()),
            "median_main_nonwarp_semantic_cosine": float(s.main_nonwarp_semantic_cosine.median()),
            "median_nonwarp_semantic_rms": float(s.nonwarp_semantic_rms.median()),
        })
    return pd.DataFrame(rows)


def make_figure(functional: pd.DataFrame, direction: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    q = functional[(functional.scope.eq("middle_IT_5_areas")) &
                   functional.null.eq("monkey_conditioned")].set_index("feature_view")
    views = ["evoked_curve", "stimulus_temporal_covariance", "model_descriptors", "combined_balanced"]
    axes[0].bar(np.arange(4), q.reindex(views).observed_over_null, color="#4C78A8")
    axes[0].axhline(1, color="#777", lw=1)
    axes[0].set_xticks(np.arange(4), ["evoked", "covariance", "model", "combined"], rotation=25)
    axes[0].set(title="Middle-IT functional areas\nvs monkey-matched random groups", ylabel="Observed / null dispersion")
    x = np.arange(len(direction))
    axes[1].bar(x, direction.median_nonwarp_delta_r2, color="#F58518")
    axes[1].axhline(0, color="#777", lw=1)
    axes[1].set_xticks(x, direction.roi, rotation=25)
    axes[1].set(title="Derivative-orthogonal mode", ylabel="Held-out median ΔR²")
    axes[2].bar(x, direction.median_nonwarp_minus_main_depth, color="#54A24B")
    axes[2].axhline(0, color="#777", lw=1)
    axes[2].set_xticks(x, direction.roi, rotation=25)
    axes[2].set(title="Network depth of non-warp mode", ylabel="Layer centroid difference")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "01_functional_it_and_two_directions.png", dpi=300)
    plt.close(fig)


def write_report(functional: pd.DataFrame, pairs: pd.DataFrame, confound: pd.DataFrame,
                 direction: pd.DataFrame) -> None:
    lines = [
        "# IT功能区时间动力学与两个后续方向", "",
        "## 1. 功能区差异是否超过随机unit集群", "",
        "置换统计量是功能区标签解释的多变量时间动力学方差（R²）。随机组严格保持每组8个unit；",
        "`monkey_conditioned` 进一步保持每只猴内部的功能区样本构成。", "",
        "| 范围 | 特征 | 随机模型 | observed R² | 随机均值 | 倍数 | p | 可交换标签比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in functional.itertuples():
        lines.append(f"| {x.scope} | {x.feature_view} | {x.null} | {x.observed_r2:.3f} | "
                     f"{x.null_mean_r2:.3f} | {x.observed_over_null:.2f} | {x.permutation_p:.5f} | "
                     f"{x.median_fraction_labels_changed:.1%} |")
    lines += ["", "### 最强的功能区配对（猴子条件置换）", "",
              "| ROI | 配对 | 距离² | 随机均值 | p | FDR q |", "|---|---|---:|---:|---:|---:|"]
    for x in pairs.head(12).itertuples():
        lines.append(f"| {x.roi} | {x.area_a}–{x.area_b} | {x.distance2:.3f} | "
                     f"{x.null_mean_distance2:.3f} | {x.monkey_conditioned_p:.5f} | {x.fdr_q:.5f} |")
    lines += ["", "### session混杂复核", "",
              "猴子条件置换并不等于跨猴复现。进一步在单只猴内保持session后，中IT差异消失：", "",
              "| monkey | 特征 | 模型 | 倍数 | p | 可交换标签比例 |",
              "|---|---|---|---:|---:|---:|"]
    for x in confound.itertuples():
        lines.append(f"| {x.monkey} | {x.feature_view} | {x.null} | {x.observed_over_null:.2f} | "
                     f"{x.permutation_p:.5f} | {x.median_fraction_labels_changed:.1%} |")
    lines += ["", "因此当前数据不能把中IT omnibus差异归因于功能区；它可能由session主导。M1仅session 24同时含MF/MB，"
              "M3的CLC与MB完全位于不同session。功能区结论必须降为未验证。"]
    lines += [
        "", "## 2. 方向一：去除潜伏期/宽度后的非形变模式", "",
        "先用主模式、其一阶导数和二阶导数构成潜伏期/宽度 nuisance 子空间；再从训练图像残差中学习与其正交的新模式，"
        "并在互斥的500张图像上验证。", "",
        "| ROI | unit | 额外ΔR² | unit正比例 | session p | 系数预测r | half模式稳定度 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in direction.itertuples():
        lines.append(f"| {x.roi} | {x.n_units} | {x.median_nonwarp_delta_r2:+.4f} | "
                     f"{x.nonwarp_positive_fraction:.1%} | {x.nonwarp_session_signflip_p:.5f} | "
                     f"{x.median_nonwarp_coefficient_r:.3f} | {x.median_nonwarp_half_stability:.3f} |")
    lines += [
        "", "## 3. 方向二：时间模式是否调用不同网络深度", "",
        "将冻结的主模式和非形变模式投影到五折PCA64逐层权重，比较两种模式的ResNet层能量质心。", "",
        "| ROI | 非形变−主模式层深 | session p | 层profile cosine | 语义profile cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in direction.itertuples():
        lines.append(f"| {x.roi} | {x.median_nonwarp_minus_main_depth:+.3f} | {x.depth_session_signflip_p:.5f} | "
                     f"{x.median_layer_profile_cosine:.3f} | {x.median_main_nonwarp_semantic_cosine:+.3f} |")
    lines += [
        "", "## 解释边界", "",
        "- 功能区pilot每区只有8个unit，且AO、CLC、MF分别高度绑定特定猴子；猴子条件置换比普通置换更可信，但仍不能替代新猴/新session复现。",
        "- 时间子空间的500/500图像互斥；底层逐窗视觉代理仍来自原五折OOF，并非整条端到端管线重新按500/500训练。",
        "- 网络层深来自标准化PCA64线性权重能量，是模型归因量，不等于生物突触层级。",
    ]
    (OUT / "FUNCTIONAL_IT_AND_TWO_DIRECTIONS_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, blocks = functional_features()
    functional = functional_permutations(cohort, blocks)
    pairs = pairwise_functional(cohort, blocks["combined_balanced"])
    confound = session_confound_audit(cohort, blocks)
    evaluation, depth, sem = nonwarp_and_depth()
    direction = direction_summary(evaluation, depth, sem)
    functional.to_csv(OUT / "functional_area_permutation_tests.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "functional_area_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    confound.to_csv(OUT / "functional_area_session_confound_audit.csv", index=False, encoding="utf-8-sig")
    evaluation.to_csv(OUT / "nonwarp_mode_disjoint_validation.csv", index=False, encoding="utf-8-sig")
    depth.to_csv(OUT / "temporal_mode_network_depth_by_fold.csv", index=False, encoding="utf-8-sig")
    sem.to_csv(OUT / "nonwarp_mode_semantics.csv", index=False, encoding="utf-8-sig")
    direction.to_csv(OUT / "two_direction_summary.csv", index=False, encoding="utf-8-sig")
    make_figure(functional, direction)
    write_report(functional, pairs, confound, direction)
    audit = {"n_functional_units": len(cohort), "areas": cohort.analysis_group.value_counts().to_dict(),
             "n_permutations": N_PERM, "seed": SEED, "target_rois": list(TARGET_ROIS)}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(functional.to_string(index=False), flush=True)
    print(direction.to_string(index=False), flush=True)
    print(pairs.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
