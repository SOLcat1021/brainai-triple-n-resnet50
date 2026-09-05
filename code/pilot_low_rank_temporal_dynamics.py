"""Concept-blind low-rank temporal dynamics pilot on the frozen 113 units.

The existing PCA64 models independently predict every 10-ms response window.
This script asks whether those out-of-fold prediction curves are improved by a
small number of temporal modes learned only from neural responses in the
corresponding training images.  Temporal hyperparameters are selected in a
leave-one-image-fold-out meta-validation, and semantics are attached only
after the temporal model has been selected.
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
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
OLD_BANK = PROJECT / "proxy_bank_10ms_all_windows"
SEMANTIC = PROJECT / "results" / "semantic_64d_high_quality_pilot_113_2026-08-26" / "fold_concept_scores.csv.gz"
OUT = PROJECT / "results" / "low_rank_temporal_dynamics_pilot_113_2026-08-27"

ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
ROI_FILE = {
    "V1": "v1_proxy_bank_64d.h5", "V2": "v2_proxy_bank_64d.h5",
    "V4": "v4_proxy_bank_64d.h5", "posterior IT": "posterior_it_proxy_bank_64d.h5",
    "middle IT": "middle_it_proxy_bank_64d.h5", "anterior IT": "anterior_it_proxy_bank_64d.h5",
}
ONSET = {"V1": 50, "V2": 60, "V4": 70, "posterior IT": 80, "middle IT": 90, "anterior IT": 100}
N_IMAGES = 1000
N_FOLDS = 5
SEED = 20260818 + 901
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
MAX_RANK = 10


def fixed_folds(n: int = N_IMAGES) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.random.default_rng(SEED).permutation(n)
    chunks = np.array_split(order, N_FOLDS)
    return [(np.concatenate([chunks[j] for j in range(N_FOLDS) if j != k]), chunks[k])
            for k in range(N_FOLDS)]


def smoother(n_time: int, lam: float) -> np.ndarray:
    if lam == 0:
        return np.eye(n_time)
    d2 = np.diff(np.eye(n_time), n=2, axis=0)
    return np.linalg.inv(np.eye(n_time) + lam * (d2.T @ d2))


def smooth_fixed_basis(n_time: int) -> np.ndarray:
    d2 = np.diff(np.eye(n_time), n=2, axis=0)
    _, vectors = np.linalg.eigh(d2.T @ d2)
    return vectors


def orient_columns(h: np.ndarray) -> np.ndarray:
    h = h.copy()
    for k in range(h.shape[1]):
        peak = int(np.argmax(np.abs(h[:, k])))
        if h[peak, k] < 0:
            h[:, k] *= -1
    return h


def svd_basis(curves: np.ndarray, lam: float, rank: int = MAX_RANK) -> np.ndarray:
    smoothed = curves @ smoother(curves.shape[1], lam)
    _, _, vt = np.linalg.svd(smoothed, full_matrices=False)
    return orient_columns(vt[:rank].T)


def roi_curves(residual: np.ndarray) -> np.ndarray:
    """images x time x units -> equally weighted unit-image x time rows."""
    scale = np.sqrt(np.mean(residual ** 2, axis=(0, 1))).clip(min=1e-6)
    return np.transpose(residual / scale[None, None], (0, 2, 1)).reshape(-1, residual.shape[1])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, float).ravel(); bb = np.asarray(b, float).ravel()
    aa -= aa.mean(); bb -= bb.mean()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(aa @ bb / den) if den > 1e-12 else np.nan


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, float).ravel(); bb = np.asarray(b, float).ravel()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(aa @ bb / den) if den > 1e-12 else np.nan


def score_curve(y: np.ndarray, pred: np.ndarray, mean_curve: np.ndarray) -> dict[str, float]:
    baseline_error = float(np.sum((y - mean_curve[None]) ** 2))
    error = float(np.sum((y - pred) ** 2))
    residual_y = y - mean_curve[None]
    residual_pred = pred - mean_curve[None]
    per_window = [corr(y[:, t], pred[:, t]) for t in range(y.shape[1])]
    return {
        "stimulus_r2": 1.0 - error / max(baseline_error, 1e-12),
        "residual_flat_r": corr(residual_y, residual_pred),
        "mean_window_r": float(np.nanmean(per_window)),
        "normalized_mse": error / max(baseline_error, 1e-12),
    }


def project_prediction(pred: np.ndarray, mean_curve: np.ndarray, h: np.ndarray) -> np.ndarray:
    return mean_curve[None] + ((pred - mean_curve[None]) @ h) @ h.T


def load_roi(roi: str, responses: np.memmap, time_index: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selected = pd.read_csv(BANK / "selected_units.csv")
    meta = selected[selected.roi.eq(roi)].copy().reset_index(drop=True)
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[time_index][:, :, units], np.float32).transpose(1, 0, 2)
    with h5py.File(BANK / ROI_FILE[roi], "r") as h:
        pred = h["oof_prediction"][:].astype(np.float32)[time_index].transpose(1, 0, 2)
        if not np.array_equal(h["unit_global"][:].astype(int), units):
            raise RuntimeError(f"unit order mismatch: {roi}")
    return meta, y, pred


def add_score(rows: list[dict], base: dict, values: dict[str, float]) -> None:
    rows.append({**base, **values})


def sweep_roi(
    roi: str, meta: pd.DataFrame, y: np.ndarray, raw_pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]], fixed_basis: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    n_time = y.shape[1]
    for fi, (train, test) in enumerate(folds):
        ytr = y[train]
        yte = y[test]
        pte = raw_pred[test]
        mean_curve = ytr.mean(axis=0)  # time x unit
        residual = ytr - mean_curve[None]

        shared_bases = {lam: svd_basis(roi_curves(residual), lam) for lam in LAMBDAS}
        unit_bases = {
            (ui, lam): svd_basis(residual[:, :, ui], lam)
            for ui in range(y.shape[2]) for lam in LAMBDAS
        }
        for ui, unit in meta.iterrows():
            base = {
                "roi": roi, "unit_global": int(unit.unit_global),
                "native_area": str(unit.native_area), "monkey": str(unit.monkey),
                "session": int(unit.session), "fold": fi,
            }
            target = yte[:, :, ui]
            pred = pte[:, :, ui]
            mu = mean_curve[:, ui]
            add_score(rows, {**base, "method": "independent_windows", "lambda": np.nan, "rank": n_time},
                      score_curve(target, pred, mu))
            for rank in range(1, min(MAX_RANK, n_time) + 1):
                h = fixed_basis[:, :rank]
                add_score(rows, {**base, "method": "fixed_smooth_basis", "lambda": np.nan, "rank": rank},
                          score_curve(target, project_prediction(pred, mu, h), mu))
                for lam in LAMBDAS:
                    for method, basis in (
                        ("roi_shared_svd", shared_bases[lam]),
                        ("unit_svd", unit_bases[(ui, lam)]),
                    ):
                        h = basis[:, :rank]
                        add_score(rows, {**base, "method": method, "lambda": lam, "rank": rank},
                                  score_curve(target, project_prediction(pred, mu, h), mu))
    return pd.DataFrame(rows)


def aggregate_grid(scores: pd.DataFrame) -> pd.DataFrame:
    session = scores.groupby(
        ["roi", "method", "lambda", "rank", "monkey", "session"], dropna=False, as_index=False
    ).stimulus_r2.mean()
    rows = []
    for key, q in scores.groupby(["roi", "method", "lambda", "rank"], dropna=False):
        s = session[
            session.roi.eq(key[0]) & session.method.eq(key[1])
            & ((session["lambda"].isna()) if pd.isna(key[2]) else session["lambda"].eq(key[2]))
            & session["rank"].eq(key[3])
        ].stimulus_r2.to_numpy(float)
        rows.append({
            "roi": key[0], "method": key[1], "lambda": key[2], "rank": int(key[3]),
            "mean_unit_fold_r2": float(q.stimulus_r2.mean()),
            "median_unit_fold_r2": float(q.stimulus_r2.median()),
            "mean_residual_flat_r": float(q.residual_flat_r.mean()),
            "mean_window_r": float(q.mean_window_r.mean()),
            "n_units": int(q.unit_global.nunique()), "n_sessions": int(len(s)),
            "session_mean_r2": float(np.mean(s)),
            "session_sem_r2": float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def config_mask(frame: pd.DataFrame, method: str, lam: float, rank: int) -> pd.Series:
    same_lambda = frame["lambda"].isna() if pd.isna(lam) else frame["lambda"].eq(lam)
    return frame.method.eq(method) & same_lambda & frame["rank"].eq(rank)


def nested_model_comparison(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    unit_rows = []
    for roi in ROIS:
        r = scores[scores.roi.eq(roi)]
        for heldout_fold in range(N_FOLDS):
            train = r[r.fold.ne(heldout_fold)]
            grid = train.groupby(["method", "lambda", "rank"], dropna=False).stimulus_r2.mean().reset_index()
            low = grid[(grid.method.ne("independent_windows")) & grid["rank"].eq(1)].sort_values(
                "stimulus_r2", ascending=False).iloc[0]
            dynamic = grid[(grid.method.ne("independent_windows")) & grid["rank"].ge(2)].sort_values(
                "stimulus_r2", ascending=False).iloc[0]
            choices.extend([
                {"roi": roi, "heldout_fold": heldout_fold, "model_class": "rank1_gain",
                 "method": low.method, "lambda": low["lambda"], "rank": int(low["rank"])},
                {"roi": roi, "heldout_fold": heldout_fold, "model_class": "dynamic_rank_ge2",
                 "method": dynamic.method, "lambda": dynamic["lambda"], "rank": int(dynamic["rank"])},
            ])
            test = r[r.fold.eq(heldout_fold)]
            rank1 = test[config_mask(test, low.method, low["lambda"], int(low["rank"]))]
            dyn = test[config_mask(test, dynamic.method, dynamic["lambda"], int(dynamic["rank"]))]
            raw = test[test.method.eq("independent_windows")]
            merged = rank1[["unit_global", "monkey", "session", "stimulus_r2"]].rename(
                columns={"stimulus_r2": "rank1_r2"}).merge(
                dyn[["unit_global", "stimulus_r2"]].rename(columns={"stimulus_r2": "dynamic_r2"}),
                on="unit_global", validate="one_to_one").merge(
                raw[["unit_global", "stimulus_r2"]].rename(columns={"stimulus_r2": "independent_r2"}),
                on="unit_global", validate="one_to_one")
            merged["roi"] = roi; merged["fold"] = heldout_fold
            merged["dynamic_minus_rank1"] = merged.dynamic_r2 - merged.rank1_r2
            merged["dynamic_minus_independent"] = merged.dynamic_r2 - merged.independent_r2
            unit_rows.append(merged)
    return pd.DataFrame(choices), pd.concat(unit_rows, ignore_index=True)


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, float)
    observed = abs(float(np.mean(values)))
    if len(values) <= 18:
        null = np.asarray([abs(float(np.mean(values * np.asarray(s))))
                           for s in product((-1, 1), repeat=len(values))])
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1, 1), size=(200_000, len(values)))
        null = np.abs((signs * values[None]).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def comparison_summary(unit: pd.DataFrame) -> pd.DataFrame:
    per_unit = unit.groupby(
        ["roi", "unit_global", "monkey", "session"], as_index=False
    )[["rank1_r2", "dynamic_r2", "independent_r2", "dynamic_minus_rank1", "dynamic_minus_independent"]].mean()
    rows = []
    for roi, q in per_unit.groupby("roi", sort=False):
        for ci, column in enumerate(("dynamic_minus_rank1", "dynamic_minus_independent")):
            delta = q[column].to_numpy(float)
            cluster = q.groupby(["monkey", "session"])[column].mean().to_numpy(float)
            rows.append({
                "roi": roi, "comparison": column, "n_units": int(len(q)),
                "median_delta_r2": float(np.median(delta)), "mean_delta_r2": float(np.mean(delta)),
                "positive_unit_fraction": float(np.mean(delta > 0)),
                "unit_wilcoxon_p": float(wilcoxon(delta).pvalue),
                "n_sessions": int(len(cluster)), "session_positive_fraction": float(np.mean(cluster > 0)),
                "session_signflip_p": signflip_p(cluster, SEED + len(rows) + ci),
            })
    return pd.DataFrame(rows)


def choose_shared_configs(grid: pd.DataFrame) -> pd.DataFrame:
    q = grid[grid.method.eq("roi_shared_svd")].copy()
    best = q.sort_values("session_mean_r2", ascending=False).groupby("roi", as_index=False).first()
    return best[["roi", "method", "lambda", "rank", "session_mean_r2", "session_sem_r2"]]


def full_shared_modes(
    configs: pd.DataFrame, responses: np.memmap, time_index: np.ndarray,
    times: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    mode_rows = []
    feature_rows = []
    bases = {}
    for _, row in configs.iterrows():
        roi = str(row["roi"]); rank = int(row["rank"]); lam = float(row["lambda"])
        meta, y, _ = load_roi(roi, responses, time_index)
        mean_curve = y.mean(axis=0)
        h = svd_basis(roi_curves(y - mean_curve[None]), lam, rank)[:, :rank]
        bases[roi] = h
        for k in range(h.shape[1]):
            v = h[:, k]
            energy = v ** 2 / max(float(np.sum(v ** 2)), 1e-12)
            centroid = float(np.sum(times * energy))
            width = float(np.sqrt(np.sum((times - centroid) ** 2 * energy)))
            feature_rows.append({
                "roi": roi, "mode": k + 1, "rank_selected": rank,
                "lambda_selected": lam,
                "peak_abs_time_ms": float(times[np.argmax(np.abs(v))]),
                "energy_centroid_ms": centroid, "energy_width_ms": width,
                "early_energy_fraction_lt80ms": float(energy[times < 80].sum()),
                "late_energy_fraction_ge120ms": float(energy[times >= 120].sum()),
                "sign_changes": int(np.sum(np.sign(v[1:]) != np.sign(v[:-1]))),
            })
            for t, value in zip(times, v):
                mode_rows.append({"roi": roi, "mode": k + 1, "time_ms": int(t), "amplitude": float(value)})
    return pd.DataFrame(mode_rows), bases, pd.DataFrame(feature_rows)


def fold_subspace_stability(
    configs: pd.DataFrame, responses: np.memmap, time_index: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]], bases: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for _, row in configs.iterrows():
        roi = str(row["roi"]); rank = int(row["rank"]); lam = float(row["lambda"])
        _, y, _ = load_roi(roi, responses, time_index)
        full = bases[roi]
        for fi, (train, _) in enumerate(folds):
            m = y[train].mean(axis=0)
            h = svd_basis(roi_curves(y[train] - m[None]), lam, rank)[:, :rank]
            singular = np.linalg.svd(full.T @ h, compute_uv=False)
            rows.append({
                "roi": roi, "fold": fi, "rank": rank,
                "mean_squared_canonical_correlation": float(np.mean(singular ** 2)),
                "worst_canonical_correlation": float(np.min(singular)),
            })
    return pd.DataFrame(rows)


def semantic_mode_mapping(
    bases: dict[str, np.ndarray], time_starts: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(SEMANTIC)
    scores = scores[scores.window_start_ms.isin(time_starts)].groupby(
        ["unit_global", "roi", "window_start_ms", "concept"], as_index=False
    ).concept_score.mean()
    rows = []
    audit = []
    for roi, h in bases.items():
        q = scores[scores.roi.eq(roi)]
        for unit, z in q.groupby("unit_global"):
            matrix = z.pivot(index="window_start_ms", columns="concept", values="concept_score").reindex(
                index=time_starts).sort_index(axis=1)
            c = matrix.to_numpy(float)
            profiles = h.T @ c
            reconstruction = h @ profiles
            explained = 1.0 - np.sum((c - reconstruction) ** 2) / max(float(np.sum(c ** 2)), 1e-12)
            audit.append({"roi": roi, "unit_global": int(unit), "semantic_curve_variance_explained": float(explained)})
            for k in range(h.shape[1]):
                for concept, value in zip(matrix.columns, profiles[k]):
                    rows.append({
                        "roi": roi, "unit_global": int(unit), "mode": k + 1,
                        "concept": concept, "mode_concept_loading": float(value),
                    })
    return pd.DataFrame(rows), pd.DataFrame(audit)


def semantic_mode_summary(mode_scores: pd.DataFrame, mode_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi, q in mode_scores.groupby("roi", sort=False):
        wide = q.pivot(index=["unit_global", "mode"], columns="concept", values="mode_concept_loading")
        features = mode_features[mode_features.roi.eq(roi)].set_index("mode")
        for mode, z in wide.groupby(level="mode"):
            values = z.to_numpy(float)
            rms = np.sqrt(np.mean(values ** 2, axis=1))
            rows.append({
                "roi": roi, "mode": int(mode), "n_units": int(len(z)),
                "semantic_rms_median": float(np.median(rms)),
                "semantic_rms_mean": float(np.mean(rms)),
                "energy_centroid_ms": float(features.loc[int(mode), "energy_centroid_ms"]),
                "top_group_concept": str(z.mean(axis=0).abs().idxmax()),
                "top_group_loading": float(z.mean(axis=0).loc[z.mean(axis=0).abs().idxmax()]),
            })
        for a, b in combinations(range(1, wide.index.get_level_values("mode").max() + 1), 2):
            aa = wide.xs(a, level="mode"); bb = wide.xs(b, level="mode")
            common = aa.index.intersection(bb.index)
            similarities = [cosine(aa.loc[u].to_numpy(float), bb.loc[u].to_numpy(float)) for u in common]
            rows.append({
                "roi": roi, "mode": f"{a}_vs_{b}", "n_units": int(len(common)),
                "semantic_rms_median": np.nan, "semantic_rms_mean": np.nan,
                "energy_centroid_ms": np.nan,
                "top_group_concept": "pairwise_profile_cosine",
                "top_group_loading": float(np.nanmedian(similarities)),
            })
    return pd.DataFrame(rows)


def make_figures(
    grid: pd.DataFrame, comparison: pd.DataFrame, configs: pd.DataFrame,
    modes: pd.DataFrame, semantic: pd.DataFrame,
) -> None:
    colors = dict(zip(ROIS, ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756", "#B279A2"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for roi in ROIS:
        q = grid[(grid.roi.eq(roi)) & grid.method.eq("roi_shared_svd")]
        best_by_rank = q.sort_values("session_mean_r2", ascending=False).groupby("rank", as_index=False).first()
        axes[0].plot(best_by_rank["rank"], best_by_rank.session_mean_r2, marker="o", color=colors[roi], label=roi)
    axes[0].set(title="ROI-shared temporal rank sweep", xlabel="Temporal rank K", ylabel="Session-mean held-out stimulus R²")
    axes[0].grid(alpha=.2); axes[0].legend(ncol=2, fontsize=8)

    c = comparison[comparison.comparison.eq("dynamic_minus_rank1")].set_index("roi").reindex(ROIS)
    axes[1].bar(range(len(c)), c.median_delta_r2, color=[colors[r] for r in ROIS])
    axes[1].axhline(0, color="#777777", lw=.8)
    axes[1].set_xticks(range(len(c)), c.index, rotation=35, ha="right")
    axes[1].set(title="Nested dynamic model advantage over rank-1 gain", ylabel="Median held-out ΔR²")
    axes[1].grid(axis="y", alpha=.2)
    fig.savefig(OUT / "01_temporal_rank_and_nested_advantage.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), constrained_layout=True)
    for ax, roi in zip(axes.flat, ROIS):
        q = modes[modes.roi.eq(roi)]
        for mode, z in q.groupby("mode"):
            ax.plot(z.time_ms, z.amplitude, marker="o", ms=2.5, label=f"mode {mode}")
        cfg = configs[configs.roi.eq(roi)].iloc[0]
        ax.set(title=f"{roi}: shared K={int(cfg['rank'])}, λ={cfg['lambda']:g}", xlabel="Time (ms)", ylabel="Mode amplitude")
        ax.axhline(0, color="#888888", lw=.7); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(OUT / "02_roi_shared_temporal_modes.png", dpi=300)
    plt.close(fig)

    numeric = semantic[pd.to_numeric(semantic["mode"], errors="coerce").notna()].copy()
    numeric["mode_num"] = numeric["mode"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for roi in ROIS:
        q = numeric[numeric.roi.eq(roi)].sort_values("energy_centroid_ms")
        ax.plot(q.energy_centroid_ms, q.semantic_rms_median, marker="o", color=colors[roi], label=roi)
    ax.set(title="Semantic loading strength of response-defined temporal modes",
           xlabel="Temporal-mode energy centroid (ms)", ylabel="Median concept-profile RMS")
    ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8)
    fig.savefig(OUT / "03_temporal_mode_semantic_strength.png", dpi=300)
    plt.close(fig)


def write_report(
    comparison: pd.DataFrame, configs: pd.DataFrame, stability: pd.DataFrame,
    mode_features: pd.DataFrame, semantic_summary: pd.DataFrame,
) -> None:
    lines = [
        "# 113 个高质量 unit：低秩时间动力学 pilot",
        "",
        "## 设计",
        "",
        "- 时间模式仅由神经响应学习，概念标签和 CAV 不参与模型选择。",
        "- 使用 0–189 ms 的 19 个连续 10-ms 窗；每折时间基底只看该折的 800 张训练图像。",
        "- 扫描 rank K=1–10、unit-specific/ROI-shared SVD、固定平滑基底，以及 9 个二阶平滑强度。",
        "- 在五个图像折上进行元交叉验证：每次用另外四折选择超参数，只在未参与选择的一折报告动态模型相对 rank-1 的增益。",
        "- 语义映射在时间模型冻结后进行，只用于解释，不参与显著性选择。",
        "",
        "## 嵌套检验：动态 rank≥2 相对固定特征×时间增益 rank=1",
        "",
        "| ROI | unit | ΔR²中位数 | 正向unit | unit p | session | session正向 | session p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in comparison[comparison.comparison.eq("dynamic_minus_rank1")].itertuples():
        lines.append(
            f"| {x.roi} | {x.n_units} | {x.median_delta_r2:+.4f} | {x.positive_unit_fraction:.1%} | "
            f"{x.unit_wilcoxon_p:.6f} | {x.n_sessions} | {x.session_positive_fraction:.1%} | {x.session_signflip_p:.6f} |"
        )
    lines += [
        "",
        "## ROI共享时间模式",
        "",
        "| ROI | K | λ | held-out R² | 跨折子空间稳定度 | 最差canonical r |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, x in configs.iterrows():
        s = stability[stability.roi.eq(x["roi"])]
        lines.append(
            f"| {x['roi']} | {int(x['rank'])} | {x['lambda']:g} | {x['session_mean_r2']:.4f} | "
            f"{s.mean_squared_canonical_correlation.mean():.3f} | {s.worst_canonical_correlation.median():.3f} |"
        )
    lines += [
        "",
        "## 模式时间尺度与语义解释",
        "",
        "| ROI | mode | 时间质心(ms) | 宽度(ms) | <80ms能量 | ≥120ms能量 | 语义RMS | 最大群体概念 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    numeric = semantic_summary[pd.to_numeric(semantic_summary["mode"], errors="coerce").notna()].copy()
    numeric["mode_num"] = numeric["mode"].astype(int)
    merged = mode_features.merge(numeric[["roi", "mode_num", "semantic_rms_median", "top_group_concept"]],
                                 left_on=["roi", "mode"], right_on=["roi", "mode_num"], how="left")
    for x in merged.itertuples():
        lines.append(
            f"| {x.roi} | {x.mode} | {x.energy_centroid_ms:.1f} | {x.energy_width_ms:.1f} | "
            f"{x.early_energy_fraction_lt80ms:.1%} | {x.late_energy_fraction_ge120ms:.1%} | "
            f"{x.semantic_rms_median:.3f} | {x.top_group_concept} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 当前 released response_matrix_img 已经是图像条件平均曲线，不包含单试次维度；因此结果是刺激依赖的平均动力学，不是 trial-to-trial latent dynamics。",
        "- 低秩模型在已有逐窗 OOF 预测上做严格训练折内的时间投影；它是判断是否值得联合重训视觉×时间模型的筛查，而不是最终端到端模型。",
        "- 模式的符号和顺序本身不具有生理意义；可靠结论依赖子空间、预测增益及跨折/跨session复现。",
        "- CAV 只解释模式在预定义概念轴上的投影；语义抽象仍需跨场景留出刺激验证。",
        "",
    ]
    (OUT / "LOW_RANK_TEMPORAL_DYNAMICS_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    responses = np.load(OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(OLD_BANK / "windows_10ms.npy").astype(int)
    time_index = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    times = windows[time_index].mean(axis=1)
    folds = fixed_folds()
    fixed_basis = smooth_fixed_basis(len(time_index))

    score_parts = []
    for roi in ROIS:
        meta, y, pred = load_roi(roi, responses, time_index)
        score_parts.append(sweep_roi(roi, meta, y, pred, folds, fixed_basis))
        print(f"swept {roi}: {len(meta)} units", flush=True)
    scores = pd.concat(score_parts, ignore_index=True)
    scores.to_csv(OUT / "temporal_rank_cv_scores.csv.gz", index=False, compression="gzip")
    grid = aggregate_grid(scores)
    grid.to_csv(OUT / "temporal_rank_grid_summary.csv", index=False, encoding="utf-8-sig")
    choices, nested = nested_model_comparison(scores)
    comparison = comparison_summary(nested)
    choices.to_csv(OUT / "nested_meta_cv_choices.csv", index=False, encoding="utf-8-sig")
    nested.to_csv(OUT / "nested_meta_cv_unit_scores.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "nested_model_comparison_by_roi.csv", index=False, encoding="utf-8-sig")

    configs = choose_shared_configs(grid)
    configs.to_csv(OUT / "selected_roi_shared_configs.csv", index=False, encoding="utf-8-sig")
    modes, bases, mode_features = full_shared_modes(configs, responses, time_index, times)
    stability = fold_subspace_stability(configs, responses, time_index, folds, bases)
    modes.to_csv(OUT / "roi_shared_temporal_modes.csv", index=False, encoding="utf-8-sig")
    mode_features.to_csv(OUT / "temporal_mode_features.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(OUT / "temporal_subspace_stability.csv", index=False, encoding="utf-8-sig")

    semantic_modes, semantic_audit = semantic_mode_mapping(bases, windows[time_index, 0])
    semantic_summary = semantic_mode_summary(semantic_modes, mode_features)
    semantic_modes.to_csv(OUT / "temporal_mode_concept_loadings.csv.gz", index=False, compression="gzip")
    semantic_audit.to_csv(OUT / "semantic_curve_reconstruction_by_unit.csv", index=False, encoding="utf-8-sig")
    semantic_summary.to_csv(OUT / "temporal_mode_semantic_summary.csv", index=False, encoding="utf-8-sig")

    make_figures(grid, comparison, configs, modes, semantic_summary)
    write_report(comparison, configs, stability, mode_features, semantic_summary)
    audit = {
        "cohort": "frozen response-blind high-quality pilot 113",
        "response_target": "released image-condition mean firing rate; no trial dimension",
        "time_range_ms": [0, 189], "n_time_bins": int(len(time_index)),
        "rank_grid": [1, MAX_RANK], "lambdas": list(LAMBDAS),
        "methods": ["unit_svd", "roi_shared_svd", "fixed_smooth_basis"],
        "selection": "leave-one-image-fold-out meta-CV over already OOF PCA64 predictions",
        "semantics_used_for_temporal_selection": False,
        "purpose": "screen whether end-to-end low-rank visual-temporal retraining is justified",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(configs.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
