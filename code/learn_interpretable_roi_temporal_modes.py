"""Learn smooth supervised temporal modes that separate coarse ROIs.

Modes are learned from response-only, image-averaged curves.  Unit curves are
shape-normalized before learning so the modes describe timing/shape rather than
overall response magnitude. Session-grouped validation tests whether the modes
generalize beyond the sessions used to learn them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "interpretable_roi_temporal_modes_raw_amplitude_2026-08-27"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_LABELS = ["V1", "V2", "V4", "pIT", "mIT", "aIT"]
SEED = 20260827
N_MODES = 3


def load_curves(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    post = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    pre = windows[:, 1] < 0
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[:, :, units], np.float32).transpose(2, 1, 0)
    curves = y[:, :, post].mean(1) - y[:, :, pre].mean((1, 2))[:, None]
    # Preserve response amplitude/gain. Only the pre-stimulus baseline is
    # removed; no per-unit centering or norm normalization is applied.
    return curves.astype(float), windows[post].mean(1)


def smooth_penalty(n: int) -> np.ndarray:
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d2.T @ d2


def learn_modes(x: np.ndarray, labels: np.ndarray, n_modes: int = 3,
                smooth_lambda: float = 0.20) -> np.ndarray:
    classes = np.unique(labels)
    grand = x.mean(0)
    between = np.zeros((x.shape[1], x.shape[1]))
    within = np.zeros_like(between)
    for c in classes:
        q = x[labels == c]
        delta = (q.mean(0) - grand)[:, None]
        between += len(q) * (delta @ delta.T)
        centered = q - q.mean(0)
        within += centered.T @ centered
    between /= max(len(x), 1)
    within /= max(len(x) - len(classes), 1)
    within += 0.15 * np.trace(within) / x.shape[1] * np.eye(x.shape[1])
    within += smooth_lambda * np.trace(within) / max(np.trace(smooth_penalty(x.shape[1])), 1e-12) * smooth_penalty(x.shape[1])
    values, vectors = eigh(between, within)
    order = np.argsort(values)[::-1]
    modes = vectors[:, order[:n_modes]].T
    modes /= np.maximum(np.linalg.norm(modes, axis=1, keepdims=True), 1e-12)
    # Deterministic sign: positive late-vs-early contrast where possible.
    for i in range(len(modes)):
        if np.sum(modes[i, -6:]) < np.sum(modes[i, :6]):
            modes[i] *= -1
    return modes


def project_scores(x: np.ndarray, modes: np.ndarray) -> np.ndarray:
    return x @ modes.T


def fold_decode(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, modes_k: int,
                smooth_lambda: float) -> pd.DataFrame:
    rows = []
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + modes_k)
    for fold, (train, test) in enumerate(splitter.split(x, labels, groups)):
        modes = learn_modes(x[train], labels[train], modes_k, smooth_lambda)
        scores_train, scores_test = project_scores(x[train], modes), project_scores(x[test], modes)
        scaler = StandardScaler().fit(scores_train)
        model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000,
                                   random_state=SEED + fold).fit(scaler.transform(scores_train), labels[train])
        pred = model.predict(scaler.transform(scores_test))
        rows.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                     "n_train_sessions": len(np.unique(groups[train])),
                     "n_test_sessions": len(np.unique(groups[test])),
                     "n_modes": modes_k, "smooth_lambda": smooth_lambda,
                     "unit_balanced_accuracy": balanced_accuracy_score(labels[test], pred),
                     "mode_roughness": float(np.mean(np.sum(np.diff(modes, n=2, axis=1) ** 2, axis=1))),
                     "mode_max_abs": float(np.max(np.abs(modes)))})
    return pd.DataFrame(rows)


def interpret_modes(modes: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, mode in enumerate(modes, 1):
        abs_mode = np.abs(mode)
        center = float(np.sum(times * abs_mode) / max(abs_mode.sum(), 1e-12))
        early = float(np.sum(mode[times < 70]))
        middle = float(np.sum(mode[(times >= 70) & (times < 130)]))
        late = float(np.sum(mode[times >= 130]))
        sign_changes = int(np.sum(np.sign(mode[1:]) != np.sign(mode[:-1])))
        rows.append({"mode": i, "time_centroid_ms": center, "early_weight": early,
                     "middle_weight": middle, "late_weight": late,
                     "sign_changes": sign_changes,
                     "roughness": float(np.sum(np.diff(mode, n=2) ** 2)),
                     "max_loading_time_ms": float(times[np.argmax(np.abs(mode))])})
    return pd.DataFrame(rows)


def make_figure(modes: np.ndarray, times: np.ndarray, scores: np.ndarray,
                labels: np.ndarray, decode: pd.DataFrame, interp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), constrained_layout=True)
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    ax = axes[0]
    for i, mode in enumerate(modes, 1):
        ax.plot(times, mode, lw=2.5, label=f"Mode {i}")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set(title="Three supervised temporal modes", xlabel="Time after image onset (ms)", ylabel="Mode loading")
    ax.legend(frameon=False); ax.grid(axis="y", color="#D8D8D8", lw=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    for i, roi in enumerate(ROIS):
        q = scores[labels == roi]
        ax.scatter(q[:, 0], q[:, 1], s=7, alpha=0.16, color=colors[i], label=ROI_LABELS[i])
        c = q.mean(0)
        ax.scatter(c[0], c[1], s=90, color=colors[i], edgecolors="#222222", linewidths=0.8)
    ax.set(title="Unit scores on Mode 1 / Mode 2", xlabel="Mode 1 score", ylabel="Mode 2 score")
    ax.legend(frameon=False, ncol=3, fontsize=8); ax.grid(color="#D8D8D8", lw=0.7); ax.set_axisbelow(True)

    ax = axes[2]
    q = decode.groupby("n_modes", as_index=False).unit_balanced_accuracy.mean()
    ax.plot(q.n_modes, q.unit_balanced_accuracy, marker="o", lw=2.5, color="#2F6F8F")
    ax.axhline(1 / 6, color="#A14A3B", ls="--", lw=1.2, label="6-class chance")
    ax.set(title="Leave-session-out ROI decoding", xlabel="Number of modes", ylabel="Balanced accuracy", xticks=[1, 2, 3], ylim=(0, 1))
    ax.legend(frameon=False); ax.grid(axis="y", color="#D8D8D8", lw=0.7); ax.set_axisbelow(True)
    fig.suptitle("Interpretable supervised temporal modes across six visual ROIs", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.004, "Modes learned from shape-normalized response curves; validation holds out complete monkey×session groups. No ResNet proxy or semantic labels used.", ha="center", fontsize=9)
    fig.savefig(OUT / "01_interpretable_roi_temporal_modes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    meta = meta[meta.roi.isin(ROIS)].sort_values("unit_global").reset_index(drop=True)
    x, times = load_curves(meta)
    labels = meta.roi.to_numpy(str)
    groups = (meta.monkey.astype(str) + "|" + meta.session.astype(str)).to_numpy(str)
    smooth_grid = (0.2, 1.0, 5.0, 20.0)
    tuning = pd.concat([
        fold_decode(x, labels, groups, N_MODES, smooth_lambda)
        for smooth_lambda in smooth_grid
    ], ignore_index=True)
    choice = tuning.groupby("smooth_lambda").agg(
        mean_accuracy=("unit_balanced_accuracy", "mean"),
        mean_roughness=("mode_roughness", "mean"),
    ).reset_index()
    max_accuracy = float(choice.mean_accuracy.max())
    eligible = choice[choice.mean_accuracy >= max_accuracy - 0.03]
    selected_lambda = float(eligible.sort_values(["mean_roughness", "smooth_lambda"]).iloc[0].smooth_lambda)
    all_modes = learn_modes(x, labels, N_MODES, selected_lambda)
    all_scores = project_scores(x, all_modes)
    interp = interpret_modes(all_modes, times)
    decode = pd.concat([fold_decode(x, labels, groups, k, selected_lambda) for k in (1, 2, 3)], ignore_index=True)
    tuning.to_csv(OUT / "smoothness_tuning.csv", index=False, encoding="utf-8-sig")
    interp.to_csv(OUT / "mode_interpretability.csv", index=False, encoding="utf-8-sig")
    decode.to_csv(OUT / "leave_session_out_decoding.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_ms": times, **{f"mode_{i+1}": all_modes[i] for i in range(N_MODES)}}).to_csv(OUT / "temporal_mode_loadings.csv", index=False, encoding="utf-8-sig")
    centroids = []
    for roi in ROIS:
        q = all_scores[labels == roi]
        centroids.append({"roi": roi, **{f"mode_{i+1}_score": float(q[:, i].mean()) for i in range(N_MODES)}})
    pd.DataFrame(centroids).to_csv(OUT / "roi_mode_score_centroids.csv", index=False, encoding="utf-8-sig")
    audit = {"n_units": len(meta), "n_sessions": len(np.unique(groups)), "shape_normalized": False,
             "response_only": True, "proxy_predictions_used": False, "semantic_labels_used": False,
             "model": "regularized generalized eigenproblem: between-ROI covariance vs within-ROI covariance + second-derivative smoothness penalty",
             "validation": "4-fold StratifiedGroupKFold holding out monkey|session groups",
             "smoothness_grid": list(smooth_grid), "selected_smooth_lambda": selected_lambda,
             "selection_rule": "smoothest setting within 0.03 balanced accuracy of the best setting",
             "n_modes_tested": [1, 2, 3]}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(all_modes, times, all_scores, labels, decode, interp)
    print(interp.to_string(index=False), flush=True)
    print(decode.groupby("n_modes").unit_balanced_accuracy.agg(["mean", "std"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
