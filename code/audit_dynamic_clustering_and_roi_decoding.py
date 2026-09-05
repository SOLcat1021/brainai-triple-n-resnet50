"""Audit unsupervised dynamic clustering and session-held-out ROI decoding."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             confusion_matrix, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
DYNAMIC = PROJECT / "results" / "dynamic_phenotype_clustering_heldout_2026-08-27"
TEMPORAL = PROJECT / "results" / "cross_roi_observed_temporal_metrics_2026-08-27"
OUT = PROJECT / "results" / "dynamic_clustering_roi_decoding_audit_2026-08-27"
SEED = 20260827
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_LABELS = ["V1", "V2", "V4", "pIT", "mIT", "aIT"]


def clr(x: np.ndarray) -> np.ndarray:
    logx = np.log(np.maximum(np.asarray(x, float), 1e-6))
    return logx - logx.mean(1, keepdims=True)


def cramers_v(labels: np.ndarray, roi: np.ndarray) -> float:
    table = pd.crosstab(labels, roi).to_numpy(float)
    total = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / total
    chi2 = np.sum((table - expected) ** 2 / np.maximum(expected, 1e-12))
    return float(np.sqrt((chi2 / total) / max(min(table.shape) - 1, 1)))


def unsupervised_audit(data: pd.DataFrame) -> pd.DataFrame:
    train = data[["train_gain_fraction", "train_warp_fraction", "train_nonwarp_fraction"]].to_numpy(float)
    test = data[["heldout_gain_fraction", "heldout_warp_fraction", "heldout_nonwarp_fraction"]].to_numpy(float)
    scaler = StandardScaler().fit(clr(train))
    train_z, test_z = scaler.transform(clr(train)), scaler.transform(clr(test))
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(data), size=min(6000, len(data)), replace=False))
    roi = data.roi.to_numpy(str)
    rows = []
    for k in range(2, 9):
        for method in ("kmeans", "gmm_diag"):
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=SEED + k, n_init=50).fit(train_z)
                train_label, test_label = model.labels_, model.predict(test_z)
            else:
                model = GaussianMixture(n_components=k, covariance_type="diag", random_state=SEED + k,
                                        n_init=10, reg_covar=1e-5).fit(train_z)
                train_label, test_label = model.predict(train_z), model.predict(test_z)
            rows.append({
                "method": method, "k": k,
                "heldout_accuracy": float(np.mean(train_label == test_label)),
                "heldout_balanced_accuracy": float(balanced_accuracy_score(train_label, test_label)),
                "heldout_ari": float(adjusted_rand_score(train_label, test_label)),
                "heldout_nmi": float(normalized_mutual_info_score(train_label, test_label)),
                "training_silhouette": float(silhouette_score(train_z[sample], train_label[sample])),
                "roi_cluster_nmi_descriptive": float(normalized_mutual_info_score(roi, train_label)),
                "roi_cluster_cramers_v_descriptive": cramers_v(train_label, roi),
            })
    return pd.DataFrame(rows)


def grouped_roi_decode(frame: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame.roi.map({roi: i for i, roi in enumerate(ROIS)}).to_numpy(int)
    groups = (frame.monkey.astype(str) + "|" + frame.session.astype(str)).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    metric_rows, prediction_rows = [], []
    for feature_name, columns in feature_sets.items():
        x = frame[columns].to_numpy(float)
        all_pred = np.full(len(frame), -1, int)
        all_prob = np.full((len(frame), len(ROIS)), np.nan, float)
        for fold, (train, test) in enumerate(splitter.split(x, labels, groups)):
            scaler = StandardScaler().fit(x[train])
            xtr, xte = scaler.transform(x[train]), scaler.transform(x[test])
            model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000,
                                       random_state=SEED + fold).fit(xtr, labels[train])
            all_pred[test] = model.predict(xte)
            probability = model.predict_proba(xte)
            for ci, cls in enumerate(model.classes_):
                all_prob[test, int(cls)] = probability[:, ci]
        valid = all_pred >= 0
        unit_balanced = balanced_accuracy_score(labels[valid], all_pred[valid])
        unit_cm = confusion_matrix(labels[valid], all_pred[valid], labels=np.arange(6), normalize="true")

        session_rows = []
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group)
            mean_prob = np.nanmean(all_prob[idx], axis=0)
            session_rows.append({"group": group, "true": int(labels[idx[0]]), "pred": int(np.nanargmax(mean_prob))})
        session_pred = pd.DataFrame(session_rows)
        session_balanced = balanced_accuracy_score(session_pred.true, session_pred.pred)
        metric_rows.append({
            "feature_set": feature_name, "n_features": len(columns),
            "unit_level_grouped_cv_balanced_accuracy": float(unit_balanced),
            "session_level_balanced_accuracy": float(session_balanced),
            "n_session_groups": len(session_pred),
        })
        for true in range(6):
            for pred in range(6):
                prediction_rows.append({"feature_set": feature_name, "true_roi": ROIS[true],
                                        "predicted_roi": ROIS[pred], "row_fraction": float(unit_cm[true, pred])})
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def make_figure(audit: pd.DataFrame, decode: pd.DataFrame, confusion: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for method, color, marker in (("kmeans", "#3B82A0", "o"), ("gmm_diag", "#C7662A", "s")):
        q = audit[audit.method.eq(method)]
        axes[0].plot(q.k, q.heldout_balanced_accuracy, color=color, marker=marker, lw=2,
                     label=method)
        axes[1].plot(q.k, q.roi_cluster_cramers_v_descriptive, color=color, marker=marker, lw=2,
                     label=method)
    axes[0].set(title="Unsupervised phenotype stability", xlabel="Number of clusters K",
                ylabel="Held-out-image balanced accuracy", ylim=(0, 1))
    axes[1].set(title="Post-hoc ROI association", xlabel="Number of clusters K",
                ylabel="Cramer's V (descriptive)", ylim=(0, 1))
    for ax in axes[:2]:
        ax.grid(axis="y", color="#D8D8D8", lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False)

    best = decode.sort_values("session_level_balanced_accuracy", ascending=False).iloc[0]
    q = confusion[confusion.feature_set.eq(best.feature_set)]
    matrix = q.pivot(index="true_roi", columns="predicted_roi", values="row_fraction").reindex(
        index=ROIS, columns=ROIS).to_numpy(float)
    image = axes[2].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(6):
        for j in range(6):
            axes[2].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if matrix[i, j] > 0.55 else "#222222")
    axes[2].set_xticks(range(6), ROI_LABELS, rotation=30)
    axes[2].set_yticks(range(6), ROI_LABELS)
    axes[2].set(xlabel="Predicted ROI", ylabel="True ROI",
                title=f"Session-held-out ROI decoder\n{best.feature_set}; session BA={best.session_level_balanced_accuracy:.3f}")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Can clustering or supervised dynamics separate visual ROIs?", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "01_clustering_and_roi_decoding_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dynamic = pd.read_csv(DYNAMIC / "unit_dynamic_phenotypes.csv")
    temporal = pd.read_csv(TEMPORAL / "all_unit_observed_temporal_metrics.csv")
    keep_temporal = ["unit_global", "evoked_centroid_ms", "mode1_centroid_ms",
                     "effective_temporal_rank", "nonwarp_residual_fraction"]
    frame = dynamic.merge(temporal[keep_temporal], on="unit_global", how="inner", validate="one_to_one")
    frame["mean_gain_fraction"] = (frame.train_gain_fraction + frame.heldout_gain_fraction) / 2
    frame["mean_warp_fraction"] = (frame.train_warp_fraction + frame.heldout_warp_fraction) / 2
    frame["mean_nonwarp_fraction"] = (frame.train_nonwarp_fraction + frame.heldout_nonwarp_fraction) / 2

    audit = unsupervised_audit(frame)
    feature_sets = {
        "3 composition fractions": ["mean_gain_fraction", "mean_warp_fraction", "mean_nonwarp_fraction"],
        "extended observed dynamics": ["mean_gain_fraction", "mean_warp_fraction", "mean_nonwarp_fraction",
                                       "evoked_centroid_ms", "mode1_centroid_ms", "effective_temporal_rank",
                                       "nonwarp_residual_fraction"],
    }
    decode, confusion = grouped_roi_decode(frame, feature_sets)
    audit.to_csv(OUT / "unsupervised_cluster_model_audit.csv", index=False, encoding="utf-8-sig")
    decode.to_csv(OUT / "session_heldout_roi_decoding.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(OUT / "session_heldout_roi_confusion.csv", index=False, encoding="utf-8-sig")
    make_figure(audit, decode, confusion)
    report = {
        "cluster_selection_rule": "training silhouette and held-out-image stability only; ROI association post-hoc",
        "roi_decoder": "multinomial logistic regression; 4-fold StratifiedGroupKFold by monkey|session",
        "chance_balanced_accuracy": 1 / 6,
    }
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(audit.to_string(index=False), flush=True)
    print(decode.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
