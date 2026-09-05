"""Cluster response-only dynamic phenotypes and recover them on held-out images.

For each unit, one image half defines a temporal basis and three variance
fractions: gain (rank-1), phase/warp (first and second derivatives), and strict
non-warp residual. K-means is fit only to training-half compositions. The
frozen basis, scaler, and centroids classify the opposite 500 images.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, confusion_matrix, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "dynamic_phenotype_clustering_heldout_2026-08-27"
SEED = 20260827
N_CLUSTERS = 4
EPS = 1e-6
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
TYPE_ORDER = ["gain-enriched", "phase-warp-enriched", "nonwarp-enriched", "intermediate"]


def orthonormal_components(h1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1 = h1 / max(float(np.linalg.norm(h1)), 1e-12)
    nuisance = np.column_stack([h1, np.gradient(h1), np.gradient(np.gradient(h1))])
    q, r = np.linalg.qr(nuisance)
    keep = np.abs(np.diag(r)) > 1e-8
    q = q[:, keep]
    return q[:, :1], q[:, 1:]


def fractions(curves: np.ndarray, gain: np.ndarray, warp: np.ndarray) -> np.ndarray:
    centered = curves - curves.mean(0, keepdims=True)
    total = float(np.sum(centered**2))
    if total <= 1e-12:
        return np.asarray([1 / 3, 1 / 3, 1 / 3], float)
    gain_energy = float(np.sum((centered @ gain) ** 2))
    warp_energy = float(np.sum((centered @ warp) ** 2)) if warp.shape[1] else 0.0
    residual = max(total - gain_energy - warp_energy, 0.0)
    ans = np.asarray([gain_energy, warp_energy, residual], float)
    return ans / max(float(ans.sum()), 1e-12)


def split_features(responses: np.memmap, units: np.ndarray, train: np.ndarray,
                   test: np.ndarray, post: np.ndarray, batch_size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    train_features = np.empty((len(units), 3), np.float32)
    test_features = np.empty((len(units), 3), np.float32)
    for start in range(0, len(units), batch_size):
        stop = min(start + batch_size, len(units))
        block = np.asarray(responses[:, :, units[start:stop]], np.float32)[post].transpose(2, 1, 0)
        for bi, curves in enumerate(block):
            train_curves = curves[train]
            centered = train_curves - train_curves.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            gain, warp = orthonormal_components(vt[0])
            train_features[start + bi] = fractions(train_curves, gain, warp)
            test_features[start + bi] = fractions(curves[test], gain, warp)
        if start % (batch_size * 10) == 0:
            print(f"features {start}/{len(units)}", flush=True)
    return train_features, test_features


def clr(x: np.ndarray) -> np.ndarray:
    logx = np.log(np.maximum(np.asarray(x, float), EPS))
    return logx - logx.mean(1, keepdims=True)


def cluster_names(centers_fraction: np.ndarray) -> dict[int, str]:
    # Assign three distinct clusters to the three simplex vertices; the
    # remaining center is called mixed. Labels are learned without ROI data.
    rows, cols = linear_sum_assignment(-centers_fraction.T)
    names: dict[int, str] = {}
    axis_names = ["gain-enriched", "phase-warp-enriched", "nonwarp-enriched"]
    for axis, cluster in zip(rows, cols):
        names[int(cluster)] = axis_names[int(axis)]
    for cluster in range(len(centers_fraction)):
        names.setdefault(cluster, "intermediate")
    return names


def fit_predict(train_features: np.ndarray, test_features: np.ndarray, seed: int):
    scaler = StandardScaler().fit(clr(train_features))
    train_z = scaler.transform(clr(train_features))
    test_z = scaler.transform(clr(test_features))
    model = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=50).fit(train_z)
    train_label = model.labels_
    test_distance = model.transform(test_z)
    test_label = np.argmin(test_distance, axis=1)
    order = np.argsort(test_distance, axis=1)
    margin = test_distance[np.arange(len(test_label)), order[:, 1]] - test_distance[np.arange(len(test_label)), order[:, 0]]
    centers_fraction = np.vstack([train_features[train_label == k].mean(0) for k in range(N_CLUSTERS)])
    names = cluster_names(centers_fraction)
    return train_label, test_label, margin, centers_fraction, names, model.inertia_


def metrics(train_label: np.ndarray, test_label: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 99)
    null = np.empty(1000, float)
    for i in range(len(null)):
        null[i] = np.mean(train_label == rng.permutation(test_label))
    return {
        "accuracy": float(np.mean(train_label == test_label)),
        "balanced_accuracy": float(balanced_accuracy_score(train_label, test_label)),
        "adjusted_rand_index": float(adjusted_rand_score(train_label, test_label)),
        "normalized_mutual_information": float(normalized_mutual_info_score(train_label, test_label)),
        "majority_baseline": float(np.max(np.bincount(train_label)) / len(train_label)),
        "permutation_accuracy_mean": float(null.mean()),
        "permutation_p": float((1 + np.sum(null >= np.mean(train_label == test_label))) / (len(null) + 1)),
    }


def named(labels: np.ndarray, names: dict[int, str]) -> np.ndarray:
    return np.asarray([names[int(x)] for x in labels], object)


def make_figure(unit: pd.DataFrame, centers: pd.DataFrame, summary: dict[str, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = {"gain-enriched": "#4E79A7", "phase-warp-enriched": "#F28E2B",
              "nonwarp-enriched": "#59A14F", "intermediate": "#8C6BB1", "unresolved": "#999999"}

    ax = axes[0, 0]
    sample = unit.sample(min(7000, len(unit)), random_state=SEED)
    x = sample.train_warp_fraction + 0.5 * sample.train_nonwarp_fraction
    y = np.sqrt(3) / 2 * sample.train_nonwarp_fraction
    for label in TYPE_ORDER:
        q = sample.train_type.eq(label)
        ax.scatter(x[q], y[q], s=7, alpha=0.28, color=colors[label], edgecolors="none", label=label)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="#333333", lw=1)
    ax.text(-0.02, -0.035, "gain", ha="right")
    ax.text(1.02, -0.035, "phase/warp", ha="left")
    ax.text(0.5, np.sqrt(3) / 2 + 0.025, "nonwarp", ha="center")
    ax.set_title("Training-half variance compositions", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[0, 1]
    cm = confusion_matrix(unit.train_type, unit.test_type, labels=TYPE_ORDER, normalize="true")
    image = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(4), ["gain", "phase", "nonwarp", "mixed"], rotation=25)
    ax.set_yticks(range(4), ["gain", "phase", "nonwarp", "mixed"])
    ax.set_xlabel("Predicted from held-out 500 images")
    ax.set_ylabel("Cluster from training 500 images")
    ax.set_title(f"Held-out recovery | balanced accuracy={summary['balanced_accuracy']:.3f}", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")

    ax = axes[1, 0]
    c = centers.set_index("type").reindex(TYPE_ORDER)
    xx = np.arange(4); width = 0.24
    ax.bar(xx - width, c.gain_fraction, width, label="gain", color="#4E79A7")
    ax.bar(xx, c.warp_fraction, width, label="phase/warp", color="#F28E2B")
    ax.bar(xx + width, c.nonwarp_fraction, width, label="nonwarp", color="#59A14F")
    ax.set_xticks(xx, ["gain", "phase", "nonwarp", "mixed"])
    ax.set_ylabel("Mean training variance fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Learned cluster centers", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color="#D8D8D8", lw=0.7); ax.set_axisbelow(True)

    ax = axes[1, 1]
    stable = unit.assign(stable=unit.train_type.eq(unit.test_type)).groupby("roi").stable.mean().reindex(ROIS)
    ax.bar(np.arange(6), stable, color="#5B7F95")
    ax.axhline(summary["majority_baseline"], color="#A14A3B", ls="--", lw=1.5, label="global majority baseline")
    ax.set_xticks(np.arange(6), ["V1", "V2", "V4", "pIT", "mIT", "aIT"])
    ax.set_ylabel("Train/held-out class agreement")
    ax.set_ylim(0, 1)
    ax.set_title("Image-held-out stability by ROI", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#D8D8D8", lw=0.7); ax.set_axisbelow(True)

    fig.suptitle("Response-only dynamic phenotypes: clustering and held-out image recovery",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.003,
             "All 23,960 mapped units (released consistency >0.4). Temporal bases and clusters use training images only; no ROI, semantic label, or visual proxy enters clustering.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "01_dynamic_phenotype_heldout_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    meta = meta[meta.released_independent_consistency.gt(0.4)].sort_values("unit_global").reset_index(drop=True)
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    post = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    order = np.random.default_rng(SEED).permutation(1000)
    half_a, half_b = np.sort(order[:500]), np.sort(order[500:])
    units = meta.unit_global.to_numpy(int)

    a_train, b_from_a = split_features(responses, units, half_a, half_b, post)
    label_a, pred_b, margin_b, centers_a, names_a, inertia_a = fit_predict(a_train, b_from_a, SEED)
    summary_a = metrics(label_a, pred_b)

    b_train, a_from_b = split_features(responses, units, half_b, half_a, post)
    label_b, pred_a, margin_a, centers_b, names_b, inertia_b = fit_predict(b_train, a_from_b, SEED + 1)
    summary_b = metrics(label_b, pred_a)

    train_type = named(label_a, names_a)
    test_type = named(pred_b, names_a)
    unit = meta[["unit_global", "monkey", "session", "roi", "native_area", "unit_type_name",
                 "released_independent_consistency"]].copy()
    for i, component in enumerate(("gain", "warp", "nonwarp")):
        unit[f"train_{component}_fraction"] = a_train[:, i]
        unit[f"heldout_{component}_fraction"] = b_from_a[:, i]
    unit["train_type"] = train_type
    unit["test_type"] = test_type
    unit["heldout_prediction_margin"] = margin_b
    unit["stable_type"] = np.where(train_type == test_type, train_type, "unresolved")
    unit.to_csv(OUT / "unit_dynamic_phenotypes.csv", index=False, encoding="utf-8-sig")

    centers = pd.DataFrame(centers_a, columns=["gain_fraction", "warp_fraction", "nonwarp_fraction"])
    centers["cluster"] = np.arange(N_CLUSTERS)
    centers["type"] = centers.cluster.map(names_a)
    centers.to_csv(OUT / "training_cluster_centers.csv", index=False, encoding="utf-8-sig")

    stability = unit.assign(stable=unit.train_type.eq(unit.test_type)).groupby(
        ["roi", "unit_type_name"], as_index=False
    ).agg(n_units=("unit_global", "size"), heldout_accuracy=("stable", "mean"))
    stability.to_csv(OUT / "heldout_accuracy_by_roi_and_unit_type.csv", index=False, encoding="utf-8-sig")

    audit = {
        "selection": "released_independent_consistency > 0.4",
        "n_units": len(unit),
        "n_images_train": 500,
        "n_images_heldout": 500,
        "seed": SEED,
        "response_only": True,
        "proxy_predictions_used": False,
        "roi_labels_used_for_clustering": False,
        "components": ["rank1_gain", "rank1_derivative_and_second_derivative_phase_warp", "strict_residual_nonwarp"],
        "primary_A_to_B": summary_a,
        "reciprocal_B_to_A": summary_b,
        "kmeans_inertia_A": float(inertia_a),
        "kmeans_inertia_B": float(inertia_b),
        "cluster_names_A": {str(k): v for k, v in names_a.items()},
        "cluster_names_B": {str(k): v for k, v in names_b.items()},
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(unit, centers, summary_a)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(unit.stable_type.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
