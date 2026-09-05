"""Evaluate three fixed low-frequency, directly interpretable temporal modes."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from learn_interpretable_roi_temporal_modes import BANK, OUT, ROIS, ROI_LABELS, load_curves


SEED = 20260827


def fixed_modes(n: int) -> np.ndarray:
    index = np.arange(n, dtype=float) + 0.5
    modes = np.vstack([np.cos(np.pi * k * index / n) for k in (1, 2, 3)])
    modes -= modes.mean(1, keepdims=True)
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    # Signs chosen for intuitive labels below.
    modes[0] *= -1  # positive = later/sustained relative to early
    modes[1] *= -1  # positive = mid-latency transient relative to edges
    return modes


def grouped_decode(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for n_modes in (1, 2, 3):
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + n_modes)
        for fold, (train, test) in enumerate(splitter.split(scores, labels, groups)):
            scaler = StandardScaler().fit(scores[train, :n_modes])
            model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000,
                                       random_state=SEED + fold).fit(
                scaler.transform(scores[train, :n_modes]), labels[train])
            pred = model.predict(scaler.transform(scores[test, :n_modes]))
            rows.append({"n_modes": n_modes, "fold": fold,
                         "balanced_accuracy": balanced_accuracy_score(labels[test], pred)})
    return pd.DataFrame(rows)


def main() -> None:
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    meta = meta[meta.roi.isin(ROIS)].sort_values("unit_global").reset_index(drop=True)
    curves, times = load_curves(meta)
    modes = fixed_modes(curves.shape[1])
    scores = curves @ modes.T
    labels = meta.roi.to_numpy(str)
    groups = (meta.monkey.astype(str) + "|" + meta.session.astype(str)).to_numpy(str)
    decode = grouped_decode(scores, labels, groups)
    decode.to_csv(OUT / "fixed_mode_leave_session_out_decoding.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_ms": times, "late_vs_early": modes[0],
                  "mid_transient_vs_edges": modes[1], "biphasic_asymmetry": modes[2]}).to_csv(
        OUT / "fixed_interpretable_mode_loadings.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    names = ["Late vs early", "Mid transient vs edges", "Biphasic asymmetry"]
    for mode, name in zip(modes, names):
        axes[0].plot(times, mode, lw=2.5, label=name)
    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].set(title="Three fixed low-frequency temporal contrasts",
                xlabel="Time after image onset (ms)", ylabel="Mode loading")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", color="#D8D8D8", lw=0.7)
    q = decode.groupby("n_modes").balanced_accuracy.agg(["mean", "std"]).reset_index()
    axes[1].errorbar(q.n_modes, q["mean"], yerr=q["std"], marker="o", lw=2.5, capsize=4,
                     color="#2F6F8F")
    axes[1].axhline(1 / 6, color="#A14A3B", ls="--", label="6-class chance")
    axes[1].set(title="Leave-session-out ROI decoding", xlabel="Number of fixed modes",
                ylabel="Balanced accuracy", xticks=[1, 2, 3], ylim=(0, 1))
    axes[1].legend(frameon=False); axes[1].grid(axis="y", color="#D8D8D8", lw=0.7)
    fig.suptitle("Interpretability-first temporal modes", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "02_fixed_interpretable_temporal_modes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(q.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
