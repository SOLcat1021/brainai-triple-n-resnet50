"""Paired comparison of direct G/T/D prediction and windowed-curve projection."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
DIRECT_DIR = PROJECT / "results" / "resnet_depth_to_GTD_discovery1200_2026-08-27"
WINDOWED_DIR = PROJECT / "results" / "proxy_GTD_oof_recovery_discovery1200_2026-08-27"
OUT = PROJECT / "results" / "direct_GTD_vs_windowed_proxy_discovery1200_2026-08-27"

ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
ROI_LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
ROI_COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
DIRECTIONS = ["G", "T", "D"]
DIRECT_MODEL = "all_depths"
SEED = 20260827


def signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = abs(float(np.mean(values)))
    if len(values) <= 18:
        null = np.asarray([
            abs(float(np.mean(values * np.asarray(signs))))
            for signs in product((-1, 1), repeat=len(values))
        ])
    else:
        rng = np.random.default_rng(SEED + len(values))
        signs = rng.choice((-1, 1), size=(200_000, len(values)))
        null = np.abs(np.mean(signs * values[None], axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def load_paired() -> pd.DataFrame:
    direct = pd.read_csv(DIRECT_DIR / "unit_nested_depth_GTD_mapping.csv")
    direct = direct.loc[direct.model.eq(DIRECT_MODEL), [
        "unit_global", "roi", "monkey", "session", "direction", "oof_r", "oof_r2"
    ]].rename(columns={"oof_r": "direct_gtd_r", "oof_r2": "direct_gtd_r2"})

    windowed = pd.read_csv(WINDOWED_DIR / "unit_GTD_proxy_recovery.csv")
    windowed = windowed[[
        "unit_global", "roi", "monkey", "session", "direction", "direct_r", "direct_r2"
    ]].rename(columns={"direct_r": "windowed_r", "direct_r2": "windowed_r2"})

    keys = ["unit_global", "roi", "monkey", "session", "direction"]
    paired = direct.merge(windowed, on=keys, how="inner", validate="one_to_one")
    if len(paired) != 1200 * 3:
        raise RuntimeError(f"Expected 3,600 paired unit-directions, found {len(paired):,}")
    if not np.all(paired.groupby(["roi", "direction"]).size().to_numpy() == 200):
        raise RuntimeError("Frozen 200-unit-per-ROI pairing changed")
    paired["delta_r"] = paired.direct_gtd_r - paired.windowed_r
    paired["delta_r2"] = paired.direct_gtd_r2 - paired.windowed_r2
    return paired


def summarize(paired: pd.DataFrame):
    session = paired.groupby(
        ["roi", "monkey", "session", "direction"], as_index=False
    )[["direct_gtd_r", "windowed_r", "direct_gtd_r2", "windowed_r2", "delta_r", "delta_r2"]].median()

    roi = session.groupby(["roi", "direction"], as_index=False)[[
        "direct_gtd_r", "windowed_r", "direct_gtd_r2", "windowed_r2", "delta_r", "delta_r2"
    ]].median()

    tests = []
    for scope, group in [("all_rois", session)]:
        for direction in DIRECTIONS:
            values = group.loc[group.direction.eq(direction), "delta_r2"].to_numpy(float)
            tests.append({
                "scope": scope,
                "direction": direction,
                "n_session_roi_groups": len(values),
                "median_direct_gtd_r2": float(group.loc[group.direction.eq(direction), "direct_gtd_r2"].median()),
                "median_windowed_r2": float(group.loc[group.direction.eq(direction), "windowed_r2"].median()),
                "median_delta_r2": float(np.median(values)),
                "mean_delta_r2": float(np.mean(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "signflip_p": signflip_p(values),
            })
    corrected = bh(np.asarray([row["signflip_p"] for row in tests]))
    for row, value in zip(tests, corrected):
        row["signflip_p_bh_three_directions"] = float(value)
    return session, roi, pd.DataFrame(tests)


def figures(session: pd.DataFrame, roi: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    for direction in DIRECTIONS:
        fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
        subset = session[session.direction.eq(direction)]
        for xi, area in enumerate(ROIS):
            values = subset[subset.roi.eq(area)]
            jitter = rng.uniform(-0.035, 0.035, len(values))
            for offset, (_, row) in zip(jitter, values.iterrows()):
                ax.plot([xi - .14 + offset, xi + .14 + offset],
                        [row.windowed_r2, row.direct_gtd_r2],
                        color="#A8A8A8", lw=.65, alpha=.55, zorder=1)
                ax.scatter(xi - .14 + offset, row.windowed_r2, s=14,
                           facecolor="white", edgecolor=ROI_COLORS[area], lw=.7, zorder=2)
                ax.scatter(xi + .14 + offset, row.direct_gtd_r2, s=14,
                           color=ROI_COLORS[area], edgecolor="none", zorder=2)
            summary = roi[(roi.roi.eq(area)) & (roi.direction.eq(direction))].iloc[0]
            ax.plot([xi - .14, xi + .14], [summary.windowed_r2, summary.direct_gtd_r2],
                    color="#1A1A1A", lw=2.2, zorder=3)
            ax.scatter(xi - .14, summary.windowed_r2, s=58, facecolor="white",
                       edgecolor="#1A1A1A", lw=1.3, zorder=4)
            ax.scatter(xi + .14, summary.direct_gtd_r2, s=58,
                       color="#1A1A1A", edgecolor="none", zorder=4)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_xticks(range(len(ROIS)), [ROI_LABELS[x] for x in ROIS])
        ax.set_ylabel("Held-out raw R²")
        ax.set_title(f"{direction}: direct coefficient model vs windowed-curve proxy",
                     fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", lw=.6)
        ax.set_axisbelow(True)
        ax.scatter([], [], s=38, facecolor="white", edgecolor="#333333", label="Windowed curve → coefficient")
        ax.scatter([], [], s=38, color="#333333", label="Direct coefficient model")
        ax.legend(frameon=False, loc="best")
        fig.text(.5, .006,
                 "Small pairs: monkey × session × ROI medians. Large pairs: median across sessions. Same 1,200 frozen units and image folds.",
                 ha="center", fontsize=8.5)
        fig.savefig(OUT / f"0{DIRECTIONS.index(direction) + 1}_{direction}_direct_vs_windowed_R2.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired = load_paired()
    session, roi, tests = summarize(paired)
    paired.to_csv(OUT / "unit_paired_direct_vs_windowed.csv", index=False, encoding="utf-8-sig")
    session.to_csv(OUT / "session_paired_direct_vs_windowed.csv", index=False, encoding="utf-8-sig")
    roi.to_csv(OUT / "roi_paired_direct_vs_windowed.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "session_signflip_tests.csv", index=False, encoding="utf-8-sig")
    figures(session, roi)
    audit = {
        "frozen_units": 1200,
        "images": 1000,
        "folds": 5,
        "direct_model": "all_depths direct ridge readout to fold-defined G/T/D",
        "windowed_model": "existing per-window OOF response proxy projected to the same fold-defined G/T/D basis",
        "primary_metric": "raw held-out R2 without affine calibration",
        "pairing_level": "unit_global x direction; summarized within monkey x session x ROI",
        "interpretation_limit": "The feature/readout structures differ, so this is an operational pipeline comparison, not an isolated causal test of supervision target.",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tests.to_string(index=False), flush=True)
    print(roi.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
