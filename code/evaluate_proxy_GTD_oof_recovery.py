"""Five-fold OOF recovery of neural G/T/D coefficients by existing proxies."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
from plot_orthogonal_temporal_modes_2d import raw_directions, symmetric_orthogonalize  # noqa: E402
from run_trin_closed_loop_validation import fixed_folds  # noqa: E402

BANK = PROJECT / "proxy_bank_10ms_all_windows"
FROZEN = PROJECT / "results" / "orthogonal_temporal_modes_2d_high_consistency_2026-08-27"
OUT = PROJECT / "results" / "proxy_GTD_oof_recovery_discovery1200_2026-08-27"
RESPONSES = BANK / "trin_all_units_10ms_responses.npy"
WINDOWS = BANK / "windows_10ms.npy"
ROIS = ["V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT"]
LABELS = dict(zip(ROIS, ["V1", "V2", "V4", "pIT", "mIT", "aIT"]))
COLORS = dict(zip(ROIS, ["#2166AC", "#1B9E77", "#6A3D9A", "#E6AB02", "#E66101", "#C51B7D"]))
DIRECTIONS = ["G", "T", "D"]
SEED = 20260827


def slug(roi: str) -> str:
    return roi.lower().replace(" ", "_")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return np.nan
    x = a[finite] - np.mean(a[finite]); y = b[finite] - np.mean(b[finite])
    den = np.sqrt(np.sum(x**2) * np.sum(y**2))
    return float(np.sum(x * y) / den) if den > 1e-12 else np.nan


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    finite = np.isfinite(y) & np.isfinite(prediction)
    if finite.sum() < 3:
        return np.nan
    target = y[finite]; pred = prediction[finite]
    den = float(np.sum((target - np.mean(target))**2))
    return 1 - float(np.sum((target - pred)**2)) / den if den > 1e-12 else np.nan


def affine_calibration(train_prediction: np.ndarray, train_target: np.ndarray,
                       test_prediction: np.ndarray) -> np.ndarray:
    finite = np.isfinite(train_prediction) & np.isfinite(train_target)
    x = train_prediction[finite]; y = train_target[finite]
    if len(x) < 3 or np.var(x) <= 1e-12:
        return np.full_like(test_prediction, np.mean(y) if len(y) else np.nan, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return intercept + slope * test_prediction


def load_roi_curves(roi: str, units: np.ndarray, responses: np.memmap,
                    baseline_idx: np.ndarray, time_idx: np.ndarray):
    neural_raw = np.asarray(responses[:, :, units], np.float32)
    neural_base = np.nanmean(neural_raw[baseline_idx], axis=0)
    neural = (neural_raw[time_idx] - neural_base[None]).transpose(2, 1, 0)

    path = BANK / f"{slug(roi)}_proxy_bank.h5"
    with h5py.File(path, "r") as h:
        bank_units = h["unit_global"][:].astype(int)
        lookup = {int(unit): i for i, unit in enumerate(bank_units)}
        local = np.asarray([lookup[int(unit)] for unit in units], int)
        order = np.argsort(local); sorted_local = local[order]
        proxy_raw = np.empty((len(baseline_idx) + len(time_idx), 1000, len(units)), np.float32)
        all_idx = np.concatenate([baseline_idx, time_idx])
        for oi, wi in enumerate(all_idx):
            slab = np.asarray(h["oof_prediction"][wi, :, sorted_local], np.float32)
            proxy_raw[oi][:, order] = slab
    nbase = len(baseline_idx)
    proxy_base = np.nanmean(proxy_raw[:nbase], axis=0)
    proxy = (proxy_raw[nbase:] - proxy_base[None]).transpose(2, 1, 0)
    return neural, proxy


def evaluate_unit(unit_row: pd.Series, neural: np.ndarray, proxy: np.ndarray,
                  times: np.ndarray):
    fold_rows = []
    pooled = {d: {"target": [], "prediction": [], "calibrated": []} for d in DIRECTIONS}
    for fold, (train, test) in enumerate(fixed_folds(neural.shape[0])):
        template = np.mean(neural[train], axis=0)
        physical, _ = raw_directions(template, times)
        if physical is None:
            continue
        basis, _, similarity = symmetric_orthogonalize(physical)
        if basis is None:
            continue
        true_coeff = (neural - template[None]) @ basis
        proxy_coeff = (proxy - template[None]) @ basis
        for di, direction in enumerate(DIRECTIONS):
            calibrated = affine_calibration(
                proxy_coeff[train, di], true_coeff[train, di], proxy_coeff[test, di])
            item = {
                "unit_global": int(unit_row.unit_global), "roi": str(unit_row.roi),
                "monkey": str(unit_row.monkey), "session": int(unit_row.session),
                "fold": fold, "direction": direction, "n_test_images": len(test),
                "direct_r": corr(true_coeff[test, di], proxy_coeff[test, di]),
                "direct_r2": r2(true_coeff[test, di], proxy_coeff[test, di]),
                "affine_calibrated_r2": r2(true_coeff[test, di], calibrated),
                "orthogonal_raw_similarity": float(similarity[di]),
            }
            fold_rows.append(item)
            pooled[direction]["target"].append(true_coeff[test, di])
            pooled[direction]["prediction"].append(proxy_coeff[test, di])
            pooled[direction]["calibrated"].append(calibrated)
    unit_rows = []
    for direction in DIRECTIONS:
        target = np.concatenate(pooled[direction]["target"])
        prediction = np.concatenate(pooled[direction]["prediction"])
        calibrated = np.concatenate(pooled[direction]["calibrated"])
        unit_rows.append({
            "unit_global": int(unit_row.unit_global), "roi": str(unit_row.roi),
            "monkey": str(unit_row.monkey), "session": int(unit_row.session),
            "direction": direction, "n_images": len(target),
            "direct_r": corr(target, prediction), "direct_r2": r2(target, prediction),
            "affine_calibrated_r2": r2(target, calibrated),
        })
    return fold_rows, unit_rows


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    observed = abs(float(np.mean(values)))
    if len(values) <= 18:
        null = np.asarray([abs(float(np.mean(values * np.asarray(signs))))
                           for signs in product((-1, 1), repeat=len(values))])
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1, 1), size=(200_000, len(values)))
        null = np.abs(np.mean(signs * values[None], axis=1))
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.minimum(q, 1)
    return out


def summarize(unit: pd.DataFrame):
    session = unit.groupby(["roi", "monkey", "session", "direction"], as_index=False)[
        ["direct_r", "direct_r2", "affine_calibrated_r2"]].median()
    rows = []
    for direction in DIRECTIONS:
        q = unit[unit.direction.eq(direction)]
        s = session[session.direction.eq(direction)]
        rows.append({
            "direction": direction, "n_units": len(q), "n_session_roi_groups": len(s),
            "median_unit_direct_r": float(q.direct_r.median()),
            "median_unit_direct_r2": float(q.direct_r2.median()),
            "median_unit_affine_calibrated_r2": float(q.affine_calibrated_r2.median()),
            "positive_unit_r_fraction": float(np.mean(q.direct_r > 0)),
            "positive_unit_calibrated_r2_fraction": float(np.mean(q.affine_calibrated_r2 > 0)),
            "session_signflip_p_direct_r": signflip_p(s.direct_r.to_numpy(float), SEED + len(rows)),
            "session_signflip_p_calibrated_r2": signflip_p(
                s.affine_calibrated_r2.to_numpy(float), SEED + 10 + len(rows)),
        })
    summary = pd.DataFrame(rows)
    summary["session_signflip_p_direct_r_bh"] = bh(summary.session_signflip_p_direct_r.to_numpy())
    summary["session_signflip_p_calibrated_r2_bh"] = bh(
        summary.session_signflip_p_calibrated_r2.to_numpy())
    summary["passes_recovery_rule"] = (
        summary.median_unit_direct_r.ge(.10) &
        summary.median_unit_affine_calibrated_r2.gt(0) &
        summary.session_signflip_p_direct_r_bh.lt(.05) &
        summary.session_signflip_p_calibrated_r2_bh.lt(.05)
    )
    return session, summary


def metric_figure(unit: pd.DataFrame, session: pd.DataFrame, metric: str,
                  ylabel: str, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), constrained_layout=True)
    for ax, direction in zip(axes, DIRECTIONS):
        qd = unit[unit.direction.eq(direction)]
        arrays = [qd.loc[qd.roi.eq(roi), metric].to_numpy(float) for roi in ROIS]
        violin = ax.violinplot(arrays, positions=np.arange(6), widths=.82,
                               showmedians=True, showextrema=False, points=150)
        for body, roi in zip(violin["bodies"], ROIS):
            body.set_facecolor(COLORS[roi]); body.set_alpha(.28); body.set_edgecolor("none")
        violin["cmedians"].set_color("#222222")
        for ri, roi in enumerate(ROIS):
            s = session[(session.direction.eq(direction)) & session.roi.eq(roi)]
            ax.scatter(np.full(len(s), ri), s[metric], s=23, color=COLORS[roi],
                       edgecolors="#222222", linewidths=.35, alpha=.85)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_xticks(np.arange(6), [LABELS[r] for r in ROIS])
        ax.set_ylabel(ylabel); ax.set_title(direction, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", lw=.6); ax.set_axisbelow(True)
    fig.suptitle("Existing ResNet proxy: held-out recovery of neural G/T/D", fontweight="bold")
    fig.text(.5, .006,
             "Violin: 1,200 frozen discovery units. Dots: monkey x session x ROI medians. GTD bases are defined from training images only.",
             ha="center", fontsize=8.5)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def heatmap(summary_roi: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for ax, metric, title, limits in [
        (axes[0], "direct_r", "Direct coefficient correlation", (-.2, .4)),
        (axes[1], "affine_calibrated_r2", "Training-calibrated held-out R²", (-.1, .2)),
    ]:
        matrix = summary_roi.pivot(index="roi", columns="direction", values=metric).reindex(
            index=ROIS, columns=DIRECTIONS).to_numpy(float)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=limits[0], vmax=limits[1], aspect="auto")
        for i in range(6):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                        color="white" if abs(matrix[i, j]) > .7 * max(abs(limits[0]), abs(limits[1])) else "#222222")
        ax.set_xticks(range(3), DIRECTIONS); ax.set_yticks(range(6), [LABELS[r] for r in ROIS])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Session-level median proxy recovery", fontweight="bold")
    fig.savefig(OUT / "03_proxy_GTD_recovery_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(FROZEN / "unit_orthogonal_GTD_weights_and_modes.csv")
    if len(frozen) != 1200 or not np.all(frozen.groupby("roi").size().reindex(ROIS) == 200):
        raise RuntimeError("Frozen discovery cohort is not the expected 200 units per ROI")
    responses = np.load(RESPONSES, mmap_mode="r")
    windows = np.load(WINDOWS).astype(int)
    baseline_idx = np.flatnonzero(windows[:, 1] < 0)
    time_idx = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time_idx].mean(axis=1).astype(float)
    all_fold, all_unit = [], []
    for roi in ROIS:
        q = frozen[frozen.roi.eq(roi)].sort_values("unit_global").reset_index(drop=True)
        units = q.unit_global.to_numpy(int)
        neural, proxy = load_roi_curves(roi, units, responses, baseline_idx, time_idx)
        for ui, row in q.iterrows():
            fold_rows, unit_rows = evaluate_unit(row, neural[ui], proxy[ui], times)
            all_fold.extend(fold_rows); all_unit.extend(unit_rows)
        print(f"evaluated {roi}: {len(q)} units", flush=True)
    fold = pd.DataFrame(all_fold); unit = pd.DataFrame(all_unit)
    fold.to_csv(OUT / "unit_fold_GTD_proxy_recovery.csv.gz", index=False, compression="gzip")
    unit.to_csv(OUT / "unit_GTD_proxy_recovery.csv", index=False, encoding="utf-8-sig")
    session, summary = summarize(unit)
    session.to_csv(OUT / "session_roi_GTD_proxy_recovery.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "direction_recovery_decision.csv", index=False, encoding="utf-8-sig")
    roi_summary = session.groupby(["roi", "direction"], as_index=False)[
        ["direct_r", "direct_r2", "affine_calibrated_r2"]].median()
    roi_summary.to_csv(OUT / "roi_GTD_proxy_recovery_summary.csv", index=False, encoding="utf-8-sig")
    metric_figure(unit, session, "direct_r", "Held-out coefficient correlation",
                  "01_proxy_GTD_direct_correlation.png")
    metric_figure(unit, session, "affine_calibrated_r2", "Held-out calibrated R²",
                  "02_proxy_GTD_calibrated_R2.png")
    heatmap(roi_summary)
    audit = {
        "frozen_cohort": str(FROZEN / "unit_orthogonal_GTD_weights_and_modes.csv"),
        "n_units": len(frozen), "n_images": 1000, "folds": 5,
        "fold_source": "same fixed five image folds used by existing proxy OOF training",
        "basis_definition": "neural training images only per fold; raw G/T/D then Lowdin orthogonalization",
        "test_target": "held-out neural coefficient relative to training neural template",
        "proxy_prediction": "existing image-OOF 10-ms firing-rate predictions, baseline subtracted and projected into the same fold basis",
        "affine_calibration": "intercept and slope fit on training-image OOF coefficients only",
        "recovery_rule": "median unit r >= 0.10, median calibrated R2 > 0, and session sign-flip BH p < 0.05 for both",
        "proxy_retraining_performed": False,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
