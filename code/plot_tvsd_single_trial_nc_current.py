"""Audit and plot NSD-style single-trial NC for the current TVSD proxy bank."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_consistency_ge_0p4_all_units_2026-08-30"
NC_PATH = PROJECT / "support" / "single_trial_nc1_by_window.csv"
OUT = PROJECT / "results" / "tvsd_single_trial_nc_current_2026-08-31"
FIG = OUT / "figures"
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
MONKEYS = ("M1", "M2", "M3", "M4", "M5")
LABEL = {"posterior IT": "Posterior IT", "middle IT": "Middle IT", "anterior IT": "Anterior IT"}


def load_proxy() -> pd.DataFrame:
    rows = []
    for path in sorted(BANK.glob("*_proxy_bank_64d.h5")):
        roi = path.name.removesuffix("_proxy_bank_64d.h5").replace("_", " ")
        roi = {"posterior it": "posterior IT", "middle it": "middle IT", "anterior it": "anterior IT"}.get(roi, roi.upper())
        with h5py.File(path, "r") as h:
            units = h["unit_global"][:].astype(np.int64)
            windows = h["windows_ms"][:].astype(int)
            rr = h["oof_r"][:].astype(np.float32)
        for wi, (start, end) in enumerate(windows):
            for unit, value in zip(units, rr[wi]):
                rows.append((roi, unit, int(start), int(end), float(value)))
    return pd.DataFrame(rows, columns=["roi", "unit_global", "window_start_ms", "window_end_ms", "repeat_mean_oof_r"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    nc = pd.read_csv(NC_PATH)
    proxy = load_proxy()
    keys = ["roi", "unit_global", "window_start_ms", "window_end_ms"]
    frame = nc.merge(proxy, on=keys, how="inner", validate="one_to_one")
    if frame.empty:
        raise RuntimeError("No NC/proxy window matches")
    # Match the archived single-trial conversion used by the existing figures.
    # The archived field is already the variance-component signal term.
    # Do not subtract repeat noise a second time.
    signal = np.maximum(frame["signal_variance"], 0.0)
    frame["single_trial_r_ceiling"] = np.sqrt(np.maximum(signal, 0.0) /
        np.maximum(signal + frame["single_trial_noise_variance"], 1e-20))
    attenuation = np.sqrt((signal + frame["repeat_mean_noise_variance"]) /
                          np.maximum(signal + frame["single_trial_noise_variance"], 1e-20))
    frame["single_trial_equivalent_oof_r"] = frame["repeat_mean_oof_r"] * attenuation
    frame["nc1_variance_fraction_recomputed"] = signal / np.maximum(
        signal + frame["single_trial_noise_variance"], 1e-20)
    frame.to_csv(OUT / "tvsd_current_single_trial_nc_long.csv.gz", index=False, compression="gzip")

    def scatter(ax, q, title):
        x = q.single_trial_r_ceiling.to_numpy(float); y = q.single_trial_equivalent_oof_r.to_numpy(float)
        ax.scatter(x, y, s=3, alpha=.14, color="#377eb8", edgecolors="none", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#d73027", lw=1)
        ax.scatter([np.nanmean(x)], [np.nanmean(y)], marker="+", s=90, color="#b2182b", linewidths=2)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.text(.04, .96, f"n={len(q):,}\nmean=({np.nanmean(x):.3f}, {np.nanmean(y):.3f})\nSpearman ρ={rho:.3f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set(xlim=(0, 1), ylim=(-.25, 1), title=title,
               xlabel="Single-trial correlation ceiling, √NC(1)",
               ylabel="Single-trial-equivalent OOF Pearson r")
        ax.grid(True, color="#dddddd", lw=.4)

    fig, ax = plt.subplots(figsize=(5.2, 4.7), constrained_layout=True)
    scatter(ax, frame, "TVSD current proxies — all matched windows")
    fig.savefig(FIG / "01_all_tvsd_single_trial_nc.png", dpi=320); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, roi in zip(axes.flat, ROIS): scatter(ax, frame[frame.roi.eq(roi)], LABEL.get(roi, roi))
    fig.savefig(FIG / "02_tvsd_single_trial_nc_by_roi.png", dpi=320); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, monkey in zip(axes, MONKEYS): scatter(ax, frame[frame.monkey.eq(monkey)], monkey)
    fig.savefig(FIG / "03_tvsd_single_trial_nc_by_monkey.png", dpi=320); plt.close(fig)

    summary = frame.groupby("roi", sort=False).agg(n=("unit_global", "size"),
        mean_nc=("single_trial_r_ceiling", "mean"), median_nc=("single_trial_r_ceiling", "median"),
        mean_single_trial_r=("single_trial_equivalent_oof_r", "mean")).reset_index()
    summary.to_csv(OUT / "summary_by_roi.csv", index=False, encoding="utf-8-sig")
    by_monkey = frame.groupby("monkey", sort=False).agg(n=("unit_global", "size"),
        mean_nc=("single_trial_r_ceiling", "mean"), median_nc=("single_trial_r_ceiling", "median"),
        mean_single_trial_r=("single_trial_equivalent_oof_r", "mean")).reset_index()
    by_monkey.to_csv(OUT / "summary_by_monkey.csv", index=False, encoding="utf-8-sig")
    audit = {"source_nc": str(NC_PATH), "source_proxy_bank": str(BANK),
             "single_trial": True, "nc_definition": "NSD NC(1) variance-component ceiling",
             "nc_formula": "sqrt(signal_variance/(signal_variance+single_trial_noise_variance)); signal_variance is already the decomposed signal term",
             "matched_windows": sorted(frame.window_start_ms.unique().tolist()),
             "n_rows": int(len(frame)), "n_units": int(frame.unit_global.nunique()),
             "selection_rule": "official released_independent_consistency >= 0.4 only"}
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(by_monkey.to_string(index=False)); print(OUT)


if __name__ == "__main__":
    main()
