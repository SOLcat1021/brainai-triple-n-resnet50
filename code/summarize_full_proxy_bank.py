"""Generate compact CSV summaries from the complete proxy data (no report)."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
SIM = BANK / "similarities"
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")


def slug(s: str) -> str:
    return s.lower().replace(" ", "_")


def main() -> None:
    meta = pd.read_csv(BANK / "unit_metadata.csv").sort_values("unit_global")
    windows = np.load(BANK / "windows_10ms.npy")
    rows = []
    for roi in ROIS:
        with h5py.File(BANK / f"{slug(roi)}_proxy_bank.h5", "r") as h:
            r = h["oof_r"][:]
            depth = h["depth"][:]
            full = h["full_fit_r"][:]
        for wi, (start, end) in enumerate(windows):
            rows.append({
                "roi": roi, "t": int(start), "window_end_ms": int(end),
                "n_units": int(r.shape[1]), "mean_oof_r": float(np.nanmean(r[wi])),
                "median_oof_r": float(np.nanmedian(r[wi])),
                "q25_oof_r": float(np.nanquantile(r[wi], .25)),
                "q75_oof_r": float(np.nanquantile(r[wi], .75)),
                "mean_full_fit_r": float(np.nanmean(full[wi])),
                "mean_depth": float(np.nanmean(depth[wi])),
                "std_depth": float(np.nanstd(depth[wi])),
            })
    window_summary = pd.DataFrame(rows)
    window_summary.to_csv(BANK / "proxy_summary_by_roi_and_window.csv", index=False)

    adjacent = pd.read_csv(SIM / "same_unit_adjacent_window_similarity.csv.gz")
    adjacent.groupby(["roi", "t", "next_t"], sort=False).agg(
        n=("unit_global", "size"),
        mean_functional_similarity=("functional_oof_similarity", "mean"),
        median_functional_similarity=("functional_oof_similarity", "median"),
        mean_layer_gate_cosine=("layer_gate_cosine", "mean"),
        mean_signed_weight_cosine=("signed_weight_cosine", "mean"),
        mean_spatial_overlap=("spatial_gaussian_overlap", "mean"),
        mean_absolute_depth_difference=("absolute_depth_difference", "mean"),
    ).reset_index().to_csv(SIM / "same_unit_adjacent_summary_by_roi_and_time.csv", index=False)

    roi_code = {r: i for i, r in enumerate(ROIS)}
    with h5py.File(SIM / "time_matched_cross_unit_top32.h5", "r") as h:
        neighbors = h["neighbor_unit_global"]
        similarities = h["neighbor_similarity"]
        rows = []
        unit_roi = meta.roi.to_numpy(str)
        for wi, (start, end) in enumerate(windows):
            for roi in ROIS:
                units = meta.loc[meta.roi.eq(roi), "unit_global"].to_numpy(np.int32)
                nn = neighbors[wi, units]
                sim = similarities[wi, units].astype(np.float32)
                rows.append({
                    "roi": roi, "t": int(start), "window_end_ms": int(end),
                    "n_units": len(units), "mean_top32_similarity": float(sim.mean()),
                    "median_top32_similarity": float(np.median(sim)),
                    "fraction_neighbors_same_roi": float((unit_roi[nn] == roi).mean()),
                })
        pd.DataFrame(rows).to_csv(SIM / "cross_unit_top32_summary_by_roi_and_time.csv", index=False)


if __name__ == "__main__":
    main()
