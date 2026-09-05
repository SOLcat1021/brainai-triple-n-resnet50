"""Locked replication of pilot temporal dynamics on new strict units."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

import pilot_low_rank_temporal_dynamics as pilot
import validate_low_rank_temporal_dynamics as validation
from explore_functional_it_temporal_directions import nonwarp_mode, orthonormal_columns


BANK = pilot.PROJECT / "proxy_bank_64d_expansion_strict_2026-08-27"
OUT = pilot.PROJECT / "results" / "temporal_dynamics_expansion_strict_2026-08-27"
ROIS = ("V2", "V4", "posterior IT")


def load_roi(roi: str, responses: np.memmap, tidx: np.ndarray):
    selected = pd.read_csv(BANK / "selected_units.csv")
    pilot_ids = set(pd.read_csv(pilot.BANK / "selected_units.csv").unit_global.astype(int))
    meta = selected[selected.roi.eq(roi) & ~selected.unit_global.astype(int).isin(pilot_ids)].copy().reset_index(drop=True)
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[tidx][:, :, units], np.float32).transpose(1, 0, 2)
    with h5py.File(BANK / pilot.ROI_FILE[roi], "r") as h:
        all_ids = h["unit_global"][:].astype(int)
        local = np.asarray([np.flatnonzero(all_ids == u)[0] for u in units])
        pred = h["oof_prediction"][:, :, local].astype(np.float32)[tidx].transpose(1, 0, 2)
    return meta, y, pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    tidx = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    configs = validation.modal_configs().set_index("roi")
    order = np.random.default_rng(validation.SEED).permutation(pilot.N_IMAGES)
    halves = (np.sort(order[:500]), np.sort(order[500:]))
    rows = []
    for roi in ROIS:
        rank, lam = int(configs.loc[roi, "rank"]), float(configs.loc[roi, "lambda"])
        meta, y, pred = load_roi(roi, responses, tidx)
        for ui, unit in meta.iterrows():
            for split, train in enumerate(halves):
                test = halves[1 - split]
                mean = y[train, :, ui].mean(0)
                centered = y[train, :, ui] - mean[None]
                h = pilot.svd_basis(centered, lam, rank)[:, :rank]
                nuisance, g = nonwarp_mode(centered, lam)
                extended = orthonormal_columns([*nuisance.T, g])
                target, raw = y[test, :, ui], pred[test, :, ui]
                rank1 = pilot.score_curve(target, pilot.project_prediction(raw, mean, h[:, :1]), mean)["stimulus_r2"]
                dynamic = pilot.score_curve(target, pilot.project_prediction(raw, mean, h), mean)["stimulus_r2"]
                derivative = pilot.score_curve(target, pilot.project_prediction(raw, mean, nuisance[:, :2]), mean)["stimulus_r2"]
                nuisance_r2 = pilot.score_curve(target, pilot.project_prediction(raw, mean, nuisance), mean)["stimulus_r2"]
                nonwarp = pilot.score_curve(target, pilot.project_prediction(raw, mean, extended), mean)["stimulus_r2"]
                tc = (target - mean[None]) @ g; pc = (raw - mean[None]) @ g
                rows.append({
                    "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                    "monkey": str(unit.monkey), "session": int(unit.session), "split": split,
                    "locked_rank": rank, "locked_lambda": lam, "rank1_r2": rank1, "dynamic_r2": dynamic,
                    "dynamic_minus_rank1": dynamic-rank1, "dynamic_minus_derivative_rank2": dynamic-derivative,
                    "nonwarp_delta_r2": nonwarp-nuisance_r2,
                    "nonwarp_coefficient_r": pilot.corr(tc, pc),
                })
    detail = pd.DataFrame(rows)
    unit = detail.groupby(["roi", "unit_global", "monkey", "session"], as_index=False).mean(numeric_only=True)
    summary = []
    for roi in ROIS:
        q = unit[unit.roi.eq(roi)]
        row = {"roi": roi, "n_new_units": len(q)}
        for ci, col in enumerate(("dynamic_minus_rank1", "dynamic_minus_derivative_rank2", "nonwarp_delta_r2")):
            cluster = q.groupby(["monkey", "session"])[col].mean().to_numpy(float)
            row[f"median_{col}"] = float(q[col].median())
            row[f"positive_fraction_{col}"] = float(np.mean(q[col] > 0))
            row[f"session_signflip_p_{col}"] = validation.signflip_p(cluster, 7000 + ci + len(summary))
        row["median_nonwarp_coefficient_r"] = float(q.nonwarp_coefficient_r.median())
        summary.append(row)
    summary = pd.DataFrame(summary)
    detail.to_csv(OUT / "locked_replication_unit_split.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "locked_replication_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "audit.json").write_text(json.dumps({"pilot_units_excluded": True, "locked_configs": True,
                                                  "image_halves": "500/500", "rois": list(ROIS)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
