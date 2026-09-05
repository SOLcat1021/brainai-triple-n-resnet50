"""Locked replication of stimulus-dependent timing on new strict units."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

import pilot_low_rank_temporal_dynamics as pilot
import validate_low_rank_temporal_dynamics as validation
from analyze_stimulus_dependent_latency import derivative_coefficients, infer_shift


BANK = pilot.PROJECT / "proxy_bank_64d_expansion_strict_2026-08-27"
OUT = pilot.PROJECT / "results" / "stimulus_dependent_latency_expansion_2026-08-27"
ROIS = ("posterior IT", "middle IT", "anterior IT")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    tidx = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    configs = validation.modal_configs().set_index("roi")
    pilot_ids = set(pd.read_csv(pilot.BANK / "selected_units.csv").unit_global.astype(int))
    selected = pd.read_csv(BANK / "selected_units.csv")
    order = np.random.default_rng(validation.SEED).permutation(pilot.N_IMAGES)
    halves = (np.sort(order[:500]), np.sort(order[500:]))
    rows = []
    for roi in ROIS:
        meta = selected[selected.roi.eq(roi) & ~selected.unit_global.astype(int).isin(pilot_ids)].copy().reset_index(drop=True)
        units = meta.unit_global.to_numpy(int)
        y = np.asarray(responses[tidx][:, :, units], np.float32).transpose(1, 0, 2)
        with h5py.File(BANK / pilot.ROI_FILE[roi], "r") as h:
            ids = h["unit_global"][:].astype(int)
            local = np.asarray([np.flatnonzero(ids == u)[0] for u in units])
            pred = h["oof_prediction"][:, :, local].astype(np.float32)[tidx].transpose(1, 0, 2)
        lam = float(configs.loc[roi, "lambda"])
        for ui, unit in meta.iterrows():
            for split, train in enumerate(halves):
                test = halves[1-split]
                mean = y[train, :, ui].mean(0)
                template = pilot.svd_basis(y[train, :, ui] - mean[None], lam, 1)[:, 0]
                target, raw = y[test, :, ui], pred[test, :, ui]
                ts, ps = infer_shift(target, mean, template), infer_shift(raw, mean, template)
                tc, pc = derivative_coefficients(target, mean, template), derivative_coefficients(raw, mean, template)
                rows.append({"roi": roi, "unit_global": int(unit.unit_global), "monkey": str(unit.monkey),
                             "session": int(unit.session), "split": split, "shift_r": pilot.corr(ts, ps),
                             "derivative_coefficient_r": pilot.corr(tc[:, 1], pc[:, 1]),
                             "main_amplitude_r": pilot.corr(tc[:, 0], pc[:, 0])})
    detail = pd.DataFrame(rows)
    unit = detail.groupby(["roi", "unit_global", "monkey", "session"], as_index=False).mean(numeric_only=True)
    summary = []
    for roi in ROIS:
        q = unit[unit.roi.eq(roi)]
        cluster = q.groupby(["monkey", "session"]).derivative_coefficient_r.mean().to_numpy(float)
        summary.append({"roi": roi, "n_new_units": len(q), "median_shift_r": float(q.shift_r.median()),
                        "median_derivative_coefficient_r": float(q.derivative_coefficient_r.median()),
                        "positive_derivative_fraction": float(np.mean(q.derivative_coefficient_r > 0)),
                        "session_signflip_p": validation.signflip_p(cluster, 8181+len(summary)),
                        "median_main_amplitude_r": float(q.main_amplitude_r.median())})
    summary = pd.DataFrame(summary)
    detail.to_csv(OUT / "new_unit_latency_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "new_unit_latency_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "audit.json").write_text(json.dumps({"pilot_units_excluded": True, "image_halves": "500/500"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
