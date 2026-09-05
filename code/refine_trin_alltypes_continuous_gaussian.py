"""Continuous sub-grid Gaussian refinement for screened Triple-N units.

Selection is deliberately simple and frozen before spatial analysis:
mean discrete-fwRF OOF r >= 0.10 and minimum of all eight windows >= 0.

For each full-data or outer-fold discrete optimum, the objective is evaluated
at the immediate negative/positive neighbour on x, y and log(sigma).  A local
three-point parabola gives a continuous maximum on each coordinate.  This is
a deterministic sub-grid refinement of the same fwRF objective, not a new
response model.  Fold-wise continuous centres are used for fitting jitter and
for continuous OOF predictions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_trin_closed_loop_validation as fitbase  # noqa: E402
import run_tvsd_fwrf_where_pilot as statbase  # noqa: E402
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack  # noqa: E402
from pilot_tvsd_continuous_gaussian import continuous_gaussian  # noqa: E402


MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "spatial_pca8_14_tvsd_frozen" / "maps_f16.npy"
POP = ROOT / "cache" / "trin" / "population_time20_all_unittypes" / "population_20ms_all_unittypes.npz"
DISCRETE = ROOT / "ResNet50_原版fwRF_20ms时序网络剖析_2026-08-18" / "TripleN三类型冻结验证"
WINDOW_CSV = DISCRETE / "TripleN三类型逐Unit逐窗_预测与感受野.csv"
OUT = ROOT / "Spatial_Preference_Dynamics_Paper_Archive_2026-08-19" / "TripleN_三类型简单筛选_连续Gaussian"
ROIS = fitbase.ROIS
WINDOWS = fitbase.WINDOWS
MEAN_R_MIN = 0.10
MIN_WINDOW_R_MIN = 0.00
BATCH = 512


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 0)


def neighbour_ids(initial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ix, iy, iz = np.unravel_index(initial, (15, 15, 10))
    ids = np.empty((len(initial), 7), np.int64)
    ids[:, 0] = initial
    coords = [(ix - 1, iy, iz), (ix + 1, iy, iz), (ix, iy - 1, iz),
              (ix, iy + 1, iz), (ix, iy, iz - 1), (ix, iy, iz + 1)]
    valid = np.empty((len(initial), 3), bool)
    valid[:, 0] = (ix > 0) & (ix < 14)
    valid[:, 1] = (iy > 0) & (iy < 14)
    valid[:, 2] = (iz > 0) & (iz < 9)
    for j, (a, b, c) in enumerate(coords, 1):
        a = np.clip(a, 0, 14); b = np.clip(b, 0, 14); c = np.clip(c, 0, 9)
        ids[:, j] = np.ravel_multi_index((a, b, c), (15, 15, 10))
    return ids, valid, ix, iy


def refine_from_stats(initial: np.ndarray, stats, bank: torch.Tensor,
                      xs: np.ndarray, ys: np.ndarray, sigmas: np.ndarray,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, cov_fy, cov_ff, var_y = statbase.centered(stats)
    var_bank = torch.einsum("gp,dpq,gq->gd", bank, cov_ff, bank).clamp_min(1e-8)
    ids, valid, _, _ = neighbour_ids(initial)
    delta = np.zeros((len(initial), 3), np.float32)
    for b0 in range(0, len(initial), BATCH):
        b1 = min(len(initial), b0 + BATCH)
        ids_b = torch.as_tensor(ids[b0:b1], device=device)
        fields = bank[ids_b]
        numerator = torch.einsum("bkp,dpb->bkd", fields, cov_fy[:, :, b0:b1])
        vx = var_bank[ids_b]
        energy = (numerator.square() / (vx * var_y[b0:b1, None, None])).sum(2).detach().cpu().numpy()
        e0 = energy[:, 0]
        for axis, (minus, plus) in enumerate([(1, 2), (3, 4), (5, 6)]):
            denominator = energy[:, minus] - 2 * e0 + energy[:, plus]
            ok = valid[b0:b1, axis] & (denominator < -1e-10)
            estimate = np.zeros(b1 - b0, np.float32)
            estimate[ok] = .5 * (energy[ok, minus] - energy[ok, plus]) / denominator[ok]
            delta[b0:b1, axis] = np.clip(estimate, -1, 1)
    x0 = xs[initial]; y0 = ys[initial]; log_s0 = np.log(sigmas[initial])
    dx = float(np.unique(np.diff(np.unique(xs)))[0])
    dy = float(np.unique(np.diff(np.unique(ys)))[0])
    dlog = float(np.diff(np.log(np.unique(sigmas))).mean())
    x = np.clip(x0 + delta[:, 0] * dx, -10, 10)
    y = np.clip(y0 + delta[:, 1] * dy, -10, 10)
    sigma = np.clip(np.exp(log_s0 + delta[:, 2] * dlog), .7, 8.0)
    return x.astype(np.float32), y.astype(np.float32), sigma.astype(np.float32)


def predict_batch(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, stats,
                  ftest: torch.Tensor, target_y: np.ndarray, device: torch.device) -> np.ndarray:
    mean_f, mean_y, cov_fy, cov_ff, _ = statbase.centered(stats)
    out = np.empty((len(ftest), len(x)), np.float32)
    for b0 in range(0, len(x), BATCH):
        b1 = min(len(x), b0 + BATCH)
        fields = continuous_gaussian(
            torch.as_tensor(x[b0:b1], device=device),
            torch.as_tensor(y[b0:b1], device=device),
            torch.as_tensor(sigma[b0:b1], device=device),
            int(round(math.sqrt(ftest.shape[-1]))),
        )
        vx = torch.einsum("bp,dpq,bq->bd", fields, cov_ff, fields).clamp_min(1e-8)
        numerator = torch.einsum("bp,dpb->bd", fields, cov_fy[:, :, b0:b1])
        weight = numerator / (vx + .05 * vx.mean(1, keepdim=True))
        xt = torch.einsum("ndp,bp->nbd", ftest, fields)
        mean_x = torch.einsum("dp,bp->bd", mean_f, fields)
        prediction = torch.einsum("nbd,bd->nb", xt - mean_x[None], weight) + mean_y[b0:b1][None]
        out[:, b0:b1] = prediction.detach().cpu().numpy()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    window = pd.read_csv(WINDOW_CSV)
    unit = window.groupby(["roi", "unit_global", "session", "monkey", "unit_index_zero_based",
                           "unit_type", "unit_type_name"], as_index=False).agg(
        mean_discrete_oof_r=("dynamic_oof_r", "mean"),
        min_discrete_oof_r=("dynamic_oof_r", "min"),
        max_discrete_oof_r=("dynamic_oof_r", "max"),
    )
    unit["passes_simple_filter"] = (
        (unit.mean_discrete_oof_r >= MEAN_R_MIN) &
        (unit.min_discrete_oof_r >= MIN_WINDOW_R_MIN)
    )
    unit.to_csv(OUT / "TripleN三类型全部Unit简单筛选索引.csv", index=False, encoding="utf-8-sig")

    maps = torch.as_tensor(np.asarray(np.load(MAPS, mmap_mode="r"), np.float32), device=device)
    f = maps.flatten(2)
    pop = np.load(POP, allow_pickle=True)
    xs, ys, sigmas = candidates()
    bank = torch.as_tensor(gaussian_mass_stack(xs, ys, sigmas, maps.shape[-1]).reshape(len(xs), -1), device=device)
    folds = fitbase.fixed_folds(len(f))
    rows = []
    for roi in ROIS:
        chosen = unit[(unit.roi == roi) & unit.passes_simple_filter].sort_values("unit_global")
        selected_global = chosen.unit_global.to_numpy(int)
        core = np.load(DISCRETE / f"{roi.replace(' ', '_')}_核心数组.npz")
        local_lookup = {int(g): i for i, g in enumerate(core["unit_global"])}
        local = np.asarray([local_lookup[int(g)] for g in selected_global], int)
        y_np = pop["responses"][1:9, :, selected_global].astype(np.float32)
        nw, n, nu = y_np.shape
        y_flat = y_np.transpose(1, 2, 0).reshape(n, nu * nw)
        y_t = torch.as_tensor(y_flat, device=device)
        test_parts = [statbase.stats_for_indices(f, y_t, torch.as_tensor(test, device=device)) for _, test in folds]
        total = statbase.add_many(test_parts)
        full_initial = core["full_candidates"][:, local].T.reshape(-1)
        full_x, full_y, full_sigma = refine_from_stats(full_initial, total, bank, xs, ys, sigmas, device)
        fold_x = np.empty((5, len(full_x)), np.float32)
        fold_y = np.empty_like(fold_x); fold_sigma = np.empty_like(fold_x)
        pred = np.empty((n, len(full_x)), np.float32)
        for k, (_, test) in enumerate(folds):
            train_stats = statbase.subtract(total, test_parts[k])
            init = core["fold_candidates"][k][:, local].T.reshape(-1)
            fold_x[k], fold_y[k], fold_sigma[k] = refine_from_stats(init, train_stats, bank, xs, ys, sigmas, device)
            p = predict_batch(fold_x[k], fold_y[k], fold_sigma[k], train_stats,
                              f[torch.as_tensor(test, device=device)], y_flat[test], device)
            pred[test] = p
            print(f"{roi}: continuous fold {k + 1}/5", flush=True)
        continuous_r = corr_columns(pred, y_flat).reshape(nu, nw).T
        full_x = full_x.reshape(nu, nw).T; full_y = full_y.reshape(nu, nw).T
        full_sigma = full_sigma.reshape(nu, nw).T
        fold_x = fold_x.reshape(5, nu, nw).transpose(0, 2, 1)
        fold_y = fold_y.reshape(5, nu, nw).transpose(0, 2, 1)
        fold_sigma = fold_sigma.reshape(5, nu, nw).transpose(0, 2, 1)
        dispersion = np.mean(np.hypot(fold_x - full_x[None], fold_y - full_y[None]), axis=0)
        for u, g in enumerate(selected_global):
            meta = chosen.iloc[u]
            for wi, (start, end) in enumerate(WINDOWS):
                source = window[(window.unit_global == g) & (window.window_index == wi)].iloc[0]
                rows.append({
                    "roi": roi, "unit_global": int(g), "session": int(meta.session), "monkey": meta.monkey,
                    "unit_index_zero_based": int(meta.unit_index_zero_based), "unit_type": int(meta.unit_type),
                    "unit_type_name": meta.unit_type_name, "window_index": wi,
                    "window_start_ms": int(start), "window_end_ms": int(end),
                    "discrete_oof_r": float(source.dynamic_oof_r), "continuous_oof_r": float(continuous_r[wi, u]),
                    "continuous_x_deg": float(full_x[wi, u]), "continuous_y_deg": float(full_y[wi, u]),
                    "continuous_sigma_deg": float(full_sigma[wi, u]),
                    "continuous_fold_center_dispersion_deg": float(dispersion[wi, u]),
                    "discrete_x_deg": float(source.center_x_deg), "discrete_y_deg": float(source.center_y_deg),
                    "discrete_sigma_deg": float(source.radius_deg),
                })
        pd.DataFrame(rows).to_csv(OUT / "TripleN合格Unit逐窗_连续Gaussian.csv", index=False, encoding="utf-8-sig")
        print(f"{roi}: {nu} units completed", flush=True)
    detail = pd.DataFrame(rows)
    audit = {
        "selection": {"mean_discrete_oof_r_min": MEAN_R_MIN, "every_window_discrete_oof_r_min": MIN_WINDOW_R_MIN},
        "selection_uses_spatial_outcome": False,
        "continuous_method": "three-point local parabolic interpolation on x, y and log-sigma around each discrete optimum",
        "continuous_full_and_fold": True,
        "jitter": "mean Euclidean distance from five continuous fold centres to the continuous full-data centre",
        "prediction": "continuous fold-specific fields evaluated OOF on the same five image folds",
        "n_units": int(detail.unit_global.nunique()),
    }
    (OUT / "连续Gaussian拟合审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
