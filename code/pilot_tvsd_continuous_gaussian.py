"""Small TVSD pilot: discrete fwRF grid versus continuous Gaussian refinement.

The visual features, response targets, marginal signed feature-weight solver,
and independent 100-image test set are held fixed.  The only change is that
Gaussian center (x, y) and radius sigma are refined by gradient descent after
initialization from the best discrete candidates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_tvsd_all_blocks_where_radius as prior
import run_tvsd_fwrf_where_pilot as base
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack, VIEW_ANGLE


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "Spatial_Preference_Dynamics_Paper_Archive_2026-08-19"
OUT = ARCHIVE / "TVSD_连续Gaussian小样本审计"
SEED = 20260820
PER_BLOCK = 2
N_STARTS = 5
N_STEPS = 450


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 0)


def select_channels(meta: pd.DataFrame) -> np.ndarray:
    """Response-blind stratified sample: two channels from each physical block."""
    rng = np.random.default_rng(SEED)
    selected = []
    for _, q in meta.groupby(["roi", "monkey", "physical_block"], sort=True):
        idx = q.channel_index.to_numpy(int)
        selected.extend(rng.choice(idx, min(PER_BLOCK, len(idx)), replace=False).tolist())
    return np.asarray(sorted(selected), int)


def continuous_gaussian(x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor,
                        n_pix: int) -> torch.Tensor:
    """Differentiable port of the original Gaussian-mass construction."""
    dpix = VIEW_ANGLE / n_pix
    coord = torch.arange(n_pix, device=x.device, dtype=x.dtype) * dpix - VIEW_ANGLE / 2 + dpix / 2
    xm, ym_raw = torch.meshgrid(coord, coord, indexing="xy")
    ym = -ym_raw
    x = x[..., None, None]
    y = y[..., None, None]
    sigma = sigma[..., None, None]
    sqrt2 = math.sqrt(2.0)
    gx = 0.5 * (torch.erf((xm - x + dpix / 2) / (sqrt2 * sigma)) -
                torch.erf((xm - x - dpix / 2) / (sqrt2 * sigma)))
    gy = 0.5 * (torch.erf((ym - y + dpix / 2) / (sqrt2 * sigma)) -
                torch.erf((ym - y - dpix / 2) / (sqrt2 * sigma)))
    exact = gx * gy
    d = 2 * sigma.square()
    approx = dpix ** 2 / (d * math.pi) * torch.exp(-((xm - x).square() + (ym - y).square()) / d)
    return torch.where(sigma < dpix, exact, approx).flatten(-2)


def check_port(xs, ys, sigmas, n_pix, device):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(xs), 40, replace=False)
    torch_field = continuous_gaussian(
        torch.as_tensor(xs[idx], device=device), torch.as_tensor(ys[idx], device=device),
        torch.as_tensor(sigmas[idx], device=device), n_pix).cpu().numpy()
    numpy_field = gaussian_mass_stack(xs[idx], ys[idx], sigmas[idx], n_pix).reshape(len(idx), -1)
    return float(np.max(np.abs(torch_field - numpy_field)))


def objective_for_fields(fields, cov_fy, cov_ff, var_y):
    # fields: start x target x pixel; cov_fy: channel x pixel x target
    var_x = torch.einsum("svp,dpq,svq->svd", fields, cov_ff, fields).clamp_min(1e-8)
    numerator = torch.einsum("svp,dpv->svd", fields, cov_fy)
    energy = (numerator.square() / (var_x * var_y[None, :, None])).sum(2)
    return energy, var_x, numerator


def predict(fields, mean_f, mean_y, cov_fy, cov_ff, ft, target_y):
    var_x = torch.einsum("vp,dpq,vq->vd", fields, cov_ff, fields).clamp_min(1e-8)
    numerator = torch.einsum("vp,dpv->vd", fields, cov_fy)
    weight = numerator / (var_x + 0.05 * var_x.mean(1, keepdim=True))
    xt = torch.einsum("ndp,vp->nvd", ft, fields)
    mean_x = torch.einsum("dp,vp->vd", mean_f, fields)
    prediction = torch.einsum("nvd,vd->nv", xt - mean_x[None], weight) + mean_y[None]
    return prediction.detach().cpu().numpy(), corr_columns(prediction.detach().cpu().numpy(), target_y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(SEED)

    train_maps = torch.as_tensor(np.asarray(np.load(base.FEATURES / "train_maps_f16.npy", mmap_mode="r"),
                                            np.float32), device=device)
    test_maps = torch.as_tensor(np.asarray(np.load(base.FEATURES / "test_maps_f16.npy", mmap_mode="r"),
                                           np.float32), device=device)
    f, ft = train_maps.flatten(2), test_maps.flatten(2)
    ytr, yte, meta = prior.load_all_targets()
    selected_channels = select_channels(meta)
    nu, nw = len(selected_channels), ytr.shape[2]
    target_index = np.asarray([u * ytr.shape[2] + wi for u in selected_channels for wi in range(nw)], int)
    # source order is image x channel x window
    ytr_selected = ytr[:, selected_channels, :].reshape(len(ytr), -1)
    yte_selected = yte[:, selected_channels, :].reshape(len(yte), -1)
    y = torch.as_tensor(ytr_selected, device=device)
    stats = base.stats_for_indices(f, y, torch.arange(len(f), device=device))
    mean_f, mean_y, cov_fy, cov_ff, var_y = base.centered(stats)

    xs, ys, sigmas = candidates()
    discrete_bank = torch.as_tensor(gaussian_mass_stack(xs, ys, sigmas, train_maps.shape[-1]).reshape(len(xs), -1),
                                    device=device)
    port_max_error = check_port(xs, ys, sigmas, train_maps.shape[-1], device)

    # Rank all grid candidates under the exact same training objective.
    var_x_grid = torch.einsum("gp,dpq,gq->gd", discrete_bank, cov_ff, discrete_bank).clamp_min(1e-8)
    numerator_grid = torch.einsum("gp,dpv->gdv", discrete_bank, cov_fy)
    energy_grid = (numerator_grid.square() /
                   (var_x_grid[:, :, None] * var_y[None, None, :])).sum(1)
    top = torch.topk(energy_grid, k=N_STARTS, dim=0).indices.T  # target x start
    init_x = torch.as_tensor(xs, device=device)[top].T.contiguous()
    init_y = torch.as_tensor(ys, device=device)[top].T.contiguous()
    init_logsigma = torch.log(torch.as_tensor(sigmas, device=device)[top]).T.contiguous()

    x = torch.nn.Parameter(init_x.clone())
    yparam = torch.nn.Parameter(init_y.clone())
    logsigma = torch.nn.Parameter(init_logsigma.clone())
    optimizer = torch.optim.Adam([x, yparam, logsigma], lr=0.045)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=0.002)
    initial_best = energy_grid.max(0).values.detach().clamp_min(1e-8)
    history = []
    for step in range(N_STEPS):
        optimizer.zero_grad(set_to_none=True)
        fields = continuous_gaussian(x, yparam, logsigma.exp(), train_maps.shape[-1])
        energy, _, _ = objective_for_fields(fields, cov_fy, cov_ff, var_y)
        loss = -(energy / initial_best[None]).mean()
        loss.backward()
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            x.clamp_(-10.0, 10.0); yparam.clamp_(-10.0, 10.0)
            logsigma.clamp_(math.log(0.7), math.log(8.0))
        if step % 50 == 0 or step == N_STEPS - 1:
            history.append({"step": step, "normalized_objective": float((-loss).detach().cpu())})

    with torch.no_grad():
        all_cont_fields = continuous_gaussian(x, yparam, logsigma.exp(), train_maps.shape[-1])
        final_energy, _, _ = objective_for_fields(all_cont_fields, cov_fy, cov_ff, var_y)
        best_start = final_energy.argmax(0)
        pick = torch.arange(len(target_index), device=device)
        cont_fields = all_cont_fields[best_start, pick]
        cont_x = x[best_start, pick]
        cont_y = yparam[best_start, pick]
        cont_sigma = logsigma.exp()[best_start, pick]
        cont_energy = final_energy[best_start, pick]
        discrete_idx = energy_grid.argmax(0)
        discrete_fields = discrete_bank[discrete_idx]
        discrete_energy = energy_grid[discrete_idx, pick]

    pred_discrete, r_discrete = predict(discrete_fields, mean_f, mean_y, cov_fy, cov_ff, ft, yte_selected)
    pred_cont, r_cont = predict(cont_fields, mean_f, mean_y, cov_fy, cov_ff, ft, yte_selected)

    dot = (discrete_fields * cont_fields).sum(1)
    field_cosine = dot / (discrete_fields.norm(dim=1) * cont_fields.norm(dim=1)).clamp_min(1e-12)
    dx = cont_x - torch.as_tensor(xs, device=device)[discrete_idx]
    dy = cont_y - torch.as_tensor(ys, device=device)[discrete_idx]
    center_shift = torch.hypot(dx, dy)
    radius_ratio = cont_sigma / torch.as_tensor(sigmas, device=device)[discrete_idx]

    rows = []
    for local_u, global_u in enumerate(selected_channels):
        m = meta.iloc[global_u]
        for wi in range(nw):
            v = local_u * nw + wi
            di = int(discrete_idx[v].cpu())
            rows.append({
                "global_channel_index": int(global_u), "monkey": m.monkey, "roi": m.roi,
                "physical_block": int(m.physical_block), "raw_channel": int(m.raw_channel),
                "window_index": wi, "window_start_ms": int(base.WINDOWS[wi, 0]),
                "window_end_ms": int(base.WINDOWS[wi, 1]),
                "discrete_x_deg": float(xs[di]), "discrete_y_deg": float(ys[di]),
                "discrete_sigma_deg": float(sigmas[di]),
                "continuous_x_deg": float(cont_x[v].cpu()), "continuous_y_deg": float(cont_y[v].cpu()),
                "continuous_sigma_deg": float(cont_sigma[v].cpu()),
                "center_shift_deg": float(center_shift[v].cpu()),
                "radius_ratio": float(radius_ratio[v].cpu()),
                "field_cosine": float(field_cosine[v].cpu()),
                "train_objective_relative_gain": float((cont_energy[v] / discrete_energy[v] - 1).cpu()),
                "discrete_test_r": float(r_discrete[v]), "continuous_test_r": float(r_cont[v]),
                "delta_test_r": float(r_cont[v] - r_discrete[v]),
                "best_start": int(best_start[v].cpu()),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "TVSD连续Gaussian_逐通道逐窗.csv", index=False, encoding="utf-8-sig")

    summary = detail.groupby("roi").agg(
        channels=("global_channel_index", "nunique"), windows=("window_index", "size"),
        discrete_mean_r=("discrete_test_r", "mean"), continuous_mean_r=("continuous_test_r", "mean"),
        mean_delta_r=("delta_test_r", "mean"), median_delta_r=("delta_test_r", "median"),
        median_field_cosine=("field_cosine", "median"),
        median_center_shift_deg=("center_shift_deg", "median"),
        median_radius_ratio=("radius_ratio", "median"),
        median_train_objective_gain=("train_objective_relative_gain", "median"),
    ).reset_index()
    summary.to_csv(OUT / "TVSD连续Gaussian_ROI汇总.csv", index=False, encoding="utf-8-sig")

    audit = {
        "sample": "response-blind stratified sample, two channels per monkey x ROI x physical block",
        "n_channels": int(nu), "n_targets": int(nu * nw), "windows": base.WINDOWS.tolist(),
        "n_starts": N_STARTS, "steps": N_STEPS,
        "continuous_parameters": ["center_x", "center_y", "log_sigma"],
        "bounds": {"x_y_deg": [-10, 10], "sigma_deg": [0.7, 8.0]},
        "held_fixed": ["ResNet50 spatial PCA features", "signed marginal feature weights",
                       "training images", "independent test images", "response target"],
        "selection": "continuous parameters chosen on all training images; independent test never used for selection",
        "gaussian_port_max_abs_error": port_max_error,
        "device": str(device), "history": history,
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("port max abs error", port_max_error, flush=True)


if __name__ == "__main__":
    main()
