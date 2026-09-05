"""Exploratory unified-res4 axis stability and geometry pilot.

The experiment intentionally stays outside the main framework.  It uses the
official TVSD 20-ms windows (50--169 ms), one fixed res4 feature space, and
the same independent 100-image test set.  For each sample size, five
independent without-replacement subsets are fitted from scratch.  Subsets
are independent across repeats because five disjoint 10,000-image folds do
not fit in the 22,248-image training set.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import pilot_tvsd_axis_sample_size_sensitivity as tvsd

FEATURES = ROOT / "cache" / "tvsd" / "original_fwrf_resnet50" / "spatial_full_14"
CAV_PATH = PROJECT / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
STAGE = "res4"
CAV_NODE = "res4_b6"
SAMPLE_SIZES = (500, 1000, 5000, 10000)
RANKS = (32, 64, 128, 256, 512, 1024)
N_REPS = 5
N_BATCH = 256
RIDGE_SCALE = 0.05
SEED = 20260903


def corr_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - np.nanmean(a, axis=0, keepdims=True)
    b = b - np.nanmean(b, axis=0, keepdims=True)
    den = np.sqrt(np.nansum(a * a, axis=0) * np.nansum(b * b, axis=0))
    return np.divide(np.nansum(a * b, axis=0), den,
                     out=np.full(den.shape, np.nan), where=den > 1e-12)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), den,
                     out=np.full(a.shape[0], np.nan), where=den > 1e-12)


def pooled_features(path: Path, field: np.ndarray, device: torch.device) -> np.ndarray:
    maps = np.load(path, mmap_mode="r")
    n, d, h, w = maps.shape
    out = np.empty((n, d), np.float32)
    f = torch.as_tensor(field, device=device)
    for start in range(0, n, N_BATCH):
        stop = min(start + N_BATCH, n)
        x = torch.as_tensor(np.asarray(maps[start:stop], np.float32), device=device)
        out[start:stop] = torch.einsum("nchw,hw->nc", x, f).detach().cpu().numpy()
    return out


def fit_pca_ridge(X: np.ndarray, y: np.ndarray, ranks: tuple[int, ...], device: torch.device):
    """Return predictions and native-coordinate beta for each requested rank."""
    mean = X.mean(0).astype(np.float32)
    std = X.std(0).clip(1e-5).astype(np.float32)
    Xs = (X - mean[None]) / std[None]
    xt = torch.as_tensor(Xs, device=device)
    q = min(max(ranks), X.shape[1], X.shape[0] - 1)
    _, _, V = torch.pca_lowrank(xt, q=q, center=False)
    yt = torch.as_tensor(y, device=device)
    out = {}
    for requested in ranks:
        r = min(requested, q)
        z = xt @ V[:, :r]
        zm = z.mean(0); ym = yt.mean(0)
        zc = z - zm[None]; yc = yt - ym[None]
        gram = zc.T @ zc
        rhs = zc.T @ yc
        lam = RIDGE_SCALE * torch.trace(gram) / max(r, 1)
        beta_pca = torch.linalg.solve(
            gram + torch.eye(r, device=device, dtype=z.dtype) * lam.clamp_min(1e-6), rhs)
        beta_native = (V[:, :r] @ beta_pca).T / torch.as_tensor(std, device=device)[None]
        pred = (z - zm[None]) @ beta_pca + ym[None]
        out[requested] = {
            "effective_rank": r,
            "beta_native": beta_native.detach().cpu().numpy(),
            "pred_train": pred.detach().cpu().numpy(),
            "mean": mean,
            "std": std,
            "V": V[:, :r].detach().cpu().numpy(),
            "beta_pca": beta_pca.detach().cpu().numpy(),
            "z_mean": zm.detach().cpu().numpy(),
            "y_mean": ym.detach().cpu().numpy(),
        }
    return out


def quantiles(values: np.ndarray) -> str:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "min=nan max=nan P5=nan P95=nan median=nan"
    return (f"min={np.min(values):.3f} max={np.max(values):.3f} "
            f"P5={np.percentile(values, 5):.3f} P95={np.percentile(values, 95):.3f} "
            f"median={np.median(values):.3f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, y_test, chosen, chosen_roi, _ = tvsd.load_targets()
    n, n_time, n_units = y.shape
    target = y.reshape(n, n_time * n_units)
    target_test = y_test.reshape(tvsd.N_TEST, n_time * n_units)
    xs, ys, sigmas = tvsd.candidates()
    center = int(np.argmin((xs ** 2 + ys ** 2) + (sigmas - 8.0) ** 2))
    field = tvsd.gaussian_mass_stack(xs[center:center + 1], ys[center:center + 1],
                                     sigmas[center:center + 1], 14)[0]
    print(f"device={device}; stage={STAGE}; CAV node={CAV_NODE}; "
          f"fixed pool center=({xs[center]:.3f},{ys[center]:.3f}), sigma={sigmas[center]:.3f}")
    print("units:", list(zip(chosen.tolist(), chosen_roi.tolist())))

    X = pooled_features(FEATURES / f"train_{STAGE}_f16.npy", field, device)
    Xt = pooled_features(FEATURES / f"test_{STAGE}_f16.npy", field, device)
    print(f"pooled feature shape train={X.shape}, test={Xt.shape}")
    test_target = torch.as_tensor(target_test, device=device)

    # Full training reference at every rank.
    full = fit_pca_ridge(X, target, RANKS, device)
    full_r = {}
    for rank, fit in full.items():
        zt = ((torch.as_tensor(Xt, device=device) - torch.as_tensor(fit["mean"], device=device)[None])
              / torch.as_tensor(fit["std"], device=device)[None]) @ torch.as_tensor(fit["V"], device=device)
        pred = ((zt - torch.as_tensor(fit["z_mean"], device=device)[None])
                @ torch.as_tensor(fit["beta_pca"], device=device)
                + torch.as_tensor(fit["y_mean"], device=device)[None])
        full_r[rank] = corr_cols(pred.detach().cpu().numpy(), target_test)
    print("\\nFULL-DATA res4 TEST PEARSON r")
    for rank in RANKS:
        print(f"rank={rank:4d} effective={full[rank]['effective_rank']:4d}: "
              f"mean={np.nanmean(full_r[rank]):.4f} median={np.nanmedian(full_r[rank]):.4f}")

    rng = np.random.default_rng(SEED)
    # beta[rank][rep] is native-space axis for all unit-window targets.
    beta_reps: dict[int, list[np.ndarray]] = {r: [] for r in RANKS}
    r_reps: dict[int, list[np.ndarray]] = {r: [] for r in RANKS}
    for size in SAMPLE_SIZES:
        for rep in range(N_REPS):
            idx = rng.choice(n, size=size, replace=False)
            fits = fit_pca_ridge(X[idx], target[idx], RANKS, device)
            for rank in RANKS:
                beta_reps[rank].append(fits[rank]["beta_native"])
                # Test projection in the subset's own PCA coordinates.
                V = torch.as_tensor(fits[rank]["V"], device=device)
                mean = torch.as_tensor(fits[rank]["mean"], device=device)
                std = torch.as_tensor(fits[rank]["std"], device=device)
                zt = (torch.as_tensor(Xt, device=device) - mean[None]) / std[None] @ V
                z = torch.as_tensor((X[idx] - fits[rank]["mean"][None]) / fits[rank]["std"][None], device=device) @ V
                yt = torch.as_tensor(target[idx], device=device)
                zm = z.mean(0); ym = yt.mean(0)
                zc = z - zm[None]; yc = yt - ym[None]
                gram = zc.T @ zc; rhs = zc.T @ yc
                rr = fits[rank]["effective_rank"]
                lam = RIDGE_SCALE * torch.trace(gram) / max(rr, 1)
                bp = torch.linalg.solve(gram + torch.eye(rr, device=device) * lam.clamp_min(1e-6), rhs)
                pred = (zt - zm[None]) @ bp + ym[None]
                r_reps[rank].append(corr_cols(pred.detach().cpu().numpy(), target_test))
            print(f"sample size {size}, repeat {rep + 1}/{N_REPS} complete", flush=True)

        print(f"\\nSTABILITY size={size} (5 independent no-replacement draws)")
        for rank in RANKS:
            reps = beta_reps[rank][-N_REPS:]
            pair = np.concatenate([row_cos(reps[a], reps[b]) for a, b in combinations(range(N_REPS), 2)])
            ref = np.concatenate([row_cos(b, full[rank]["beta_native"]) for b in reps])
            rr = np.concatenate([r_reps[rank][-N_REPS:]])
            print(f"rank={rank:4d} effective={full[rank]['effective_rank']:4d}; "
                  f"pairwise_axis {quantiles(pair)}; to-full {quantiles(ref)}; "
                  f"test-r mean={np.nanmean(rr):.4f} median={np.nanmedian(rr):.4f}")

    # Geometry uses the full-data reference and only this single res4 space.
    cav = np.load(CAV_PATH, allow_pickle=True)
    concepts = cav["concepts"].astype(str)
    nodes = cav["nodes"].astype(str); dims = cav["dims"].astype(int); offsets = cav["offsets"].astype(int)
    ni = int(np.flatnonzero(nodes == CAV_NODE)[0]); a, b = int(offsets[ni]), int(offsets[ni + 1])
    cav_native = np.asarray(cav["cav_full"][..., a:b], np.float32)
    print("\\nGEOMETRY IN UNIFIED RES4 NATIVE SPACE (unit axes averaged over 50--169 ms)")
    for rank in RANKS:
        beta = full[rank]["beta_native"]
        unit_axes = np.asarray([beta[u * n_time:(u + 1) * n_time].mean(0) for u in range(n_units)])
        ua = unit_axes / np.linalg.norm(unit_axes, axis=1, keepdims=True).clip(1e-12)
        ca = cav_native / np.linalg.norm(cav_native, axis=1, keepdims=True).clip(1e-12)
        uc = (ua @ ca.T).ravel()
        uu = (ua @ ua.T)[np.triu_indices(n_units, 1)]
        cc = (ca @ ca.T)[np.triu_indices(len(ca), 1)]
        print(f"rank={rank:4d} effective={full[rank]['effective_rank']:4d}")
        print("  unit-concept:", quantiles(uc))
        print("  unit-unit:   ", quantiles(uu))
        print("  concept-concept:", quantiles(cc))
    print("concepts:", ", ".join(concepts.tolist()))
    print("NOTE: only res4 is interpreted; other stages are intentionally omitted.")


if __name__ == "__main__":
    main()
