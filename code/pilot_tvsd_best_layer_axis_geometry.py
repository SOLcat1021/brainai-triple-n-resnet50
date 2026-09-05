"""Exploratory TVSD best-layer and unit/concept geometry pilot.

No files are written.  A fixed broad Gaussian pooling field is used only to
make layer readouts comparable; the experiment is about native feature-space
axis geometry, not a new fwRF search.  For each ResNet stage, a train-only PCA
is fitted to pooled activations, ridge readouts are evaluated at 32/64/128
components (clipped by the stage width), and the best stage per unit is chosen
by the same independent 100-image test set.  Broden CAVs at the last block of
that stage are projected through the same PCA basis before cosine statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import pilot_tvsd_axis_sample_size_sensitivity as tvsd


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
FULL_FEATURES = ROOT / "cache" / "tvsd" / "original_fwrf_resnet50" / "spatial_full_14"
CAV_PATH = PROJECT / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
STAGES = ("stem", "res2", "res3", "res4", "res5")
STAGE_LAST_NODE = {"stem": "stem", "res2": "res2_b3", "res3": "res3_b4", "res4": "res4_b6", "res5": "res5_b3"}
RANKS = (32, 64, 128)
N_BATCH = 256
RIDGE_SCALE = 0.05


def corr_cols(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return tvsd.corr_columns(pred, target)


def pooled_features(path: Path, field: np.ndarray, device: torch.device) -> np.ndarray:
    maps = np.load(path, mmap_mode="r")
    n, d, h, w = maps.shape
    out = np.empty((n, d), np.float32)
    f = torch.as_tensor(field, device=device)
    for a in range(0, n, N_BATCH):
        b = min(a + N_BATCH, n)
        x = torch.as_tensor(np.asarray(maps[a:b], np.float32), device=device)
        out[a:b] = torch.einsum("nchw,hw->nc", x, f).detach().cpu().numpy()
    return out


def fit_ranked(X: np.ndarray, Xt: np.ndarray, y: np.ndarray, ranks: tuple[int, ...], device: torch.device):
    # Train-only centering/PCA; no test image participates in basis fitting.
    mean = X.mean(0, keepdims=True).astype(np.float32)
    std = X.std(0, keepdims=True).clip(1e-5).astype(np.float32)
    Xs = (X - mean) / std
    Xts = (Xt - mean) / std
    xt = torch.as_tensor(Xs, device=device)
    # q=128 is enough for all requested ranks; smaller stages are clipped.
    q = min(max(ranks), X.shape[1])
    _, _, V = torch.pca_lowrank(xt, q=q, center=False)
    V = V[:, :q]
    Z = xt @ V
    Zt = torch.as_tensor(Xts, device=device) @ V
    y_t = torch.as_tensor(y, device=device)
    out = {}
    for rank in ranks:
        r = min(rank, X.shape[1])
        z = Z[:, :r]; zt = Zt[:, :r]
        zm = z.mean(0); ym = y_t.mean(0)
        zc = z - zm; yc = y_t - ym
        gram = zc.T @ zc
        rhs = zc.T @ yc
        lam = RIDGE_SCALE * torch.trace(gram) / max(r, 1)
        eye = torch.eye(r, device=device, dtype=z.dtype)
        beta = torch.linalg.solve(gram + eye * lam.clamp_min(1e-6), rhs)
        pred = (zt - zm) @ beta + ym
        # beta_raw has one vector per target in unstandardized native channels.
        beta_raw = (V[:, :r] @ beta).T / torch.as_tensor(std[0], device=device)[None]
        out[r] = (pred.detach().cpu().numpy(), beta_raw.detach().cpu().numpy(), V[:, :r].detach().cpu().numpy(), mean[0], std[0])
    return out


def cosine_stats(unit_axes: np.ndarray, concept_axes: np.ndarray) -> dict[str, float]:
    ua = unit_axes / np.linalg.norm(unit_axes, axis=1, keepdims=True).clip(1e-12)
    ca = concept_axes / np.linalg.norm(concept_axes, axis=1, keepdims=True).clip(1e-12)
    uc = ua @ ca.T
    uu = ua @ ua.T
    cc = ca @ ca.T
    iu = np.triu_indices(len(uu), 1); ic = np.triu_indices(len(cc), 1)
    vals = {
        "unit_concept_min": float(np.nanmin(uc)), "unit_concept_max": float(np.nanmax(uc)),
        "unit_concept_p5": float(np.nanpercentile(uc, 5)), "unit_concept_p95": float(np.nanpercentile(uc, 95)),
        "unit_concept_median": float(np.nanmedian(uc)),
        "unit_unit_min": float(np.nanmin(uu[iu])), "unit_unit_max": float(np.nanmax(uu[iu])),
        "unit_unit_p5": float(np.nanpercentile(uu[iu], 5)), "unit_unit_p95": float(np.nanpercentile(uu[iu], 95)),
        "concept_concept_min": float(np.nanmin(cc[ic])), "concept_concept_max": float(np.nanmax(cc[ic])),
        "concept_concept_p5": float(np.nanpercentile(cc[ic], 5)), "concept_concept_p95": float(np.nanpercentile(cc[ic], 95)),
    }
    return vals


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, yt, chosen_channels, chosen_roi, _ = tvsd.load_targets()
    # Six official 20-ms windows from 50-169 ms, same independent test set.
    n, nt, nu = y.shape
    target = y.reshape(n, nt * nu)
    target_test = yt.reshape(tvsd.N_TEST, nt * nu)
    # Fixed broad central Gaussian: isolate feature-axis question from RF search.
    xs, ys, sg = tvsd.candidates()
    center = int(np.argmin((xs ** 2 + ys ** 2) + (sg - 8.0) ** 2))
    field = tvsd.gaussian_mass_stack(xs[center:center + 1], ys[center:center + 1], sg[center:center + 1], 14)[0]
    print(f"device={device}; fixed pool candidate center=({xs[center]:.3f},{ys[center]:.3f}), sigma={sg[center]:.3f}")
    print("channels:", list(zip(chosen_channels.tolist(), chosen_roi.tolist())))

    cav = np.load(CAV_PATH, allow_pickle=True)
    cav_concepts = cav["concepts"].astype(str)
    cav_nodes = cav["nodes"].astype(str)
    cav_dims = cav["dims"].astype(int)
    cav_offsets = cav["offsets"].astype(int)
    cav_full = cav["cav_full"].astype(np.float32)
    layer_rows = []
    best = []
    saved = {}
    for stage in STAGES:
        train_path = FULL_FEATURES / f"train_{stage}_f16.npy"
        test_path = FULL_FEATURES / f"test_{stage}_f16.npy"
        X = pooled_features(train_path, field, device)
        Xt = pooled_features(test_path, field, device)
        fits = fit_ranked(X, Xt, target, RANKS, device)
        for rank, (pred, beta_raw, *_rest) in fits.items():
            r = corr_cols(pred, target_test)
            for ui, unit in enumerate(chosen_channels):
                layer_rows.append({"stage": stage, "rank": rank, "unit_index": ui,
                                   "unit_global": int(unit), "roi": str(chosen_roi[ui]),
                                   "mean_test_pearson_r": float(np.nanmean(r[ui * nt:(ui + 1) * nt])),
                                   "median_test_pearson_r": float(np.nanmedian(r[ui * nt:(ui + 1) * nt]))})
            saved[(stage, rank)] = (beta_raw, fits[rank][2])
        print(f"finished {stage} ({X.shape[1]} native channels)")
        del X, Xt

    table = np.asarray([[x["mean_test_pearson_r"] for x in layer_rows
                         if x["stage"] == s and x["rank"] == 64] for s in STAGES])
    for ui, unit in enumerate(chosen_channels):
        scores = np.asarray([next(x["mean_test_pearson_r"] for x in layer_rows
                                  if x["stage"] == s and x["rank"] == 64 and x["unit_index"] == ui) for s in STAGES])
        bi = int(np.nanargmax(scores)); stage = STAGES[bi]
        best.append({"unit_global": int(unit), "roi": str(chosen_roi[ui]), "best_stage": stage,
                     "best_rank": 64, "best_mean_test_r": float(scores[bi]),
                     "stage_scores_r64": "; ".join(f"{s}={v:.3f}" for s, v in zip(STAGES, scores))})
    print("\nBEST STAGE PER UNIT")
    print(best)
    print("\nLAYER/RANK MEAN TEST r")
    for s in STAGES:
        for rank in RANKS:
            z = [x["mean_test_pearson_r"] for x in layer_rows if x["stage"] == s and x["rank"] == rank]
            print(f"{s:5s} rank={rank:3d}: mean={np.nanmean(z):.4f}, median={np.nanmedian(z):.4f}")

    # Geometry at each unit's best stage. Concept axes use the final residual
    # block for that stage, the closest available Broden node to the readout.
    print("\nGEOMETRY AT BEST STAGE (native coordinates and PCA truncations)")
    for rank in RANKS:
        unit_axes = []; concept_axes = []
        for ui, item in enumerate(best):
            stage = item["best_stage"]
            beta_raw, V = saved[(stage, rank)]
            unit_axes.append(beta_raw[ui * nt:(ui + 1) * nt].mean(0))
            node = STAGE_LAST_NODE[stage]
            ni = int(np.flatnonzero(cav_nodes == node)[0])
            a, b = int(cav_offsets[ni]), int(cav_offsets[ni + 1])
            cav_native = cav_full[:, a:b]
            # For rank < native D, map CAV through the same PCA basis.
            concept_axes.append((cav_native @ V[:, :min(rank, V.shape[1])]).mean(0))
        ua = np.asarray(unit_axes)
        ca = np.asarray(concept_axes)
        # The concept list is node-specific; averaging across best-stage nodes
        # is not valid, so recompute exact unit-concept matrix per unit below.
        uc_values = []
        for ui, item in enumerate(best):
            stage = item["best_stage"]; beta_raw, V = saved[(stage, rank)]
            u = beta_raw[ui * nt:(ui + 1) * nt].mean(0)
            ni = int(np.flatnonzero(cav_nodes == STAGE_LAST_NODE[stage])[0]); a, b = int(cav_offsets[ni]), int(cav_offsets[ni + 1])
            c = cav_full[:, a:b] @ V[:, :min(rank, V.shape[1])]
            u = u / np.linalg.norm(u).clip(1e-12); c = c / np.linalg.norm(c, axis=1, keepdims=True).clip(1e-12)
            uc_values.extend((c @ u).tolist())
        # Unit-unit is well-defined after each unit's own best-stage space only
        # within the same stage; report same-stage pairs explicitly.
        for stage in STAGES:
            ids = [i for i, x in enumerate(best) if x["best_stage"] == stage]
            if len(ids) >= 2:
                vectors = []
                for i in ids:
                    beta_raw, V = saved[(stage, rank)]; vectors.append(beta_raw[i * nt:(i + 1) * nt].mean(0))
                uu = np.asarray(vectors); uu /= np.linalg.norm(uu, axis=1, keepdims=True).clip(1e-12)
                vals = (uu @ uu.T)[np.triu_indices(len(ids), 1)]
                uu_stats = f"same-{stage} unit-unit cos min={np.min(vals):.3f}, max={np.max(vals):.3f}, P5={np.percentile(vals,5):.3f}, P95={np.percentile(vals,95):.3f}"
            else:
                uu_stats = "same-stage unit-unit: n/a"
        print(f"rank={rank:3d}: unit-concept min={np.min(uc_values):.3f}, max={np.max(uc_values):.3f}, P5={np.percentile(uc_values,5):.3f}, P95={np.percentile(uc_values,95):.3f}; {uu_stats}")
    print("Note: cross-stage unit-unit cosine is intentionally not reported; those axes live in different native feature spaces.")


if __name__ == "__main__":
    main()
