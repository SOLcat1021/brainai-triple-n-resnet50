"""Final all-unit Triple-N ResNet-50 encoding model.

Population: every mapped Triple-N unit released in the local population cache
(SU, MU and non-somatic; no additional consistency/performance screening).

For every unit, one 20-ms window is selected from the released 20-ms response
cache by image-wise response variance. The final model is then evaluated with five image OOF
folds.  Each fold learns, without seeing its outer test images:

1. a target-specific continuous Gaussian spatial field;
2. a 17-layer quality curve from small single-layer ridge readouts;
3. an exact smooth Gaussian gate over ResNet-50 depth;
4. one joint signed ridge over all 17 x 8 neural-blind PCA channels, scaled by
   that gate.

The same rule is used for every ROI and animal.  There is no hard top-k layer
selection, ROI-specific routing, temporal sharing, or outer-test selection.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"D:\Coding\BrainAI")
FINAL = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
OUT = FINAL / "results"
FIG = FINAL / "figures"
SRC = FINAL / "code"
POP = ROOT / "cache" / "trin" / "population_time20_all_unittypes" / "population_20ms_all_unittypes.npz"
NC = FINAL / "support" / "single_trial_nc1_by_window_20ms.csv"
MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "block_pca8_14_tvsd_frozen" / "maps_f16.npy"

sys.path.insert(0, str(SRC))
import run_trin_closed_loop_validation as foldbase  # noqa: E402
import run_tvsd_fwrf_where_pilot as statbase  # noqa: E402
from extract_trin_resnet50_modelspace import candidates, gaussian_mass_stack  # noqa: E402
from pilot_tvsd_continuous_gaussian import continuous_gaussian  # noqa: E402
from refine_trin_alltypes_continuous_gaussian import refine_from_stats  # noqa: E402


LAYERS = (
    "stem", "res2.1", "res2.2", "res2.3", "res3.1", "res3.2", "res3.3", "res3.4",
    "res4.1", "res4.2", "res4.3", "res4.4", "res4.5", "res4.6",
    "res5.1", "res5.2", "res5.3",
)
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
ROI_LABEL = {"V1": "V1", "V2": "V2", "V4": "V4", "posterior IT": "Posterior IT",
             "middle IT": "Middle IT", "anterior IT": "Anterior IT"}
MONKEYS = ("M1", "M2", "M3", "M4", "M5")
N_LAYERS = 17
DIMS = 8
SCREEN_DIMS = 3
RIDGE = 0.10
TARGET_BATCH = 32


def slug(x: str) -> str:
    return x.lower().replace(" ", "_")


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, np.float64) - np.nanmean(a, axis=0, keepdims=True)
    bb = np.asarray(b, np.float64) - np.nanmean(b, axis=0, keepdims=True)
    den = np.sqrt(np.nansum(aa * aa, 0) * np.nansum(bb * bb, 0))
    return np.divide(np.nansum(aa * bb, 0), den, out=np.full(den.shape, np.nan),
                     where=den > 1e-12).astype(np.float32)


def load_roi_targets(roi: str, pop: np.lib.npyio.NpzFile) -> tuple[np.ndarray, pd.DataFrame]:
    units = np.flatnonzero(pop["native_area"].astype(str) == roi).astype(int)
    windows = pop["windows"].astype(int)
    response = np.asarray(pop["responses"][:, :, units], np.float32)
    wi = np.nanargmax(np.nanvar(response, axis=1), axis=0).astype(int)
    target = np.empty((response.shape[1], len(units)), np.float32)
    for w in range(len(windows)):
        take = np.flatnonzero(wi == w)
        if len(take):
            target[:, take] = np.asarray(response[w, :, take], np.float32).T
    table = pd.DataFrame({
        "roi": roi, "unit_global": units, "window_index": wi,
        "window_start_ms": windows[wi, 0], "window_end_ms": windows[wi, 1],
        "selected_window_image_variance": np.nanvar(response, axis=1)[wi, np.arange(len(units))],
        "released_independent_consistency": pop["reliability"][units].astype(float),
        "session": pop["session"][units].astype(int), "monkey": pop["monkey"][units].astype(str),
        "native_area": pop["native_area"][units].astype(str),
        "unit_type": pop["unit_type"][units].astype(int),
        "unit_type_name": pop["unit_type_name"][units].astype(str),
    })
    return target, table


def normalize(raw: np.ndarray, train: np.ndarray, test: np.ndarray,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tr, te = [], []
    for layer in range(N_LAYERS):
        a = torch.as_tensor(np.asarray(raw[train, layer], np.float32), device=device).flatten(2)
        b = torch.as_tensor(np.asarray(raw[test, layer], np.float32), device=device).flatten(2)
        mean = a.mean((0, 2), keepdim=True)
        std = a.std((0, 2), keepdim=True).clamp_min(1e-5)
        tr.append((a - mean) / std); te.append((b - mean) / std)
    return torch.cat(tr, 1), torch.cat(te, 1)


def make_stats(f: torch.Tensor, y_np: np.ndarray) -> dict:
    y = torch.as_tensor(np.asarray(y_np, np.float32), device=f.device)
    return {"n": int(len(f)), "sum_f": f.sum(0), "sum_y": y.sum(0),
            "sum_y2": (y * y).sum(0), "cross": torch.einsum("ndp,nv->dpv", f, y),
            "ff": torch.einsum("ndp,ndq->dpq", f, f)}


def select_fields(stats: dict, bank: torch.Tensor, batch: int = 128) -> np.ndarray:
    _, _, cov_fy, cov_ff, var_y = statbase.centered(stats)
    var_x = torch.einsum("gp,dpq,gq->gd", bank, cov_ff, bank).clamp_min(1e-8)
    chosen = np.empty(len(var_y), np.int32)
    for t0 in range(0, len(var_y), batch):
        t1 = min(len(var_y), t0 + batch)
        num = torch.einsum("gp,dpv->gdv", bank, cov_fy[:, :, t0:t1])
        score = (num.square() / (var_x[:, :, None] * var_y[None, None, t0:t1].clamp_min(1e-8))).sum(1)
        chosen[t0:t1] = score.argmax(0).cpu().numpy()
    return chosen


def batched_ridge(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor,
                  ridge: float = RIDGE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target-specific batched ridge: x is [images, targets, channels]."""
    mx = xtr.mean(0); my = ytr.mean(0)
    xc = xtr - mx[None]; yc = ytr - my[None]
    gram = torch.einsum("nbd,nbe->bde", xc, xc)
    alpha = ridge * gram.diagonal(dim1=1, dim2=2).sum(1) / gram.shape[-1]
    gram.diagonal(dim1=1, dim2=2).add_(alpha[:, None].clamp_min(1e-6))
    rhs = torch.einsum("nbd,nb->bd", xc, yc)
    beta = torch.linalg.solve(gram, rhs)
    pred = torch.einsum("nbd,bd->nb", xte - mx[None], beta) + my[None]
    return pred, beta, mx


def corr_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a - a.mean(0, keepdim=True); bb = b - b.mean(0, keepdim=True)
    den = torch.sqrt(aa.square().sum(0) * bb.square().sum(0)).clamp_min(1e-8)
    return (aa * bb).sum(0) / den


def infer_gate(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a unimodal Gaussian gate to each fold-local layer quality curve."""
    depth = np.arange(N_LAYERS, dtype=np.float32)
    smooth = np.empty_like(scores)
    smooth[:, 0] = .75 * scores[:, 0] + .25 * scores[:, 1]
    smooth[:, -1] = .25 * scores[:, -2] + .75 * scores[:, -1]
    smooth[:, 1:-1] = .25 * scores[:, :-2] + .50 * scores[:, 1:-1] + .25 * scores[:, 2:]
    peak = np.argmax(np.nan_to_num(smooth, nan=-1), axis=1)
    mu = peak.astype(np.float32)
    tolerance = .012
    evidence = np.exp(np.clip((smooth - smooth[np.arange(len(smooth)), peak, None]) / tolerance, -40, 0))
    evidence *= np.exp(-.5 * ((depth[None] - mu[:, None]) / 4.0) ** 2)
    sigma = np.sqrt((evidence * (depth[None] - mu[:, None]) ** 2).sum(1) /
                    np.maximum(evidence.sum(1), 1e-12)).clip(.6, 3.5)
    gate = np.exp(-.5 * ((depth[None] - mu[:, None]) / sigma[:, None]) ** 2)
    gate /= gate.sum(1, keepdims=True)
    return gate.astype(np.float32), mu, sigma.astype(np.float32)


def fit_roi(target: np.ndarray, raw: np.ndarray, device: torch.device,
            roi: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    prediction = np.full_like(target, np.nan, np.float32)
    gate_all = np.zeros((5, target.shape[1], N_LAYERS), np.float32)
    field_rows: list[dict] = []
    xs, ys, sigmas = candidates(); grid = int(raw.shape[-1])
    bank = torch.as_tensor(gaussian_mass_stack(xs, ys, sigmas, grid).reshape(len(xs), -1),
                           device=device)
    for fi, (train, test) in enumerate(foldbase.fixed_folds(len(target))):
        ftr, fte = normalize(raw, train, test, device)
        screen = torch.cat([ftr[:, l * DIMS:l * DIMS + SCREEN_DIMS] for l in range(N_LAYERS)], 1)
        stats = make_stats(screen, target[train])
        initial = select_fields(stats, bank)
        cx, cy, cs = refine_from_stats(initial, stats, bank, xs, ys, sigmas, device)
        ytr_full = torch.as_tensor(target[train], device=device)
        # Deterministic inner split; spatial and gate selection use outer-train only.
        order = np.random.default_rng(20260824 + fi).permutation(len(train))
        inner_val = torch.as_tensor(order[:200], device=device)
        inner_train = torch.as_tensor(order[200:], device=device)
        for t0 in range(0, target.shape[1], TARGET_BATCH):
            t1 = min(target.shape[1], t0 + TARGET_BATCH)
            field = continuous_gaussian(torch.as_tensor(cx[t0:t1], device=device),
                                        torch.as_tensor(cy[t0:t1], device=device),
                                        torch.as_tensor(cs[t0:t1], device=device), grid)
            xtr = torch.einsum("ndp,bp->nbd", ftr, field)
            xte = torch.einsum("ndp,bp->nbd", fte, field)
            yb = ytr_full[:, t0:t1]
            layer_score = np.empty((t1 - t0, N_LAYERS), np.float32)
            for layer in range(N_LAYERS):
                sl = slice(layer * DIMS, (layer + 1) * DIMS)
                val_pred, _, _ = batched_ridge(xtr[inner_train, :, sl], yb[inner_train],
                                               xtr[inner_val, :, sl])
                layer_score[:, layer] = corr_torch(val_pred, yb[inner_val]).detach().cpu().numpy()
            gate, mu, sigma = infer_gate(layer_score)
            gate_all[fi, t0:t1] = gate
            scale = torch.as_tensor(np.sqrt(np.clip(gate, 1e-5, None) * N_LAYERS),
                                    device=device, dtype=xtr.dtype)
            scale = torch.repeat_interleave(scale, DIMS, dim=1)
            pred, _, _ = batched_ridge(xtr * scale[None], yb, xte * scale[None])
            prediction[test, t0:t1] = pred.detach().cpu().numpy()
            for j in range(t1 - t0):
                field_rows.append({"roi": roi, "fold": fi, "target_local": t0 + j,
                                   "center_x": float(cx[t0 + j]), "center_y": float(cy[t0 + j]),
                                   "sigma_space": float(cs[t0 + j]), "center_depth": float(mu[j]),
                                   "sigma_layers": float(sigma[j])})
        del ftr, fte, screen, stats, ytr_full
        torch.cuda.empty_cache()
        print(f"{roi}: completed outer fold {fi + 1}/5", flush=True)
    return prediction, gate_all, field_rows


def attach_nc(result: pd.DataFrame) -> pd.DataFrame:
    if not NC.exists():
        raise FileNotFoundError(
            f"Missing 20-ms noise-ceiling table: {NC}. "
            "The legacy 10-ms table is intentionally not used."
        )
    nc = pd.read_csv(NC)
    widths = nc.window_end_ms - nc.window_start_ms + 1
    if not (widths == 20).all():
        raise ValueError("Noise-ceiling table contains non-20-ms windows")
    keys = ["roi", "unit_global", "window_start_ms", "window_end_ms"]
    keep = keys + ["signal_variance", "single_trial_noise_variance", "repeat_mean_noise_variance",
                   "nc1_variance_fraction", "r_ceiling_1"]
    out = result.merge(nc[keep], on=keys, how="left", validate="one_to_one")
    if out.r_ceiling_1.isna().any():
        raise RuntimeError("NC(1) match missing")
    attenuation = np.sqrt((out.signal_variance + out.repeat_mean_noise_variance) /
                          (out.signal_variance + out.single_trial_noise_variance))
    out["single_trial_equivalent_r"] = out.repeat_mean_oof_r * attenuation
    out["single_trial_equivalent_r2"] = np.maximum(out.single_trial_equivalent_r, 0) ** 2
    return out


def setup_style() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": .8,
                         "figure.dpi": 160, "savefig.facecolor": "white"})


def scatter(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    x = frame.r_ceiling_1.to_numpy(float); y = frame.single_trial_equivalent_r.to_numpy(float)
    ax.scatter(x, y, s=5, alpha=.16, color="#72ADD4", edgecolors="none", rasterized=True)
    ax.plot([0, 1], [0, 1], color="#F06D6A", lw=1.25)
    ax.scatter([np.nanmean(x)], [np.nanmean(y)], marker="+", s=110, linewidths=2.3,
               color="#D6273B", zorder=5)
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    ax.text(.04, .96, f"n={len(frame):,}\nmean: ({np.nanmean(x):.3f}, {np.nanmean(y):.3f})\nSpearman ρ={rho:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.2)
    ax.set_xlim(0, 1); ax.set_ylim(-.25, 1); ax.grid(True, color="#D9D9D9", lw=.45, alpha=.75)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Single-trial correlation ceiling, √NC(1)")
    ax.set_ylabel("Single-trial-equivalent OOF Pearson r")


def make_figures(result: pd.DataFrame, gates: np.ndarray, manifest: pd.DataFrame) -> list[Path]:
    setup_style(); FIG.mkdir(parents=True, exist_ok=True); paths = []
    fig, ax = plt.subplots(figsize=(5.2, 4.7), constrained_layout=True)
    scatter(ax, result, "All 23,960 mapped units")
    fig.suptitle("Triple-N final ResNet-50 encoding model", fontweight="bold", fontsize=12)
    p = FIG / "01_all_units_single_trial_noise_ceiling.png"; fig.savefig(p, dpi=320); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, roi in zip(axes.flat, ROIS): scatter(ax, result[result.roi.eq(roi)], ROI_LABEL[roi])
    fig.suptitle("Triple-N final ResNet-50: all units by ROI", fontweight="bold", fontsize=13)
    p = FIG / "02_six_rois_single_trial_noise_ceiling.png"; fig.savefig(p, dpi=320); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, monkey in zip(axes, MONKEYS): scatter(ax, result[result.monkey.eq(monkey)], monkey)
    fig.suptitle("Triple-N final ResNet-50: all units by monkey", fontweight="bold", fontsize=13)
    p = FIG / "03_five_monkeys_single_trial_noise_ceiling.png"; fig.savefig(p, dpi=320); plt.close(fig); paths.append(p)

    roi_gate = np.stack([gates[:, manifest.roi.eq(r).to_numpy()].mean((0, 1)) for r in ROIS])
    pd.DataFrame(roi_gate, index=[ROI_LABEL[r] for r in ROIS], columns=LAYERS).to_csv(
        OUT / "roi_by_layer_mean_gaussian_gate.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 4.4), constrained_layout=True)
    im = ax.imshow(roi_gate, aspect="auto", cmap="YlGnBu", vmin=0,
                   vmax=max(.12, float(roi_gate.max())))
    ax.set_xticks(range(N_LAYERS), LAYERS, rotation=45, ha="right")
    ax.set_yticks(range(6), [ROI_LABEL[r] for r in ROIS])
    ax.set_xlabel("ResNet-50 depth (all stem / bottleneck outputs)")
    ax.set_ylabel("Ventral visual hierarchy")
    ax.set_title("Learned smooth Gaussian depth gates — all units", loc="left", fontweight="bold")
    for i in range(6):
        for j in range(N_LAYERS):
            ax.text(j, i, f"{roi_gate[i,j]:.02f}", ha="center", va="center", fontsize=7,
                    color="white" if roi_gate[i,j] > .075 else "#1A1A1A")
    fig.colorbar(im, ax=ax, fraction=.025, pad=.018, label="Mean cross-validated gate weight")
    p = FIG / "04_roi_by_resnet_layer_gaussian_gate_heatmap.png"; fig.savefig(p, dpi=320); plt.close(fig); paths.append(p)
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    started = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.set_float32_matmul_precision("high")
    pop = np.load(POP, allow_pickle=True)
    flat = np.load(MAPS, mmap_mode="r")
    raw = flat.reshape(len(flat), N_LAYERS, DIMS, flat.shape[-2], flat.shape[-1])
    results, manifests, gates_all, fields_all = [], [], [], []
    for roi in ROIS:
        cache = OUT / f"{slug(roi)}_final_oof.npz"
        target, manifest = load_roi_targets(roi, pop)
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            pred = z["prediction"].astype(np.float32); gates = z["gates"].astype(np.float32)
            field_rows = pd.read_csv(OUT / f"{slug(roi)}_fold_fields.csv").to_dict("records")
            print(f"{roi}: resumed cached final OOF", flush=True)
        else:
            pred, gates, field_rows = fit_roi(target, raw, device, roi)
            np.savez_compressed(cache, target=target, prediction=pred, gates=gates,
                                unit_global=manifest.unit_global.to_numpy(), layers=np.asarray(LAYERS))
            pd.DataFrame(field_rows).to_csv(OUT / f"{slug(roi)}_fold_fields.csv", index=False,
                                             encoding="utf-8-sig")
        manifest["repeat_mean_oof_r"] = corr_columns(pred, target)
        result = attach_nc(manifest)
        results.append(result); manifests.append(manifest); gates_all.append(gates); fields_all.extend(field_rows)
        pd.concat(results, ignore_index=True).to_csv(OUT / "all_unit_best_window_results.csv",
                                                     index=False, encoding="utf-8-sig")
        print(f"{roi}: saved {len(manifest):,} units; mean repeat r={manifest.repeat_mean_oof_r.mean():.4f}",
              flush=True)
        torch.cuda.empty_cache()
    result = pd.concat(results, ignore_index=True); manifest = pd.concat(manifests, ignore_index=True)
    gates = np.concatenate(gates_all, axis=1)
    result.to_csv(OUT / "all_unit_best_window_results.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "all_unit_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fields_all).to_csv(OUT / "all_fold_spatial_and_depth_parameters.csv", index=False,
                                    encoding="utf-8-sig")
    roi_summary = result.groupby("roi", sort=False).agg(
        n=("unit_global", "size"), mean_repeat_r=("repeat_mean_oof_r", "mean"),
        mean_single_trial_r=("single_trial_equivalent_r", "mean"),
        median_single_trial_r=("single_trial_equivalent_r", "median"),
        mean_ceiling=("r_ceiling_1", "mean")).reset_index()
    monkey_summary = result.groupby("monkey", sort=False).agg(
        n=("unit_global", "size"), mean_repeat_r=("repeat_mean_oof_r", "mean"),
        mean_single_trial_r=("single_trial_equivalent_r", "mean"),
        mean_ceiling=("r_ceiling_1", "mean")).reset_index()
    roi_summary.to_csv(OUT / "summary_by_roi.csv", index=False, encoding="utf-8-sig")
    monkey_summary.to_csv(OUT / "summary_by_monkey.csv", index=False, encoding="utf-8-sig")
    paths = make_figures(result, gates, manifest)
    audit = {
        "backbone": "torchvision ResNet-50 IMAGENET1K_V2",
        "population": "all 23,960 locally mapped Triple-N units; SU/MU/non-somatic; no additional consistency or performance screen",
        "local_population_note": "released independent consistency ranges 0.400026–1.0 because 0.4 is inherited from the released population cache, not re-applied here",
        "target": "one 20-ms window per unit, selected by image-wise response variance from the released 20-ms cache",
        "spatial_readout": "outer-fold target-specific continuous Gaussian fwRF",
        "depth_readout": "outer-train-only smooth exact Gaussian gate over all 17 nodes",
        "joint_readout": "signed ridge over 17x8 TVSD-frozen neural-blind PCA channels",
        "hard_top_k": False, "roi_specific_routing": False, "temporal_parameter_sharing": False,
        "outer_test_used_for_space_or_depth_selection": False,
        "noise_ceiling": "NSD NC(1), matched to the exact selected 20-ms target",
        "elapsed_seconds": time.time() - started, "figures": [str(x) for x in paths],
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(roi_summary.to_string(index=False), flush=True)
    print(monkey_summary.to_string(index=False), flush=True)
    print(FINAL, flush=True)


if __name__ == "__main__":
    main()
