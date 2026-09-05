"""Extended response-blind PCA-rank sweep for the final Triple-N proxy model.

The pilot isolates channel rank.  It reuses the exact outer-fold Gaussian
fields and smooth depth gates learned by the archived PCA8 model, while all
ranks are prefixes of one response-blind PCA256 basis.  Units are selected only
by NC(1), never by the existing model prediction.
"""

from __future__ import annotations

import io
import json
import math
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.decomposition import PCA


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
OUT = PROJECT / "results" / "rank_sweep_extended_2026-08-26"
CACHE = OUT / "trin_resnet50_pca256_maps_f16.npy"
BASIS = OUT / "response_blind_pca256_basis.npz"
STIMULI = ROOT / "data" / "TripleN" / "V1" / "others" / "StimuliNNN.zip"

LAYERS = (
    "stem", "res2.1", "res2.2", "res2.3", "res3.1", "res3.2", "res3.3", "res3.4",
    "res4.1", "res4.2", "res4.3", "res4.4", "res4.5", "res4.6",
    "res5.1", "res5.2", "res5.3",
)
ROIS = ("V1", "V2", "V4", "posterior IT", "middle IT", "anterior IT")
RANKS = (8, 16, 32, 64, 128, 256)
N_IMAGES = 1000
N_PER_ROI = 20
GRID = 14
MAX_RANK = 256
SEED = 20260818 + 901
FOLDS = 5
RIDGES = (0.03, 0.1, 0.3, 1.0, 3.0)


def slug(x: str) -> str:
    return x.lower().replace(" ", "_")


def fixed_folds(n: int):
    order = np.random.default_rng(SEED).permutation(n)
    chunks = np.array_split(order, FOLDS)
    return [(np.concatenate([chunks[j] for j in range(FOLDS) if j != k]), chunks[k])
            for k in range(FOLDS)]


def forward_nodes(model: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    out = []
    x = model.relu(model.bn1(model.conv1(x))); out.append(x)
    x = model.maxpool(x)
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            x = block(x); out.append(x)
    if len(out) != len(LAYERS):
        raise RuntimeError(f"expected 17 nodes, got {len(out)}")
    return out


def read_batch(archive: zipfile.ZipFile, indices: np.ndarray, transform) -> torch.Tensor:
    values = []
    for i in indices:
        with Image.open(io.BytesIO(archive.read(f"{int(i) + 1:04d}.bmp"))) as im:
            values.append(transform(im.convert("RGB")))
    return torch.stack(values)


def fit_pca_basis(model, archive, transform, device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if BASIS.exists():
        z = np.load(BASIS, allow_pickle=True)
        return [z[f"mean_{i}"] for i in range(17)], [z[f"basis_{i}"] for i in range(17)]
    rng = np.random.default_rng(20260826)
    chosen = np.sort(rng.choice(N_IMAGES, 256, replace=False))
    samples: list[list[np.ndarray]] = [[] for _ in LAYERS]
    with torch.inference_mode():
        for start in range(0, len(chosen), 16):
            ids = chosen[start:start + 16]
            nodes = forward_nodes(model, read_batch(archive, ids, transform).to(device))
            for li, node in enumerate(nodes):
                # A fixed response-blind spatial sample from every image.
                b, c, h, w = node.shape
                flat = node.permute(0, 2, 3, 1).reshape(b, h * w, c)
                pos = torch.linspace(0, h * w - 1, 16, device=device).long()
                take = flat[:, pos].reshape(-1, c).float().cpu().numpy()
                samples[li].append(take)
            del nodes
    means, bases, payload = [], [], {"layers": np.asarray(LAYERS)}
    for li, chunks in enumerate(samples):
        x = np.concatenate(chunks, axis=0).astype(np.float32)
        pca = PCA(n_components=min(MAX_RANK, x.shape[1]), svd_solver="randomized",
                  random_state=20260826, iterated_power=4)
        pca.fit(x)
        mean = pca.mean_.astype(np.float32)
        basis = pca.components_.T.astype(np.float32)
        if basis.shape[1] < MAX_RANK:
            basis = np.pad(basis, ((0, 0), (0, MAX_RANK - basis.shape[1])))
        means.append(mean); bases.append(basis)
        payload[f"mean_{li}"] = mean
        payload[f"basis_{li}"] = basis
        payload[f"explained_{li}"] = np.pad(
            pca.explained_variance_ratio_.astype(np.float32),
            (0, MAX_RANK - len(pca.explained_variance_ratio_)))
        print(f"PCA {li + 1:02d}/17 {LAYERS[li]} C={x.shape[1]}", flush=True)
    np.savez_compressed(BASIS, **payload)
    return means, bases


def extract_maps(model, archive, transform, device, means, bases) -> np.memmap:
    shape = (N_IMAGES, len(LAYERS), MAX_RANK, GRID, GRID)
    if CACHE.exists():
        out = np.load(CACHE, mmap_mode="r")
        if out.shape == shape:
            return out
    out = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float16, shape=shape)
    with torch.inference_mode():
        for start in range(0, N_IMAGES, 12):
            stop = min(start + 12, N_IMAGES)
            nodes = forward_nodes(model, read_batch(
                archive, np.arange(start, stop), transform).to(device))
            for li, node in enumerate(nodes):
                mean = torch.as_tensor(means[li], device=device)[None, :, None, None]
                basis = torch.as_tensor(bases[li], device=device)
                proj = torch.einsum("bchw,ck->bkhw", node.float() - mean, basis)
                if proj.shape[-1] != GRID:
                    proj = F.interpolate(proj, size=(GRID, GRID), mode="bilinear", align_corners=False)
                out[start:stop, li] = proj.cpu().numpy().astype(np.float16)
            if stop % 60 == 0 or stop == N_IMAGES:
                print(f"projected images {stop}/1000", flush=True)
            del nodes
    out.flush()
    return np.load(CACHE, mmap_mode="r")


def gaussian_field(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # Same 20-degree Gaussian-mass geometry used by the archived fwRF.
    dpix = 20.0 / GRID
    coord = np.arange(GRID, dtype=np.float32) * dpix - 10.0 + dpix / 2
    xm, ym0 = np.meshgrid(coord, coord)
    ym = -ym0
    result = []
    from scipy.special import erf
    for xx, yy, ss in zip(x, y, sigma):
        gx = .5 * (erf((xm - xx + dpix / 2) / (math.sqrt(2) * ss)) -
                   erf((xm - xx - dpix / 2) / (math.sqrt(2) * ss)))
        gy = .5 * (erf((ym - yy + dpix / 2) / (math.sqrt(2) * ss)) -
                   erf((ym - yy - dpix / 2) / (math.sqrt(2) * ss)))
        exact = gx * gy
        d = 2 * ss ** 2
        approx = dpix ** 2 / (d * np.pi) * np.exp(-((xm - xx) ** 2 + (ym - yy) ** 2) / d)
        result.append(exact if ss < dpix else approx)
    return np.asarray(result, np.float32).reshape(len(result), -1)


def corr_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a - a.mean(0, keepdims=True); bb = b - b.mean(0, keepdims=True)
    den = np.sqrt((aa * aa).sum(0) * (bb * bb).sum(0))
    return np.divide((aa * bb).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 1e-12)


def ridge_predict(xtr, ytr, xte, alpha):
    mx = xtr.mean(0); my = ytr.mean()
    xc = xtr - mx; yc = ytr - my
    # Preserve the original scale convention while using the mathematically
    # equivalent dual form once features outnumber training images.
    lam = float(alpha) * float(np.sum(xc * xc)) / max(1, xc.shape[1])
    lam = max(lam, 1e-6)
    if xc.shape[1] > xc.shape[0]:
        dual = xc @ xc.T
        coef = np.linalg.solve(
            dual + np.eye(dual.shape[0], dtype=np.float32) * lam, yc)
        return ((xte - mx) @ xc.T) @ coef + my
    gram = xc.T @ xc
    beta = np.linalg.solve(
        gram + np.eye(gram.shape[0], dtype=np.float32) * lam, xc.T @ yc)
    return (xte - mx) @ beta + my


def select_units() -> pd.DataFrame:
    all_results = pd.read_csv(PROJECT / "results" / "all_unit_best_window_results.csv")
    rows = []
    for roi in ROIS:
        frame = all_results[all_results.roi.eq(roi)].sort_values(
            ["r_ceiling_1", "released_independent_consistency"], ascending=False).head(N_PER_ROI).copy()
        frame["target_local"] = -1
        z = np.load(PROJECT / "results" / f"{slug(roi)}_final_oof.npz")
        lookup = {int(u): i for i, u in enumerate(z["unit_global"])}
        frame["target_local"] = [lookup[int(u)] for u in frame.unit_global]
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "selected_units.csv", index=False, encoding="utf-8-sig")
    return out


def evaluate(maps: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda")
    rows = []
    folds = fixed_folds(N_IMAGES)
    for roi in ROIS:
        sel = selected[selected.roi.eq(roi)].copy().reset_index(drop=True)
        z = np.load(PROJECT / "results" / f"{slug(roi)}_final_oof.npz")
        local = sel.target_local.to_numpy(int)
        target = z["target"][:, local].astype(np.float32)
        gates = z["gates"][:, local].astype(np.float32)
        fields_table = pd.read_csv(PROJECT / "results" / f"{slug(roi)}_fold_fields.csv")
        pred_by_rank = {rank: np.full_like(target, np.nan) for rank in RANKS}
        alpha_by_rank = {rank: [] for rank in RANKS}
        for fi, (train, test) in enumerate(folds):
            ft = fields_table[(fields_table.fold == fi) & fields_table.target_local.isin(local)]
            ft = ft.set_index("target_local").loc[local]
            fields = gaussian_field(ft.center_x.to_numpy(), ft.center_y.to_numpy(), ft.sigma_space.to_numpy())
            # Normalize once at rank256, independently per layer and fold as in the final model.
            tr = torch.as_tensor(np.asarray(maps[train], np.float32), device=device).flatten(3)
            te = torch.as_tensor(np.asarray(maps[test], np.float32), device=device).flatten(3)
            mean = tr.mean((0, 3), keepdim=True)
            std = tr.std((0, 3), keepdim=True).clamp_min(1e-5)
            tr = (tr - mean) / std; te = (te - mean) / std
            fld = torch.as_tensor(fields, device=device)
            # image x unit x layer x rank
            xtr_max = torch.einsum("nldp,up->nuld", tr, fld).cpu().numpy()
            xte_max = torch.einsum("nldp,up->nuld", te, fld).cpu().numpy()
            del tr, te
            rng = np.random.default_rng(20260826 + fi)
            perm = rng.permutation(len(train)); inner_val = perm[:160]; inner_train = perm[160:]
            for rank in RANKS:
                gate_scale = np.sqrt(np.clip(gates[fi] * len(LAYERS), 1e-5, None))
                xtr = (xtr_max[:, :, :, :rank] * gate_scale[None, :, :, None]).reshape(len(train), len(sel), -1)
                xte = (xte_max[:, :, :, :rank] * gate_scale[None, :, :, None]).reshape(len(test), len(sel), -1)
                for ui in range(len(sel)):
                    best_alpha, best_r = None, -np.inf
                    for alpha in RIDGES:
                        pv = ridge_predict(xtr[inner_train, ui], target[train][inner_train, ui],
                                           xtr[inner_val, ui], alpha)
                        rr = corr_cols(pv[:, None], target[train][inner_val, ui][:, None])[0]
                        if np.isfinite(rr) and rr > best_r:
                            best_r, best_alpha = rr, alpha
                    if best_alpha is None:
                        best_alpha = 0.1
                    pred_by_rank[rank][test, ui] = ridge_predict(
                        xtr[:, ui], target[train, ui], xte[:, ui], best_alpha)
                    alpha_by_rank[rank].append(best_alpha)
            print(f"{roi}: fold {fi + 1}/5", flush=True)
            del xtr_max, xte_max
            torch.cuda.empty_cache()
        for rank in RANKS:
            rr = corr_cols(pred_by_rank[rank], target)
            for ui, r in enumerate(rr):
                rows.append({"roi": roi, "unit_global": int(sel.unit_global.iloc[ui]),
                             # The stem has only 64 native channels; all later
                             # ResNet-50 nodes have at least 256 channels.
                             "rank_per_layer": rank, "total_features": min(rank, 64) + 16 * rank,
                             "oof_r": float(r), "r_ceiling_1": float(sel.r_ceiling_1.iloc[ui]),
                             "median_selected_ridge": float(np.median(alpha_by_rank[rank]))})
    return pd.DataFrame(rows)


def plot_results(result: pd.DataFrame) -> None:
    summary = result.groupby("rank_per_layer").agg(
        mean_r=("oof_r", "mean"), median_r=("oof_r", "median"),
        sem_r=("oof_r", lambda x: x.std(ddof=1) / np.sqrt(len(x)))).reset_index()
    wide = result.pivot(index=["roi", "unit_global"], columns="rank_per_layer", values="oof_r")
    for rank in RANKS[1:]:
        delta = wide[rank] - wide[8]
        summary.loc[summary.rank_per_layer.eq(rank), "mean_delta_vs_8"] = delta.mean()
        summary.loc[summary.rank_per_layer.eq(rank), "win_fraction_vs_8"] = (delta > 0).mean()
    summary.loc[summary.rank_per_layer.eq(8), ["mean_delta_vs_8", "win_fraction_vs_8"]] = [0, .5]
    summary.to_csv(OUT / "rank_summary.csv", index=False, encoding="utf-8-sig")
    roi = result.groupby(["roi", "rank_per_layer"]).oof_r.mean().unstack()
    roi.to_csv(OUT / "rank_by_roi_mean_r.csv", encoding="utf-8-sig")

    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    axes[0].errorbar(summary.rank_per_layer, summary.mean_r, yerr=summary.sem_r,
                     marker="o", lw=2, capsize=4, color="#2878B5")
    axes[0].set(xlabel="PCA rank per ResNet node", ylabel="Mean five-fold OOF Pearson r",
                title="Rank sweep on 120 high-NC Triple-N units")
    axes[0].grid(alpha=.3)
    for name, values in roi.iterrows():
        axes[1].plot(values.index, values.values, marker="o", label=name)
    axes[1].set(xlabel="PCA rank per ResNet node", ylabel="Mean five-fold OOF Pearson r",
                title="Rank sensitivity by ROI")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "rank_sweep_oof_r.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    with zipfile.ZipFile(STIMULI) as archive:
        means, bases = fit_pca_basis(model, archive, transform, device)
        maps = extract_maps(model, archive, transform, device, means, bases)
    selected = select_units()
    result = evaluate(maps, selected)
    result.to_csv(OUT / "unit_level_rank_sweep.csv", index=False, encoding="utf-8-sig")
    plot_results(result)
    (OUT / "audit.json").write_text(json.dumps({
        "sample": "top 20 NC(1) units per six coarse ROIs; no model-r selection",
        "n_units": int(selected.unit_global.nunique()), "ranks": list(RANKS),
        "fixed_across_ranks": ["images", "units", "targets", "outer folds", "Gaussian fwRF", "depth gates"],
        "pca": "one response-blind PCA256 basis fit on 256 Triple-N images x 16 positions; ranks are nested prefixes; native channel count caps each node",
        "ridge": "per-unit inner validation among 0.03,0.1,0.3,1,3",
        "elapsed_seconds": time.time() - started,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT / "rank_summary.csv").to_string(index=False), flush=True)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
