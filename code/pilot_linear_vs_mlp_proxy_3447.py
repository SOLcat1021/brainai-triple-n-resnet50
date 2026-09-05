"""Small paired linear-vs-MLP readout pilot for middle-IT unit 3447."""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import numpy as np
import torch

ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
import run_trin_closed_loop_validation as folds
from pilot_tvsd_continuous_gaussian import continuous_gaussian

MAPS = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "block_pca8_14_tvsd_frozen" / "maps_f16.npy"
RESP = ROOT / "cache" / "trin" / "population_time10_area_starts_all_unittypes" / "middle_it_responses.npy"
META = ROOT / "cache" / "trin" / "population_time10_area_starts_all_unittypes" / "middle_it_metadata.npz"
BANK = PROJECT / "proxy_bank_10ms_all_windows" / "middle_it_proxy_bank.h5"
UNIT, WINDOW = 3447, (150, 159)
RIDGE = 0.05
SEED = 20260908
EPOCHS = 120

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum() / np.sqrt((a*a).sum()*(b*b).sum()))

def pool_maps(maps, field):
    # maps are [N, 136, 14, 14]; field is unit-specific and normalized.
    return np.einsum("ncpq,pq->nc", maps.astype(np.float32), field).astype(np.float32)

def ridge_fit_predict(xtr, ytr, xte):
    xm, ym = xtr.mean(0), ytr.mean()
    xc = xtr - xm
    lam = RIDGE * np.trace(xc.T @ xc) / xtr.shape[1]
    beta = np.linalg.solve(xc.T @ xc + lam*np.eye(xtr.shape[1]), xc.T @ (ytr-ym))
    return (xte-xm) @ beta + ym

def mlp_fit_predict(xtr, ytr, xte, seed):
    torch.manual_seed(seed)
    xm, xs = xtr.mean(0), xtr.std(0).clip(1e-5)
    xtr = (xtr-xm)/xs; xte = (xte-xm)/xs
    ym, ys = float(ytr.mean()), float(ytr.std() or 1.0)
    yt = (ytr-ym)/ys
    model = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 64), torch.nn.ReLU(),
                                 torch.nn.Linear(64, 1)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    x = torch.as_tensor(xtr, device="cuda"); y = torch.as_tensor(yt[:,None], device="cuda")
    for _ in range(EPOCHS):
        pred = model(x); loss = torch.mean((pred-y)**2)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (model(torch.as_tensor(xte, device="cuda")).squeeze(1).cpu().numpy()*ys + ym)

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for this pilot")
    maps = np.load(MAPS, mmap_mode="r")
    responses = np.load(RESP, mmap_mode="r")
    windows = np.load(META, allow_pickle=True)["windows"]
    wi = int(np.flatnonzero(np.all(windows == WINDOW, axis=1))[0])
    with h5py.File(BANK, "r") as h:
        ui = int(np.flatnonzero(h["unit_global"][:] == UNIT)[0])
        widx = int(np.flatnonzero(np.all(h["windows_ms"][:] == WINDOW, axis=1))[0])
        cx, cy, sigma = h["spatial_parameters"][widx, ui]
        gate = h["layer_gate"][widx, ui].astype(np.float32)
        norm_mean = h["normalization_mean"][:].astype(np.float32)
        norm_std = h["normalization_std"][:].astype(np.float32)
    yy = responses[wi, :, ui].astype(np.float32)
    # Reproduce the main proxy input: normalization, unit-specific Gaussian pool,
    # and fixed layer gate absorbed as a feature scale.
    maps_n = (maps.astype(np.float32)-norm_mean[None,:,None,None]) / norm_std[None,:,None,None]
    field = continuous_gaussian(torch.as_tensor([cx], device="cuda"),
                                torch.as_tensor([cy], device="cuda"),
                                torch.as_tensor([sigma], device="cuda"), 14)
    field = field.reshape(14, 14).cpu().numpy()
    X = pool_maps(maps_n, field)
    X *= np.repeat(np.sqrt(np.clip(gate, 1e-5, None)*17.0), 8)[None]
    print(f"device=cuda; unit={UNIT}; window={WINDOW}; X={X.shape}; MLP=136-64-1 ReLU")
    lr, mr = [], []
    linear_oof = np.empty(len(X), np.float32)
    mlp_oof = np.empty(len(X), np.float32)
    for fi, (_, test) in enumerate(folds.fixed_folds(len(X))):
        train = np.setdiff1d(np.arange(len(X)), test, assume_unique=True)
        p1 = ridge_fit_predict(X[train], yy[train], X[test])
        p2 = mlp_fit_predict(X[train], yy[train], X[test], SEED+fi)
        linear_oof[test] = p1; mlp_oof[test] = p2
        lr.append(corr(p1, yy[test])); mr.append(corr(p2, yy[test]))
        print(f"fold {fi+1}/5: linear r={lr[-1]:+.4f}; MLP r={mr[-1]:+.4f}", flush=True)
    print(f"\nlinear OOF Pearson r={corr(linear_oof, yy):+.4f}")
    print(f"MLP OOF Pearson r={corr(mlp_oof, yy):+.4f}")
    print(f"linear fold mean={np.mean(lr):+.4f}, median={np.median(lr):+.4f}")
    print(f"MLP fold mean={np.mean(mr):+.4f}, median={np.median(mr):+.4f}, delta mean={np.mean(np.asarray(mr)-lr):+.4f}")

if __name__ == "__main__": main()
