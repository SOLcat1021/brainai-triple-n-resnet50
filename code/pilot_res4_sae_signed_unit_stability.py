"""Top-K=256 SAE stability using signed sparse coding of unit axes."""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path: sys.path.insert(0, str(CODE))
import pilot_res4_sae_axis_stability as base
import pilot_tvsd_axis_sample_size_sensitivity as tvsd

HIDDEN, TOPK, N_REPS, SIZE, EPOCHS, SEED = 4096, 256, 3, 5000, 140, 20260907

def align(ref, other):
    a = ref / np.linalg.norm(ref, axis=0, keepdims=True).clip(1e-12)
    b = other / np.linalg.norm(other, axis=0, keepdims=True).clip(1e-12)
    sim = a.T @ b
    rows, cols = linear_sum_assignment(-sim)  # no sign flip: decoder atoms are ReLU features
    p = np.empty(HIDDEN, dtype=int); p[rows] = cols
    return other[:, p], sim[rows, cols]

def sparse_codes(dec, directions, alpha=2e-4, steps=1200):
    """Signed L1 sparse coding via batched FISTA (same objective as Lasso)."""
    dn = dec / np.linalg.norm(dec, axis=0, keepdims=True).clip(1e-12)
    d = torch.as_tensor(dn, dtype=torch.float32)
    v = torch.as_tensor(directions, dtype=torch.float32)
    # A power iteration gives a safe Lipschitz constant for grad ||Dz-v||^2/2.
    q = torch.randn(d.shape[1], 1); q /= torch.linalg.norm(q)
    for _ in range(30):
        q = d.T @ (d @ q); q /= torch.linalg.norm(q).clamp_min(1e-8)
    L = float((q.T @ d.T @ d @ q).item()) * 1.05
    z = torch.zeros(v.shape[0], d.shape[1]); z_prev = z.clone(); t = 1.0
    for _ in range(steps):
        momentum = z + ((t - 1) / (t + 1)) * (z - z_prev)
        grad = (momentum @ d.T - v) @ d
        z_new = torch.sign(momentum - grad / L) * torch.relu(torch.abs(momentum - grad / L) - alpha / L)
        z_prev, z = z, z_new
        t = (1 + np.sqrt(1 + 4*t*t)) / 2
    return z.numpy().astype(np.float32)

def cosine_rows(a, b):
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a*b, axis=1), den, out=np.full(a.shape[0], np.nan), where=den > 1e-12)

def summary(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f"median={np.median(x):.3f}, P5={np.percentile(x,5):.3f}, P95={np.percentile(x,95):.3f}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.HIDDEN = HIDDEN
    y, _, chosen, chosen_roi, _ = tvsd.load_targets(); n, nt, nu = y.shape
    xs, ys, sigmas = tvsd.candidates(); c = int(np.argmin(xs**2 + ys**2 + (sigmas-8)**2))
    field = tvsd.gaussian_mass_stack(xs[c:c+1], ys[c:c+1], sigmas[c:c+1], 14)[0]
    X = base.pooled(base.FEATURES / "train_res4_f16.npy", field, device)
    Z, mean, std, V, zstd = base.frozen_pca(X, 64, device)
    target = y.reshape(n, nt*nu).astype(np.float32)
    zt = torch.as_tensor(Z, device=device); yt = torch.as_tensor(target, device=device)
    zc, yc = zt-zt.mean(0), yt-yt.mean(0)
    gram, rhs = zc.T@zc, zc.T@yc
    beta = torch.linalg.solve(gram + torch.eye(64,device=device)*(0.05*torch.trace(gram)/64), rhs).cpu().numpy().T
    directions = np.asarray([beta[u*nt:(u+1)*nt].mean(0) for u in range(nu)], np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
    rng = np.random.default_rng(SEED)
    idxs = [rng.choice(n, SIZE, replace=False) for _ in range(N_REPS)]
    models=[]; mses=[]
    for rep, idx in enumerate(idxs):
        model, mse, _ = base.train_sae(Z[idx], device, SEED + 2000*TOPK, TOPK)
        models.append(model); mses.append(mse)
        print(f"rep={rep+1}/{N_REPS} same initialization, different data, recon_mse={mse:.6f}", flush=True)
    ref = base.decoder_matrix(models[0]); codes=[]; dec_scores=[]
    for model in models:
        d, score = align(ref, base.decoder_matrix(model))
        codes.append(sparse_codes(d, directions)); dec_scores.append(score)
    print("\nTOPK=256 signed sparse coding of unit axes")
    print("reconstruction MSE:", summary(mses))
    print("matched decoder cosine:", summary(np.concatenate(dec_scores)))
    pair_cos=[]; pair_jac=[]; l0=[]
    for i,j in combinations(range(N_REPS),2):
        a,b=codes[i],codes[j]
        pair_cos.append(cosine_rows(a,b))
        aa=np.abs(a)>1e-3; bb=np.abs(b)>1e-3
        pair_jac.append(np.sum(aa&bb,axis=1)/np.maximum(1,np.sum(aa|bb,axis=1)))
    pair_cos=np.stack(pair_cos); pair_jac=np.stack(pair_jac)
    print("all unit-axis coefficient cosine:", summary(pair_cos.ravel()))
    print("all unit-axis support Jaccard:", summary(pair_jac.ravel()))
    for u in range(nu):
        print(f"unit_{int(chosen[u])}_{chosen_roi[u]}: cosine {summary(pair_cos[:,u])}; "
              f"Jaccard {summary(pair_jac[:,u])}; median L0={np.median([np.sum(np.abs(x[u])>1e-3) for x in codes]):.0f}")

if __name__ == "__main__": main()
