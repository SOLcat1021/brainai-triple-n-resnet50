"""Small SAE stability pilot on the fixed TVSD res4+PCA64 space.

The upstream PCA basis and fixed Gaussian pool are frozen.  Five independent
without-replacement subsets (1,000 and 5,000 images) train a 64->128 linear
ReLU sparse autoencoder.  Decoder-column Hungarian alignment removes latent
permutation ambiguity before comparing unit/concept latent directions.
No files are written.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

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
N_BATCH = 512
HIDDEN = 128
SAMPLE_SIZES = (1000, 5000)
N_REPS = 5
EPOCHS = 220
LR = 2e-3
TOPK_VALUES = (8, 16)
SEED = 20260904


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), den,
                     out=np.full(a.shape[0], np.nan), where=den > 1e-12)


def qsummary(x: np.ndarray) -> str:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (f"min={np.min(x):.3f} max={np.max(x):.3f} "
            f"P5={np.percentile(x, 5):.3f} P95={np.percentile(x, 95):.3f} "
            f"median={np.median(x):.3f}") if len(x) else "n/a"


def pooled(path: Path, field: np.ndarray, device: torch.device) -> np.ndarray:
    maps = np.load(path, mmap_mode="r")
    n, d, _, _ = maps.shape
    out = np.empty((n, d), np.float32)
    f = torch.as_tensor(field, device=device)
    for a in range(0, n, N_BATCH):
        b = min(a + N_BATCH, n)
        x = torch.as_tensor(np.asarray(maps[a:b], np.float32), device=device)
        out[a:b] = torch.einsum("nchw,hw->nc", x, f).detach().cpu().numpy()
    return out


def frozen_pca(X: np.ndarray, q: int = 64, device: torch.device | None = None):
    device = device or torch.device("cpu")
    mean = X.mean(0).astype(np.float32)
    std = X.std(0).clip(1e-5).astype(np.float32)
    xt = torch.as_tensor((X - mean[None]) / std[None], device=device)
    _, _, V = torch.pca_lowrank(xt, q=q, center=False)
    Z = (xt @ V).detach().cpu().numpy().astype(np.float32)
    # Normalize PCA coordinates so SAE sparsity is not dominated by variance.
    zstd = Z.std(0).clip(1e-5).astype(np.float32)
    return Z / zstd[None], mean, std, V.detach().cpu().numpy(), zstd


class SAE(torch.nn.Module):
    def __init__(self, d: int, hidden: int, topk: int):
        super().__init__()
        self.enc = torch.nn.Linear(d, hidden)
        self.dec = torch.nn.Linear(hidden, d)
        self.topk = topk

    def forward(self, x):
        pre = torch.relu(self.enc(x))
        # Top-K SAE: exactly K active latent features per sample, making the
        # sparsity constraint explicit rather than relying on a tunable L1.
        values, indices = torch.topk(pre, k=self.topk, dim=1)
        z = torch.zeros_like(pre).scatter(1, indices, values)
        return self.dec(z), z


def train_sae(X: np.ndarray, device: torch.device, seed: int, topk: int) -> tuple[SAE, float, float]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = SAE(X.shape[1], HIDDEN, topk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.as_tensor(X, device=device)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), N_BATCH):
            xb = x[order[start:start + N_BATCH]]
            rec, z = model(xb)
            loss = torch.mean((rec - xb) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rec, z = model(x)
        mse = float(torch.mean((rec - x) ** 2).cpu())
        active = float((z > 1e-3).float().mean().cpu())
    return model, mse, active


def decoder_matrix(model: SAE) -> np.ndarray:
    # Columns are latent decoder directions in the 64D input space.
    # torch.nn.Linear stores weight as [input_features, output_features]
    # for this decoder's 128->64 mapping, so it is already 64 x 128.
    return model.dec.weight.detach().cpu().numpy().astype(np.float32)


def align_to_reference(ref_dec: np.ndarray, dec: np.ndarray) -> np.ndarray:
    a = ref_dec / np.linalg.norm(ref_dec, axis=0, keepdims=True).clip(1e-12)
    b = dec / np.linalg.norm(dec, axis=0, keepdims=True).clip(1e-12)
    sim = a.T @ b
    rows, cols = linear_sum_assignment(-np.abs(sim))
    perm = np.empty_like(cols)
    perm[rows] = cols
    out = dec[:, perm]
    signs = np.sign(np.sum(ref_dec * out, axis=0)); signs[signs == 0] = 1
    return out * signs[None]


def align_encoder_codes(ref_dec: np.ndarray, dec: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Permute encoder codes using signed decoder-column matching.

    No sign flip is applied to ReLU codes: unlike a linear coefficient vector,
    an encoder activation is nonnegative and its sign is part of the model.
    """
    a = ref_dec / np.linalg.norm(ref_dec, axis=0, keepdims=True).clip(1e-12)
    b = dec / np.linalg.norm(dec, axis=0, keepdims=True).clip(1e-12)
    rows, cols = linear_sum_assignment(-(a.T @ b))
    perm = np.empty_like(cols); perm[rows] = cols
    return codes[:, perm]


def latent_direction(dec: np.ndarray, direction: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Minimum-norm latent vector whose decoder reconstruction matches direction."""
    gram = dec.T @ dec + ridge * np.eye(dec.shape[1], dtype=np.float32)
    return np.linalg.solve(gram, dec.T @ direction).astype(np.float32)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, _, chosen, chosen_roi, _ = tvsd.load_targets()
    n, n_time, n_units = y.shape
    xs, ys, sigmas = tvsd.candidates()
    center = int(np.argmin((xs ** 2 + ys ** 2) + (sigmas - 8.0) ** 2))
    field = tvsd.gaussian_mass_stack(xs[center:center + 1], ys[center:center + 1], sigmas[center:center + 1], 14)[0]
    print(f"device={device}; frozen stage={STAGE}; fixed pool sigma={sigmas[center]:.3f}; units={chosen.tolist()}")

    Xnative = pooled(FEATURES / f"train_{STAGE}_f16.npy", field, device)
    Z, mean, std, V, zstd = frozen_pca(Xnative, 64, device)
    # Full-data ridge unit axes in the same normalized PCA coordinates.
    target = y.reshape(n, n_time * n_units).astype(np.float32)
    zt = torch.as_tensor(Z, device=device); yt = torch.as_tensor(target, device=device)
    zm = zt.mean(0); ym = yt.mean(0); zc = zt - zm[None]; yc = yt - ym[None]
    gram = zc.T @ zc; rhs = zc.T @ yc
    lam = 0.05 * torch.trace(gram) / 64
    beta = torch.linalg.solve(gram + torch.eye(64, device=device) * lam, rhs).detach().cpu().numpy().T
    unit_axes = np.asarray([beta[u * n_time:(u + 1) * n_time].mean(0) for u in range(n_units)], np.float32)

    cav = np.load(CAV_PATH, allow_pickle=True)
    concepts = cav["concepts"].astype(str); nodes = cav["nodes"].astype(str)
    dims = cav["dims"].astype(int); offsets = cav["offsets"].astype(int)
    ni = int(np.flatnonzero(nodes == CAV_NODE)[0]); a, b = int(offsets[ni]), int(offsets[ni + 1])
    cav_native = np.asarray(cav["cav_full"][..., a:b], np.float32)
    # Transform native CAVs into the frozen normalized PCA64 coordinates.
    cav_pca = (cav_native * std[None]) @ V / zstd[None]
    # A small, interpretable subset for stability readout.
    wanted = ["head", "face", "eye", "wall", "person", "wood", "building", "texture"]
    cidx = [int(np.flatnonzero(concepts == x)[0]) for x in wanted if np.any(concepts == x)]
    concept_axes = cav_pca[cidx]
    axis_names = [concepts[i] for i in cidx]
    directions = np.concatenate([unit_axes, concept_axes], axis=0)
    direction_names = [f"unit_{int(u)}_{chosen_roi[u]}" for u in range(n_units)] + axis_names

    rng = np.random.default_rng(SEED)
    for topk in TOPK_VALUES:
        for size in SAMPLE_SIZES:
            models = []; metrics = []
            for rep in range(N_REPS):
                idx = rng.choice(n, size=size, replace=False)
                model, mse, active = train_sae(Z[idx], device, SEED + topk * 10000 + size + rep, topk)
                models.append(model); metrics.append((mse, active))
                print(f"topk={topk}, size={size}, repeat={rep + 1}/{N_REPS}, recon_mse={mse:.5f}, "
                      f"active_fraction={active:.4f}", flush=True)
            ref_dec = decoder_matrix(models[0])
            latent = []
            for model in models:
                dec = align_to_reference(ref_dec, decoder_matrix(model))
                latent.append(np.asarray([latent_direction(dec, d) for d in directions]))
            latent = np.stack(latent)
            # Compare each axis across repeats after decoder alignment.
            pair = np.stack([cosine_rows(latent[i], latent[j]) for i, j in combinations(range(N_REPS), 2)])
            # Sparsity is reported for latent direction coefficients, not just images.
            l0 = (np.abs(latent) > 1e-3).mean(axis=2)
            print(f"\\nSAE SUMMARY topk={topk}, size={size}")
            print(f"reconstruction MSE median={np.median([m[0] for m in metrics]):.6f}; "
                  f"image active fraction median={np.median([m[1] for m in metrics]):.4f}")
            print("all-axis latent cosine:", qsummary(pair.ravel()))
            for k, name in enumerate(direction_names):
                print(f"  {name:18s}: cosine {qsummary(pair[:, k])}; latent L0 fraction={np.mean(l0[:, k]):.3f}")
            print("decoder column cosine to repeat 1:", qsummary(np.concatenate([
                cosine_rows(ref_dec.T, align_to_reference(ref_dec, decoder_matrix(m)).T) for m in models[1:]])))
            # Correct axis representation: pass each 64D direction through
            # the trained encoder, so Top-K is applied to the axis itself.
            axis_codes = []
            for model in models:
                with torch.no_grad():
                    d = torch.as_tensor(directions, device=device)
                    pre = torch.relu(model.enc(d))
                    val, ind = torch.topk(pre, k=model.topk, dim=1)
                    code = torch.zeros_like(pre).scatter(1, ind, val).cpu().numpy()
                code = align_encoder_codes(ref_dec, decoder_matrix(model), code)
                axis_codes.append(code)
            axis_codes = np.stack(axis_codes)
            code_pair = np.stack([cosine_rows(axis_codes[i], axis_codes[j])
                                  for i, j in combinations(range(N_REPS), 2)])
            print(f"encoder Top-K axis representation: exactly {model.topk} active latent per axis")
            print("encoder-axis cross-repeat cosine:", qsummary(code_pair.ravel()))
            for k, name in enumerate(direction_names):
                supports = [set(np.flatnonzero(axis_codes[r, k])) for r in range(N_REPS)]
                jac = [len(supports[i] & supports[j]) / max(1, len(supports[i] | supports[j]))
                       for i, j in combinations(range(N_REPS), 2)]
                print(f"  encoder {name:18s}: cosine {qsummary(code_pair[:, k])}; support Jaccard median={np.median(jac):.3f}")
            if topk == 256 and size == 5000:
                # Pairwise concept similarity in the aligned 4096D latent
                # space; report support overlap alongside cosine.
                concept_start = n_units
                concept_latent = latent[:, concept_start:]
                names = axis_names
                print("concept-concept similarity (5000 images, 4096D, per-repeat median):")
                for i, j in combinations(range(len(names)), 2):
                    cs = []
                    ov = []
                    for rep_latent in concept_latent:
                        ai = rep_latent[i]; aj = rep_latent[j]
                        cs.append(float(np.dot(ai, aj) / (np.linalg.norm(ai) * np.linalg.norm(aj) + 1e-12)))
                        ki = np.argsort(np.abs(ai))[::-1][:20]
                        kj = np.argsort(np.abs(aj))[::-1][:20]
                        ov.append(len(set(ki.tolist()) & set(kj.tolist())))
                    print(f"  {names[i]} vs {names[j]}: cosine={np.median(cs):.3f} "
                          f"(range {np.min(cs):.3f}..{np.max(cs):.3f}); top20 overlap={np.median(ov):.1f}")

    print("\\nNOTE: SAE is trained on frozen res4+PCA64 image activations; native CAVs are mapped through the same frozen PCA.")
    print("NOTE: cross-repeat latent comparisons use decoder-column Hungarian alignment; without it, permutation would make stability meaningless.")


if __name__ == "__main__":
    main()
