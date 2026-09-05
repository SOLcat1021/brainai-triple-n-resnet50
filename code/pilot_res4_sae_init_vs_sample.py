"""Disentangle SAE initialization instability from training-sample instability.

This is an exploratory diagnostic.  It freezes the full-data res4 PCA64 basis
and all evaluated unit/concept directions, then compares 4096-wide Top-K SAEs
under two conditions:

1. identical 5,000 images with different initializations;
2. different 5,000-image draws with identical initialization.

No result files or feature caches are written.
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

import pilot_res4_sae_axis_stability as base
import pilot_tvsd_axis_sample_size_sensitivity as tvsd

HIDDEN = 4096
TOPKS = (16, 32, 64, 256)
SAMPLE_SIZE = 5000
N_REPS = 3
EPOCHS = 140
BATCH_SIZE = 512
LR = 2e-3
SEED = 20260906


class WideSAE(torch.nn.Module):
    def __init__(self, input_dim: int, topk: int):
        super().__init__()
        self.enc = torch.nn.Linear(input_dim, HIDDEN)
        self.dec = torch.nn.Linear(HIDDEN, input_dim)
        self.topk = topk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(self.enc(x))
        values, indices = torch.topk(pre, k=self.topk, dim=1)
        return torch.zeros_like(pre).scatter(1, indices, values)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.dec(z), z


def train_sae(X: np.ndarray, topk: int, seed: int, device: torch.device) -> tuple[WideSAE, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WideSAE(X.shape[1], topk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.as_tensor(X, device=device)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), BATCH_SIZE):
            xb = x[order[start:start + BATCH_SIZE]]
            rec, _ = model(xb)
            loss = torch.mean((rec - xb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        rec, _ = model(x)
        mse = float(torch.mean((rec - x) ** 2).cpu())
    return model, mse


def decoder(model: WideSAE) -> np.ndarray:
    return model.dec.weight.detach().cpu().numpy().astype(np.float32)


def alignment(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return other->ref permutation and matched signed decoder cosine."""
    a = ref / np.linalg.norm(ref, axis=0, keepdims=True).clip(1e-12)
    b = other / np.linalg.norm(other, axis=0, keepdims=True).clip(1e-12)
    sim = a.T @ b
    # ReLU latents have no sign symmetry: matching an atom to its negative
    # would make the decoder look aligned while invalidating its code support.
    rows, cols = linear_sum_assignment(-sim)
    perm = np.empty(HIDDEN, dtype=np.int64)
    perm[rows] = cols
    matched = sim[rows, cols]
    return perm, matched


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), den,
                     out=np.full(a.shape[0], np.nan), where=den > 1e-12)


def signed_axis_codes(model: WideSAE, directions: np.ndarray, device: torch.device) -> np.ndarray:
    """Preserve signed evidence by concatenating enc(d) and enc(-d)."""
    with torch.no_grad():
        d = torch.as_tensor(directions, device=device)
        positive = model.encode(d).cpu().numpy()
        negative = model.encode(-d).cpu().numpy()
    return np.concatenate([positive, negative], axis=1)


def align_codes(codes: np.ndarray, perm: np.ndarray) -> np.ndarray:
    h = codes.shape[1] // 2
    return np.concatenate([codes[:, :h][:, perm], codes[:, h:][:, perm]], axis=1)


def support_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a != 0
    bb = b != 0
    inter = np.sum(aa & bb, axis=1)
    union = np.sum(aa | bb, axis=1)
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def summarize(x: np.ndarray) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (f"median={np.median(x):.3f}, P5={np.percentile(x, 5):.3f}, "
            f"P95={np.percentile(x, 95):.3f}")


def fixed_directions(Z: np.ndarray, pca_transform: tuple, device: torch.device) -> tuple[np.ndarray, list[str]]:
    y, _, chosen, chosen_roi, _ = tvsd.load_targets()
    n, n_time, n_units = y.shape
    target = y.reshape(n, n_time * n_units).astype(np.float32)
    z = torch.as_tensor(Z, device=device)
    yt = torch.as_tensor(target, device=device)
    zc = z - z.mean(0, keepdim=True)
    yc = yt - yt.mean(0, keepdim=True)
    gram = zc.T @ zc
    lam = 0.05 * torch.trace(gram) / Z.shape[1]
    beta = torch.linalg.solve(gram + torch.eye(Z.shape[1], device=device) * lam,
                              zc.T @ yc).detach().cpu().numpy().T
    unit_axes = np.asarray([
        beta[u * n_time:(u + 1) * n_time].mean(0) for u in range(n_units)
    ], np.float32)

    _, std, V, zstd = pca_transform
    cav = np.load(base.CAV_PATH, allow_pickle=True)
    concepts = cav["concepts"].astype(str)
    nodes = cav["nodes"].astype(str)
    offsets = cav["offsets"].astype(int)
    node = int(np.flatnonzero(nodes == base.CAV_NODE)[0])
    start, stop = int(offsets[node]), int(offsets[node + 1])
    cav_native = np.asarray(cav["cav_full"][..., start:stop], np.float32)
    cav_pca = (cav_native * std[None]) @ V / zstd[None]
    wanted = ("head", "face", "eye", "wall", "person", "wood", "building", "texture")
    cidx = [int(np.flatnonzero(concepts == name)[0]) for name in wanted
            if np.any(concepts == name)]
    directions = np.concatenate([unit_axes, cav_pca[cidx]], axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
    names = [f"unit_{int(chosen[u])}_{chosen_roi[u]}" for u in range(n_units)]
    names += [str(concepts[i]) for i in cidx]
    return directions.astype(np.float32), names


def evaluate_group(models: list[WideSAE], directions: np.ndarray,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decoders = [decoder(model) for model in models]
    raw_codes = [signed_axis_codes(model, directions, device) for model in models]
    decoder_scores = []
    axis_cosines = []
    axis_jaccards = []
    for i, j in combinations(range(len(models)), 2):
        perm, matched = alignment(decoders[i], decoders[j])
        aligned = align_codes(raw_codes[j], perm)
        decoder_scores.append(np.abs(matched))
        axis_cosines.append(cosine_rows(raw_codes[i], aligned))
        axis_jaccards.append(support_jaccard(raw_codes[i], aligned))
    return np.concatenate(decoder_scores), np.stack(axis_cosines), np.stack(axis_jaccards)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, _, _, _, _ = tvsd.load_targets()
    n = len(y)
    xs, ys, sigmas = tvsd.candidates()
    center = int(np.argmin((xs ** 2 + ys ** 2) + (sigmas - 8.0) ** 2))
    field = tvsd.gaussian_mass_stack(xs[center:center + 1], ys[center:center + 1],
                                     sigmas[center:center + 1], 14)[0]
    Xnative = base.pooled(base.FEATURES / "train_res4_f16.npy", field, device)
    Z, mean, std, V, zstd = base.frozen_pca(Xnative, 64, device)
    directions, names = fixed_directions(Z, (mean, std, V, zstd), device)

    rng = np.random.default_rng(SEED)
    fixed_idx = rng.choice(n, size=SAMPLE_SIZE, replace=False)
    varied_idx = [rng.choice(n, size=SAMPLE_SIZE, replace=False) for _ in range(N_REPS)]
    print(f"device={device}; input=PCA64 frozen on all {n} images; hidden={HIDDEN}; "
          f"sample_size={SAMPLE_SIZE}; epochs={EPOCHS}")
    print("directions:", ", ".join(names), flush=True)

    for topk in TOPKS:
        same_data = []
        same_init = []
        same_data_mse = []
        same_init_mse = []
        for rep in range(N_REPS):
            model, mse = train_sae(Z[fixed_idx], topk, SEED + 1000 * topk + rep, device)
            same_data.append(model)
            same_data_mse.append(mse)
            print(f"topk={topk:3d} same-data rep={rep + 1}/{N_REPS} mse={mse:.6f}", flush=True)
        for rep in range(N_REPS):
            model, mse = train_sae(Z[varied_idx[rep]], topk, SEED + 2000 * topk, device)
            same_init.append(model)
            same_init_mse.append(mse)
            print(f"topk={topk:3d} different-data rep={rep + 1}/{N_REPS} mse={mse:.6f}", flush=True)

        print(f"\nTOPK={topk} RESULTS")
        for label, models, mses in (
            ("same data, different init", same_data, same_data_mse),
            ("different data, same init", same_init, same_init_mse),
        ):
            dec, cos, jac = evaluate_group(models, directions, device)
            print(f"  {label}")
            print(f"    reconstruction MSE: {summarize(np.asarray(mses))}")
            print(f"    matched decoder |cos|: {summarize(dec)}")
            print(f"    signed-axis cosine: {summarize(cos.ravel())}")
            print(f"    signed-axis support Jaccard: {summarize(jac.ravel())}")
            unit_cos = cos[:, :10].ravel()
            concept_cos = cos[:, 10:].ravel()
            print(f"    unit-axis cosine: {summarize(unit_cos)}")
            print(f"    concept-axis cosine: {summarize(concept_cos)}")
        print("", flush=True)


if __name__ == "__main__":
    main()
