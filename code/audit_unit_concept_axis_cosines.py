"""Sampled cosine geometry of TVSD unit and Broden concept axes."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_consistency_ge_0p4_all_units_2026-08-30"
BASIS = PROJECT / "results" / "rank_sweep_extended_2026-08-26" / "response_blind_pca256_basis.npz"
CAV_PATH = PROJECT / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
OUT = PROJECT / "results" / "unit_concept_axis_cosine_audit_2026-09-03"
WINDOW = (150, 159)
SEED = 20260903
MAX_UNITS_PER_LAYER = 500
MAX_PAIRS = 20_000
MAX_PLOT_SAMPLES_PER_LAYER = 5_000


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def sample_pairs(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.empty(0, int), np.empty(0, int)
    total = n * (n - 1) // 2
    if total <= count:
        i, j = np.triu_indices(n, 1)
        return i, j
    i = rng.integers(0, n, count)
    j = rng.integers(0, n - 1, count)
    j += (j >= i)
    return i, j


def stats(values: np.ndarray) -> dict:
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    return {"n": int(v.size), "min": float(v.min()), "p05": float(np.quantile(v, .05)),
            "q1": float(np.quantile(v, .25)), "median": float(np.median(v)),
            "q3": float(np.quantile(v, .75)), "p95": float(np.quantile(v, .95)),
            "max": float(v.max()), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "positive_fraction": float((v > 0).mean()),
            "near_zero_fraction": float((np.abs(v) < .1).mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    basis_npz = np.load(BASIS, allow_pickle=True)
    norm = np.load(BANK / "feature_normalization_full.npz", allow_pickle=True)["std"].astype(float)
    cav = np.load(CAV_PATH, allow_pickle=True)
    concepts = cav["concepts"].astype(str)
    nodes = cav["nodes"].astype(str)
    offsets = cav["offsets"].astype(int)
    cav_by_layer = {}
    for li, node in enumerate(nodes):
        c = cav["cav_full"][:, offsets[li]:offsets[li + 1]].astype(float)
        cav_by_layer[li] = normalize(c)

    unit_vectors: dict[int, list[tuple[int, np.ndarray]]] = {li: [] for li in range(len(nodes))}
    unit_meta = []
    wi = None
    for path in sorted(BANK.glob("*_proxy_bank_64d.h5")):
        with h5py.File(path, "r") as h:
            windows = h["windows_ms"][:].astype(int)
            if wi is None:
                hit = np.flatnonzero(np.all(windows == WINDOW, axis=1))
                if len(hit) != 1:
                    raise RuntimeError(f"window {WINDOW} not found")
                wi = int(hit[0])
            units = h["unit_global"][:].astype(int)
            weights = h["channel_weights"][wi].astype(float).reshape(len(units), len(nodes), 64)
            gates = h["layer_gate"][wi].astype(float)
        for ui, unit in enumerate(units):
            li = int(np.argmax(gates[ui]))
            b = basis_npz[f"basis_{li}"][:, :64].astype(float)
            uv = b @ (weights[ui, li] / np.maximum(norm[li], 1e-12))
            if np.linalg.norm(uv) <= 1e-10:
                continue
            unit_vectors[li].append((int(unit), normalize(uv[None])[0]))
            unit_meta.append((int(unit), li, path.stem))

    rows = []
    summaries = []
    for li, node in enumerate(nodes):
        entries = unit_vectors[li]
        if not entries:
            continue
        take = rng.choice(len(entries), min(MAX_UNITS_PER_LAYER, len(entries)), replace=False)
        uv = np.stack([entries[int(i)][1] for i in take])
        ui, uj = sample_pairs(len(uv), MAX_PAIRS, rng)
        unit_cos = (uv[ui] * uv[uj]).sum(1)
        cv = cav_by_layer[li]
        ci, cj = sample_pairs(len(cv), MAX_PAIRS, rng)
        concept_cos = (cv[ci] * cv[cj]).sum(1)
        uc_i = rng.integers(0, len(uv), min(MAX_PAIRS, len(uv) * len(cv)))
        uc_j = rng.integers(0, len(cv), len(uc_i))
        unit_concept_cos = (uv[uc_i] * cv[uc_j]).sum(1)
        for relation, vals in (("unit_unit", unit_cos), ("concept_concept", concept_cos), ("unit_concept", unit_concept_cos)):
            # Keep the full sampled values for summary statistics, but only a
            # bounded subset for the distribution figure and CSV.
            keep = np.arange(len(vals)) if len(vals) <= MAX_PLOT_SAMPLES_PER_LAYER else rng.choice(len(vals), MAX_PLOT_SAMPLES_PER_LAYER, replace=False)
            for v in vals[keep]:
                rows.append({"layer_index": li, "layer": str(node), "relation": relation, "cosine": float(v),
                             "angle_degrees": float(np.degrees(np.arccos(np.clip(v, -1, 1))))})
            s = stats(vals); s.update({"layer_index": li, "layer": str(node), "relation": relation,
                                       "n_units_available": len(entries), "n_units_sampled": len(uv)})
            summaries.append(s)

    samples = pd.DataFrame(rows)
    samples.to_csv(OUT / "cosine_samples.csv.gz", index=False, compression="gzip")
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary_by_layer.csv", index=False, encoding="utf-8-sig")
    pooled_rows = []
    for relation, q in samples.groupby("relation", sort=False):
        s = stats(q.cosine.to_numpy()); s.update({"layer": "ALL_SAME_LAYER_POOLED", "relation": relation})
        pooled_rows.append(s)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "summary_pooled.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, relation in zip(axes, ("unit_unit", "concept_concept", "unit_concept")):
        q = samples[samples.relation.eq(relation)]
        ax.hist(q.cosine, bins=60, range=(-1, 1), color="#4c8dbc", alpha=.85)
        s = pooled[pooled.relation.eq(relation)].iloc[0]
        ax.axvline(float(s["median"]), color="#d73027", lw=1.5)
        ax.set(title=relation.replace("_", "–"), xlabel="Signed cosine", ylabel="Sample count")
        ax.text(.04, .96, f"median={s['median']:.3f}\n5–95%={s['p05']:.3f}…{s['p95']:.3f}", transform=ax.transAxes, va="top", fontsize=8)
    fig.savefig(OUT / "cosine_distributions.png", dpi=280); plt.close(fig)
    audit = {"window": list(WINDOW), "window_index": wi, "unit_axis": "final channel_weights at 150–159 ms; layer=argmax layer_gate; PCA64 weights mapped to native channels and normalized",
             "concept_axis": "500 full Broden CAV axes in native channel space; normalized per layer", "cross_layer_comparison": False,
             "n_unit_axes": int(sum(len(v) for v in unit_vectors.values())), "layers": list(nodes),
             "sampling": {"max_units_per_layer": MAX_UNITS_PER_LAYER, "max_pairs": MAX_PAIRS, "seed": SEED},
             "selection_rule": "official released_independent_consistency >= 0.4 only"}
    (OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False)); print(summary.to_string(index=False)); print(OUT)


if __name__ == "__main__":
    main()
