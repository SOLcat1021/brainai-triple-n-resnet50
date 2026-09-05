"""Find the nearest Broden concept axis for each MF proxy unit."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

P = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
B = P / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
PROXY = B / "middle_it_proxy_bank_64d.h5"
META = B / "selected_units.csv"
CAV = P / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
BASIS = P / "results" / "rank_sweep_extended_2026-08-26" / "response_blind_pca256_basis.npz"
OUT = P / "results" / "mf_unit_nearest_concepts_2026-08-30"
WINDOW = (150, 159)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META).set_index("unit_global")
    bank = np.load(CAV, allow_pickle=True)
    concepts = bank["concepts"].astype(str)
    nodes = bank["nodes"].astype(str)
    offsets = bank["offsets"].astype(int)
    pca = np.load(BASIS, allow_pickle=True)
    stds = np.load(B / "feature_normalization_full.npz", allow_pickle=True)["std"]
    with h5py.File(PROXY, "r") as h:
        units_all = h["unit_global"][:].astype(int)
        wi = int(np.flatnonzero(np.all(h["windows_ms"][:] == WINDOW, axis=1))[0])
        gates = h["layer_gate"][wi].astype(float)
        weights = h["channel_weights"][wi].astype(float).reshape(len(units_all), 17, 64)
    rows = []
    for ui, unit in enumerate(units_all):
        if str(meta.loc[int(unit), "native_area"]) != "MF":
            continue
        li = int(np.argmax(gates[ui])); node = nodes[li]
        start, stop = offsets[li], offsets[li + 1]
        basis = pca[f"basis_{li}"][:, :64].astype(float)
        uv = basis @ (weights[ui, li] / np.maximum(stds[li], 1e-12))
        uv /= np.linalg.norm(uv).clip(1e-12)
        cav = bank["cav_full"][:, start:stop].astype(float)
        cav /= np.linalg.norm(cav, axis=1, keepdims=True).clip(1e-12)
        cos = cav @ uv
        rep = bank["cav_repeated"][:, :, start:stop].astype(float)
        rep /= np.linalg.norm(rep, axis=2, keepdims=True).clip(1e-12)
        rep_cos = np.einsum("crd,d->cr", rep, uv)
        order = np.argsort(-cos)
        for rank, ci in enumerate(order[:10], 1):
            rows.append({"unit_global": int(unit), "rank": rank, "window_start_ms": WINDOW[0], "window_end_ms": WINDOW[1], "best_layer": node, "best_layer_gate": float(gates[ui, li]), "concept": concepts[ci], "cosine_full": float(cos[ci]), "angle_degrees_full": float(np.degrees(np.arccos(np.clip(cos[ci], -1, 1)))), "cosine_repeat_mean": float(rep_cos[ci].mean()), "cosine_repeat_sd": float(rep_cos[ci].std(ddof=1))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mf_units_top10_nearest_concepts.csv", index=False, encoding="utf-8-sig")
    top = out[out["rank"] == 1].copy()
    top.to_csv(OUT / "mf_units_nearest_concept_summary.csv", index=False, encoding="utf-8-sig")
    print(top.to_string(index=False))

if __name__ == "__main__":
    main()
