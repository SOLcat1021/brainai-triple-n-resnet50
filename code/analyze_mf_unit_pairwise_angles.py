"""Pairwise proxy-direction angles for MF units at a fixed ResNet layer."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd

PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
PROXY = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "middle_it_proxy_bank_64d.h5"
META = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "selected_units.csv"
BASIS = PROJECT / "results" / "rank_sweep_extended_2026-08-26" / "response_blind_pca256_basis.npz"
FULL_NORM = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "feature_normalization_full.npz"
OUT = PROJECT / "results" / "mf_unit_pairwise_angles_2026-08-30"
TARGET = 3447
WINDOW = (150, 159)
LAYER_INDEX = 10  # res4_b3, fixed to target unit's best layer

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROXY, "r") as h:
        units = h["unit_global"][:].astype(int)
        wi = int(np.flatnonzero(np.all(h["windows_ms"][:] == WINDOW, axis=1))[0])
        weights = h["channel_weights"][wi].astype(np.float64).reshape(len(units), 17, 64)[:, LAYER_INDEX]
    metadata = pd.read_csv(META).set_index("unit_global")
    mf_units = np.asarray([u for u in units if str(metadata.loc[int(u), "native_area"]) == "MF"], dtype=int)
    keep = np.asarray([int(u) in set(mf_units) for u in units])
    units = units[keep]
    weights = weights[keep]
    p = np.load(BASIS, allow_pickle=True)[f"basis_{LAYER_INDEX}"][:, :64].astype(np.float64)
    std = np.load(FULL_NORM, allow_pickle=True)["std"][LAYER_INDEX].astype(np.float64)
    native = (p @ (weights / np.maximum(std[None], 1e-12)).T).T
    native /= np.linalg.norm(native, axis=1, keepdims=True).clip(1e-12)
    sim = native @ native.T
    ti = int(np.flatnonzero(units == TARGET)[0])
    rows = []
    for i, u in enumerate(units):
        if i == ti: continue
        c = float(sim[ti, i])
        rows.append({"target_unit": TARGET, "other_unit": int(u), "layer": "res4_b3", "cosine": c, "angle_degrees": float(np.degrees(np.arccos(np.clip(c, -1, 1))))})
    table = pd.DataFrame(rows).sort_values("cosine", ascending=False)
    table.to_csv(OUT / "mf_unit3447_pairwise_angles.csv", index=False, encoding="utf-8-sig")
    matrix = pd.DataFrame(sim, index=units, columns=units)
    matrix.to_csv(OUT / "mf_all_unit_pairwise_cosine_res4_b3.csv", encoding="utf-8-sig")
    nearest = table.iloc[0].to_dict(); farthest = table.iloc[-1].to_dict()
    summary = {"target_unit": TARGET, "n_mf_units": int(len(units)), "window_ms": list(WINDOW), "fixed_layer": "res4_b3", "nearest_unit": nearest, "farthest_unit": farthest, "median_cosine_to_target": float(table.cosine.median()), "median_angle_to_target_degrees": float(table.angle_degrees.median()), "all_pair_median_cosine": float(sim[np.triu_indices(len(units), 1)].mean())}
    (OUT / "mf_unit_pairwise_angles_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nNearest:")
    print(table.head(5).to_string(index=False))
    print("\nFarthest:")
    print(table.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
