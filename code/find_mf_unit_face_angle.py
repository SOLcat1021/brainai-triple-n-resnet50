"""Find MF units whose best-layer proxy direction is close to the face CAV."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
PROXY = BANK / "middle_it_proxy_bank_64d.h5"
META = BANK / "selected_units.csv"
CAV = PROJECT / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
BASIS = PROJECT / "results" / "rank_sweep_extended_2026-08-26" / "response_blind_pca256_basis.npz"
NORM = BANK / "feature_normalization_full.npz"
OUT = PROJECT / "results" / "mf_face_angle_search_2026-08-30"
WINDOW = (150, 159)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META).set_index("unit_global")
    bank = np.load(CAV, allow_pickle=True)
    concepts = bank["concepts"].astype(str)
    ci = int(np.flatnonzero(concepts == "face")[0])
    nodes = bank["nodes"].astype(str)
    offsets = bank["offsets"].astype(int)
    pca_bank = np.load(BASIS, allow_pickle=True)
    norm_bank = np.load(NORM, allow_pickle=True)
    rows = []
    with h5py.File(PROXY, "r") as h:
        units_all = h["unit_global"][:].astype(int)
        wi = int(np.flatnonzero(np.all(h["windows_ms"][:] == WINDOW, axis=1))[0])
        gates = h["layer_gate"][wi].astype(np.float64)
        weights = h["channel_weights"][wi].astype(np.float64).reshape(len(units_all), 17, 64)
    for ui, unit in enumerate(units_all):
        if str(meta.loc[int(unit), "native_area"]) != "MF":
            continue
        li = int(np.argmax(gates[ui]))
        node = nodes[li]
        start, stop = offsets[li], offsets[li + 1]
        p = pca_bank[f"basis_{li}"][:, :64].astype(np.float64)
        std = norm_bank["std"][li].astype(np.float64)
        unit_native = p @ (weights[ui, li] / np.maximum(std, 1e-12))
        cav = bank["cav_full"][ci, start:stop].astype(np.float64)
        c = float(np.dot(unit_native, cav) / (np.linalg.norm(unit_native) * np.linalg.norm(cav)))
        repeats = bank["cav_repeated"][ci, :, start:stop].astype(np.float64)
        rc = np.array([np.dot(unit_native, x) / (np.linalg.norm(unit_native) * np.linalg.norm(x)) for x in repeats])
        rows.append({"unit_global": int(unit), "native_area": "MF", "window_start_ms": WINDOW[0], "window_end_ms": WINDOW[1], "best_layer": node, "best_layer_gate": float(gates[ui, li]), "face_cosine_full": c, "face_angle_full_degrees": float(np.degrees(np.arccos(np.clip(c, -1, 1)))), "face_cosine_repeat_mean": float(rc.mean()), "face_cosine_repeat_sd": float(rc.std(ddof=1)), "face_angle_repeat_mean_degrees": float(np.degrees(np.arccos(np.clip(rc.mean(), -1, 1))))})
    out = pd.DataFrame(rows).sort_values("face_angle_full_degrees")
    out.to_csv(OUT / "mf_units_face_angles.csv", index=False, encoding="utf-8-sig")
    close = out[out.face_angle_full_degrees <= 10].copy()
    close.to_csv(OUT / "mf_units_face_angle_within10deg.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nWithin 10 degrees: {len(close)}")
    if close.empty:
        print("No MF unit meets the 10-degree criterion; closest unit:")
        print(out.head(1).to_string(index=False))

if __name__ == "__main__":
    main()
