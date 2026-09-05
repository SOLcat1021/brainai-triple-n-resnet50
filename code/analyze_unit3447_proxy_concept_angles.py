"""Find the Broden concept axes most aligned/anti-aligned with MF unit 3447.

The comparison is made in one shared 64-D native-channel space: the unit's
final signed proxy readout at its representative 150-159 ms window versus
the CAVs at the proxy's best (maximum layer-gate) ResNet layer.
"""

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
PROXY = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "middle_it_proxy_bank_64d.h5"
META = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "selected_units.csv"
CAV = PROJECT / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
QUALITY = PROJECT / "tcav_broden500" / "broden500_cav_quality_by_layer.csv"
BASIS = PROJECT / "results" / "rank_sweep_extended_2026-08-26" / "response_blind_pca256_basis.npz"
FULL_NORM = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26" / "feature_normalization_full.npz"
OUT = PROJECT / "results" / "unit3447_proxy_concept_angles_2026-08-30"
UNIT = 3447
REP_START, REP_END = 150, 159


def cosine_rows(axis: np.ndarray, vector: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    vector = np.asarray(vector, np.float64)
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True).clip(1e-12)
    vector = vector / np.linalg.norm(vector).clip(1e-12)
    return axis @ vector


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(META)
    meta = metadata.loc[metadata.unit_global.eq(UNIT)].iloc[0]
    with h5py.File(PROXY, "r") as h:
        units = h["unit_global"][:].astype(int)
        ui = int(np.flatnonzero(units == UNIT)[0])
        windows = h["windows_ms"][:].astype(int)
        wi = int(np.flatnonzero((windows[:, 0] == REP_START) & (windows[:, 1] == REP_END))[0])
        gates = h["layer_gate"][wi, ui].astype(np.float64)
        weights = h["channel_weights"][wi, ui].astype(np.float64)

    bank = np.load(CAV, allow_pickle=True)
    concepts = bank["concepts"].astype(str)
    nodes = bank["nodes"].astype(str)
    offsets = bank["offsets"].astype(int)
    best_layer_index = int(np.argmax(gates))
    node = nodes[best_layer_index]
    start, stop = offsets[best_layer_index], offsets[best_layer_index + 1]
    pca = np.load(BASIS, allow_pickle=True)[f"basis_{best_layer_index}"][:, :64].astype(np.float64)
    norm = np.load(FULL_NORM, allow_pickle=True)["std"][best_layer_index].astype(np.float64)
    if pca.shape[0] != stop - start or len(norm) != 64:
        raise RuntimeError(f"PCA/native dimension mismatch at {node}: {pca.shape}, native={stop-start}, std={len(norm)}")
    # The proxy stores weights in standardized PCA64 coordinates.  Set all
    # discarded PCA components to zero, then map the retained vector back to
    # native channels so it can be compared with the native-channel CAV.
    pca_weight = weights.reshape(17, 64)[best_layer_index] / np.maximum(norm, 1e-12)
    unit_vec = pca @ pca_weight

    cav_full = bank["cav_full"][:, start:stop].astype(np.float64)
    cav_repeated = bank["cav_repeated"][:, :, start:stop].astype(np.float64)
    cosine = cosine_rows(cav_full, unit_vec)
    repeat_cosine = np.stack([cosine_rows(cav_repeated[:, r], unit_vec) for r in range(cav_repeated.shape[1])], axis=1)
    quality = pd.read_csv(QUALITY)
    q = quality[(quality.concept.isin(concepts)) & quality.node.eq(node)].copy()
    q = q.drop_duplicates("concept").set_index("concept")

    rows = []
    for ci, concept in enumerate(concepts):
        c = repeat_cosine[ci]
        rows.append({
            "unit_global": UNIT,
            "representative_window_start_ms": REP_START,
            "representative_window_end_ms": REP_END,
            "proxy_window_index": wi,
            "best_layer_index": best_layer_index,
            "best_layer_node": node,
            "concept": concept,
            "cosine_alignment_full_cav": float(cosine[ci]),
            "angle_degrees_full_cav": float(np.degrees(np.arccos(np.clip(cosine[ci], -1, 1)))),
            "cosine_alignment_repeat_mean": float(c.mean()),
            "cosine_alignment_repeat_sd": float(c.std(ddof=1)),
            "angle_degrees_repeat_mean": float(np.degrees(np.arccos(np.clip(c.mean(), -1, 1)))),
            "cav_heldout_accuracy": float(q.loc[concept, "heldout_accuracy_mean"]) if concept in q.index else np.nan,
            "unit_vector_norm_native_after_pca64_zeroing": float(np.linalg.norm(unit_vec)),
            "best_layer_gate": float(gates[best_layer_index]),
        })
    table = pd.DataFrame(rows).sort_values("cosine_alignment_full_cav", ascending=False)
    table.to_csv(OUT / "unit3447_all500_concept_angles.csv", index=False, encoding="utf-8-sig")
    top = pd.concat([table.head(1).assign(extreme="minimum_angle"), table.tail(1).assign(extreme="maximum_angle")])
    top.to_csv(OUT / "unit3447_extreme_concept_axes.csv", index=False, encoding="utf-8-sig")
    summary = {
        "unit_global": UNIT,
        "native_area": str(meta.native_area),
        "proxy_roi": str(meta.roi),
        "representative_window_ms": [REP_START, REP_END],
        "best_layer_node": node,
        "best_layer_index": best_layer_index,
        "best_layer_gate": float(gates[best_layer_index]),
        "unit_vector_definition": "proxy PCA64 weights standardized by full-data std, mapped through native PCA basis; discarded PCA components set to zero",
        "max_alignment_concept": str(table.iloc[0].concept),
        "max_alignment_cosine": float(table.iloc[0].cosine_alignment_full_cav),
        "max_alignment_angle_degrees": float(table.iloc[0].angle_degrees_full_cav),
        "min_alignment_concept": str(table.iloc[-1].concept),
        "min_alignment_cosine": float(table.iloc[-1].cosine_alignment_full_cav),
        "min_alignment_angle_degrees": float(table.iloc[-1].angle_degrees_full_cav),
        "interpretation": "signed CAV orientation is treated as concept-positive; maximum angle is the most anti-aligned concept axis",
    }
    (OUT / "unit3447_extreme_concept_axes.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nExtreme axes:")
    print(top[["extreme", "concept", "best_layer_node", "cosine_alignment_full_cav", "angle_degrees_full_cav", "cosine_alignment_repeat_mean", "cosine_alignment_repeat_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
