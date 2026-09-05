"""AF1 correlation with three face-part concepts at every ResNet stage.

Uses the same 1000 TriN images, cross-fitted fwRFs, PCA32 bases, CAV bank, and
actual neural responses as the univariate concept audit. Prints only; writes no
result files.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).with_name("pilot_hierarchical_concept155_fwrf.py")
CONCEPTS = ("mouth", "eyebrow", "nose")


def load_source():
    spec = importlib.util.spec_from_file_location("hierarchical_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64) - np.mean(x)
    y = np.asarray(y, np.float64) - np.mean(y)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(x @ y / denominator) if denominator > 1e-12 else np.nan


def main() -> None:
    source = load_source()
    selected = source.select_units()
    af1 = selected[
        (selected.native_area == "AF") & (selected.area_unit_number == 1)
    ].reset_index(drop=True)
    if len(af1) != 1 or int(af1.iloc[0].unit_global) != 22610:
        raise RuntimeError("Frozen AF1 selection no longer resolves to unit 22610")

    metadata = pd.read_csv(source.AXIS_DIR / "concepts_1000.csv").iloc[:500]
    chosen = metadata[metadata.name.isin(CONCEPTS)].copy()
    if set(chosen.name) != set(CONCEPTS):
        raise RuntimeError("One or more requested concepts are missing")

    rows = []
    for layer_index, layer in enumerate(source.LAYERS):
        for concept in CONCEPTS:
            item = chosen[chosen.name == concept].iloc[0]
            rows.append({
                "concept_index": int(item.concept_index),
                "concept": concept,
                "category": str(item.primary_category),
                "layer_index": layer_index,
                "layer": layer,
            })
    assignments = pd.DataFrame(rows)

    parameters, targets = source.load_fields(af1)
    device = source.torch.device("cuda" if source.torch.cuda.is_available() else "cpu")
    print(f"device={device}; AF1=22610; axes={len(assignments)}", flush=True)
    fields = source.continuous_gaussian(parameters, 14, device)
    scores = source.extract_scores(assignments, fields, device)

    responses = np.load(source.RESPONSES, mmap_mode="r")
    unit_metadata = pd.read_csv(source.METADATA)
    response_index = int(np.flatnonzero(unit_metadata.unit_global.to_numpy() == 22610)[0])
    folds = source.fixed_folds(source.N_IMAGES)
    quality = pd.read_csv(source.AXIS_DIR / "pca32_concept_axis_quality_1000.csv")

    print("RESULT|window|concept|layer|r_oof|r_fullfield|fold_sign|axis_stability|axis_accuracy")
    for target_index, (_, window) in enumerate(targets):
        time_index = (window + 20) // 10
        y = np.asarray(responses[time_index, :, response_index], np.float32)
        for concept_row, axis in assignments.iterrows():
            crossfit = np.empty(source.N_IMAGES, np.float32)
            fullfield = np.asarray(scores[:, target_index * (source.N_FOLDS + 1) + source.N_FOLDS, concept_row], np.float32)
            fold_r = []
            for fold, (_, test) in enumerate(folds):
                field_index = target_index * (source.N_FOLDS + 1) + fold
                crossfit[test] = np.asarray(scores[test, field_index, concept_row], np.float32)
                fold_r.append(correlation(crossfit[test], y[test]))
            sign_fraction = max(
                np.mean(np.asarray(fold_r) > 0), np.mean(np.asarray(fold_r) < 0)
            )
            node = source.NODES[int(axis.layer_index)]
            q = quality[
                (quality.concept_index == int(axis.concept_index)) & (quality.node == node)
            ].iloc[0]
            print(
                f"RESULT|{window}-{window + 9}|{axis.concept}|{axis.layer}|"
                f"{correlation(crossfit, y):+.6f}|{correlation(fullfield, y):+.6f}|{sign_fraction:.1f}|"
                f"{float(q.repeat_axis_cosine_mean):.6f}|"
                f"{float(q.heldout_accuracy_mean):.6f}"
            )


if __name__ == "__main__":
    main()
