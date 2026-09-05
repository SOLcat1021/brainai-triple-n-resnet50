"""Cross-fitted univariate concept alignment for the hierarchical fwRF pilot.

Each concept is evaluated separately. For every held-out TriN image, its concept
score is computed with the fwRF fitted without that image, then correlated with
the actual neural response. No multiconcept regression or coefficient ranking is
used, and no result files are written.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SOURCE = Path(__file__).with_name("pilot_hierarchical_concept155_fwrf.py")


def load_source():
    spec = importlib.util.spec_from_file_location("hierarchical_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def strictly_stable_assignments(source) -> pd.DataFrame:
    """Apply the frozen main-analysis gate before testing neural associations."""
    quality = pd.read_csv(
        source.AXIS_DIR / "pca32_concept_axis_quality_1000.csv"
    )
    enough_examples = quality.groupby("concept_index").positive_count.first() > 1000
    stable_all_layers = (
        quality.groupby("concept_index").repeat_axis_cosine_mean.min() > 0.9
    )
    eligible = enough_examples.index[enough_examples & stable_all_layers]
    assignments = source.concept_assignments()
    result = assignments[
        assignments.concept_index.isin(eligible) & (assignments.accuracy >= 0.75)
    ].reset_index(drop=True)
    if len(result) != 71:
        raise RuntimeError(f"Expected 71 strictly stable concepts, found {len(result)}")
    return result


def column_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
    return np.divide(
        x.T @ y,
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=denominator > 1e-12,
    )


def bonferroni_r_threshold(n_images: int, n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided Pearson-r threshold controlling FWER across all tested profiles."""
    critical_t = student_t.ppf(1.0 - alpha / (2.0 * n_tests), df=n_images - 2)
    return float(critical_t / np.sqrt(critical_t * critical_t + n_images - 2))


def main() -> None:
    source = load_source()
    selected = source.select_units()
    assignments = strictly_stable_assignments(source)
    parameters, targets = source.load_fields(selected)
    device = source.torch.device("cuda" if source.torch.cuda.is_available() else "cpu")
    print(
        f"device={device}; units={len(selected)}; targets={len(targets)}; "
        f"concepts={len(assignments)}",
        flush=True,
    )
    fields = source.continuous_gaussian(parameters, 14, device)
    scores = source.extract_scores(assignments, fields, device)

    response_bank = np.load(source.RESPONSES, mmap_mode="r")
    metadata = pd.read_csv(source.METADATA)
    response_lookup = {int(unit): i for i, unit in enumerate(metadata.unit_global)}
    outer_folds = source.fixed_folds(source.N_IMAGES)
    records = []
    for target_index, (selected_index, window) in enumerate(targets):
        unit = selected.loc[selected_index]
        response_index = (window + 20) // 10
        y = np.asarray(
            response_bank[response_index, :, response_lookup[int(unit.unit_global)]],
            np.float32,
        )
        crossfit_scores = np.empty((source.N_IMAGES, len(assignments)), np.float32)
        fold_r = np.empty((source.N_FOLDS, len(assignments)), np.float32)
        for fold, (_, test) in enumerate(outer_folds):
            field_index = target_index * (source.N_FOLDS + 1) + fold
            crossfit_scores[test] = np.asarray(scores[test, field_index], np.float32)
            fold_r[fold] = column_correlations(crossfit_scores[test], y[test])
        correlation = column_correlations(crossfit_scores, y)
        sign_fraction = np.maximum(
            np.mean(fold_r > 0, axis=0), np.mean(fold_r < 0, axis=0)
        )
        for concept_index, r in enumerate(correlation):
            concept = assignments.iloc[concept_index]
            records.append({
                "unit_global": int(unit.unit_global),
                "native_area": str(unit.native_area),
                "area_unit_number": int(unit.area_unit_number),
                "window": int(window),
                "concept": str(concept.concept),
                "category": str(concept.category),
                "layer": str(concept.layer),
                "r": float(r),
                "fold_sign_fraction": float(sign_fraction[concept_index]),
            })
        print(f"correlated {target_index + 1}/{len(targets)}", flush=True)

    result = pd.DataFrame(records)
    n_tests = len(targets) * len(assignments)
    r_threshold = bonferroni_r_threshold(source.N_IMAGES, n_tests)
    stable = result[
        (result.fold_sign_fraction >= 0.8) & (result.r.abs() >= r_threshold)
    ]
    positive_pool = stable[stable.r > 0]
    negative_pool = stable[stable.r < 0]
    positive = (
        positive_pool.sort_values(
            ["unit_global", "window", "r"], ascending=[True, True, False]
        )
        .groupby(["unit_global", "window"], sort=False).head(10)
    )
    negative = (
        negative_pool.sort_values(
            ["unit_global", "window", "r"], ascending=[True, True, True]
        )
        .groupby(["unit_global", "window"], sort=False).head(10)
    )

    print("\nAUDIT")
    print({
        "metric": "Pearson r across 1000 images: cross-fitted fwRF concept score vs actual response",
        "concepts_jointly_fitted": False,
        "concept_gate": (
            "positive_count > 1000 and repeat-axis mean cosine > 0.9 "
            "independently in every layer from res2 through res5, with assigned-layer "
            "held-out accuracy >= 0.75"
        ),
        "concept_score_sign_preserved": True,
        "fwrf_image_leakage": "prevented by five-fold image cross-fitting",
        "windows_ms": source.WINDOWS,
        "familywise_alpha": 0.05,
        "familywise_scope_tests": n_tests,
        "absolute_r_threshold": r_threshold,
        "required_image_fold_sign_fraction": 0.8,
        "passing_associations": int(len(stable)),
        "profiles_with_at_least_one_passing_concept": int(
            stable.groupby(["unit_global", "window"]).ngroups
        ),
        "files_written": False,
    })
    print("\nCANDIDATE_CONCEPTS_BY_LAYER")
    print(assignments.layer.value_counts().sort_index().to_string())
    print("\nTOP10_POSITIVE_BY_LAYER")
    print(positive.layer.value_counts().sort_index().to_string())
    print("\nTOP10_NEGATIVE_BY_LAYER")
    print(negative.layer.value_counts().sort_index().to_string())

    print("\nTOP3_PER_UNIT_WINDOW")
    for (area, number, unit, window), group in result.groupby(
        ["native_area", "area_unit_number", "unit_global", "window"], sort=True
    ):
        good = group[
            (group.fold_sign_fraction >= 0.8) & (group.r.abs() >= r_threshold)
        ]
        pos = good[good.r > 0].nlargest(3, "r")
        neg = good[good.r < 0].nsmallest(3, "r")
        fmt = lambda frame: ", ".join(
            f"{row.concept}[{row.layer}] {row.r:+.3f} ({row.fold_sign_fraction:.0%})"
            for row in frame.itertuples(index=False)
        )
        print(f"{area}-{int(number)} unit={int(unit)} {int(window)}-{int(window)+9}ms")
        print(f"  positive: {fmt(pos)}")
        print(f"  negative: {fmt(neg)}")

    print("\nCOMPACT_TOP5_POSITIVE")

    def unit_label(area: str, number: int) -> str:
        separator = "-" if any(character.isdigit() for character in area) else ""
        return f"{area}{separator}{number}"

    for (area, number, unit), group in result.groupby(
        ["native_area", "area_unit_number", "unit_global"], sort=True
    ):
        cells = []
        for window in source.WINDOWS:
            current = group[
                (group.window == window)
                & (group.r > 0)
                & (group.fold_sign_fraction >= 0.8)
                & (group.r >= r_threshold)
            ].nlargest(5, "r")
            values = ", ".join(
                f"{row.concept}[{row.layer}] {row.r:.3f}"
                for row in current.itertuples(index=False)
            ) or "--"
            cells.append(values)
        print(
            "TABLE|"
            + "|".join([
                unit_label(str(area), int(number)),
                str(int(unit)),
                *cells,
            ])
        )


if __name__ == "__main__":
    main()
