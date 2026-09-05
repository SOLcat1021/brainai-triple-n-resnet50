"""Select response-stable Triple-N units before any semantic analysis.

Selection uses only released neural consistency and the archived PCA8 proxy's
cross-validated image prediction across the predeclared reliable time range.
No TCAV, concept label, semantic trajectory, or PCA64 result is consulted.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
OUT = PROJECT / "cohorts" / "high_quality_64d_pilot_2026-08-26"
START_MS = {
    "V1": 50, "V2": 60, "V4": 70,
    "posterior IT": 80, "middle IT": 90, "anterior IT": 100,
}
ROI_ORDER = list(START_MS)
FUNCTIONAL = ["MF", "MB", "MO", "CLC", "LPP", "AB", "AF", "AO"]
RANK_COLUMNS = ["min_window_r", "mean_window_r", "released_independent_consistency"]


def build_quality_table() -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT / "results" / "all_unit_manifest.csv").set_index("unit_global")
    rows = []
    for path in sorted((PROJECT / "proxy_bank_10ms_all_windows").glob("*_proxy_bank.h5")):
        with h5py.File(path, "r") as h:
            units = h["unit_global"][:].astype(int)
            windows = h["windows_ms"][:]
            roi = str(manifest.loc[units[0], "roi"])
            keep = (windows[:, 0] >= START_MS[roi]) & (windows[:, 1] <= 169)
            r = h["oof_r"][:][keep]
        frame = manifest.loc[units].copy().reset_index()
        frame["analysis_start_ms"] = START_MS[roi]
        frame["analysis_end_ms"] = 169
        frame["n_eligible_windows"] = int(keep.sum())
        frame["min_window_r"] = np.nanmin(r, axis=0)
        frame["mean_window_r"] = np.nanmean(r, axis=0)
        frame["median_window_r"] = np.nanmedian(r, axis=0)
        frame["max_window_r"] = np.nanmax(r, axis=0)
        frame["fraction_windows_r_ge_020"] = np.nanmean(r >= 0.20, axis=0)
        frame["fraction_windows_r_ge_025"] = np.nanmean(r >= 0.25, axis=0)
        rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    consistent = table.released_independent_consistency >= 0.40
    table["tier_A_strict"] = consistent & (table.min_window_r >= 0.25) & (table.mean_window_r >= 0.35)
    table["tier_B_core"] = (
        consistent & (table.min_window_r >= 0.10) & (table.mean_window_r >= 0.30)
        & (table.fraction_windows_r_ge_020 >= 0.80)
    )
    table["tier_C_reserve"] = consistent & (table.mean_window_r >= 0.25) & (table.max_window_r >= 0.35)
    return table


def diverse_take(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Quality-ranked round-robin over monkey/session/unit type strata."""
    frame = frame.sort_values(RANK_COLUMNS, ascending=False).copy()
    groups = []
    for key, group in frame.groupby(["monkey", "session", "unit_type_name"], sort=False):
        groups.append((key, group.sort_values(RANK_COLUMNS, ascending=False).reset_index(drop=True)))
    groups.sort(key=lambda item: tuple(-float(item[1].iloc[0][c]) for c in RANK_COLUMNS))
    chosen, depth = [], 0
    while len(chosen) < n:
        added = False
        for _, group in groups:
            if depth < len(group):
                chosen.append(group.iloc[depth])
                added = True
                if len(chosen) == n:
                    break
        if not added:
            break
        depth += 1
    return pd.DataFrame(chosen).reset_index(drop=True)


def select_with_fallback(group: pd.DataFrame, n: int) -> pd.DataFrame:
    strict = diverse_take(group[group.tier_A_strict], n)
    strict["selection_tier"] = "A_strict"
    if len(strict) >= n:
        return strict
    used = set(strict.unit_global.astype(int))
    core = diverse_take(group[group.tier_B_core & ~group.unit_global.isin(used)], n - len(strict))
    core["selection_tier"] = "B_core_fallback"
    return pd.concat([strict, core], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    quality = build_quality_table()
    quality.to_csv(OUT / "all_unit_quality_audit.csv", index=False, encoding="utf-8-sig")
    quality[quality.tier_A_strict].to_csv(OUT / "tier_A_strict_pool.csv", index=False, encoding="utf-8-sig")

    coarse_small = pd.concat([
        select_with_fallback(quality[quality.roi.eq(roi)], 12).assign(analysis_group=roi)
        for roi in ROI_ORDER
    ], ignore_index=True)
    coarse_expand = pd.concat([
        select_with_fallback(quality[quality.roi.eq(roi)], 50).assign(analysis_group=roi)
        for roi in ROI_ORDER
    ], ignore_index=True)
    functional_small = pd.concat([
        select_with_fallback(quality[quality.native_area.eq(area)], 8).assign(analysis_group=area)
        for area in FUNCTIONAL
    ], ignore_index=True)
    functional_expand = pd.concat([
        select_with_fallback(quality[quality.native_area.eq(area)], 20).assign(analysis_group=area)
        for area in FUNCTIONAL
    ], ignore_index=True)

    coarse_small.to_csv(OUT / "pilot_coarse_roi_12_each.csv", index=False, encoding="utf-8-sig")
    coarse_expand.to_csv(OUT / "expansion_coarse_roi_50_each.csv", index=False, encoding="utf-8-sig")
    functional_small.to_csv(OUT / "pilot_functional_it_8_each.csv", index=False, encoding="utf-8-sig")
    functional_expand.to_csv(OUT / "expansion_functional_it_20_each.csv", index=False, encoding="utf-8-sig")

    counts = quality.groupby("roi")[["tier_A_strict", "tier_B_core", "tier_C_reserve"]].sum().astype(int)
    functional_counts = quality[quality.native_area.isin(FUNCTIONAL)].groupby("native_area")[[
        "tier_A_strict", "tier_B_core", "tier_C_reserve"]].sum().astype(int).reindex(FUNCTIONAL)
    audit = {
        "semantic_outcome_used_for_selection": False,
        "pca64_result_used_for_selection": False,
        "screening_proxy": "archived response-independent PCA8 ResNet50 proxy; five-fold OOF r",
        "released_consistency_min": 0.40,
        "reliable_time_range_ms": START_MS,
        "tier_A_strict": "minimum eligible-window r >= .25 and mean r >= .35",
        "tier_B_core": "minimum r >= .10, mean r >= .30, and >=80% windows r >= .20",
        "tier_C_reserve": "mean r >= .25 and best-window r >= .35",
        "ranking": RANK_COLUMNS,
        "diversity": "round-robin over monkey x session x unit type after quality ranking",
        "coarse_counts": counts.to_dict(orient="index"),
        "functional_counts": functional_counts.fillna(0).astype(int).to_dict(orient="index"),
    }
    (OUT / "selection_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Tier-A pool by ROI:\n", counts.to_string(), flush=True)
    print("\nSmall pilot unique units:", pd.concat([coarse_small, functional_small]).unit_global.nunique())
    print("Expansion unique units:", pd.concat([coarse_expand, functional_expand]).unit_global.nunique())
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
