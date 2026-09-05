"""Fast query/load helpers for the complete Triple-N ResNet proxy bank."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
DB = BANK / "proxy_index.sqlite"


def query(sql_where: str = "1=1", parameters: Iterable[object] = (),
          limit: int | None = None) -> pd.DataFrame:
    """Query proxies joined to their complete unit metadata.

    Example: ``query("p.roi=? AND p.t=? AND p.r>?", ("V4", 70, .25))``.
    """
    statement = "SELECT p.*, u.* FROM proxies p JOIN units u USING(unit_global) WHERE " + sql_where
    if limit is not None:
        statement += f" LIMIT {int(limit)}"
    with sqlite3.connect(DB) as connection:
        frame = pd.read_sql_query(statement, connection, params=tuple(parameters))
    return frame.loc[:, ~frame.columns.duplicated(keep="first")]


def load_proxy(proxy_id: int, include_predictions: bool = False) -> dict[str, object]:
    """Load one complete unit-window proxy and its metadata by proxy_id."""
    row = query("p.proxy_id=?", (int(proxy_id),), limit=1)
    if row.empty:
        raise KeyError(proxy_id)
    record = row.iloc[0].to_dict()
    wi = int(record["window_index"])
    ui = int(record["roi_local_unit"])
    with h5py.File(record["weight_file"], "r") as h:
        record.update({
            "spatial_center_x": float(h["spatial_parameters"][wi, ui, 0]),
            "spatial_center_y": float(h["spatial_parameters"][wi, ui, 1]),
            "spatial_sigma": float(h["spatial_parameters"][wi, ui, 2]),
            "layer_gate": h["layer_gate"][wi, ui].astype(np.float32),
            "signed_channel_weights": h["channel_weights"][wi, ui].astype(np.float32),
            "intercept": float(h["intercept"][wi, ui]),
            "normalization_mean": h["normalization_mean"][:].astype(np.float32),
            "normalization_std": h["normalization_std"][:].astype(np.float32),
        })
        if include_predictions:
            record["oof_prediction"] = h["oof_prediction"][wi, :, ui].astype(np.float32)
            record["fitted_prediction"] = h["fitted_prediction"][wi, :, ui].astype(np.float32)
    return record


def load_unit_trajectory(unit_global: int, include_predictions: bool = False) -> list[dict[str, object]]:
    """Load all 21 independent time-window proxies for one unit."""
    rows = query("p.unit_global=?", (int(unit_global),)).sort_values("t")
    return [load_proxy(int(proxy_id), include_predictions) for proxy_id in rows.proxy_id]
