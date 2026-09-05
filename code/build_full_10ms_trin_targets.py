"""Build the uncensored Triple-N 10-ms target bank and unit metadata.

All released/mapped units are retained.  Windows span -20..189 ms in adjacent
10-ms bins, so later analyses can choose their own onset without rebuilding the
neural targets.  The response cache is reconstructed from the released 1-ms
``response_matrix_img`` arrays; the first 1,000 image columns are the NNN image
set used by the ResNet feature cache.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
RAW = ROOT / "data" / "TripleN" / "V1" / "Raw" / "H5FILES"
PROCESSED = ROOT / "data" / "TripleN" / "V1" / "Processed"
OTHERS = ROOT / "data" / "TripleN" / "V1" / "others"
POP_FILE = ROOT / "cache" / "trin" / "population_time20_all_unittypes" / "population_20ms_all_unittypes.npz"
OUT = PROJECT / "proxy_bank_10ms_all_windows"
RESP_FILE = OUT / "trin_all_units_10ms_responses.npy"
META_FILE = OUT / "unit_metadata.csv"

WINDOWS = np.asarray([(s, s + 9) for s in range(-20, 190, 10)], dtype=np.int16)
TIME_ZERO_INDEX = 49


def indexed_files(folder: Path, suffix: str) -> dict[int, Path]:
    ans: dict[int, Path] = {}
    for p in folder.glob(f"*{suffix}"):
        m = re.search(r"ses(\d+)_", p.name, flags=re.I)
        if m:
            ans[int(m.group(1))] = p
    return ans


def functional_prefix(label: object) -> str:
    s = str(label).upper().strip()
    if s.startswith("UNKNOWN"):
        return "UNKNOWN"
    m = re.match(r"([A-Z]+)", s)
    return m.group(1) if m else s


def match_site(session: int, position: float, native_area: str,
               boundaries: pd.DataFrame, xyz: pd.DataFrame) -> dict[str, object]:
    q = boundaries[(boundaries.SesIdx == session) &
                   (boundaries.y1.astype(float) <= position) &
                   (boundaries.y2.astype(float) >= position)].copy()
    if q.empty:
        return {}
    want = functional_prefix(native_area)
    prefix = q.AREALABEL.map(functional_prefix)
    exact = q[prefix.eq(want)]
    if not exact.empty:
        q = exact
    row = q.iloc[0]
    site = xyz[xyz.AreaIDX.astype(int).eq(int(row.RoiIndex))]
    out = {
        "boundary_area_label": str(row.AREALABEL),
        "boundary_category": str(row.Categoty),
        "boundary_anatomical_area": str(row.Area),
        "recording_site_index": int(row.RoiIndex),
        "position_range_start_um": float(row.y1),
        "position_range_end_um": float(row.y2),
    }
    if not site.empty:
        s = site.iloc[0]
        out.update({
            "site_subject": int(s.Subject),
            "x_r_mm": float(s.R),
            "y_a_mm": float(s.A),
            "z_s_mm": float(s.S),
            "site_coarse_label": str(s.Label),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pop = np.load(POP_FILE, allow_pickle=True)
    n_units = len(pop["session"])
    h5_files = indexed_files(RAW, ".h5")
    mat_files = indexed_files(PROCESSED, ".mat")
    sessions = sorted(np.unique(pop["session"]).astype(int).tolist())
    missing_h5 = sorted(set(sessions) - set(h5_files))
    missing_mat = sorted(set(sessions) - set(mat_files))
    if missing_h5 or missing_mat:
        raise FileNotFoundError({"missing_h5": missing_h5, "missing_processed": missing_mat})

    boundaries = pd.read_excel(OTHERS / "exclude_area.xls")
    xyz = pd.read_excel(OTHERS / "AreaXYZ.xlsx")
    responses = np.lib.format.open_memmap(
        RESP_FILE, mode="w+", dtype=np.float32,
        shape=(len(WINDOWS), 1000, n_units),
    )
    responses[:] = np.nan

    base = pd.DataFrame({
        "unit_global": np.arange(n_units, dtype=np.int32),
        "session": pop["session"].astype(int),
        "monkey": pop["monkey"].astype(str),
        "unit_index_zero_based": pop["unit_index_zero_based"].astype(int),
        "roi": pop["area"].astype(str),
        "native_area": pop["native_area"].astype(str),
        "released_independent_consistency": pop["reliability"].astype(float),
        "unit_type": pop["unit_type"].astype(int),
        "unit_type_name": pop["unit_type_name"].astype(str),
    })
    extra_rows: list[dict[str, object]] = []
    scalar_fields = (
        "B_SI", "F_SI", "O_SI", "best_r_time1", "best_r_time2", "pos",
        "snr", "snrmax", "reliability_basic", "reliability_best",
        "reliability_find_testset",
    )

    for si, session in enumerate(sessions, start=1):
        globals_ = np.flatnonzero(pop["session"].astype(int) == session)
        unit_idx = pop["unit_index_zero_based"][globals_].astype(int)
        order = np.argsort(unit_idx)
        globals_sorted = globals_[order]
        unit_sorted = unit_idx[order]

        with h5py.File(h5_files[session], "r") as handle:
            source = handle["response_matrix_img"]
            if source.shape[1] < 1000:
                raise RuntimeError(f"session {session}: fewer than 1000 images")
            for wi, (start, end) in enumerate(WINDOWS):
                a = TIME_ZERO_INDEX + int(start)
                b = TIME_ZERO_INDEX + int(end) + 1
                # Reading the short contiguous time slab first is much faster
                # than issuing thousands of small HDF5 fancy-index requests.
                slab = np.asarray(source[a:b, :1000, :], np.float32)
                responses[wi][:, globals_sorted] = slab[:, :, unit_sorted].mean(0)

        mat = loadmat(mat_files[session], squeeze_me=True, struct_as_record=False)
        for g, u in zip(globals_sorted.tolist(), unit_sorted.tolist()):
            row: dict[str, object] = {
                "unit_global": int(g),
                "source_h5": str(h5_files[session]),
                "source_processed": str(mat_files[session]),
            }
            for field in scalar_fields:
                value = mat.get(field)
                if value is not None and np.ndim(value) == 1 and u < len(value):
                    v = value[u]
                    row[field if field != "pos" else "electrode_position_um"] = (
                        float(v) if np.issubdtype(np.asarray(v).dtype, np.number) else str(v)
                    )
            row.update(match_site(
                session, float(row.get("electrode_position_um", np.nan)),
                str(pop["native_area"][g]), boundaries, xyz,
            ))
            extra_rows.append(row)
        responses.flush()
        print(f"rebuilt session {session} ({si}/{len(sessions)}), units={len(globals_)}", flush=True)

    extra = pd.DataFrame(extra_rows)
    meta = base.merge(extra, on="unit_global", how="left", validate="one_to_one")
    meta.to_csv(META_FILE, index=False, encoding="utf-8-sig")
    np.save(OUT / "windows_10ms.npy", WINDOWS)

    # Exact reconstruction audit against the earlier 20-ms cache.
    old_windows = pop["windows"].astype(int)
    check = []
    for start, end in old_windows:
        idx = np.flatnonzero((WINDOWS[:, 0] >= start) & (WINDOWS[:, 1] <= end))
        if len(idx) == 2:
            rebuilt = np.asarray(responses[idx], np.float32).mean(0)
            old_i = np.flatnonzero((old_windows[:, 0] == start) & (old_windows[:, 1] == end))[0]
            check.append(float(np.nanmax(np.abs(rebuilt - pop["responses"][old_i]))))
    audit = {
        "n_units": int(n_units),
        "n_images": 1000,
        "windows": WINDOWS.tolist(),
        "response_shape": list(responses.shape),
        "max_abs_reconstruction_error_vs_released_20ms": max(check) if check else None,
        "xyz_coverage_fraction": float(meta[["x_r_mm", "y_a_mm", "z_s_mm"]].notna().all(1).mean()),
        "all_units_retained": True,
        "roi_onset_filter": False,
    }
    (OUT / "target_build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
