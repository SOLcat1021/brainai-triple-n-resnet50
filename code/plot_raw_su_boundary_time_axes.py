"""Plot raw unit temporal metrics and fitted spatial trends without binning."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
BANK = PROJECT / "proxy_bank_10ms_all_windows"
OUT = PROJECT / "results" / "functional_boundary_gradients_2026-08-27"
SESSIONS = [24, 29, 30, 60, 61]
PAIRS = {24: ("MB", "MF"), 29: ("AF", "AB"), 30: ("AF", "AB"), 60: ("AF", "AB"), 61: ("AF", "AB")}
SESSION_TITLES = {
    24: "M1 s24 | site 8 | MB-MF",
    29: "M3 s29 | site 18 | AF-AB",
    30: "M3 s30 | site 18 | AF-AB",
    60: "M1 s60 | site 30 | AF-AB",
    61: "M1 s61 | site 30 | AF-AB",
}
METRICS = [
    ("evoked_centroid_ms", "Evoked-curve centroid", "Time (ms)"),
    ("mode1_centroid_ms", "Mode-1 temporal centroid", "Time (ms)"),
    ("effective_temporal_rank", "Effective temporal rank", "Rank"),
    ("nonwarp_residual_fraction", "Non-warp residual fraction", "Fraction"),
]
COLORS = {"MB": "#2B7A9B", "MF": "#C35D2E", "AF": "#2B7A9B", "AB": "#C35D2E"}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-12 else 0.0


def orthonormal_columns(columns: list[np.ndarray]) -> np.ndarray:
    x = np.column_stack(columns)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > max(float(s[0]) * 1e-7, 1e-10)]


def boundary_midpoint(q: pd.DataFrame, first: str, second: str) -> float:
    first_end = q.loc[q.native_area.eq(first), "position_range_end_um"].astype(float).median()
    second_start = q.loc[q.native_area.eq(second), "position_range_start_um"].astype(float).median()
    return float((first_end + second_start) / 2.0)


def compute_metrics(meta: pd.DataFrame) -> pd.DataFrame:
    responses = np.load(BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(BANK / "windows_10ms.npy").astype(int)
    post = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    pre = windows[:, 1] < 0
    times = windows[post].mean(1)
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[:, :, units], np.float32).transpose(2, 1, 0)
    rows = []
    for ui, yy_all in enumerate(y):
        baseline = float(yy_all[:, pre].mean())
        evoked = yy_all[:, post].mean(0) - baseline
        evoked_energy = evoked**2
        evoked_centroid = float(times @ evoked_energy / max(float(evoked_energy.sum()), 1e-12))
        yy = yy_all[:, post]
        centered = yy - yy.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        frac = s**2 / max(float(np.sum(s**2)), 1e-12)
        h1, h2 = vt[0], vt[1]
        h1_energy = h1**2 / max(float(np.sum(h1**2)), 1e-12)
        nuisance = orthonormal_columns([h1, np.gradient(h1), np.gradient(np.gradient(h1))])
        residual = centered - (centered @ nuisance) @ nuisance.T
        rows.append({
            "evoked_centroid_ms": evoked_centroid,
            "mode1_centroid_ms": float(times @ h1_energy),
            "mode2_derivative_abs_r": abs(corr(h2, np.gradient(h1))),
            "effective_temporal_rank": float(np.exp(-np.sum(frac * np.log(np.maximum(frac, 1e-12))))),
            "nonwarp_residual_fraction": float(np.sum(residual**2) / max(float(np.sum(centered**2)), 1e-12)),
        })
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def fit_line(x: np.ndarray, y: np.ndarray, side: np.ndarray, jump: bool) -> tuple[np.ndarray, np.ndarray]:
    # Robustness comes from displaying both fits on all raw observations; the
    # quadratic form matches the earlier cross-validated model comparison.
    X = np.column_stack([np.ones_like(x), x, x**2])
    if jump:
        X = np.column_stack([X, side])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    grid = np.linspace(float(x.min()), float(x.max()), 400)
    grid_side = (grid >= 0).astype(float)
    G = np.column_stack([np.ones_like(grid), grid, grid**2])
    if jump:
        G = np.column_stack([G, grid_side])
    return grid, G @ coef


def make_figure(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(METRICS), len(SESSIONS), figsize=(18, 12), sharex="col", constrained_layout=True)
    for col, session in enumerate(SESSIONS):
        q = data[data.session.eq(session)].copy()
        first, second = PAIRS[session]
        boundary = boundary_midpoint(q, first, second)
        q["relative_position_mm"] = (q.electrode_position_um.astype(float) - boundary) / 1000.0
        x = q.relative_position_mm.to_numpy(float)
        side = (x >= 0).astype(float)
        for row, (metric, title, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for area in (first, second):
                a = q[q.native_area.eq(area)]
                ax.scatter(a.relative_position_mm, a[metric], s=10, alpha=0.22,
                           color=COLORS[area], edgecolors="none", label=f"{area} units (n={len(a)})")
            y = q[metric].to_numpy(float)
            grid, continuous = fit_line(x, y, side, jump=False)
            _, jump = fit_line(x, y, side, jump=True)
            ax.plot(grid, continuous, color="#222222", lw=2.0, label="Continuous quadratic")
            left = grid < 0
            right = ~left
            ax.plot(grid[left], jump[left], color="#6D3D8C", lw=2.2, ls="--", label="Boundary-step fit")
            ax.plot(grid[right], jump[right], color="#6D3D8C", lw=2.2, ls="--")
            ax.axvline(0, color="#222222", lw=1.1, ls=":")
            ax.grid(axis="y", color="#D8D8D8", lw=0.65, alpha=0.75)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(SESSION_TITLES[session], fontsize=10, fontweight="bold")
                ax.legend(frameon=False, fontsize=7, loc="best")
            if col == 0:
                ax.set_ylabel(ylabel)
                ax.text(-0.31, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10, fontweight="bold")
            if row == len(METRICS) - 1:
                ax.set_xlabel("Unit position relative to boundary (mm)")
    fig.suptitle("Raw all-unit temporal metrics along cross-area penetration axes",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.002,
             "Each point is one released unit (SU, MU, or non-somatic); no filtering or spatial binning. Solid black: continuous quadratic fit. Dashed purple: fit allowing a boundary step.",
             ha="center", fontsize=9)
    fig.savefig(OUT / "04_raw_all_units_temporal_metrics_with_boundary_fits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = pd.read_csv(BANK / "unit_metadata.csv")
    keep = meta.session.isin(SESSIONS)
    keep &= np.asarray([
        int(session) in PAIRS and area in PAIRS[int(session)]
        for session, area in zip(meta.session, meta.native_area)
    ])
    meta = meta[keep].sort_values(["session", "electrode_position_um", "unit_global"]).reset_index(drop=True)
    data = compute_metrics(meta)
    data.to_csv(OUT / "raw_all_unit_temporal_metrics.csv", index=False, encoding="utf-8-sig")
    make_figure(data)
    print(data.groupby(["session", "native_area"]).size().to_string())
    print(OUT / "04_raw_all_units_temporal_metrics_with_boundary_fits.png")


if __name__ == "__main__":
    main()
