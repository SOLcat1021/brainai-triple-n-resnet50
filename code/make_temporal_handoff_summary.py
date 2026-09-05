"""Create the compact handoff figure for the temporal-dynamics follow-up."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
OUT = ROOT / "results" / "temporal_dynamics_morning_handoff_2026-08-27"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(ROOT / "results/same_session_functional_pairs_2026-08-27/same_session_pair_permutation.csv")
    repl = pd.read_csv(ROOT / "results/temporal_dynamics_expansion_strict_2026-08-27/locked_replication_summary.csv")
    sem = pd.read_csv(ROOT / "results/cross_scene_temporal_semantics_2026-08-27/cross_scene_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    q = pairs[(pairs.feature_view.eq("combined_balanced")) & pairs["null"].eq("within_session_unit_type")]
    labels = ["MF–MB\nM1 S24", "AB–AF\nshared S", "AB–AF\nM1 S60"]
    axes[0].bar(np.arange(3), q.observed_over_null, color="#4C78A8")
    axes[0].axhline(1, color="#777", lw=1); axes[0].set_xticks(np.arange(3), labels)
    axes[0].set(title="Strict same-session comparisons", ylabel="Observed / shuffled dispersion")

    x = np.arange(3); width = .36
    axes[1].bar(x-width/2, repl.median_dynamic_minus_derivative_rank2, width, label="dynamic − derivative")
    axes[1].bar(x+width/2, repl.median_nonwarp_delta_r2, width, label="orthogonal non-warp")
    axes[1].axhline(0, color="#777", lw=1); axes[1].set_xticks(x, repl.roi, rotation=25)
    axes[1].set(title="Locked replication on new units", ylabel="Median held-out ΔR²"); axes[1].legend(fontsize=8)

    roi_order = ["V2", "V4", "posterior IT", "middle IT", "anterior IT"]
    x = np.arange(len(roi_order)); width = .36
    a = sem[sem["mode"].eq("nonwarp")].set_index("roi").reindex(roi_order)
    b = sem[sem["mode"].eq("time_phase_residual")].set_index("roi").reindex(roi_order)
    axes[2].bar(x-width/2, a.median_cross_scene_ridge_r, width, label="non-warp")
    axes[2].bar(x+width/2, b.median_cross_scene_ridge_r, width, label="time phase")
    axes[2].axhline(0, color="#777", lw=1); axes[2].set_xticks(x, roi_order, rotation=28)
    axes[2].set(title="Cross-scene semantic prediction", ylabel="Median held-out r"); axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "01_temporal_dynamics_handoff_summary.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
