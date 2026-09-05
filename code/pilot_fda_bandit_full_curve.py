"""Small offline bandit pilot: FDA attention, fixed full-curve evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
RESP = P / "proxy_bank_10ms_all_windows" / "trin_all_units_10ms_responses.npy"
WIN = P / "proxy_bank_10ms_all_windows" / "windows_10ms.npy"
MAPS = P / "results" / "rank_sweep_extended_2026-08-26" / "trin_resnet50_pca256_maps_f16.npy"
META = P / "proxy_bank_64d_consistency_ge_0p4_all_units_2026-08-30" / "selected_units.csv"
OUT = P / "results" / "pilot_fda_bandit_full_curve_2026-09-03"


def corr(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else np.nan


def curve_basis(y, k=6):
    z = y - y.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(z, full_matrices=False)
    return vh[:k].T


def fit_reward(xtr, ytr, xva, yva, basis, weights):
    ctrain = (ytr - ytr.mean(0, keepdims=True)) @ basis
    cva = (yva - ytr.mean(0, keepdims=True)) @ basis
    target = ctrain * weights[None]
    # Shared rank-4 readout: target weighting changes the shared subspace,
    # unlike independent scalar regressions.
    xtx = xtr.T @ xtr + 1e-2 * np.eye(xtr.shape[1])
    beta = np.linalg.solve(xtx, xtr.T @ target)
    u, s, vh = np.linalg.svd(beta, full_matrices=False)
    rank = min(4, len(s))
    beta = (u[:, :rank] * s[:rank]) @ vh[:rank]
    predc = (xva @ beta) / weights[None]
    pred = predc @ basis.T + ytr.mean(0, keepdims=True)
    return corr(yva, pred), pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resp = np.load(RESP, mmap_mode="r").astype(np.float32)
    windows = np.load(WIN).astype(float)
    tid = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    # unit x image x time; the final reward is evaluated on the full curve.
    yall = np.asarray(resp[tid], np.float32).transpose(2, 1, 0)
    maps = np.load(MAPS, mmap_mode="r")[:, :, :64].astype(np.float32).mean((-1, -2)).reshape(1000, -1)
    xall = (maps - maps.mean(0, keepdims=True)) / np.maximum(maps.std(0, keepdims=True), 1e-6)
    meta = pd.read_csv(META)
    units = meta.unit_global.drop_duplicates().head(8).to_numpy(int)
    actions = np.asarray([[1, 1, 1, 1, 1, 1], [1, .6, .6, .6, .6, .6],
                          [.6, 1, .6, .6, .6, .6], [.6, .6, 1, .6, .6, .6],
                          [.6, .6, .6, 1, .6, .6], [.6, .6, .6, .6, 1, .6],
                          [.6, .6, .6, .6, .6, 1], [1, .3, .3, 1, .3, .3]], float)
    rows = []
    rng = np.random.default_rng(20260903)
    for unit in units:
        y = yall[unit]
        order = rng.permutation(1000); tr, va, te = order[:600], order[600:800], order[800:]
        basis = curve_basis(y[tr])
        scores = []
        for ai, w in enumerate(actions):
            r, _ = fit_reward(xall[tr], y[tr], xall[va], y[va], basis, w)
            scores.append(r)
        chosen = int(np.nanargmax(scores))
        r_base, _ = fit_reward(xall[tr], y[tr], xall[te], y[te], basis, actions[0])
        r_bandit, _ = fit_reward(xall[tr], y[tr], xall[te], y[te], basis, actions[chosen])
        rows.append({"unit_global": int(unit), "roi": str(meta.loc[meta.unit_global.eq(unit), "roi"].iloc[0]),
                     "validation_best_action": chosen, "validation_best_r": scores[chosen],
                     "test_r_unweighted": r_base, "test_r_bandit": r_bandit,
                     "test_delta_r": r_bandit - r_base})
    d = pd.DataFrame(rows); d.to_csv(OUT / "unit_results.csv", index=False)
    summary = pd.DataFrame([{"n_units": len(d), "mean_test_r_unweighted": d.test_r_unweighted.mean(),
                             "mean_test_r_bandit": d.test_r_bandit.mean(), "mean_delta_r": d.test_delta_r.mean(),
                             "positive_delta_fraction": float((d.test_delta_r > 0).mean())}])
    summary.to_csv(OUT / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.8)); ax.scatter(d.test_r_unweighted, d.test_r_bandit, s=35, color="#377eb8")
    lo = min(d.test_r_unweighted.min(), d.test_r_bandit.min()); hi = max(d.test_r_unweighted.max(), d.test_r_bandit.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#d73027"); ax.set(xlabel="Test full-curve r: uniform FDA", ylabel="Test full-curve r: bandit FDA", title="FDA attention bandit pilot")
    fig.tight_layout(); fig.savefig(OUT / "bandit_vs_uniform_full_curve_r.png", dpi=250); plt.close(fig)
    (OUT / "AUDIT.json").write_text(json.dumps({"n_units": len(d), "actions": actions.tolist(),
        "supervision_target": "complete response curve", "selection": "validation full-curve Pearson r",
        "evaluation": "independent test full-curve Pearson r", "model": "rank-4 reduced-rank ridge",
        "note": "offline contextual-bandit upper-bound pilot; no test-set action selection"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(d.to_string(index=False)); print(OUT)


if __name__ == "__main__": main()
