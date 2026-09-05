"""Strict within-monkey/session temporal comparisons for identifiable IT pairs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import pilot_low_rank_temporal_dynamics as pilot
from explore_functional_it_temporal_directions import reduce_block, permutation_test


POOL = pilot.PROJECT / "cohorts" / "high_quality_64d_pilot_2026-08-26" / "tier_A_strict_pool.csv"
OUT = pilot.PROJECT / "results" / "same_session_functional_pairs_2026-08-27"
N_PERM = 100_000


def build_features(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[:, :, units], np.float32).transpose(2, 1, 0)
    pre = windows[:, 1] < 0
    post = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    times = windows[post].mean(1)
    evoked = y.mean(1) - y[:, :, pre].mean((1, 2))[:, None]
    evoked /= np.maximum(np.linalg.norm(evoked, axis=1, keepdims=True), 1e-8)
    cov, desc = [], []
    tri = np.triu_indices(len(post))
    for yy in y[:, :, post]:
        c = np.corrcoef(yy, rowvar=False)
        cov.append(np.nan_to_num(c[tri], nan=0.0))
        centered = yy - yy.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        frac = s ** 2 / max(float(np.sum(s ** 2)), 1e-12)
        h1, h2 = vt[0], vt[1]
        centroid1 = float(times @ (h1 ** 2 / np.sum(h1 ** 2)))
        centroid2 = float(times @ (h2 ** 2 / np.sum(h2 ** 2)))
        deriv = abs(pilot.corr(h2, np.gradient(h1)))
        entropy_rank = float(np.exp(-np.sum(frac * np.log(np.maximum(frac, 1e-12)))))
        desc.append([centroid1, centroid2, deriv, entropy_rank, *frac[:5]])
    blocks = {
        "evoked_curve": reduce_block(evoked),
        "stimulus_temporal_covariance": reduce_block(np.asarray(cov)),
        "response_dynamic_descriptors": reduce_block(np.asarray(desc)),
    }
    blocks["combined_balanced"] = np.concatenate([v / np.sqrt(v.shape[1]) for v in blocks.values()], axis=1)
    return blocks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL)
    selections = []
    mfmb = pool[(pool.monkey.eq("M1")) & (pool.session.eq(24)) & pool.native_area.isin(["MF", "MB"])].copy()
    selections.append(("MF_vs_MB_M1_session24", mfmb))
    abaf = pool[pool.native_area.isin(["AB", "AF"])].copy()
    shared = abaf.groupby(["monkey", "session"]).native_area.nunique()
    shared = set(shared[shared.eq(2)].index.tolist())
    abaf = abaf[[tuple(x) in shared for x in zip(abaf.monkey, abaf.session)]]
    selections.append(("AB_vs_AF_all_shared_sessions", abaf))
    selections.append(("AB_vs_AF_M1_session60", abaf[(abaf.monkey.eq("M1")) & abaf.session.eq(60)].copy()))
    rows = []
    for name, q in selections:
        q = q.sort_values("unit_global").reset_index(drop=True)
        blocks = build_features(q)
        labels = q.native_area.to_numpy(str)
        session = (q.monkey.astype(str) + "|" + q.session.astype(str)).to_numpy()
        session_type = (q.monkey.astype(str) + "|" + q.session.astype(str) + "|" + q.unit_type_name.astype(str)).to_numpy()
        for view, x in blocks.items():
            for null_name, strata in (("within_session", session), ("within_session_unit_type", session_type)):
                result = permutation_test(x, labels, strata, 20260827 + len(rows))
                rows.append({"comparison": name, "feature_view": view, "null": null_name,
                             "n_units": len(q), "n_a": int(np.sum(labels == sorted(set(labels))[0])),
                             "n_b": int(np.sum(labels == sorted(set(labels))[1])), **result})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "same_session_pair_permutation.csv", index=False, encoding="utf-8-sig")
    lines = ["# 同猴同session功能区配对检验", "",
             "仅使用严格梯队，并只检验原始数据中实际共同出现的MF–MB和AB–AF。", "",
             "| 比较 | 特征 | 置换 | n | observed/null | p | 可交换比例 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for x in result.itertuples():
        lines.append(f"| {x.comparison} | {x.feature_view} | {x.null} | {x.n_units} | "
                     f"{x.observed_over_null:.2f} | {x.permutation_p:.5f} | {x.median_fraction_labels_changed:.1%} |")
    lines += ["", "统计单位仍是unit；同一session内unit可能相关，因此这些p值是探索性证据，而非独立动物层面的确认。"]
    (OUT / "SAME_SESSION_FUNCTIONAL_PAIR_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps({"strict_only": True, "n_permutations": N_PERM,
                                                  "allowed_pairs": ["MF-MB", "AB-AF"]}, indent=2), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
