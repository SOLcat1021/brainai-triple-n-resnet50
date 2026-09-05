"""Cross-scene semantic interpretation of response-defined temporal modes."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import pilot_low_rank_temporal_dynamics as pilot
import validate_low_rank_temporal_dynamics as validation
from explore_functional_it_temporal_directions import nonwarp_mode


BANK = pilot.PROJECT / "proxy_bank_64d_expansion_strict_2026-08-27"
ATLAS = pilot.PROJECT / "Authoritative_Semantic_SOM"
CAPTIONS = pilot.PROJECT / "SOM_语义地图_原方法复现" / "triple_n_1000_captions.jsonl"
OUT = pilot.PROJECT / "results" / "cross_scene_temporal_semantics_2026-08-27"
ROIS = ("V2", "V4", "posterior IT", "middle IT", "anterior IT")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 20260827 + 457


def semantic_features() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    image = np.load(ATLAS / "triple_n_1000_mpnet.npy").astype(np.float64)
    anchor = np.load(ATLAS / "concept_anchor_mpnet.npy").astype(np.float64)
    records = json.loads((ATLAS / "concept_anchors.json").read_text(encoding="utf-8"))
    captions = [json.loads(x) for x in CAPTIONS.read_text(encoding="utf-8").splitlines()]
    captions.sort(key=lambda x: int(x["image_index"]))
    families = list(dict.fromkeys(str(x["family"]) for x in records))
    family_vectors = []
    for family in families:
        idx = [i for i, row in enumerate(records) if row["family"] == family]
        v = anchor[idx].mean(0); v /= max(np.linalg.norm(v), 1e-12)
        family_vectors.append(v)
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    x = image @ np.stack(family_vectors).T
    scenes = KMeans(n_clusters=10, random_state=SEED, n_init=50).fit_predict(image)
    return x, families, scenes, [str(x["caption"]) for x in captions]


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
    den = np.sqrt(np.sum(a*a, 0) * np.sum(b*b, 0))
    return np.divide(np.sum(a*b, 0), den, out=np.full(a.shape[1], np.nan), where=den > 1e-12)


def scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(0), train.std(0)
    sd = np.maximum(sd, 1e-8)
    return (train-mu)/sd, (test-mu)/sd


def choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    nsplit = min(4, len(unique))
    if nsplit < 2:
        return 1.0
    scores = {a: [] for a in ALPHAS}
    for tr, va in GroupKFold(n_splits=nsplit).split(x, groups=groups):
        xtr, xva = scale(x[tr], x[va])
        for alpha in ALPHAS:
            pred = Ridge(alpha=alpha).fit(xtr, y[tr]).predict(xva)
            scores[alpha].append(float(np.nanmean(corr_columns(y[va], pred))))
    return max(ALPHAS, key=lambda a: np.nanmean(scores[a]))


def load_roi(roi: str, responses: np.memmap, tidx: np.ndarray):
    selected = pd.read_csv(BANK / "selected_units.csv")
    pilot_ids = set(pd.read_csv(pilot.BANK / "selected_units.csv").unit_global.astype(int))
    meta = selected[selected.roi.eq(roi) & ~selected.unit_global.astype(int).isin(pilot_ids)].copy().reset_index(drop=True)
    units = meta.unit_global.to_numpy(int)
    y = np.asarray(responses[tidx][:, :, units], np.float32).transpose(1, 0, 2)
    return meta, y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, families, scenes, captions = semantic_features()
    responses = np.load(pilot.OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(pilot.OLD_BANK / "windows_10ms.npy").astype(int)
    tidx = np.flatnonzero((windows[:, 0] >= 0) & (windows[:, 1] <= 189))
    configs = validation.modal_configs().set_index("roi")
    outer = list(GroupKFold(n_splits=5).split(x, groups=scenes))
    rows, preference_rows = [], []
    full_nonwarp = {}
    for roi in ROIS:
        meta, y = load_roi(roi, responses, tidx)
        lam = float(configs.loc[roi, "lambda"])
        for fold, (train, test) in enumerate(outer):
            targets = {name: [] for name in ("main_amplitude", "nonwarp", "time_phase", "time_phase_residual")}
            train_targets = {name: [] for name in targets}
            for ui, unit in meta.iterrows():
                mean = y[train, :, ui].mean(0)
                nuisance, g = nonwarp_mode(y[train, :, ui] - mean[None], lam)
                h1 = pilot.svd_basis(y[train, :, ui] - mean[None], lam, 1)[:, 0]
                derivative = np.gradient(h1)
                derivative -= h1 * float(h1 @ derivative)
                derivative /= max(np.linalg.norm(derivative), 1e-12)
                train_main = (y[train, :, ui] - mean[None]) @ h1
                test_main = (y[test, :, ui] - mean[None]) @ h1
                train_nonwarp = (y[train, :, ui] - mean[None]) @ g
                test_nonwarp = (y[test, :, ui] - mean[None]) @ g
                train_phase = (y[train, :, ui] - mean[None]) @ derivative
                test_phase = (y[test, :, ui] - mean[None]) @ derivative
                slope = float(train_main @ train_phase / max(train_main @ train_main, 1e-12))
                for name, a, b in (
                    ("main_amplitude", train_main, test_main),
                    ("nonwarp", train_nonwarp, test_nonwarp),
                    ("time_phase", train_phase, test_phase),
                    ("time_phase_residual", train_phase - slope * train_main, test_phase - slope * test_main),
                ):
                    train_targets[name].append(a)
                    targets[name].append(b)
            for name in targets:
                ytr = np.column_stack(train_targets[name]); yte = np.column_stack(targets[name])
                alpha = choose_alpha(x[train], ytr, scenes[train])
                xtr, xte = scale(x[train], x[test])
                pred = Ridge(alpha=alpha).fit(xtr, ytr).predict(xte)
                ridge_r = corr_columns(yte, pred)
                train_corr = np.column_stack([corr_columns(ytr, x[train, j:j+1].repeat(ytr.shape[1], axis=1))
                                              for j in range(x.shape[1])])
                best = np.nanargmax(np.abs(train_corr), axis=1)
                for ui, unit in meta.iterrows():
                    test_r = pilot.corr(yte[:, ui], x[test, best[ui]])
                    signed_test_r = float(np.sign(train_corr[ui, best[ui]]) * test_r)
                    rows.append({
                        "roi": roi, "unit_global": int(unit.unit_global), "native_area": str(unit.native_area),
                        "monkey": str(unit.monkey), "session": int(unit.session), "fold": fold,
                        "mode": name, "ridge_alpha": alpha, "cross_scene_ridge_r": float(ridge_r[ui]),
                        "selected_family": families[best[ui]],
                        "train_selected_family_abs_r": float(abs(train_corr[ui, best[ui]])),
                        "cross_scene_selected_family_signed_r": signed_test_r,
                    })

        # Response-blind image examples are selected only after the full-data
        # temporal mode is frozen; captions are reported for interpretation.
        for ui, unit in meta.iterrows():
            mean = y[:, :, ui].mean(0)
            _, g = nonwarp_mode(y[:, :, ui] - mean[None], lam)
            coefficient = (y[:, :, ui] - mean[None]) @ g
            full_nonwarp[(roi, int(unit.unit_global))] = coefficient

    result = pd.DataFrame(rows)
    unit = result.groupby(["roi", "unit_global", "native_area", "monkey", "session", "mode"], as_index=False).agg(
        cross_scene_ridge_r=("cross_scene_ridge_r", "mean"),
        selected_family_generalization_r=("cross_scene_selected_family_signed_r", "mean"),
    )
    summary = unit.groupby(["roi", "mode"], as_index=False).agg(
        n_units=("unit_global", "size"), median_cross_scene_ridge_r=("cross_scene_ridge_r", "median"),
        positive_ridge_fraction=("cross_scene_ridge_r", lambda z: float(np.mean(z > 0))),
        median_selected_family_generalization_r=("selected_family_generalization_r", "median"),
        positive_selected_family_fraction=("selected_family_generalization_r", lambda z: float(np.mean(z > 0))),
    )
    for metric in ("cross_scene_ridge_r", "selected_family_generalization_r"):
        p_values = []
        for row in summary.itertuples():
            q = unit[(unit.roi.eq(row.roi)) & unit["mode"].eq(row.mode)]
            cluster = q.groupby(["monkey", "session"])[metric].mean().to_numpy(float)
            p_values.append(validation.signflip_p(cluster, SEED + len(p_values)))
        summary[f"session_signflip_p_{metric}"] = p_values

    v2 = unit[(unit.roi.eq("V2")) & unit["mode"].eq("nonwarp")].sort_values("cross_scene_ridge_r", ascending=False)
    for rank, row in enumerate(v2.head(5).itertuples(), start=1):
        coef = full_nonwarp[("V2", int(row.unit_global))]
        for pole, idx in (("positive", np.argsort(coef)[-10:][::-1]), ("negative", np.argsort(coef)[:10])):
            for order_i, image_i in enumerate(idx, start=1):
                preference_rows.append({"unit_global": int(row.unit_global), "unit_rank": rank, "pole": pole,
                                        "rank_within_pole": order_i, "image_index": int(image_i),
                                        "coefficient": float(coef[image_i]), "scene_cluster": int(scenes[image_i]),
                                        "caption": captions[image_i]})

    result.to_csv(OUT / "cross_scene_fold_results.csv", index=False, encoding="utf-8-sig")
    unit.to_csv(OUT / "cross_scene_unit_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "cross_scene_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(preference_rows).to_csv(OUT / "v2_top_nonwarp_image_captions.csv", index=False, encoding="utf-8-sig")
    (OUT / "audit.json").write_text(json.dumps({"scene_clusters": 10, "outer_folds": 5,
                                                  "semantic_families": families, "pilot_units_excluded": True,
                                                  "temporal_modes_learned_without_semantics": True}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
