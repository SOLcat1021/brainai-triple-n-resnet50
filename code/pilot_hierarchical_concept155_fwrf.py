"""Minimal hierarchical-concept encoding demo with frozen fwRF fields.

The concept layer assignment is response-blind. Spatial fields are reused from
the established 17-layer proxy bank, including fold-local fields for OOF
evaluation. Each concept contributes only from its assigned ResNet stage.
"""
from __future__ import annotations

import io
import math
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


ROOT = Path(r"D:/Coding/BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
COHORT = PROJECT / "cohorts/high_quality_64d_pilot_2026-08-26"
AXIS_DIR = PROJECT / "tcav_broden1000_pca32"
PROXY_DIR = PROJECT / "proxy_bank_10ms_all_windows"
IMAGE_ZIP = ROOT / "data/TripleN/V1/others/StimuliNNN.zip"
RESPONSES = PROXY_DIR / "trin_all_units_10ms_responses.npy"
METADATA = PROXY_DIR / "unit_metadata.csv"
REPORT = PROJECT / "hierarchical_concept155_fwrf_demo.md"

LAYERS = ("res2", "res3", "res4", "res5")
NODES = ("res2_b3", "res3_b4", "res4_b6", "res5_b3")
WINDOWS = (70, 110, 150)
N_IMAGES = 1000
N_FOLDS = 5
SEED = 20260818 + 901
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
BATCH_SIZE = 24

BANK_BY_AREA = {
    "V1": "v1_proxy_bank.h5",
    "V1/V2": "v2_proxy_bank.h5",
    "V2": "v2_proxy_bank.h5",
    "V4": "v4_proxy_bank.h5",
    "PF": "posterior_it_proxy_bank.h5",
    "PITP": "posterior_it_proxy_bank.h5",
    "MF": "middle_it_proxy_bank.h5",
    "MB": "middle_it_proxy_bank.h5",
    "MO": "middle_it_proxy_bank.h5",
    "CLC": "middle_it_proxy_bank.h5",
    "LPP": "middle_it_proxy_bank.h5",
    "AF": "anterior_it_proxy_bank.h5",
    "AB": "anterior_it_proxy_bank.h5",
    "AO": "anterior_it_proxy_bank.h5",
    "AMC": "anterior_it_proxy_bank.h5",
}
CATEGORY_ZH = {
    "color": "颜色", "material": "材质", "object": "物体",
    "part": "部件", "scene": "场景", "texture": "纹理",
}


def fixed_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.random.default_rng(SEED).permutation(n)
    chunks = np.array_split(order, N_FOLDS)
    return [
        (np.concatenate([chunks[j] for j in range(N_FOLDS) if j != k]), chunks[k])
        for k in range(N_FOLDS)
    ]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, float) - np.mean(a)
    bb = np.asarray(b, float) - np.mean(b)
    denominator = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    return float(np.sum(aa * bb) / denominator) if denominator > 1e-12 else np.nan


def select_units() -> pd.DataFrame:
    coarse = pd.read_csv(COHORT / "pilot_coarse_roi_12_each.csv")
    functional = pd.read_csv(COHORT / "pilot_functional_it_8_each.csv")
    quality = pd.concat([coarse, functional], ignore_index=True)
    quality = quality.sort_values(
        ["unit_global", "min_window_r", "mean_window_r", "released_independent_consistency"],
        ascending=[True, False, False, False],
    ).drop_duplicates("unit_global")
    metadata = pd.read_csv(METADATA)[["unit_global", "native_area"]]
    quality = quality.drop(columns=["native_area"], errors="ignore").merge(
        metadata, on="unit_global", how="left", validate="one_to_one"
    )
    selected = []
    for area, group in quality.groupby("native_area", sort=True):
        chosen = group.sort_values(
            ["min_window_r", "mean_window_r", "released_independent_consistency"],
            ascending=False,
        ).head(2).copy()
        chosen["area_unit_number"] = np.arange(1, len(chosen) + 1)
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True)
    return result.sort_values(["native_area", "area_unit_number"]).reset_index(drop=True)


def concept_assignments() -> pd.DataFrame:
    concepts = pd.read_csv(AXIS_DIR / "concepts_1000.csv").iloc[:500].set_index("concept_index")
    rows = []
    for concept_index, metadata in concepts.iterrows():
        shard = next((AXIS_DIR / "axis_shards").glob(f"{concept_index:04d}_*.npz"))
        saved = np.load(shard)
        accuracy = saved["heldout_accuracy"].astype(float)
        repeated = saved["cav_repeated"].astype(float)
        means = accuracy.mean(axis=0)
        standard_errors = accuracy.std(axis=0, ddof=1) / np.sqrt(accuracy.shape[0])
        stability = []
        for layer_index in range(len(LAYERS)):
            vectors = repeated[:, layer_index]
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            pairwise = vectors @ vectors.T
            stability.append(float(pairwise[np.triu_indices(len(vectors), 1)].mean()))
        stable = np.asarray(stability) > 0.9
        if not stable.any():
            continue
        stable_indices = np.flatnonzero(stable)
        best = stable_indices[np.argmax(means[stable_indices])]
        threshold = means[best] - standard_errors[best]
        chosen = int(np.flatnonzero(stable & (means >= threshold))[0])
        rows.append({
            "concept_index": int(concept_index),
            "concept": str(metadata["name"]),
            "category": str(metadata["primary_category"]),
            "layer_index": chosen,
            "layer": LAYERS[chosen],
            "accuracy": float(means[chosen]),
            "stability": float(stability[chosen]),
        })
    result = pd.DataFrame(rows)
    if len(result) != 155:
        raise RuntimeError(f"Expected 155 assigned concepts, found {len(result)}")
    return result


def continuous_gaussian(parameters: np.ndarray, n_pix: int, device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(parameters, dtype=torch.float32, device=device)
    x, y, sigma = values.T
    view_angle = 20.0
    dpix = view_angle / n_pix
    coord = torch.arange(n_pix, device=device, dtype=torch.float32) * dpix - view_angle / 2 + dpix / 2
    xm, ym_raw = torch.meshgrid(coord, coord, indexing="xy")
    ym = -ym_raw
    x = x[:, None, None]
    y = y[:, None, None]
    sigma = sigma[:, None, None]
    sqrt2 = math.sqrt(2.0)
    gx = 0.5 * (torch.erf((xm - x + dpix / 2) / (sqrt2 * sigma)) -
                torch.erf((xm - x - dpix / 2) / (sqrt2 * sigma)))
    gy = 0.5 * (torch.erf((ym - y + dpix / 2) / (sqrt2 * sigma)) -
                torch.erf((ym - y - dpix / 2) / (sqrt2 * sigma)))
    exact = gx * gy
    d = 2 * sigma.square()
    approx = dpix ** 2 / (d * math.pi) * torch.exp(-((xm - x).square() + (ym - y).square()) / d)
    return torch.where(sigma < dpix, exact, approx).flatten(1)


def load_fields(selected: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int]]]:
    parameters = []
    targets = []
    for selected_index, row in selected.iterrows():
        bank_path = PROXY_DIR / BANK_BY_AREA[str(row.native_area)]
        with h5py.File(bank_path, "r") as bank:
            units = bank["unit_global"][:].astype(int)
            unit_matches = np.flatnonzero(units == int(row.unit_global))
            if len(unit_matches) != 1:
                raise RuntimeError(f"Unit {row.unit_global} missing or duplicated in {bank_path.name}")
            local_unit = int(unit_matches[0])
            bank_windows = bank["windows_ms"][:]
            for window in WINDOWS:
                window_matches = np.flatnonzero(bank_windows[:, 0] == window)
                if len(window_matches) != 1:
                    raise RuntimeError(f"Window {window} missing in {bank_path.name}")
                wi = int(window_matches[0])
                for fold in range(N_FOLDS):
                    parameters.append(bank["fold_spatial_parameters"][fold, wi, local_unit].astype(float))
                parameters.append(bank["spatial_parameters"][wi, local_unit].astype(float))
                targets.append((selected_index, window))
    return np.asarray(parameters, np.float32), targets


def stage_maps(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    x = model.relu(model.bn1(model.conv1(images)))
    x = model.maxpool(x)
    x = model.layer1(x)
    result = {"res2": x}
    x = model.layer2(x)
    result["res3"] = x
    x = model.layer3(x)
    result["res4"] = x
    x = model.layer4(x)
    result["res5"] = x
    return result


def extract_scores(assignments: pd.DataFrame, fields: torch.Tensor, device: torch.device) -> np.ndarray:
    bank = np.load(AXIS_DIR / "pca32_concept_axis_bank_1000.npz", allow_pickle=True)
    axes = bank["cav_pca32"].astype(np.float32)
    bases = np.load(AXIS_DIR / "pca32_bases.npz")
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    model = torchvision.models.resnet50(weights=weights).eval().to(device)
    scores = np.empty((N_IMAGES, len(fields), len(assignments)), np.float16)
    with zipfile.ZipFile(IMAGE_ZIP) as archive, torch.inference_mode():
        for start in range(0, N_IMAGES, BATCH_SIZE):
            stop = min(N_IMAGES, start + BATCH_SIZE)
            images = torch.stack([
                transform(Image.open(io.BytesIO(archive.read(f"{index + 1:04d}.bmp"))).convert("RGB"))
                for index in range(start, stop)
            ]).to(device)
            maps = stage_maps(model, images)
            for layer_index, (layer, node) in enumerate(zip(LAYERS, NODES)):
                concept_rows = np.flatnonzero(assignments.layer_index.to_numpy() == layer_index)
                if not len(concept_rows):
                    continue
                concept_indices = assignments.iloc[concept_rows].concept_index.to_numpy(int)
                mean = torch.as_tensor(bases[f"mean_{node}"].astype(np.float32), device=device)
                scale = torch.as_tensor(bases[f"scale_{node}"].astype(np.float32), device=device)
                components = torch.as_tensor(bases[f"components_{node}"].astype(np.float32).T, device=device)
                channel_projection = components / scale[:, None]
                bias = -(mean / scale) @ components
                feature_map = torch.nn.functional.interpolate(maps[layer].float(), size=(14, 14), mode="area")
                projected_map = torch.einsum("bchw,cd->bdhw", feature_map, channel_projection).flatten(2)
                pca_scores = torch.einsum("bdp,fp->bfd", projected_map, fields) + bias[None, None]
                concept_axes = torch.as_tensor(axes[concept_indices, layer_index], device=device)
                concept_scores = torch.einsum("bfd,kd->bfk", pca_scores, concept_axes)
                scores[start:stop, :, concept_rows] = concept_scores.cpu().numpy().astype(np.float16)
            print(f"feature scoring {stop}/{N_IMAGES}", flush=True)
    return scores


def choose_alpha(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    folds = KFold(4, shuffle=True, random_state=seed)
    best = (-np.inf, ALPHAS[0])
    for alpha in ALPHAS:
        prediction = np.empty(len(y), np.float32)
        for train, test in folds.split(x):
            model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8).fit(x[train], y[train])
            prediction[test] = model.predict(x[test])
        score = corr(prediction, y)
        if score > best[0]:
            best = (score, alpha)
    return float(best[1])


def fit_models(
    selected: pd.DataFrame,
    assignments: pd.DataFrame,
    targets: list[tuple[int, int]],
    scores: np.ndarray,
) -> list[dict]:
    response_bank = np.load(RESPONSES, mmap_mode="r")
    metadata = pd.read_csv(METADATA)
    response_lookup = {int(unit): i for i, unit in enumerate(metadata.unit_global)}
    outer_folds = fixed_folds(N_IMAGES)
    results = []
    for target_index, (selected_index, window) in enumerate(targets):
        row = selected.loc[selected_index]
        response_index = (window + 20) // 10
        y = np.asarray(response_bank[response_index, :, response_lookup[int(row.unit_global)]], np.float32)
        prediction = np.empty(N_IMAGES, np.float32)
        for fold, (train, test) in enumerate(outer_folds):
            x = np.asarray(scores[:, target_index * (N_FOLDS + 1) + fold], np.float32)
            mean = x[train].mean(axis=0)
            scale = x[train].std(axis=0).clip(1e-6)
            z = (x - mean) / scale
            y_mean = y[train].mean()
            y_scale = max(float(y[train].std()), 1e-6)
            y_standardized = (y - y_mean) / y_scale
            alpha = choose_alpha(z[train], y_standardized[train], SEED + target_index * 101 + fold)
            prediction[test] = Ridge(alpha=alpha, solver="lsqr", tol=1e-8).fit(
                z[train], y_standardized[train]
            ).predict(z[test])

        full_x = np.asarray(scores[:, target_index * (N_FOLDS + 1) + N_FOLDS], np.float32)
        full_mean = full_x.mean(axis=0)
        full_scale = full_x.std(axis=0).clip(1e-6)
        full_z = (full_x - full_mean) / full_scale
        standardized_y = (y - y.mean()) / max(float(y.std()), 1e-6)
        final_alpha = choose_alpha(full_z, standardized_y, SEED + target_index * 101 + 97)
        final_model = Ridge(alpha=final_alpha, solver="lsqr", tol=1e-8).fit(full_z, standardized_y)
        top = np.argsort(np.abs(final_model.coef_))[::-1][:10]
        results.append({
            "selected_index": selected_index,
            "unit_global": int(row.unit_global),
            "native_area": str(row.native_area),
            "area_unit_number": int(row.area_unit_number),
            "window": window,
            "oof_r": corr(prediction, y),
            "alpha": final_alpha,
            "top": [
                {
                    "concept": str(assignments.iloc[index].concept),
                    "category": str(assignments.iloc[index].category),
                    "layer": str(assignments.iloc[index].layer),
                    "weight": float(final_model.coef_[index]),
                }
                for index in top
            ],
        })
        print(f"fit {target_index + 1}/{len(targets)}", flush=True)
    return results


def write_report(selected: pd.DataFrame, assignments: pd.DataFrame, results: list[dict]) -> None:
    lines = [
        "# 分层概念混合编码最小Demo",
        "",
        "## 配置",
        "",
        "- 候选unit：既有113个高质量unit；按原生ROI内的min-window r、mean-window r和一致性排序，每区最多2个。",
        "- 时间窗：70–79、110–119、150–159 ms分别独立拟合；每个unit最终只展示概念模型OOF r最高的窗口。",
        "- 概念：Broden高频500轴中具有至少一个稳定层的155个；单层稳定性门槛为重复CAV平均余弦>0.9。",
        "- 层级：每个概念只保留按held-out分类准确率1-SE规则得到的最早近最优层。",
        "- 感受野：直接复用既有17层代理模型保存的FWRF；OOF使用对应外折的fold-local FWRF，全数据权重使用full-data FWRF。",
        "- 模型：感受野内概念得分逐列标准化后进行岭回归；Top-10按标准化系数绝对值排序，保留正负号。",
        "",
        "## Unit选择",
        "",
    ]
    for area, group in selected.groupby("native_area", sort=True):
        labels = [f"{area}-{int(row.area_unit_number)}=unit {int(row.unit_global)}" for _, row in group.iterrows()]
        suffix = "（113集合中仅此1个）" if len(group) == 1 else ""
        lines.append(f"- {area}：{'，'.join(labels)}{suffix}")
    lines.extend([
        "",
        "## 结果",
        "",
        "概念得分和响应均已标准化，所列为无量纲标准化岭回归系数。正负号表示在其他概念得分不变的线性条件下，对预测响应的正向或负向贡献。",
        "",
    ])
    for area, area_results in pd.DataFrame(results).groupby("native_area", sort=True):
        lines.extend([f"## {area}", ""])
        units = sorted({(int(item["area_unit_number"]), int(item["unit_global"])) for item in area_results.to_dict("records")})
        for number, unit in units:
            lines.extend([f"### {area}-{number}（unit {unit}）", ""])
            unit_results = [item for item in results if item["native_area"] == area and item["unit_global"] == unit]
            item = max(unit_results, key=lambda value: np.nan_to_num(value["oof_r"], nan=-np.inf))
            lines.append(
                f"**最佳窗 {item['window']}–{item['window'] + 9} ms：OOF r={item['oof_r']:.3f}，ridge alpha={item['alpha']:g}**"
            )
            lines.append("")
            for rank, entry in enumerate(item["top"], 1):
                lines.append(
                    f"{rank}. {entry['concept']} [{entry['layer']}; {CATEGORY_ZH[entry['category']]}]：{entry['weight']:+.4f}"
                )
            lines.append("")
    displayed = []
    for unit in selected.unit_global.astype(int):
        candidates = [item for item in results if item["unit_global"] == unit]
        displayed.append(max(candidates, key=lambda value: np.nan_to_num(value["oof_r"], nan=-np.inf)))
    mean_r = float(np.nanmean([item["oof_r"] for item in displayed]))
    median_r = float(np.nanmedian([item["oof_r"] for item in displayed]))
    lines.extend([
        "## 汇总与限制",
        "",
        f"- 共{len(selected)}个unit、内部拟合{len(results)}个unit×window模型；按unit取最佳窗后，平均OOF r={mean_r:.3f}，中位数={median_r:.3f}。",
        "- 这是最小可行性demo。155个概念在同一层内高度相关，岭回归系数是唯一的最小范数解，但Top-10不能被解释为独立因果贡献。",
        "- Unit来自既有代理性能筛选，而且展示窗口由三个候选窗的OOF r择优；当前r仅用于确认概念模型可预测，不用于对总体神经元群体作无偏性能估计。",
    ])
    REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    selected = select_units()
    assignments = concept_assignments()
    parameters, targets = load_fields(selected)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; units={len(selected)}; targets={len(targets)}; concepts={len(assignments)}", flush=True)
    fields = continuous_gaussian(parameters, 14, device)
    scores = extract_scores(assignments, fields, device)
    results = fit_models(selected, assignments, targets, scores)
    write_report(selected, assignments, results)
    print(f"DONE {REPORT}", flush=True)


if __name__ == "__main__":
    main()
