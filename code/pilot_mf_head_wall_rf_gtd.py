"""RF-local head/wall scores versus G/T/D in five high-predictability MF units."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata
from torchvision.models import ResNet50_Weights, resnet50


ROOT = Path(r"D:\Coding\BrainAI")
PROJECT = ROOT / "ResNet50_Final_AllUnits_2026-08-24"
CODE = PROJECT / "code"
sys.path.insert(0, str(CODE))
from plot_orthogonal_temporal_modes_2d import raw_directions, symmetric_orthogonalize  # noqa: E402
from train_64d_high_quality_proxy_batch import fixed_folds, gaussian_fields  # noqa: E402

BANK = PROJECT / "proxy_bank_64d_high_quality_pilot_113_2026-08-26"
OLD_BANK = PROJECT / "proxy_bank_10ms_all_windows"
CAV_BANK = PROJECT / "tcav_broden500" / "broden500_native_channel_cav_bank.npz"
CAV_QUALITY = PROJECT / "tcav_broden500" / "broden500_cav_quality_by_layer.csv"
STIMULI = ROOT / "data" / "TripleN" / "V1" / "others" / "StimuliNNN.zip"
OUT = PROJECT / "results" / "mf_head_wall_rf_gtd_pilot_2026-08-28"

CONCEPTS = ("head", "wall")
DIRECTIONS = ("G", "T", "D")
N_IMAGES = 1000
GRID = 14


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep] - np.mean(x[keep]); y = y[keep] - np.mean(y[keep])
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / den) if den > 1e-12 else np.nan


def choose_units(selected: pd.DataFrame, h: h5py.File) -> pd.DataFrame:
    middle = selected[selected.roi.eq("middle IT")].reset_index(drop=True)
    units = h["unit_global"][:].astype(int)
    if not np.array_equal(units, middle.unit_global.to_numpy(int)):
        raise RuntimeError("middle-IT unit order mismatch")
    windows = h["windows_ms"][:].astype(int)
    reliable = (windows[:, 0] >= 90) & (windows[:, 1] <= 169)
    mean_r = h["oof_r"][:][reliable].mean(axis=0)
    middle["mean_oof_r_90_169"] = mean_r
    return middle[middle.native_area.eq("MF")].nlargest(5, "mean_oof_r_90_169").copy()


def choose_layers() -> dict[str, str]:
    quality = pd.read_csv(CAV_QUALITY)
    chosen = {}
    for concept in CONCEPTS:
        row = quality[quality.concept.eq(concept)].sort_values(
            ["heldout_accuracy_mean", "repeat_axis_cosine_mean"], ascending=False,
        ).iloc[0]
        chosen[concept] = str(row.node)
    return chosen


def read_batch(archive: zipfile.ZipFile, start: int, stop: int, transform) -> torch.Tensor:
    images = []
    for index in range(start, stop):
        with Image.open(io.BytesIO(archive.read(f"{index + 1:04d}.bmp"))) as source:
            images.append(transform(source.convert("RGB")))
    return torch.stack(images)


def extract_concept_maps(chosen_layers: dict[str, str]) -> np.ndarray:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "head_wall_native_cav_maps_f32.npy"
    if cache.exists():
        maps = np.load(cache)
        if maps.shape == (len(CONCEPTS), N_IMAGES, GRID, GRID):
            return maps

    bank = np.load(CAV_BANK, allow_pickle=True)
    names = bank["concepts"].tolist(); nodes = bank["nodes"].tolist()
    offsets = bank["offsets"].astype(int)
    axes = {}
    for concept in CONCEPTS:
        ci = names.index(concept); li = nodes.index(chosen_layers[concept])
        axes[concept] = torch.as_tensor(
            bank["cav_full"][ci, offsets[li]:offsets[li + 1]], dtype=torch.float32,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval().to(device)
    modules = {
        "res5_b1": model.layer4[0], "res5_b2": model.layer4[1], "res5_b3": model.layer4[2],
    }
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for node in set(chosen_layers.values()):
        hooks.append(modules[node].register_forward_hook(
            lambda _module, _inputs, output, key=node: activations.__setitem__(key, output.detach())
        ))
    result = np.empty((len(CONCEPTS), N_IMAGES, GRID, GRID), np.float32)
    try:
        with zipfile.ZipFile(STIMULI) as archive, torch.inference_mode():
            for start in range(0, N_IMAGES, 32):
                stop = min(start + 32, N_IMAGES)
                activations.clear()
                batch = read_batch(archive, start, stop, weights.transforms()).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    model(batch)
                for ci, concept in enumerate(CONCEPTS):
                    activation = activations[chosen_layers[concept]].float()
                    axis = axes[concept].to(device)
                    score_map = torch.einsum("bchw,c->bhw", activation, axis)[:, None]
                    score_map = F.interpolate(
                        score_map, size=(GRID, GRID), mode="bilinear", align_corners=False,
                    )[:, 0]
                    result[ci, start:stop] = score_map.cpu().numpy()
                if stop % 160 == 0 or stop == N_IMAGES:
                    print(f"native concept maps {stop}/{N_IMAGES}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    np.save(cache, result)
    return result


def rf_scores(concept_maps: np.ndarray, units: pd.DataFrame,
              middle: pd.DataFrame, h: h5py.File) -> np.ndarray:
    windows = h["windows_ms"][:].astype(int)
    folds = fixed_folds(N_IMAGES)
    scores = np.full((len(CONCEPTS), len(units), N_IMAGES), np.nan, np.float32)
    for ui, unit in enumerate(units.itertuples(index=False)):
        local = int(middle.index[middle.unit_global.eq(unit.unit_global)][0])
        match = np.flatnonzero(
            (windows[:, 0] == int(unit.window_start_ms))
            & (windows[:, 1] == int(unit.window_end_ms))
        )
        if len(match) != 1:
            raise RuntimeError(f"Representative window missing for unit {unit.unit_global}")
        wi = int(match[0])
        for fold, (_train, test) in enumerate(folds):
            field = gaussian_fields(h["fold_spatial_parameters"][fold, wi, local][None])[0]
            field = field.reshape(GRID, GRID)
            for ci in range(len(CONCEPTS)):
                scores[ci, ui, test] = np.einsum(
                    "nhw,hw->n", concept_maps[ci, test], field,
                )
    if not np.isfinite(scores).all():
        raise RuntimeError("RF concept scores are incomplete")
    return scores


def gtd_coefficients(units: pd.DataFrame, middle: pd.DataFrame,
                     h: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    responses = np.load(OLD_BANK / "trin_all_units_10ms_responses.npy", mmap_mode="r")
    windows = np.load(OLD_BANK / "windows_10ms.npy").astype(int)
    h_windows = h["windows_ms"][:].astype(int)
    if not np.array_equal(windows, h_windows):
        raise RuntimeError("Response and proxy windows differ")
    baseline = np.flatnonzero(windows[:, 1] < 0)
    time = np.flatnonzero((windows[:, 0] >= 70) & (windows[:, 1] <= 189))
    times = windows[time].mean(axis=1).astype(float)
    observed = np.empty((len(units), N_IMAGES, len(DIRECTIONS)), np.float32)
    predicted = np.empty_like(observed)
    folds = fixed_folds(N_IMAGES)
    proxy_all = h["oof_prediction"][:]
    for ui, unit in enumerate(units.itertuples(index=False)):
        local = int(middle.index[middle.unit_global.eq(unit.unit_global)][0])
        raw = np.asarray(responses[:, :, int(unit.unit_global)], np.float32)
        raw_base = raw[baseline].mean(axis=0)
        neural = (raw[time] - raw_base[None]).T
        proxy_raw = np.asarray(proxy_all[:, :, local], np.float32)
        proxy_base = proxy_raw[baseline].mean(axis=0)
        proxy = (proxy_raw[time] - proxy_base[None]).T
        for train, test in folds:
            template = neural[train].mean(axis=0)
            physical, _center = raw_directions(template, times)
            basis, _eigenvalue, _similarity = symmetric_orthogonalize(physical)
            if basis is None:
                raise RuntimeError(f"GTD directions not identifiable for unit {unit.unit_global}")
            observed[ui, test] = (neural[test] - template[None]) @ basis
            predicted[ui, test] = (proxy[test] - template[None]) @ basis
    return observed, predicted


def correlations(scores: np.ndarray, observed: np.ndarray, predicted: np.ndarray,
                 units: pd.DataFrame, chosen_layers: dict[str, str]) -> pd.DataFrame:
    rows = []
    for ci, concept in enumerate(CONCEPTS):
        for ui, unit in enumerate(units.itertuples(index=False)):
            x = scores[ci, ui]
            for di, direction in enumerate(DIRECTIONS):
                rows.append({
                    "concept": concept, "cav_layer": chosen_layers[concept],
                    "unit_global": int(unit.unit_global), "unit_type": str(unit.unit_type_name),
                    "mean_window_oof_r": float(unit.mean_oof_r_90_169),
                    "direction": direction,
                    "predicted_pearson_r": corr(x, predicted[ui, :, di]),
                    "predicted_spearman_r": corr(rankdata(x), rankdata(predicted[ui, :, di])),
                    "observed_pearson_r": corr(x, observed[ui, :, di]),
                    "observed_spearman_r": corr(rankdata(x), rankdata(observed[ui, :, di])),
                })
    return pd.DataFrame(rows)


def heatmap(table: pd.DataFrame, column: str, title: str, path: Path) -> None:
    units = table.unit_global.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    vmax = max(.15, float(np.nanmax(np.abs(table[column]))))
    for ax, concept in zip(axes, CONCEPTS):
        q = table[table.concept.eq(concept)]
        matrix = q.pivot(index="unit_global", columns="direction", values=column).reindex(
            index=units, columns=DIRECTIONS,
        ).to_numpy(float)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for y in range(len(units)):
            for x in range(3):
                ax.text(x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center", fontsize=10)
        median = np.nanmedian(matrix, axis=0)
        ax.set_title(f"{concept} | {q.cav_layer.iloc[0]}\nmedian: " +
                     "  ".join(f"{d} {v:+.2f}" for d, v in zip(DIRECTIONS, median)),
                     fontweight="bold")
        ax.set_xticks(range(3), DIRECTIONS)
        ax.set_yticks(range(len(units)), [f"unit {unit}" for unit in units])
        fig.colorbar(image, ax=ax, fraction=.045, pad=.03)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(BANK / "selected_units.csv")
    with h5py.File(BANK / "middle_it_proxy_bank_64d.h5", "r") as h:
        middle = selected[selected.roi.eq("middle IT")].reset_index(drop=True)
        units = choose_units(selected, h)
        chosen_layers = choose_layers()
        concept_maps = extract_concept_maps(chosen_layers)
        scores = rf_scores(concept_maps, units, middle, h)
        observed, predicted = gtd_coefficients(units, middle, h)
    table = correlations(scores, observed, predicted, units, chosen_layers)
    table.to_csv(OUT / "head_wall_rf_score_GTD_correlations.csv", index=False)
    np.savez_compressed(
        OUT / "head_wall_rf_scores_and_GTD.npz",
        concepts=np.asarray(CONCEPTS), units=units.unit_global.to_numpy(int),
        image_scores=scores, observed_gtd=observed, predicted_gtd=predicted,
    )
    heatmap(
        table, "predicted_pearson_r",
        "RF-local concept score vs OOF proxy-predicted G/T/D (Pearson r)",
        OUT / "01_head_wall_vs_predicted_GTD.png",
    )
    heatmap(
        table, "observed_pearson_r",
        "RF-local concept score vs observed neural G/T/D (Pearson r)",
        OUT / "02_head_wall_vs_observed_GTD.png",
    )
    audit = {
        "units": units.unit_global.astype(int).tolist(),
        "unit_selection": "top five MF units by mean five-fold OOF firing-rate r over 90-169 ms",
        "concept_layers": chosen_layers,
        "concept_layer_selection": "maximum Broden held-out CAV accuracy, before neural correlation",
        "concept_map": "native-channel CAV dot native ResNet activation at every spatial location; no PCA",
        "spatial_pooling": "fold-specific frozen Gaussian fwRF at each unit's preselected representative window",
        "gtd": "fold-specific neural-training template and Lowdin-orthogonalized G/T/D; held-out observed and OOF proxy curves projected into the same basis",
        "n_images": N_IMAGES,
        "correlation": ["Pearson", "Spearman"],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = table.groupby(["concept", "direction"])[
        ["predicted_pearson_r", "observed_pearson_r"]
    ].median()
    print(summary.to_string(), flush=True)
    print(f"Saved pilot to {OUT}", flush=True)


if __name__ == "__main__":
    main()
