"""Build the original Deepnet-fwRF model-space tensor for Triple-N.

Faithful fwRF details (St-Yves & Naselaris, 2018):
* one isotropic Gaussian feature-pooling field shared by every feature map;
* 15 x 15 candidate centers and 10 log-spaced radii (2250 candidates);
* Gaussian pixel mass evaluated in a common 20-degree coordinate system;
* log(1 + sqrt(abs(x))) compressive nonlinearity after spatial pooling.

The intended experimental substitution is CaffeNet -> ImageNet-V2 ResNet-50.
Five spatial ResNet scales plus the global-average and classifier outputs are
retained, matching the original paper's use of convolutional and fully
connected features. They are concatenated during fwRF training and do not
receive independent receptive fields.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[2]
STIMULI = ROOT / "data" / "TripleN" / "V1" / "others" / "StimuliNNN.zip"
OUT = ROOT / "cache" / "trin" / "original_fwrf_resnet50" / "modelspace_standard_v2"
SPATIAL_LAYERS = ("stem", "res2", "res3", "res4", "res5")
GLOBAL_LAYERS = ("avgpool", "fc")
LAYERS = SPATIAL_LAYERS + GLOBAL_LAYERS
CHANNELS = {
    "stem": 64, "res2": 256, "res3": 512, "res4": 1024, "res5": 2048,
    "avgpool": 2048, "fc": 1000,
}
VIEW_ANGLE = 20.0
NX = NY = 15
NS = 10
SIGMA_MIN = 0.7
SIGMA_MAX = 8.0


def original_linspace(n: int, center: float, width: float) -> np.ndarray:
    d = width / n
    vmin, vmax = center + (d - width) / 2, center + width / 2
    return np.arange(vmin, vmax + 1e-12, d, dtype=np.float32)


def original_logspace(n: int, start: float, stop: float) -> np.ndarray:
    return np.exp(np.linspace(np.log(start + 1e-12), np.log(stop + 1e-12), n)).astype(np.float32)


def candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = original_linspace(NX, 0.0, VIEW_ANGLE)
    ys = original_linspace(NY, 0.0, VIEW_ANGLE)
    ss = original_logspace(NS, SIGMA_MIN, SIGMA_MAX)
    # np.unravel_index in the author code uses (nx, ny, ns): sigma changes fastest.
    ix, iy, isigma = np.unravel_index(np.arange(NX * NY * NS), (NX, NY, NS))
    return xs[ix], ys[iy], ss[isigma]


def gaussian_mass_stack(xs: np.ndarray, ys: np.ndarray, sigmas: np.ndarray, n_pix: int) -> np.ndarray:
    """Python-3 port of the author's numpy_utility.make_gaussian_mass_stack."""
    deg = np.float32(VIEW_ANGLE)
    dpix = deg / n_pix
    pix_min = -deg / 2 + 0.5 * dpix
    coords = np.arange(pix_min, deg / 2, dpix, dtype=np.float32)
    xm, ym_raw = np.meshgrid(coords, coords)
    ym = -ym_raw
    result = np.empty((len(xs), n_pix, n_pix), np.float32)
    for i, (x, y, sigma) in enumerate(zip(xs, ys, sigmas)):
        if sigma < dpix:
            gx = 0.5 * (erf((xm - x + dpix / 2) / (math.sqrt(2) * sigma)) -
                        erf((xm - x - dpix / 2) / (math.sqrt(2) * sigma)))
            gy = 0.5 * (erf((ym - y + dpix / 2) / (math.sqrt(2) * sigma)) -
                        erf((ym - y - dpix / 2) / (math.sqrt(2) * sigma)))
            result[i] = gx * gy
        else:
            d = 2 * np.float32(sigma) ** 2
            amp = 1.0 / (d * np.pi)
            result[i] = dpix ** 2 * amp * np.exp(-((xm - x) ** 2 + (ym - y) ** 2) / d)
    return result


def read_images(indices: list[int], archive: zipfile.ZipFile) -> torch.Tensor:
    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    values = []
    for index in indices:
        with Image.open(io.BytesIO(archive.read(f"{index + 1:04d}.bmp"))) as source:
            values.append(transform(source.convert("RGB")))
    return torch.stack(values)


def forward_maps(model: torch.nn.Module, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    x = model.relu(model.bn1(model.conv1(batch)))
    out["stem"] = x
    x = model.maxpool(x)
    x = model.layer1(x); out["res2"] = x
    x = model.layer2(x); out["res3"] = x
    x = model.layer3(x); out["res4"] = x
    x = model.layer4(x); out["res5"] = x
    x = model.avgpool(x).flatten(1); out["avgpool"] = x
    x = model.fc(x); out["fc"] = x
    return out


def main(batch_size: int = 2, candidate_batch: int = 150) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "complete.json"
    expected = [OUT / f"{layer}_g2250_n1000_f16.npy" for layer in SPATIAL_LAYERS]
    expected += [OUT / f"{layer}_n1000_f16.npy" for layer in GLOBAL_LAYERS]
    if marker.exists() and all(path.exists() for path in expected):
        print(marker)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    ).eval().to(device)
    xs, ys, sigmas = candidates()
    spatial_needed = {
        layer for layer in SPATIAL_LAYERS
        if not (OUT / f"{layer}_g2250_n1000_f16.npy").exists()
    }
    global_needed = {
        layer for layer in GLOBAL_LAYERS
        if not (OUT / f"{layer}_n1000_f16.npy").exists()
    }
    spatial_arrays = {}
    for layer in SPATIAL_LAYERS:
        path = OUT / f"{layer}_g2250_n1000_f16.npy"
        shape = (len(xs), 1000, CHANNELS[layer])
        spatial_arrays[layer] = (np.load(path, mmap_mode="r+") if path.exists()
                                 else np.lib.format.open_memmap(
                                     path, mode="w+", dtype=np.float16, shape=shape))
        assert spatial_arrays[layer].shape == shape
    global_arrays = {}
    for layer in GLOBAL_LAYERS:
        path = OUT / f"{layer}_n1000_f16.npy"
        shape = (1000, CHANNELS[layer])
        global_arrays[layer] = (np.load(path, mmap_mode="r+") if path.exists()
                                else np.lib.format.open_memmap(
                                    path, mode="w+", dtype=np.float16, shape=shape))
        assert global_arrays[layer].shape == shape
    gaussian_cache = {}
    with zipfile.ZipFile(STIMULI) as archive, torch.inference_mode():
        for start in range(0, 1000, batch_size):
            stop = min(start + batch_size, 1000)
            maps = forward_maps(model, read_images(list(range(start, stop)), archive).to(device))
            for layer in spatial_needed:
                fmap = maps[layer]
                resolution = int(fmap.shape[-1])
                if resolution not in gaussian_cache:
                    gaussian_cache[resolution] = gaussian_mass_stack(xs, ys, sigmas, resolution)
                for g0 in range(0, len(xs), candidate_batch):
                    g1 = min(g0 + candidate_batch, len(xs))
                    fields = torch.as_tensor(gaussian_cache[resolution][g0:g1], device=device)
                    pooled = torch.einsum("bchw,ghw->gbc", fmap.float(), fields)
                    pooled = torch.log1p(torch.sqrt(torch.abs(pooled)))
                    spatial_arrays[layer][g0:g1, start:stop] = pooled.cpu().numpy().astype(np.float16)
            for layer in global_needed:
                values = torch.log1p(torch.sqrt(torch.abs(maps[layer].float())))
                global_arrays[layer][start:stop] = values.cpu().numpy().astype(np.float16)
            if stop % 20 == 0:
                print(f"model-space images {stop}/1000", flush=True)
    for array in list(spatial_arrays.values()) + list(global_arrays.values()):
        array.flush()
    metadata = {
        "source": "St-Yves & Naselaris 2018 Deepnet-fwRF",
        "network_substitution": "torchvision ResNet50 IMAGENET1K_V2",
        "preprocessing": repr(torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()),
        "layers": list(LAYERS), "channels": CHANNELS,
        "view_angle": VIEW_ANGLE, "nx": NX, "ny": NY, "ns": NS,
        "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
        "candidate_count": int(len(xs)),
        "centers_x": xs.tolist(), "centers_y": ys.tolist(), "sigmas": sigmas.tolist(),
        "nonlinearity": "log(1+sqrt(abs(x)))",
        "dtype": "float16 storage; float32 fitting",
    }
    marker.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(marker)


if __name__ == "__main__":
    main()
