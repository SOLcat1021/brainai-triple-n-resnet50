"""Extract full-channel ResNet-50 spatial maps for the TVSD image order.

The MAT file supplies the authoritative THINGS order; no directory sorting is
used. Maps are resized to the existing 14x14 fwRF grid and stored as float16
memmaps so extraction can resume after interruption.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMG_ROOT = Path(r"G:\BrainAI_Data\shared\THINGS\images_extracted\object_images")
DATA = ROOT / "data" / "TVSD" / "monkeyN" / "things_imgs.mat"
OUT = ROOT / "cache" / "tvsd" / "original_fwrf_resnet50" / "spatial_full_14"
N_TRAIN, N_TEST, GRID = 22248, 100, 14
LAYERS = {"stem": 64, "res2": 256, "res3": 512, "res4": 1024, "res5": 2048}


def mat_string(handle: h5py.File, ref) -> str:
    values = np.asarray(handle[ref]).reshape(-1)
    return "".join(chr(int(v)) for v in values)


def image_paths(group, handle):
    refs = group["things_path"][()]
    return [IMG_ROOT / mat_string(handle, refs[i, 0]).replace("\\", "/") for i in range(len(refs))]


def forward(model, x):
    out = {}
    x = model.relu(model.bn1(model.conv1(x))); out["stem"] = x
    x = model.maxpool(x); x = model.layer1(x); out["res2"] = x
    x = model.layer2(x); out["res3"] = x
    x = model.layer3(x); out["res4"] = x
    x = model.layer4(x); out["res5"] = x
    return out


def main(batch_size: int = 32):
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA, "r") as mat:
        train_paths = image_paths(mat["train_imgs"], mat)
        test_paths = image_paths(mat["test_imgs"], mat)
    assert len(train_paths) == N_TRAIN and len(test_paths) == N_TEST
    assert all(p.exists() for p in train_paths + test_paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(weights=None)
    state = torch.load(ROOT / "models" / "vision" / "resnet50_imagenet1k_v2.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state); model.eval().to(device)
    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    arrays = {}
    for split, n, paths in (("train", N_TRAIN, train_paths), ("test", N_TEST, test_paths)):
        for layer, channels in LAYERS.items():
            path = OUT / f"{split}_{layer}_f16.npy"
            arrays[(split, layer)] = np.load(path, mmap_mode="r+") if path.exists() else np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, channels, GRID, GRID))
        marker = OUT / f"{split}_complete.json"
        start = 0
        if marker.exists():
            start = int(json.loads(marker.read_text())["n_images"])
        with torch.inference_mode():
            for i in range(start, n, batch_size):
                j = min(i + batch_size, n)
                batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:j]]).to(device)
                maps = forward(model, batch)
                for layer, fmap in maps.items():
                    pooled = torch.nn.functional.adaptive_avg_pool2d(fmap.float(), (GRID, GRID))
                    arrays[(split, layer)][i:j] = pooled.cpu().numpy().astype(np.float16)
                if j % 320 == 0 or j == n:
                    for a in arrays.values(): a.flush()
                    marker.write_text(json.dumps({"n_images": j, "device": str(device), "image_order": "things_imgs.mat things_path"}, indent=2), encoding="utf-8")
                    print(f"{split} {j}/{n}", flush=True)
    (OUT / "metadata.json").write_text(json.dumps({"network": "ResNet50 IMAGENET1K_V2", "layers": LAYERS, "grid": GRID, "dtype": "float16", "source_mat": str(DATA), "image_root": str(IMG_ROOT)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
