"""GAMUS tile reader + PyTorch dataset.

Verified format (Phase 0, see docs/phase0_data_report.md):
  - one HDF5 file per tile and modality, single dataset named 'image', no attributes
    (=> tiles carry NO georeferencing: no CRS, no transform)
  - images/<split>/<id>_{RGB|IMG}.h5  uint8   (1024, 1024, 3)
  - heights/<split>/<id>_AGL.h5       float32 (1024, 1024)  nDSM, metres above ground
  - classes/<split>/<id>_CLS.h5       float32 (DC) or uint8 (NYC, PHL), labels 0..6
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "gamus"

# Documented GAMUS ground sampling distance. Only valid for GAMUS tiles; never
# assumed for arbitrary user images.
GAMUS_GSD_M = 0.33
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["image"][()]


def tile_paths(split: str, tid: str, root: Path = DATA_DIR) -> dict[str, Path]:
    img = root / "images" / split / f"{tid}_RGB.h5"
    if not img.exists():
        img = root / "images" / split / f"{tid}_IMG.h5"
    return {
        "rgb": img,
        "ndsm": root / "heights" / split / f"{tid}_AGL.h5",
        "cls": root / "classes" / split / f"{tid}_CLS.h5",
    }


# NoData rules derived from the Phase 0 survey (docs/phase0_data_report.md):
#  - DC tiles clip unreliable LiDAR (mostly over water) to exactly -5.0, with a few
#    values in (-5, -4.9). Anything below NODATA_BELOW_M is treated as NoData.
#    Small negatives (NYC, down to about -1.6 m) are ordinary DSM-DTM noise and stay valid.
#  - Isolated LiDAR spikes (birds, wires, speckle, up to ~200 m on flat ground) are
#    removed by comparing each pixel with its 5x5 median. Real tall buildings are
#    larger than the window and survive (only a few corner pixels are lost).
NODATA_BELOW_M = -2.0
SPIKE_WINDOW = 5
SPIKE_THRESHOLD_M = 20.0


def valid_mask(ndsm: np.ndarray) -> np.ndarray:
    """Pixels that may enter losses/metrics. The single place the NoData rule lives."""
    from scipy.ndimage import median_filter
    valid = np.isfinite(ndsm) & (ndsm >= NODATA_BELOW_M)
    med = median_filter(np.where(valid, ndsm, 0), size=SPIKE_WINDOW, mode="nearest")
    return valid & ((ndsm - med) <= SPIKE_THRESHOLD_M)


# Cached masks live in a directory named after the rule parameters, so changing
# the rule can never silently reuse stale masks.
VALID_CACHE_DIR = ROOT / "data" / "cache" / (
    f"valid_below{NODATA_BELOW_M:g}_w{SPIKE_WINDOW}_t{SPIKE_THRESHOLD_M:g}")


def cached_valid_mask(split: str, tid: str, ndsm: np.ndarray) -> np.ndarray:
    path = VALID_CACHE_DIR / split / f"{tid}.npy"
    if path.exists():
        return np.unpackbits(np.load(path))[: ndsm.size].reshape(ndsm.shape).astype(bool)
    v = valid_mask(ndsm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.packbits(v))
    return v


def read_rgb(split: str, tid: str, root: Path = DATA_DIR) -> np.ndarray:
    return _read(tile_paths(split, tid, root)["rgb"])[..., :3].astype(np.uint8)


def read_tile(split: str, tid: str, root: Path = DATA_DIR, cache: bool = True,
              window: tuple[int, int, int] | None = None) -> dict[str, np.ndarray]:
    """Read a tile, or only a square window (y, x, size) of it.

    Window reads slice the HDF5 datasets directly (only the needed rows are read).
    The validity mask is always computed on the FULL tile (the spike filter needs
    neighbours) and then sliced, so a window's mask equals the full-tile mask there.
    """
    p = tile_paths(split, tid, root)
    if window is None:
        sl = (slice(None), slice(None))
    else:
        y, x, c = window
        sl = (slice(y, y + c), slice(x, x + c))
    with h5py.File(p["rgb"], "r") as f:
        rgb = f["image"][sl]
    with h5py.File(p["cls"], "r") as f:
        cls = f["image"][sl]
    use_cache = cache and root == DATA_DIR
    cache_file = VALID_CACHE_DIR / split / f"{tid}.npy"
    with h5py.File(p["ndsm"], "r") as f:
        d = f["image"]
        if use_cache and cache_file.exists():
            ndsm = d[sl].astype(np.float32)
            H, W = d.shape
            valid = np.unpackbits(np.load(cache_file))[: H * W].reshape(H, W).astype(bool)[sl]
        else:
            full = d[()].astype(np.float32)
            valid = (cached_valid_mask(split, tid, full) if use_cache else valid_mask(full))[sl]
            ndsm = full[sl]
    assert rgb.shape[:2] == ndsm.shape == cls.shape == valid.shape, (tid, rgb.shape, ndsm.shape)
    return {
        "rgb": rgb[..., :3].astype(np.uint8),
        "ndsm": ndsm,
        "cls": np.rint(cls).astype(np.uint8),
        "valid": valid,
    }


def load_subset(name: str) -> dict[str, list[str]]:
    with open(ROOT / "configs" / "subsets" / f"{name}.yaml") as f:
        return yaml.safe_load(f)["splits"]


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_TO_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)


def color_jitter(rgb: np.ndarray, rng: np.random.Generator, brightness: float = 0.1,
                 contrast: float = 0.1, saturation: float = 0.1, hue: float = 0.02) -> np.ndarray:
    """Mild photometric jitter on a uint8 HxWx3 image (geometry untouched).

    brightness/contrast/saturation: multiplicative factors from U[1-x, 1+x];
    hue: rotation of the chroma plane in YIQ space by 2*pi*U(-hue, hue).
    """
    x = rgb.astype(np.float32)
    x = x * rng.uniform(1 - brightness, 1 + brightness)
    m = float((x @ _LUMA).mean())
    x = (x - m) * rng.uniform(1 - contrast, 1 + contrast) + m
    gray = (x @ _LUMA)[..., None]
    x = gray + (x - gray) * rng.uniform(1 - saturation, 1 + saturation)
    if hue:
        t = 2 * np.pi * rng.uniform(-hue, hue)
        rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]], np.float32)
        x = x @ (np.linalg.inv(_TO_YIQ) @ rot @ _TO_YIQ).T.astype(np.float32)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own numpy stream (derived from torch's
    per-worker seed); otherwise workers could draw identical crops."""
    import torch
    np.random.seed(torch.initial_seed() % 2**32)


try:
    import torch
    from torch.utils.data import Dataset

    class GAMUSDataset(Dataset):
        """Returns dict(image[3,H,W] ImageNet-normalised, ndsm[1,H,W] metres,
        valid[1,H,W] bool, cls[H,W] long, id).

        train=True: random `crop` window (read directly from disk) + random flips /
        90° rotations (height is invariant to in-plane rotation for nadir imagery)
        + optional colour jitter (dict of color_jitter kwargs).
        train=False: full tile (or centre crop if `crop` is set), no augmentation.

        scale_aug (Phase 6, train only), e.g. {"p": 0.5, "min": 1.0, "max": 1.95}: with
        probability p draw s ~ U(min, max), read a round(crop * s) px window and coarsen it
        to `crop` px (RGB area average, nDSM valid-weighted area average with >= 50% valid,
        classes nearest; depthwizard.data.coarsen = the Phase 4 simulation). The sample then
        shows the scene at a ground resolution of 0.33 * s m/px with the same tensor size.
        `scale` in the returned dict is the factor used (1.0 = native).
        """

        def __init__(self, subset: str, split: str, crop: int | None = None,
                     train: bool = False, root: Path = DATA_DIR, jitter: dict | None = None,
                     tile_size: int = 1024, scale_aug: dict | None = None):
            self.ids = load_subset(subset)[split]
            self.split, self.crop, self.train, self.root = split, crop, train, root
            self.jitter, self.tile_size = jitter, tile_size
            self.scale_aug = scale_aug if train else None
            if self.scale_aug:
                assert crop and round(crop * self.scale_aug["max"]) <= tile_size, 'scale_aug window exceeds the tile'

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, i: int) -> dict:
            tid = self.ids[i]
            window = None
            scale = 1.0
            if self.scale_aug and np.random.rand() < self.scale_aug["p"]:
                scale = float(np.random.uniform(self.scale_aug["min"], self.scale_aug["max"]))
            if self.crop:
                c, S = self.crop, self.tile_size
                c_out = c
                c = int(round(c * scale))            # native window that becomes `crop` px after coarsening
                if self.train:
                    y, x = np.random.randint(0, S - c + 1), np.random.randint(0, S - c + 1)
                else:
                    y, x = (S - c) // 2, (S - c) // 2
                window = (y, x, c)
            t = read_tile(self.split, tid, self.root, window=window)
            rgb, ndsm, cls, valid = t["rgb"], t["ndsm"], t["cls"], t["valid"]
            if self.crop and c != c_out:
                from .coarsen import coarsen_cls, coarsen_gt, coarsen_rgb
                rgb = coarsen_rgb(rgb, c_out)
                ndsm, valid = coarsen_gt(ndsm, valid, c_out)
                cls = coarsen_cls(cls, c_out)

            if self.train:
                k = np.random.randint(4)
                flip = np.random.rand() < 0.5

                def aug(a):
                    a = np.rot90(a, k, axes=(0, 1))
                    return a[:, ::-1] if flip else a
                rgb, ndsm, cls, valid = aug(rgb), aug(ndsm), aug(cls), aug(valid)
                if self.jitter:
                    rgb = color_jitter(np.ascontiguousarray(rgb),
                                       np.random.default_rng(np.random.randint(2**31)), **self.jitter)

            img = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            return {
                "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                "ndsm": torch.from_numpy(np.ascontiguousarray(ndsm))[None],
                "valid": torch.from_numpy(np.ascontiguousarray(valid))[None],
                "cls": torch.from_numpy(np.ascontiguousarray(cls)).long(),
                "id": tid,
                "scale": scale,
            }
except ImportError:  # torch-free use (inspection scripts)
    pass
