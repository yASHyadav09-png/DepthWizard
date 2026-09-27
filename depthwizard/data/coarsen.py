"""Simulating coarser ground resolution from 0.33 m GAMUS tiles (Phase 4 study, Phase 6 training).

Identical definitions to scripts/phase4_gsd_study.py (tested for equality):
    RGB   : area average (PIL BOX)                         - a coarser sensor integrates radiance
    nDSM  : area average of VALID pixels only; a coarse pixel is valid if >= 50% of its area is valid
    class : nearest neighbour                              - labels are categorical
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def box(a: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    wh = (size, size) if isinstance(size, int) else (size[1], size[0])
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(wh, Image.Resampling.BOX), dtype=np.float32)


def coarsen_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    if size == rgb.shape[0]:
        return rgb
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((size, size), Image.Resampling.BOX),
                      dtype=np.uint8)


def coarsen_gt(nd: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if size == nd.shape[0]:
        return nd, valid
    num = box(np.where(valid, nd, 0.0), size)
    den = box(valid.astype(np.float32), size)
    v = den >= 0.5
    return np.where(v, num / np.maximum(den, 1e-6), 0.0).astype(np.float32), v


def coarsen_cls(cls: np.ndarray, size: int) -> np.ndarray:
    if size == cls.shape[0]:
        return cls
    return np.asarray(Image.fromarray(np.ascontiguousarray(cls)).resize((size, size), Image.Resampling.NEAREST))
