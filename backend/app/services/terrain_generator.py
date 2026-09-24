"""Build a compact 3D terrain grid from the relative height field.

A 4000x3000 photo is 12M pixels -- far too many vertices for a browser. We
resample the rDSM onto a small grid (256 on the longest side by default) and
ship the heights as a base64 float32 buffer. The frontend displaces a
PlaneGeometry with it and projects the original RGB as the texture.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TerrainGrid:
    """Everything Three.js needs to build the mesh."""

    width: int                 # grid columns
    height: int                # grid rows
    heights_b64: str           # row-major float32 [0, 1], length width*height
    encoding: str              # buffer format hint for the client
    min_height: float
    max_height: float
    mean_height: float
    aspect_ratio: float        # source image width / height
    plane_width: float         # world units for the mesh footprint
    plane_depth: float
    source_width: int          # pixel size of the rDSM the grid came from
    source_height: int
    height_units: str = "relative (0-1, unitless)"

    def to_dict(self) -> dict:
        return asdict(self)


def _resample(field01: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Area-average down / bilinear up, in float space (no 8-bit quantisation)."""
    src_h, src_w = field01.shape
    if (src_w, src_h) == (target_w, target_h):
        return field01.astype(np.float32)
    resample = (
        Image.Resampling.BOX
        if target_w <= src_w and target_h <= src_h
        else Image.Resampling.BILINEAR
    )
    img = Image.fromarray(field01.astype(np.float32), mode="F")
    return np.asarray(img.resize((target_w, target_h), resample), dtype=np.float32)


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur in float space.

    Pillow's ``ImageFilter.GaussianBlur`` refuses mode "F" images, and going
    through 8-bit would quantise the height field, so this is done by hand:
    a 1-D kernel applied as shift-and-add along each axis, with edge padding.
    """
    if sigma <= 0:
        return field
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    padded = np.pad(field, ((0, 0), (radius, radius)), mode="edge")
    width = field.shape[1]
    horizontal = np.zeros_like(field)
    for i, weight in enumerate(kernel):
        horizontal += weight * padded[:, i : i + width]

    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    height = field.shape[0]
    blurred = np.zeros_like(field)
    for i, weight in enumerate(kernel):
        blurred += weight * padded[i : i + height, :]
    return blurred.astype(np.float32)


def _grid_shape(src_w: int, src_h: int, resolution: int) -> tuple[int, int]:
    """Longest side becomes ``resolution``; the other side keeps the aspect."""
    resolution = max(16, int(resolution))
    if src_w >= src_h:
        width = min(resolution, src_w)
        height = max(16, round(width * src_h / src_w))
    else:
        height = min(resolution, src_h)
        width = max(16, round(height * src_w / src_h))
    return int(width), int(height)


def generate_terrain(
    height01: np.ndarray,
    resolution: int = 256,
    smooth: float = 0.6,
) -> TerrainGrid:
    """Relative height field -> downsampled terrain grid for the 3D viewer.

    ``smooth`` applies a small Gaussian blur after resampling. Monocular depth
    is noisy at object boundaries and a touch of smoothing removes the spiky
    artefacts without flattening real structure.
    """
    field = np.asarray(height01, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D height field, got shape {field.shape}.")

    src_h, src_w = field.shape
    grid_w, grid_h = _grid_shape(src_w, src_h, resolution)
    grid = _resample(field, grid_w, grid_h)

    if smooth > 0:
        grid = _gaussian_blur(grid, float(smooth))

    grid = np.clip(np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    # Normalise the mesh footprint so the longest side is 1.0 world unit; the
    # viewer then scales it up once and every image frames identically.
    aspect = src_w / float(src_h)
    if aspect >= 1.0:
        plane_w, plane_d = 1.0, 1.0 / aspect
    else:
        plane_w, plane_d = aspect, 1.0

    buffer = np.ascontiguousarray(grid, dtype="<f4").tobytes()

    return TerrainGrid(
        width=grid_w,
        height=grid_h,
        heights_b64=base64.b64encode(buffer).decode("ascii"),
        encoding="float32-le-base64",
        min_height=float(grid.min()),
        max_height=float(grid.max()),
        mean_height=float(grid.mean()),
        aspect_ratio=float(aspect),
        plane_width=float(plane_w),
        plane_depth=float(plane_d),
        source_width=int(src_w),
        source_height=int(src_h),
    )
