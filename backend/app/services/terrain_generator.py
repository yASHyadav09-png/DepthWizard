"""Build a compact 3D terrain grid from the metric height field.

A 4000x3000 image is 12M pixels, far too many vertices for a browser. The nDSM is
area-averaged onto a grid (TERRAIN_RESOLUTION on the longest side) and shipped as a
base64 float32 buffer of heights IN METRES. The footprint is also in metres
(pixels x ground resolution), so the viewer can show the scene at true proportions.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TerrainGrid:
    """Everything Three.js needs to build the mesh."""

    width: int                 # grid columns
    height: int                # grid rows
    heights_b64: str           # row-major float32 metres, length width*height
    encoding: str              # buffer format hint for the client
    min_height: float
    max_height: float
    mean_height: float
    aspect_ratio: float        # source image width / height
    plane_width: float         # footprint in metres (source width  x gsd)
    plane_depth: float         # footprint in metres (source height x gsd)
    source_width: int          # pixel size of the nDSM the grid came from
    source_height: int
    gsd_m: float               # metres per source pixel used for the footprint
    display_min: float         # colour-scale range (m), shared with the legend
    display_max: float
    height_units: str = "m"

    def to_dict(self) -> dict:
        return asdict(self)


def _resample(field: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Area-average down / bilinear up, in float space."""
    src_h, src_w = field.shape
    if (src_w, src_h) == (target_w, target_h):
        return field.astype(np.float32)
    resample = (Image.Resampling.BOX if target_w <= src_w and target_h <= src_h
                else Image.Resampling.BILINEAR)
    img = Image.fromarray(field.astype(np.float32), mode="F")
    return np.asarray(img.resize((target_w, target_h), resample), dtype=np.float32)


def _grid_shape(src_w: int, src_h: int, resolution: int) -> tuple[int, int]:
    """Longest side becomes `resolution` (never upsampled); the other keeps the aspect."""
    resolution = max(16, int(resolution))
    if src_w >= src_h:
        width = min(resolution, src_w)
        height = max(2, round(width * src_h / src_w))
    else:
        height = min(resolution, src_h)
        width = max(2, round(height * src_w / src_h))
    return int(width), int(height)


def generate_terrain(heights_m: np.ndarray, gsd_m: float, display: tuple[float, float],
                     resolution: int = 512) -> TerrainGrid:
    field = np.asarray(heights_m, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D height field, got shape {field.shape}.")
    if not gsd_m > 0:
        raise ValueError(f"gsd_m must be positive, got {gsd_m}.")

    src_h, src_w = field.shape
    grid_w, grid_h = _grid_shape(src_w, src_h, resolution)
    grid = np.nan_to_num(_resample(field, grid_w, grid_h), nan=0.0, posinf=0.0, neginf=0.0)
    buffer = np.ascontiguousarray(grid, dtype="<f4").tobytes()

    return TerrainGrid(
        width=grid_w, height=grid_h,
        heights_b64=base64.b64encode(buffer).decode("ascii"), encoding="float32-le-base64",
        min_height=float(grid.min()), max_height=float(grid.max()), mean_height=float(grid.mean()),
        aspect_ratio=float(src_w / src_h),
        plane_width=float(src_w * gsd_m), plane_depth=float(src_h * gsd_m),
        source_width=int(src_w), source_height=int(src_h), gsd_m=float(gsd_m),
        display_min=float(display[0]), display_max=float(display[1]),
    )
