"""Turn a raw relative depth map into a normalised relative height / rDSM.

STAGE 1 SCOPE: everything produced here is *relative and unitless*. No metric
scaling, DEM fusion or georeferencing happens yet -- those hooks land in later
stages (SRTM / GCP calibration -> absolute DSM).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
from matplotlib.colors import LightSource
from PIL import Image

# Percentile clip used to keep a handful of outlier pixels from flattening the
# whole scene. Classic robust min/max stretch.
CLIP_LOW = 1.0
CLIP_HIGH = 99.0


@dataclass(frozen=True)
class HeightStats:
    """Statistics over the relative height field. Unitless, never metres."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    p05: float
    p95: float
    units: str = "relative (0-1, unitless)"

    def to_dict(self) -> dict:
        return asdict(self)


def sanitize(depth: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with finite values so downstream maths is safe."""
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if finite.all():
        return depth
    if not finite.any():
        return np.zeros_like(depth)
    fill = float(depth[finite].mean())
    return np.where(finite, depth, fill).astype(np.float32)


def normalize(depth: np.ndarray, robust: bool = True) -> np.ndarray:
    """Scale an arbitrary depth array into [0, 1]."""
    depth = sanitize(depth)
    if robust:
        lo, hi = np.percentile(depth, [CLIP_LOW, CLIP_HIGH])
    else:
        lo, hi = float(depth.min()), float(depth.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-8:
        return np.zeros_like(depth)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_relative_height(depth: np.ndarray, is_inverse_depth: bool = True) -> np.ndarray:
    """Relative depth -> relative height field (rDSM), normalised to [0, 1].

    Depth Anything V2 regresses *inverse* depth: a large value means the pixel
    is close to the camera. For a downward-looking / oblique scene, closer to
    the sensor means higher off the ground, so the normalised inverse depth is
    already a relative height. When a future model emits true metric depth,
    pass ``is_inverse_depth=False`` and the field is flipped instead.
    """
    normalized = normalize(depth)
    if not is_inverse_depth:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def compute_stats(height: np.ndarray) -> HeightStats:
    h = sanitize(height).astype(np.float64)
    p05, median, p95 = np.percentile(h, [5, 50, 95])
    return HeightStats(
        min=float(h.min()),
        max=float(h.max()),
        mean=float(h.mean()),
        median=float(median),
        std=float(h.std()),
        p05=float(p05),
        p95=float(p95),
    )


def colorize(field01: np.ndarray, cmap_name: str = "inferno") -> Image.Image:
    """Apply a matplotlib colour map to a [0, 1] field and return an RGB image."""
    field01 = np.clip(sanitize(field01), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap_name](field01)
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def hillshaded_dsm(
    height01: np.ndarray,
    cmap_name: str = "gist_earth",
    vertical_exaggeration: float = 25.0,
) -> Image.Image:
    """Shaded-relief render of the rDSM -- the classic geospatial DSM look."""
    height01 = np.clip(sanitize(height01), 0.0, 1.0)
    light = LightSource(azdeg=315, altdeg=45)
    try:
        shaded = light.shade(
            height01.astype(np.float64),
            cmap=matplotlib.colormaps[cmap_name],
            vert_exag=vertical_exaggeration,
            blend_mode="soft",
        )
        rgb = (shaded[..., :3] * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:  # noqa: BLE001 - degenerate (flat) fields
        return colorize(height01, cmap_name)


def to_grayscale_png(field01: np.ndarray) -> Image.Image:
    """16-bit-ish grayscale preview; useful for quick visual QA."""
    field01 = np.clip(sanitize(field01), 0.0, 1.0)
    return Image.fromarray((field01 * 255.0).round().astype(np.uint8), mode="L")
