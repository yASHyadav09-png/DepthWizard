"""Statistics and 2D visualisations of a metric height field (nDSM, metres)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib
import numpy as np
from matplotlib.colors import LightSource
from PIL import Image


@dataclass(frozen=True)
class HeightStats:
    """Statistics of the predicted height above ground, in metres."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    p05: float
    p95: float
    p99: float
    frac_above_2m: float
    units: str = "m"

    def to_dict(self) -> dict:
        return asdict(self)


def sanitize(field: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 m so downstream maths is safe."""
    return np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def compute_stats(heights_m: np.ndarray) -> HeightStats:
    h = sanitize(heights_m).astype(np.float64)
    p05, median, p95, p99 = np.percentile(h, [5, 50, 95, 99])
    return HeightStats(min=float(h.min()), max=float(h.max()), mean=float(h.mean()),
                       median=float(median), std=float(h.std()), p05=float(p05), p95=float(p95),
                       p99=float(p99), frac_above_2m=float((h > 2.0).mean()))


def display_range(heights_m: np.ndarray) -> tuple[float, float]:
    """Colour-scale range shared by the 2D map, the 3D elevation mode and the legend:
    0 m up to the 99th percentile (at least 5 m, so flat scenes aren't all red)."""
    return 0.0, max(5.0, float(np.percentile(sanitize(heights_m), 99)))


def colorize(heights_m: np.ndarray, vmin: float, vmax: float, cmap_name: str = "magma") -> Image.Image:
    t = np.clip((sanitize(heights_m) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (matplotlib.colormaps[cmap_name](t)[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def hillshade(heights_m: np.ndarray, gsd_m: float, vmin: float, vmax: float,
              cmap_name: str = "gist_earth") -> Image.Image:
    """Shaded relief with TRUE vertical scale: heights and pixel spacing are both metres."""
    h = sanitize(heights_m).astype(np.float64)
    light = LightSource(azdeg=315, altdeg=45)
    try:
        shaded = light.shade(h, cmap=matplotlib.colormaps[cmap_name], vmin=vmin, vmax=vmax,
                             vert_exag=1.0, dx=gsd_m, dy=gsd_m, blend_mode="soft")
        return Image.fromarray((shaded[..., :3] * 255.0).round().astype(np.uint8), mode="RGB")
    except Exception:  # noqa: BLE001 - degenerate (flat) fields
        return colorize(h, vmin, vmax, cmap_name)
