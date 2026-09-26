"""DEM -> approximate DTM on the image grid (Phase 5).

Source: Copernicus GLO-30 (30 m, heights relative to the EGM2008 geoid), cached locally.
GLO-30 is a *surface* model from X-band radar: it partly contains buildings and canopy,
smoothed to 30 m. It is NOT a bare-earth DTM. We therefore approximate the ground by:

  1. fill no-data (e.g. tile seams) by nearest-neighbour,
  2. morphological OPENING (grey erosion then dilation) with a window of OPEN_WINDOW_M
     metres, which removes raised features narrower than the window (buildings, tree
     clumps) while keeping terrain wider than it; the window is set in metres and
     converted per axis to DEM pixels (degree pixels are not square),
  3. Gaussian smoothing (SMOOTH_SIGMA_PX DEM pixels) to remove the opening's plateaus,
  4. reprojection onto the image grid with BILINEAR resampling (continuous field).

The window (150 m) is fixed a priori (larger than typical building footprints, smaller than
terrain features) and is not tuned on the evaluation area. Limitations: canopy or
buildings wider than the window survive; narrow ridges and hilltops are lowered.
Vertical datum: EGM2008 (inherited from GLO-30); horizontal: the DEM's EPSG:4326.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy import ndimage

import os

# Ground-filter window. Default fixed a priori at 150 m (Phase 5 plan). Configurable because
# the Pittsburgh evaluation found the filter HURTS on steep terrain (docs/phase5_geospatial.md);
# the choice is pending a decision and multi-area evidence (Phase 6). 0 disables the filter.
OPEN_WINDOW_M = float(os.environ.get("DW_DEM_OPEN_WINDOW_M", "150"))
SMOOTH_SIGMA_PX = 1.0
VERTICAL_DATUM = "EGM2008 geoid height (EPSG:3855)"
SOURCE = "Copernicus GLO-30 DEM"


@dataclass
class DemOnGrid:
    dtm: np.ndarray            # ground estimate on the image grid (m, EGM2008)
    dem_raw: np.ndarray        # unfiltered GLO-30 on the image grid (m, EGM2008), for diagnostics
    covered: np.ndarray        # bool: image pixels inside the DEM's valid coverage
    info: dict


def _fill_nodata(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return a
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def ground_filter(dem: np.ndarray, px_x_m: float, px_y_m: float,
                  window_m: float = OPEN_WINDOW_M, sigma_px: float = SMOOTH_SIGMA_PX) -> np.ndarray:
    if window_m <= 0:
        return dem.astype(np.float32)
    size = (max(1, int(round(window_m / px_y_m)) | 1), max(1, int(round(window_m / px_x_m)) | 1))  # odd, (rows, cols)
    opened = ndimage.grey_opening(dem, size=size, mode="nearest")
    return ndimage.gaussian_filter(opened, sigma_px, mode="nearest").astype(np.float32) if sigma_px > 0 else opened


def dem_pixel_size_m(src_transform: Affine, src_crs: CRS, lat_deg: float) -> tuple[float, float]:
    if src_crs.is_geographic:
        m_per_deg = 111_320.0
        return abs(src_transform.a) * m_per_deg * np.cos(np.radians(lat_deg)), abs(src_transform.e) * m_per_deg
    unit = src_crs.linear_units_factor[1] if src_crs.linear_units_factor else 1.0
    return abs(src_transform.a) * unit, abs(src_transform.e) * unit


def dtm_on_grid(dem_path, dst_crs: CRS, dst_transform: Affine, shape: tuple[int, int],
                window_m: float = OPEN_WINDOW_M) -> DemOnGrid:
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        src_crs, src_t = src.crs, src.transform
        lat = src.xy(src.height // 2, src.width // 2)[1] if src_crs.is_geographic else 0.0
    valid = np.isfinite(dem) & ((dem != nodata) if nodata is not None else True)
    n_filled = int((~valid).sum())
    dem = _fill_nodata(dem, valid)
    px_x, px_y = dem_pixel_size_m(src_t, src_crs, lat)
    ground = ground_filter(dem, px_x, px_y, window_m)

    def to_grid(a, resampling=Resampling.bilinear, fill=np.nan):
        out = np.full(shape, fill, np.float32)
        reproject(a, out, src_transform=src_t, src_crs=src_crs, dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=resampling, src_nodata=None, dst_nodata=fill)
        return out

    dtm = to_grid(ground)
    raw = to_grid(dem)
    cov = to_grid(valid.astype(np.float32), Resampling.nearest, 0.0) > 0.5
    covered = cov & np.isfinite(dtm)
    info = {"source": SOURCE, "file": str(dem_path), "vertical_datum": VERTICAL_DATUM,
            "horizontal_crs": src_crs.to_string(), "dem_pixel_m": [round(px_x, 2), round(px_y, 2)],
            "ground_filter": {"method": "grey opening + gaussian", "window_m": window_m, "sigma_px": SMOOTH_SIGMA_PX},
            "resampling": "bilinear", "nodata_filled_px": n_filled, "coverage_fraction": float(covered.mean())}
    return DemOnGrid(dtm=dtm, dem_raw=raw, covered=covered, info=info)
