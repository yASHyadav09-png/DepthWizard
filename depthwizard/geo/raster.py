"""Georeferenced image input and GeoTIFF output (Phase 5).

Rules (verified from the data, never assumed):
  * CRS and affine transform come from the file. The ground resolution (GSD) is
    |pixel size| x the CRS's linear unit factor (metres, US survey feet, ...).
  * Rotated/sheared transforms are rejected (the viewer and metrics assume north-up
    pixels). Pixels must be square within 1%.
  * Geographic CRSs (degrees) have no metric pixel size: the image is reprojected to
    its local UTM zone at the equivalent ground resolution (bilinear for RGB). This is
    the ONLY case in which user imagery is resampled, and it is reported.
  * RGB = bands 1-3. uint8 is used as is; uint16 is stretched to uint8 with the
    0.2-99.8 percentiles per band (reported).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject


class GeoInputError(ValueError):
    pass


@dataclass
class GeoImage:
    rgb: np.ndarray                  # H x W x 3 uint8
    valid: np.ndarray                # H x W bool (data present in all 3 bands)
    crs: CRS                         # projected, metric (after optional reprojection)
    transform: Affine                # north-up, square pixels
    gsd_m: float
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in the image CRS."""
        h, w = self.shape
        return _bounds_lrbt(self.transform, w, h)


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def _to_uint8(bands: np.ndarray, notes: list[str]) -> np.ndarray:
    if bands.dtype == np.uint8:
        return bands
    out = np.empty(bands.shape, np.uint8)
    for k in range(bands.shape[0]):
        b = bands[k].astype(np.float64)
        lo, hi = np.percentile(b, [0.2, 99.8])
        out[k] = np.clip((b - lo) / max(hi - lo, 1e-9) * 255, 0, 255).round().astype(np.uint8)
    notes.append(f"{bands.dtype} bands stretched to uint8 (0.2-99.8 percentile per band)")
    return out


def read_geo_image(path) -> GeoImage:
    notes: list[str] = []
    with rasterio.open(path) as src:
        if src.crs is None:
            raise GeoInputError("The GeoTIFF has no CRS; it cannot be treated as georeferenced.")
        if src.count < 3:
            raise GeoInputError(f"Expected at least 3 bands (R, G, B), found {src.count}.")
        t = src.transform
        if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
            raise GeoInputError("Rotated or sheared GeoTIFFs are not supported (need a north-up transform).")
        bands = src.read([1, 2, 3])
        mask = src.dataset_mask() > 0
        if src.count > 3:
            notes.append(f"{src.count} bands: using bands 1-3 as R, G, B")
        crs, px, py = src.crs, abs(t.a), abs(t.e)

    rgb = _to_uint8(bands, notes)
    if crs.is_geographic:
        h, w = rgb.shape[1:]
        lon_c, lat_c = rasterio.transform.xy(t, h // 2, w // 2)
        dst_crs = utm_crs_for(lon_c, lat_c)
        dt, dw, dh = calculate_default_transform(crs, dst_crs, w, h, *_bounds_lrbt(t, w, h))
        out = np.zeros((3, dh, dw), np.uint8)
        for k in range(3):
            reproject(rgb[k], out[k], src_transform=t, src_crs=crs, dst_transform=dt, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
        vm = np.zeros((dh, dw), np.uint8)
        reproject(mask.astype(np.uint8), vm, src_transform=t, src_crs=crs, dst_transform=dt, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        notes.append(f"geographic CRS {crs.to_string()} reprojected to {dst_crs.to_string()} (bilinear RGB)")
        rgb, mask, crs, t = out, vm > 0, dst_crs, dt
        px, py = abs(t.a), abs(t.e)

    unit = crs.linear_units_factor[1] if crs.linear_units_factor else 1.0
    gsd_x, gsd_y = px * unit, py * unit
    if abs(gsd_x - gsd_y) / max(gsd_x, gsd_y) > 0.01:
        raise GeoInputError(f"Non-square pixels ({gsd_x:.3f} x {gsd_y:.3f} m) are not supported.")
    if unit != 1.0:
        notes.append(f"CRS linear unit '{crs.linear_units}' ({unit} m): pixel size converted to metres")
    return GeoImage(rgb=np.ascontiguousarray(rgb.transpose(1, 2, 0)), valid=mask, crs=crs, transform=t,
                    gsd_m=float(gsd_x), notes=notes)


def _bounds_lrbt(t: Affine, w: int, h: int) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of a north-up raster, as calculate_default_transform expects."""
    left, top = t.c, t.f
    right, bottom = left + t.a * w, top + t.e * h
    return left, min(top, bottom), right, max(top, bottom)


def write_geotiff(path, array: np.ndarray, crs: CRS, transform: Affine, nodata: float = -9999.0,
                  tags: dict | None = None, band_description: str | None = None) -> None:
    arr = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    prof = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32", crs=crs,
                transform=transform, nodata=nodata, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        if band_description:
            dst.set_band_description(1, band_description)
        dst.update_tags(1, units="metre")


def pixel_to_world(transform: Affine, col: float, row: float) -> tuple[float, float]:
    x, y = transform * (col, row)
    return float(x), float(y)


def gsd_ok(gsd_m: float) -> bool:
    return math.isfinite(gsd_m) and gsd_m > 0
