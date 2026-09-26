"""Local Copernicus GLO-30 cache for the app (Phase 5).

The live demo must not depend on internet access, so DEMs are read from local files:
  * data/dem/*.tif                 (clips fetched by the app or by scripts/fetch_geo_aoi.py)
  * data/geo/<aoi>/glo30_dem.tif   (evaluation areas)
A cached DEM is used if it covers the image bounds plus the ground-filter margin.
If none does and fetching is allowed, a clip (image bounds + margin) is fetched once from
Microsoft Planetary Computer (anonymous) and cached under data/dem/. Otherwise the caller
falls back to an nDSM-only product and says so.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from ..data.gamus import ROOT

DEM_DIR = ROOT / "data" / "dem"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _candidates() -> list[Path]:
    return sorted(DEM_DIR.glob("*.tif")) + sorted((ROOT / "data" / "geo").glob("*/glo30_dem.tif"))


def needed_bounds_lonlat(crs, bounds, margin_m: float) -> tuple[float, float, float, float]:
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    lat = (s + n) / 2
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * max(np.cos(np.radians(lat)), 0.1))
    return w - dlon, s - dlat, e + dlon, n + dlat


def find_cached(bounds_ll) -> Path | None:
    w, s, e, n = bounds_ll
    for p in _candidates():
        try:
            with rasterio.open(p) as src:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        except Exception:  # noqa: BLE001
            continue
        if b[0] <= w and b[1] <= s and b[2] >= e and b[3] >= n:
            return p
    return None


def fetch_glo30(bounds_ll) -> Path:
    import planetary_computer
    import pystac_client
    from rasterio.merge import merge
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps([round(v, 4) for v in bounds_ll]).encode()).hexdigest()[:10]
    out = DEM_DIR / f"glo30_{bounds_ll[0]:.3f}_{bounds_ll[1]:.3f}_{key}.tif"
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=list(bounds_ll)).items())
    if not items:
        raise FileNotFoundError("No Copernicus GLO-30 tile covers this area.")
    srcs = [rasterio.open(i.assets["data"].href) for i in items]
    try:
        mosaic, tr = merge(srcs, bounds=bounds_ll, nodata=-32767.0)
        prof = srcs[0].profile.copy()
        prof.update(driver="GTiff", width=mosaic.shape[2], height=mosaic.shape[1], transform=tr, nodata=-32767.0,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(mosaic)
            dst.update_tags(SOURCE="Copernicus GLO-30 (Planetary Computer)", ITEMS=",".join(i.id for i in items),
                            VERTICAL_DATUM="EGM2008")
    finally:
        for s in srcs:
            s.close()
    return out


def get_dem(crs, bounds, margin_m: float, allow_fetch: bool) -> tuple[Path | None, str]:
    """Returns (path or None, how) with how in {'cache', 'fetched', 'unavailable: ...'}."""
    need = needed_bounds_lonlat(crs, bounds, margin_m)
    p = find_cached(need)
    if p is not None:
        return p, "cache"
    if not allow_fetch:
        return None, "unavailable: no cached DEM covers this area and fetching is disabled"
    try:
        return fetch_glo30(need), "fetched"
    except Exception as exc:  # noqa: BLE001
        return None, f"unavailable: {exc}"
