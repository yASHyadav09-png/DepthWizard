"""Fetch a small georeferenced test area (Phase 5) from Microsoft Planetary Computer
(anonymous access) and cache it locally, so demos and tests run offline.

For an AOI (lon/lat centre + half-size in metres) writes to data/geo/<name>/:
    naip_rgb.tif      NAIP RGB, native CRS/resolution, windowed COG read (bands 1-3 = R, G, B)
    lidar_dsm.tif     USGS 3DEP LiDAR DSM (2 m) clipped to the AOI  -- reference
    lidar_dtm.tif     USGS 3DEP LiDAR DTM (2 m) clipped to the AOI  -- reference
    glo30_dem.tif     Copernicus GLO-30 DEM (30 m, EGM2008 heights), AOI + margin -- the DEM the app uses
    manifest.json     item ids, dates, CRSs, resolutions, source URLs (without SAS tokens)

Usage:  python scripts/fetch_geo_aoi.py --name pittsburgh --lon -79.985 --lat 40.425 --half 650 --naip-year 2019
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def aoi_bounds_lonlat(lon: float, lat: float, half_m: float) -> tuple[float, float, float, float]:
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * np.cos(np.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def clip(href: str, bounds_ll, out: Path, bands=None) -> dict:
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bounds_ll, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        idx = bands or list(range(1, src.count + 1))
        data = src.read(idx, window=win)
        prof = src.profile.copy()
        prof.update(driver="GTiff", width=win.width, height=win.height, count=len(idx),
                    transform=src.window_transform(win), compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256)
        prof.pop("photometric", None)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(data)
            dst.update_tags(**{k: v for k, v in src.tags().items() if len(str(v)) < 2000})
        return {"file": out.name, "crs": src.crs.to_string(), "res": list(src.res), "shape": [len(idx), win.height, win.width],
                "dtype": str(data.dtype), "nodata": src.nodata, "units": src.units or None,
                "source": href.split("?")[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half", type=float, default=650, help="AOI half-size in metres")
    ap.add_argument("--dem-margin", type=float, default=3000, help="extra DEM margin (m) for the ground filter")
    ap.add_argument("--naip-year", type=str)
    a = ap.parse_args()

    out = ROOT / "data" / "geo" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    bb = aoi_bounds_lonlat(a.lon, a.lat, a.half)
    bb_dem = aoi_bounds_lonlat(a.lon, a.lat, a.half + a.dem_margin)
    man = {"aoi": {"name": a.name, "lon": a.lon, "lat": a.lat, "half_m": a.half, "bounds_lonlat": bb}}

    naip = [i for i in cat.search(collections=["naip"], bbox=bb).items()
            if a.naip_year is None or i.properties.get("naip:year") == a.naip_year]
    naip.sort(key=lambda i: i.properties.get("datetime", ""))
    if not naip:
        sys.exit("no NAIP item")
    it = naip[-1]
    man["naip"] = {"item": it.id, "datetime": it.properties.get("datetime"), "gsd": it.properties.get("gsd"),
                   **clip(it.assets["image"].href, bb, out / "naip_rgb.tif", bands=[1, 2, 3])}
    print("NAIP", man["naip"]["item"], man["naip"]["crs"], man["naip"]["res"], man["naip"]["shape"])

    for kind in ("dsm", "dtm"):
        items = list(cat.search(collections=[f"3dep-lidar-{kind}"], bbox=bb).items())
        if not items:
            sys.exit(f"no 3DEP {kind}")
        it = items[0]
        man[f"lidar_{kind}"] = {"item": it.id, "datetime": it.properties.get("start_datetime") or it.properties.get("datetime"),
                                **clip(it.assets["data"].href, bb, out / f"lidar_{kind}.tif")}
        print(kind.upper(), it.id, man[f"lidar_{kind}"]["crs"][:60], man[f"lidar_{kind}"]["res"])

    # GLO-30 comes in 1x1 degree tiles; the AOI + margin can span several -> mosaic them
    from rasterio.merge import merge
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=bb_dem).items())
    srcs = [rasterio.open(i.assets["data"].href) for i in items]
    try:
        mosaic, tr = merge(srcs, bounds=bb_dem, nodata=-32767.0)
        prof = srcs[0].profile.copy()
        prof.update(driver="GTiff", width=mosaic.shape[2], height=mosaic.shape[1], transform=tr, nodata=-32767.0,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out / "glo30_dem.tif", "w", **prof) as dst:
            dst.write(mosaic)
        man["glo30"] = {"items": [i.id for i in items], "file": "glo30_dem.tif", "crs": srcs[0].crs.to_string(),
                        "res": list(srcs[0].res), "shape": list(mosaic.shape),
                        "vertical_datum": "EGM2008 geoid (Copernicus DEM specification)",
                        "nodata_px": int((mosaic == -32767.0).sum())}
    finally:
        for s in srcs:
            s.close()
    print("GLO-30", man["glo30"]["items"], man["glo30"]["crs"], man["glo30"]["res"], man["glo30"]["shape"],
          "nodata px", man["glo30"]["nodata_px"])
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
