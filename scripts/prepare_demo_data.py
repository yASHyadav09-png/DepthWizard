"""Assemble the offline demo inputs in sample_data/ from data that is already on disk (no downloads).

    sample_data/gamus_val/     GAMUS VAL tiles (0.33 m PNG + LiDAR nDSM reference)
                               -> created with scripts/export_demo_samples.py if missing
    sample_data/geo/           GeoTIFF demos from the Phase 5/6 evaluation areas:
        <aoi>_naip.tif             NAIP RGB, native CRS and 0.6 m resolution (upload this)
        <aoi>_lidar_dsm.tif        USGS 3DEP LiDAR DSM 2 m, NAVD88 (reference for "Validate")
        <aoi>_gcps_example.csv     5 example ground control points (x, y in the image CRS; z = LiDAR
                                   DTM converted to EGM2008), to demonstrate the GCP option

The DEMs the app needs for these areas stay in data/geo/<aoi>/glo30_dem.tif, where the backend's
DEM cache finds them; with DW_DEM_ALLOW_FETCH=0 the demo then works fully offline.
Evaluation areas are used here as DEMO inputs only; their numbers are already reported in Phase 5/6.

Usage:  python scripts/prepare_demo_data.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
GEO = ["pittsburgh", "carmel_suburb", "del_rio_arid"]     # hilly urban, suburban, arid
OUT = ROOT / "sample_data" / "geo"


def example_gcps(aoi_dir: Path, n: int = 5, seed: int = 0) -> str:
    from phase5_eval_aoi import navd88_to_egm2008_offset
    man = json.loads((aoi_dir / "manifest.json").read_text())
    off = navd88_to_egm2008_offset(man["aoi"]["lon"], man["aoi"]["lat"])["offset_m"]
    with rasterio.open(aoi_dir / "lidar_dtm.tif") as s:
        dtm = s.read(1, masked=True).filled(np.nan); t = s.transform
    with rasterio.open(aoi_dir / "lidar_dsm.tif") as s:
        dsm = s.read(1, masked=True).filled(np.nan)
    # open ground: DSM == DTM (no object), away from the edges
    ok = np.isfinite(dtm) & np.isfinite(dsm) & (np.abs(dsm - dtm) < 0.3)
    ok[:50] = ok[-50:] = False; ok[:, :50] = ok[:, -50:] = False
    cand = np.argwhere(ok)
    pick = cand[np.random.default_rng(seed).choice(len(cand), n, replace=False)]
    lines = ["x,y,z"]
    for r, c in pick:
        x, y = t * (c + 0.5, r + 0.5)
        lines.append(f"{x:.2f},{y:.2f},{dtm[r, c] + off:.2f}")
    return "\n".join(lines) + "\n"


def main():
    if not (ROOT / "sample_data" / "gamus_val").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_demo_samples.py")], check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for aoi in GEO:
        d = ROOT / "data" / "geo" / aoi
        if not (d / "naip_rgb.tif").exists():
            print(f"skip {aoi}: data/geo/{aoi} missing (scripts/fetch_geo_aoi.py)"); continue
        assert (d / "glo30_dem.tif").exists(), f"{aoi}: DEM missing, the demo would not be offline"
        shutil.copy2(d / "naip_rgb.tif", OUT / f"{aoi}_naip.tif")
        shutil.copy2(d / "lidar_dsm.tif", OUT / f"{aoi}_lidar_dsm.tif")
        (OUT / f"{aoi}_gcps_example.csv").write_text(example_gcps(d))
        print("wrote", aoi)


if __name__ == "__main__":
    main()
