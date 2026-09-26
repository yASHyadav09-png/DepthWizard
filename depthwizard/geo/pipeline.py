"""Georeferenced pipeline (Phase 5): GeoTIFF RGB -> nDSM -> DTM (DEM) -> absolute DSM.

    nDSM = trained model on the image's native grid (direct inference, Phase 4 decision)
    DTM  = ground-filtered GLO-30 on the same grid (dem.py), optionally GCP-corrected
    DSM  = DTM + nDSM            [m, EGM2008 geoid heights]

Outputs share the input's CRS and affine transform exactly (same grid, same pixel
registration). Pixels without image data or DEM coverage are no-data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dem import VERTICAL_DATUM, dtm_on_grid
from .gcp import correction_surface, fit_correction
from .raster import GeoImage, read_geo_image, write_geotiff


@dataclass
class GeoResult:
    image: GeoImage
    ndsm: np.ndarray
    dtm: np.ndarray
    dsm: np.ndarray
    valid: np.ndarray
    dem_info: dict
    gcp: dict | None = None
    notes: list[str] = field(default_factory=list)


def run_geo(image_path, predict, dem_path, gcps: np.ndarray | None = None, gcp_model: str = "offset") -> GeoResult:
    """`predict`: HxWx3 uint8 -> HxW nDSM (m), e.g. NDSMPredictor.predict."""
    img = read_geo_image(image_path)
    ndsm = predict(img.rgb).astype(np.float32)
    dem = dtm_on_grid(dem_path, img.crs, img.transform, img.shape)
    dtm = dem.dtm
    gcp = None
    if gcps is not None and len(gcps):
        gcp = fit_correction(dtm, img.transform, gcps, gcp_model)
        dtm = dtm + correction_surface(gcp, img.transform, img.shape)
    valid = img.valid & dem.covered
    nan = np.float32(np.nan)
    ndsm_o = np.where(img.valid, ndsm, nan)
    dtm_o = np.where(dem.covered, dtm, nan)
    dsm_o = np.where(valid, dtm + ndsm, nan)
    return GeoResult(image=img, ndsm=ndsm_o, dtm=dtm_o, dsm=dsm_o, valid=valid, dem_info=dem.info, gcp=gcp,
                     notes=list(img.notes))


def write_outputs(res: GeoResult, out_dir, stem: str, extra_tags: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {"GSD_M": res.image.gsd_m, "DEM_SOURCE": res.dem_info["source"],
            "GROUND_FILTER": res.dem_info["ground_filter"], "GCP": res.gcp["params"] if res.gcp else "none",
            **(extra_tags or {})}
    files = {}
    for name, arr, desc, datum in (
            ("dsm", res.dsm, "DSM: surface elevation", VERTICAL_DATUM),
            ("dtm", res.dtm, "DTM: ground elevation (filtered GLO-30)", VERTICAL_DATUM),
            ("ndsm", res.ndsm, "nDSM: height above ground (model)", "height above local ground (not a datum)")):
        p = out_dir / f"{stem}_{name}.tif"
        write_geotiff(p, arr, res.image.crs, res.image.transform, tags={**base, "PRODUCT": name.upper(),
                                                                          "VERTICAL_DATUM": datum, "UNITS": "metre"},
                      band_description=desc)
        files[name] = p
    return files
