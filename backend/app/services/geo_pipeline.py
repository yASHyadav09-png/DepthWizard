"""Phase 5: georeferenced (GeoTIFF) uploads -> absolute DSM.

    GeoTIFF RGB --(CRS, transform, GSD from the file)--> nDSM (trained model, native grid)
              --(cached Copernicus GLO-30, ground filter, bilinear onto the grid)--> DTM
              --(optional GCP correction: offset or plane)--> DTM'
    DSM = DTM' + nDSM   [m, EGM2008 geoid heights]; GeoTIFF outputs keep the input CRS/transform.

If no DEM covers the image (offline and not cached), the product falls back to a
georeferenced nDSM (kind "ndsm") and says so.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from app.config import settings
from app.services import height_processor, storage, terrain_generator
from app.services.height_estimator import get_height_estimator
from app.utils.errors import InvalidImageError

logger = logging.getLogger(__name__)


def parse_gcps(data: bytes | None, crs) -> np.ndarray | None:
    """CSV with header x,y,z (image CRS) or lon,lat,z (WGS 84). z in metres, EGM2008."""
    if not data:
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"GCP file is not a readable CSV: {exc}") from exc
    if not rows:
        raise InvalidImageError("GCP file has no rows.")
    cols = {c.strip().lower(): c for c in rows[0]}
    try:
        if {"x", "y", "z"} <= cols.keys():
            pts = np.array([[float(r[cols["x"]]), float(r[cols["y"]]), float(r[cols["z"]])] for r in rows])
        elif {"lon", "lat", "z"} <= cols.keys():
            from pyproj import Transformer
            ll = np.array([[float(r[cols["lon"]]), float(r[cols["lat"]]), float(r[cols["z"]])] for r in rows])
            x, y = Transformer.from_crs("EPSG:4326", crs.to_wkt(), always_xy=True).transform(ll[:, 0], ll[:, 1])
            pts = np.column_stack([x, y, ll[:, 2]])
        else:
            raise InvalidImageError("GCP CSV needs columns x,y,z (image CRS) or lon,lat,z.")
    except ValueError as exc:
        raise InvalidImageError(f"GCP CSV contains non-numeric values: {exc}") from exc
    return pts


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def run_geo_pipeline(data: bytes, filename: str | None, gcp_csv: bytes | None = None,
                     gcp_model: str = "offset") -> dict:
    from depthwizard.geo import dem as demmod
    from depthwizard.geo.dem_cache import get_dem
    from depthwizard.geo.gcp import correction_surface, fit_correction
    from depthwizard.geo.pipeline import GeoResult, write_outputs
    from depthwizard.geo.raster import GeoInputError, read_geo_image
    from depthwizard.inference import metric_validity
    from pyproj import Transformer

    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    job_id = storage.new_job_id()
    jdir = storage.job_dir(job_id, create=True)
    src_path = jdir / storage.INPUT_TIF
    src_path.write_bytes(data)
    try:
        img = read_geo_image(src_path)
    except GeoInputError as exc:
        raise InvalidImageError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Could not read the GeoTIFF: {exc}") from exc
    h, w = img.shape
    if h * w > settings.MAX_INFERENCE_PIXELS:
        raise InvalidImageError(f"Image has {h * w / 1e6:.1f} MP; the limit is {settings.MAX_INFERENCE_PIXELS / 1e6:.0f} MP.")
    timings["load_image_ms"] = (time.perf_counter() - t0) * 1000

    estimator = get_height_estimator()
    ndsm, infer_ms = estimator.predict(img.rgb)
    timings["ndsm_inference_ms"] = infer_ms

    step = time.perf_counter()
    margin = max(3 * demmod.OPEN_WINDOW_M, 500.0)
    dem_path, dem_how = get_dem(img.crs, img.bounds, margin, settings.DEM_ALLOW_FETCH)
    gcps = parse_gcps(gcp_csv, img.crs)
    gcp_fit = None
    notes = list(img.notes)
    if dem_path is not None:
        demg = demmod.dtm_on_grid(dem_path, img.crs, img.transform, img.shape)
        dtm = demg.dtm
        if gcps is not None:
            try:
                gcp_fit = fit_correction(dtm, img.transform, gcps, gcp_model)
            except ValueError as exc:
                raise InvalidImageError(f"GCP correction failed: {exc}") from exc
            dtm = dtm + correction_surface(gcp_fit, img.transform, img.shape)
        valid = img.valid & demg.covered
        res = GeoResult(image=img, ndsm=np.where(img.valid, ndsm, np.nan).astype(np.float32),
                        dtm=np.where(demg.covered, dtm, np.nan).astype(np.float32),
                        dsm=np.where(valid, dtm + ndsm, np.nan).astype(np.float32), valid=valid,
                        dem_info={**demg.info, "how": dem_how}, gcp=gcp_fit, notes=notes)
        kind = "dsm"
    else:
        if gcps is not None:
            notes.append("GCPs ignored: no DEM available, so there is no DTM to correct")
        res = None
        kind = "ndsm"
        notes.append(f"No DEM: {dem_how}. Product is a georeferenced nDSM (height above ground) only.")
    timings["dem_dtm_ms"] = (time.perf_counter() - step) * 1000

    # ---- artefacts
    step = time.perf_counter()
    rgb_img = Image.fromarray(img.rgb)
    tags = {"MODEL_RUN": estimator.model_info()["run"], "MODEL_SHA256": estimator.model_info()["sha256"]}
    geotiffs = {}
    if res is not None:
        files = write_outputs(res, jdir, "depthwizard", tags)
        geotiffs = {k: storage.asset_url(job_id, p.name) for k, p in files.items()}
        surface = res.dsm
        storage.save_array(job_id, storage.DSM_NPY, res.dsm)
    else:
        from depthwizard.geo.raster import write_geotiff
        p = jdir / "depthwizard_ndsm.tif"
        write_geotiff(p, np.where(img.valid, ndsm, np.nan), img.crs, img.transform,
                      tags={**tags, "PRODUCT": "NDSM", "UNITS": "metre", "VERTICAL_DATUM": "height above local ground"},
                      band_description="nDSM: height above ground (model)")
        geotiffs = {"ndsm": storage.asset_url(job_id, p.name)}
        surface = np.where(img.valid, ndsm, np.nan).astype(np.float32)
    ndsm_arr = np.where(img.valid, ndsm, np.nan).astype(np.float32)
    height_url = storage.save_array(job_id, storage.HEIGHT_NPY, ndsm_arr)
    fin = np.isfinite(surface)
    fill = float(np.nanmin(surface)) if fin.any() else 0.0
    surf_filled = np.where(fin, surface, fill).astype(np.float32)
    ndsm_filled = np.nan_to_num(ndsm_arr, nan=0.0)

    if kind == "dsm":
        lo, hi = (float(np.percentile(surf_filled[fin], 1)), float(np.percentile(surf_filled[fin], 99)))
        display = (lo, max(hi, lo + 5.0))
    else:
        display = height_processor.display_range(ndsm_filled)
    terrain = terrain_generator.generate_terrain(surf_filled, img.gsd_m, display, settings.TERRAIN_RESOLUTION).to_dict()
    ndsm_grid = terrain_generator._resample(ndsm_filled, terrain["width"], terrain["height"])
    terrain["ndsm_b64"] = _b64(ndsm_grid)
    terrain["surface_kind"] = kind
    texture = rgb_img.copy()
    texture.thumbnail((settings.MAX_TEXTURE_SIDE, settings.MAX_TEXTURE_SIDE))
    ndsm_disp = height_processor.display_range(ndsm_filled)
    assets = {
        "original": storage.save_image(job_id, storage.ORIGINAL_PNG, rgb_img),
        "texture": storage.save_image(job_id, storage.TEXTURE_JPG, texture),
        "height_map": storage.save_image(job_id, storage.HEIGHT_PNG, height_processor.colorize(ndsm_filled, *ndsm_disp)),
        "hillshade": storage.save_image(job_id, storage.HILLSHADE_PNG,
                                        height_processor.hillshade(surf_filled, img.gsd_m, *display)),
        "height_array": height_url,
    }
    if kind == "dsm":
        assets["dsm_array"] = storage.asset_url(job_id, storage.DSM_NPY)
    timings["artefact_export_ms"] = (time.perf_counter() - step) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    to_ll = Transformer.from_crs(img.crs.to_wkt(), "EPSG:4326", always_xy=True)
    tr = img.transform
    corners = [tr * (0, 0), tr * (w, 0), tr * (w, h), tr * (0, h)]
    corners_ll = [list(to_ll.transform(x, y)) for x, y in corners]
    validity, validity_note = metric_validity(img.gsd_m)
    stats = height_processor.compute_stats(ndsm_filled)
    model = estimator.model_info()
    height_product = {
        "kind": kind,
        "description": "surface elevation (DSM = DTM + nDSM)" if kind == "dsm" else "height above ground (georeferenced)",
        "height_units": "m", "width": w, "height": h,
        "height_array": assets["dsm_array"] if kind == "dsm" else height_url,
        "ndsm_array": height_url,
        "gsd_m": img.gsd_m, "gsd_source": "geotiff", "metric_validity": validity, "validity_note": validity_note,
        "nodata": "NaN in arrays, -9999 in GeoTIFFs",
        "crs": img.crs.to_string(), "crs_epsg": img.crs.to_epsg(), "crs_wkt": img.crs.to_wkt(),
        "transform": [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f], "bounds": list(img.bounds), "corners_lonlat": corners_ll,
        "vertical_datum": demmod.VERTICAL_DATUM if kind == "dsm" else None,
        "dem": res.dem_info if res is not None else {"how": dem_how},
        "gcp": ({k: v for k, v in gcp_fit.items() if k != "residuals_before"} if gcp_fit else None),
        "geotiffs": geotiffs, "notes": notes,
        "model": {"run": model["run"], "checkpoint_sha256": model["sha256"], "git_commit": model["git_commit"],
                  "val_rmse_m": model["val_rmse"]},
    }
    elev = {}
    if kind == "dsm":
        e = res.dsm[np.isfinite(res.dsm)]
        g = res.dtm[np.isfinite(res.dtm)]
        elev = {"dsm_min": float(e.min()), "dsm_max": float(e.max()), "dtm_min": float(g.min()),
                "dtm_max": float(g.max()), "relief_m": float(g.max() - g.min())}
    payload = {
        "job_id": job_id, "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": 5, "stage_label": "Phase 5 - georeferenced input, absolute DSM",
        "source": {"filename": filename or "upload.tif", "bytes": len(data), "width": w, "height": h},
        "model": {"name": settings.MODEL_LABEL, "run": model["run"], "checkpoint": model["checkpoint"],
                  "checkpoint_sha256": model["sha256"], "epoch": model["epoch"], "val_rmse_m": model["val_rmse"],
                  "git_commit": model["git_commit"], "type": "supervised monocular nDSM regression",
                  "precision": "bf16" if model["device"].startswith("cuda") else "fp32", "device": model["device"],
                  "device_label": model["device_label"]},
        "height_product": height_product,
        "statistics": {**stats.to_dict(), "display_min": ndsm_disp[0], "display_max": ndsm_disp[1], "is_metric": True,
                       "metric_validity": validity, "georeferenced": True, "elevation": elev},
        "assets": assets, "terrain": terrain, "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "disclaimer": ("Surface elevation (m, EGM2008)" if kind == "dsm" else "Estimated height above ground (m)")
                      + ("" if validity == "valid" else " - resolution uncertain"),
        "metadata": {"phase": 5, "app_version": settings.VERSION},
    }
    storage.save_metadata(job_id, payload)
    logger.info("[%s] geo %s %dx%d gsd %.2f m, DEM %s, %.0f ms", job_id, kind, w, h, img.gsd_m, dem_how,
                timings["total_ms"])
    return payload
