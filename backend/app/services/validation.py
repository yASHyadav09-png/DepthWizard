"""Validate a job's predicted nDSM against a user-supplied reference height raster (Phase 7d).

Reference formats:
  * NumPy .npy float array in METRES on exactly the prediction's pixel grid; NaN = no data.
  * GeoTIFF (georeferenced jobs only): reprojected onto the job grid (bilinear, continuous
    field). Vertical datum: if the reference CRS declares NAVD88 it is converted to EGM2008
    with PROJ (constant offset at the image centre); otherwise it is ASSUMED to be EGM2008,
    and the response says so.
Target: the job's DSM (elevation) for georeferenced jobs with a DEM, otherwise its nDSM;
`target` can force "ndsm" or "dsm".

Metrics use the same code and definitions as the model evaluation
(depthwizard.eval.metrics): pooled over valid pixels,
    MAE = mean|p-g|, RMSE = sqrt(mean (p-g)^2), bias = mean(p-g), Pearson r.
"""

from __future__ import annotations

import base64
import io

import matplotlib
import numpy as np
from PIL import Image

from app.services import storage, terrain_generator
from app.utils.errors import InvalidImageError

ERROR_LIMIT_M = 10.0          # colour scale of the error map and 3D error layer: +-10 m
VALIDATION_JSON = "validation.json"
ERROR_PNG = "validation_error.png"


def load_reference(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    try:
        ref = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Reference must be a NumPy .npy height array in metres: {exc}") from exc
    if ref.ndim == 3 and ref.shape[0] == 1:
        ref = ref[0]
    if ref.ndim != 2 or ref.dtype.kind not in "fiu":
        raise InvalidImageError(f"Reference must be a 2-D numeric array, got shape {ref.shape}, dtype {ref.dtype}.")
    if tuple(ref.shape) != tuple(shape):
        raise InvalidImageError(
            f"Reference grid {ref.shape[1]}x{ref.shape[0]} px does not match the prediction grid "
            f"{shape[1]}x{shape[0]} px. Phase 7d needs the same pixel grid (GeoTIFF reprojection: Phase 5).")
    return ref.astype(np.float32)


def load_geotiff_reference(data: bytes, hp: dict, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    import pyproj
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    if not hp.get("crs_wkt"):
        raise InvalidImageError("GeoTIFF references need a georeferenced job (upload a GeoTIFF image).")
    try:
        with MemoryFile(data) as mem, mem.open() as src:
            if src.crs is None:
                raise InvalidImageError("Reference GeoTIFF has no CRS.")
            arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            src_t, src_crs_full = src.transform, pyproj.CRS(src.crs.to_wkt())
    except InvalidImageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Reference is not a readable GeoTIFF: {exc}") from exc
    horiz, vert = src_crs_full, None
    if src_crs_full.is_compound:
        horiz, vert = src_crs_full.sub_crs_list[0], src_crs_full.sub_crs_list[1]
    info = {"crs": horiz.to_string(), "vertical_crs": vert.name if vert is not None else None}
    if vert is not None and "NAVD88" in vert.name:
        pyproj.network.set_network_enabled(True)
        lon, lat = np.mean(np.array(hp["corners_lonlat"]), axis=0)
        t = pyproj.Transformer.from_crs("EPSG:4269+5703", "EPSG:4326+3855", always_xy=True)
        off = t.transform(float(lon), float(lat), 300.0)[2] - 300.0
        arr = arr + np.float32(off)
        info["datum_conversion"] = f"NAVD88 -> EGM2008, constant offset {off:+.3f} m at the image centre (PROJ)"
    elif vert is not None and "EGM2008" in vert.name:
        info["datum_conversion"] = "none (reference already EGM2008)"
    else:
        info["datum_conversion"] = "none: reference vertical datum not declared; ASSUMED EGM2008"
    dst = np.full(shape, np.nan, np.float32)
    reproject(arr, dst, src_transform=src_t, src_crs=rasterio.crs.CRS.from_wkt(horiz.to_wkt()),
              dst_transform=Affine(*hp["transform"]), dst_crs=rasterio.crs.CRS.from_wkt(hp["crs_wkt"]),
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    info["resampling"] = "bilinear onto the job grid"
    return dst, info


def validate(job_id: str, data: bytes, filename: str | None, target: str = "auto") -> dict:
    from depthwizard.eval.evaluate import BIN_NAMES, HEIGHT_EDGES
    from depthwizard.eval.metrics import HeightMetrics

    meta = storage.load_result(job_id)
    hp = meta["height_product"]
    kind = hp.get("kind", "ndsm")
    tgt = kind if target == "auto" else target
    if tgt == "dsm" and kind != "dsm":
        raise InvalidImageError("This job has no DSM (not georeferenced or no DEM); validate against 'ndsm'.")
    arr_name = storage.DSM_NPY if tgt == "dsm" else storage.HEIGHT_NPY
    pred = np.load(storage.job_dir(job_id) / arr_name).astype(np.float32)
    ref_info: dict = {}
    if (filename or "").lower().endswith((".tif", ".tiff")):
        ref, ref_info = load_geotiff_reference(data, hp, pred.shape)
    else:
        ref = load_reference(data, pred.shape)
    valid = np.isfinite(ref) & np.isfinite(pred)
    if valid.sum() < 100:
        raise InvalidImageError("Reference has fewer than 100 valid (finite) pixels.")

    ref0 = np.where(valid, ref, 0.0).astype(np.float32)
    pred0 = np.where(valid, pred, 0.0).astype(np.float32)
    if tgt == "ndsm":
        bins = np.digitize(ref0, HEIGHT_EDGES[1:-1]).astype(np.uint8)
        hm = HeightMetrics(label_names=BIN_NAMES)
    else:  # elevation: height bands are meaningless; one overall block
        bins = np.zeros(ref0.shape, np.uint8)
        hm = HeightMetrics(label_names={0: "all"})
    hm.update(pred0, ref0, valid, bins)
    res = hm.result()

    err = np.where(valid, pred0 - ref0, np.nan).astype(np.float32)
    t = np.clip((np.nan_to_num(err, nan=0.0) + ERROR_LIMIT_M) / (2 * ERROR_LIMIT_M), 0, 1)
    rgb = matplotlib.colormaps["RdBu_r"](t)[..., :3]
    rgb[~valid] = 0.35
    storage.save_image(job_id, ERROR_PNG, Image.fromarray((rgb * 255).astype(np.uint8)))

    grid = meta["terrain"]
    # error on the SAME grid as the 3D mesh (area-averaged over valid pixels), NaN where no data
    num = terrain_generator._resample(np.nan_to_num(err, nan=0.0), grid["width"], grid["height"])
    den = terrain_generator._resample(valid.astype(np.float32), grid["width"], grid["height"])
    err_grid = np.where(den >= 0.5, num / np.maximum(den, 1e-6), np.nan).astype("<f4")

    out = {
        "job_id": job_id,
        "reference": {"filename": filename or "reference.npy", "valid_pixels": int(valid.sum()),
                      "valid_fraction": float(valid.mean()), "units": "m", **ref_info},
        "target": tgt,
        "definition": f"error = predicted {tgt.upper()} - reference, pooled over valid pixels",
        "overall": res["overall"],
        "per_height_band": res["per_class"],
        "error_map": storage.asset_url(job_id, ERROR_PNG),
        "error_limit_m": ERROR_LIMIT_M,
        "error_grid": {"width": grid["width"], "height": grid["height"], "encoding": "float32-le-base64, NaN = no data",
                       "values_b64": base64.b64encode(np.ascontiguousarray(err_grid).tobytes()).decode("ascii")},
    }
    storage.save_metadata_file(job_id, VALIDATION_JSON, {k: v for k, v in out.items() if k != "error_grid"})
    return out
