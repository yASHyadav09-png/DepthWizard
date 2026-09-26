"""Validate a job's predicted nDSM against a user-supplied reference height raster (Phase 7d).

Reference format (Phase 7d): a NumPy .npy float array in METRES with exactly the
prediction's pixel grid (H x W); NaN / +-inf = no data. GeoTIFF references with
reprojection arrive with the geospatial work (Phase 5).

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


def validate(job_id: str, data: bytes, filename: str | None) -> dict:
    from depthwizard.eval.evaluate import BIN_NAMES, HEIGHT_EDGES
    from depthwizard.eval.metrics import HeightMetrics

    pred = storage.load_height_array(job_id).astype(np.float32)
    ref = load_reference(data, pred.shape)
    valid = np.isfinite(ref)
    if valid.sum() < 100:
        raise InvalidImageError("Reference has fewer than 100 valid (finite) pixels.")

    ref0 = np.where(valid, ref, 0.0).astype(np.float32)
    bins = np.digitize(ref0, HEIGHT_EDGES[1:-1]).astype(np.uint8)
    hm = HeightMetrics(label_names=BIN_NAMES)
    hm.update(pred, ref0, valid, bins)
    res = hm.result()

    err = np.where(valid, pred - ref0, np.nan).astype(np.float32)
    t = np.clip((np.nan_to_num(err, nan=0.0) + ERROR_LIMIT_M) / (2 * ERROR_LIMIT_M), 0, 1)
    rgb = matplotlib.colormaps["RdBu_r"](t)[..., :3]
    rgb[~valid] = 0.35
    storage.save_image(job_id, ERROR_PNG, Image.fromarray((rgb * 255).astype(np.uint8)))

    meta = storage.load_result(job_id)
    grid = meta["terrain"]
    # error on the SAME grid as the 3D mesh (area-averaged over valid pixels), NaN where no data
    num = terrain_generator._resample(np.nan_to_num(err, nan=0.0), grid["width"], grid["height"])
    den = terrain_generator._resample(valid.astype(np.float32), grid["width"], grid["height"])
    err_grid = np.where(den >= 0.5, num / np.maximum(den, 1e-6), np.nan).astype("<f4")

    out = {
        "job_id": job_id,
        "reference": {"filename": filename or "reference.npy", "valid_pixels": int(valid.sum()),
                      "valid_fraction": float(valid.mean()), "units": "m"},
        "definition": "error = predicted - reference, pooled over valid reference pixels",
        "overall": res["overall"],
        "per_height_band": res["per_class"],
        "error_map": storage.asset_url(job_id, ERROR_PNG),
        "error_limit_m": ERROR_LIMIT_M,
        "error_grid": {"width": grid["width"], "height": grid["height"], "encoding": "float32-le-base64, NaN = no data",
                       "values_b64": base64.b64encode(np.ascontiguousarray(err_grid).tobytes()).decode("ascii")},
    }
    storage.save_metadata_file(job_id, VALIDATION_JSON, {k: v for k, v in out.items() if k != "error_grid"})
    return out
