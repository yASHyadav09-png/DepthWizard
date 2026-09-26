"""Phase 3 pipeline orchestrator.

    RGB image (JPG/PNG, native resolution, never resampled)
      -> trained DA-V2-S nDSM model, tiled   (services.height_estimator)
      -> nDSM in metres + statistics          (services.height_processor)
      -> terrain grid in metres               (services.terrain_generator)
      -> artefacts on disk                    (services.storage)

The response carries a `height_product` block, the hand-off contract between the
model side and the viewer (docs/phase3_demo.md). Later phases fill the fields that
are null here:
  [phase-4] resampling to the training GSD      -> metric_validity "valid" for more inputs
  [phase-5] GeoTIFF IO, DEM/GCP calibration     -> kind "dsm", crs, transform, vertical_datum
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np

from app.config import settings
from app.services import height_processor, storage, terrain_generator
from app.services.height_estimator import get_height_estimator
from app.utils import image_io
from app.utils.errors import InvalidImageError

logger = logging.getLogger(__name__)

PHASE = 3
PIPELINE_STEPS = ["load_image", "ndsm_inference_tiled", "statistics", "terrain_generation", "artefact_export"]
FUTURE_STEPS = ["gsd_resampling", "geotiff_io", "dem_gcp_calibration", "absolute_dsm", "reference_validation"]


def resolve_gsd(gsd_m: float | None) -> tuple[float, str]:
    """Ground resolution used for the footprint, and where it came from."""
    from depthwizard.data.gamus import GAMUS_GSD_M
    if gsd_m is None:
        return GAMUS_GSD_M, "assumed_training_gsd"
    return float(gsd_m), "user"


def run_pipeline(data: bytes, filename: str | None, content_type: str | None,
                 gsd_m: float | None = None) -> dict:
    """Run the full Phase 3 pipeline and return the API response payload."""
    from depthwizard.inference import metric_validity

    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    # --- 1. decode + validate -------------------------------------------
    step = time.perf_counter()
    image_io.validate_upload(filename, content_type, data)
    if gsd_m is not None and not (0.01 <= gsd_m <= 100.0):
        raise InvalidImageError(f"gsd_m must be between 0.01 and 100 metres per pixel, got {gsd_m}.")
    original = image_io.load_rgb_image(data)
    if original.width * original.height > settings.MAX_INFERENCE_PIXELS:
        raise InvalidImageError(
            f"Image has {original.width * original.height / 1e6:.1f} MP; the limit is "
            f"{settings.MAX_INFERENCE_PIXELS / 1e6:.0f} MP (inference runs at native resolution).")
    rgb = np.asarray(original, dtype=np.uint8)
    timings["load_image_ms"] = (time.perf_counter() - step) * 1000.0

    job_id = storage.new_job_id()
    logger.info("[%s] %s (%dx%d, %.1f KB)", job_id, filename or "upload",
                original.width, original.height, len(data) / 1024)

    # --- 2. nDSM inference (native resolution, tiled) ------------------
    estimator = get_height_estimator()
    heights, infer_ms = estimator.predict(rgb)
    timings["ndsm_inference_ms"] = infer_ms

    # --- 3. statistics ---------------------------------------------------
    step = time.perf_counter()
    stats = height_processor.compute_stats(heights)
    display = height_processor.display_range(heights)
    gsd_used, gsd_source = resolve_gsd(gsd_m)
    validity, validity_note = metric_validity(gsd_m)
    timings["statistics_ms"] = (time.perf_counter() - step) * 1000.0

    # --- 4. terrain grid -------------------------------------------------
    step = time.perf_counter()
    terrain = terrain_generator.generate_terrain(heights, gsd_used, display,
                                                 resolution=settings.TERRAIN_RESOLUTION)
    timings["terrain_generation_ms"] = (time.perf_counter() - step) * 1000.0

    # --- 5. artefacts ----------------------------------------------------
    step = time.perf_counter()
    texture = image_io.limit_side(original, settings.MAX_TEXTURE_SIDE)
    assets = {
        "original": storage.save_image(job_id, storage.ORIGINAL_PNG, original),
        "texture": storage.save_image(job_id, storage.TEXTURE_JPG, texture),
        "height_map": storage.save_image(job_id, storage.HEIGHT_PNG,
                                         height_processor.colorize(heights, *display)),
        "hillshade": storage.save_image(job_id, storage.HILLSHADE_PNG,
                                        height_processor.hillshade(heights, gsd_used, *display)),
        "height_array": storage.save_array(job_id, storage.HEIGHT_NPY, heights),
    }
    timings["artefact_export_ms"] = (time.perf_counter() - step) * 1000.0
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0

    model = estimator.model_info()
    height_product = {
        "kind": "ndsm",
        "description": "height above ground",
        "height_units": "m",
        "width": int(heights.shape[1]),
        "height": int(heights.shape[0]),
        "height_array": assets["height_array"],
        "gsd_m": gsd_used,
        "gsd_source": gsd_source,
        "metric_validity": validity,
        "validity_note": validity_note,
        "nodata": None,             # every pixel receives a prediction
        "crs": None,                # [phase-5]
        "transform": None,          # [phase-5]
        "vertical_datum": None,     # [phase-5]
        "model": {"run": model["run"], "checkpoint_sha256": model["sha256"],
                  "git_commit": model["git_commit"], "val_rmse_m": model["val_rmse"]},
    }

    payload = {
        "job_id": job_id,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": PHASE,
        "stage_label": settings.APP_STAGE,
        "source": {"filename": filename or "upload", "bytes": len(data),
                   "width": original.width, "height": original.height},
        "model": {"name": settings.MODEL_LABEL, "run": model["run"], "checkpoint": model["checkpoint"],
                  "checkpoint_sha256": model["sha256"], "epoch": model["epoch"],
                  "val_rmse_m": model["val_rmse"], "git_commit": model["git_commit"],
                  "type": "supervised monocular nDSM regression",
                  "precision": "bf16" if model["device"].startswith("cuda") else "fp32",
                  "device": model["device"], "device_label": model["device_label"]},
        "height_product": height_product,
        "statistics": {**stats.to_dict(), "display_min": display[0], "display_max": display[1],
                       "is_metric": True, "metric_validity": validity, "georeferenced": False},
        "assets": assets,
        "terrain": terrain.to_dict(),
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "disclaimer": ("Estimated height above ground (m)" if validity == "valid"
                       else "Estimated height above ground (m) - resolution uncertain"),
        "metadata": {"pipeline": PIPELINE_STEPS, "planned_future_steps": FUTURE_STEPS,
                     "app_version": settings.VERSION, "phase": PHASE},
    }

    storage.save_metadata(job_id, payload)
    logger.info("[%s] done in %.0f ms (inference %.0f ms)", job_id, timings["total_ms"], infer_ms)
    return payload
