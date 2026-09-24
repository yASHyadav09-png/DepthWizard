"""Stage 1 pipeline orchestrator.

    RGB image
      -> Depth Anything V2 Small          (services.depth_estimator)
      -> relative depth                   (services.depth_processor)
      -> relative height / rDSM           (services.depth_processor)
      -> terrain grid                     (services.terrain_generator)
      -> artefacts on disk                (services.storage)

EXTENSION POINTS for later stages are marked with `# [stage-N]`. Each future
step slots in as another service module called from `run_pipeline` without
touching the API layer:
  [stage-2] GAMUS semantic segmentation + RGB/geometry feature fusion
  [stage-3] SRTM / GCP scale calibration -> absolute metric DSM
  [stage-4] GeoTIFF IO + RMSE/MAE validation against LiDAR
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.config import settings
from app.services import depth_processor, storage, terrain_generator
from app.services.depth_estimator import get_depth_estimator
from app.utils import image_io

logger = logging.getLogger(__name__)

STAGE = 1
PIPELINE_STEPS = [
    "load_image",
    "depth_estimation",
    "relative_height",
    "terrain_generation",
    "artefact_export",
]
FUTURE_STEPS = [
    "gamus_semantic_refinement",
    "dem_gcp_scale_calibration",
    "absolute_dsm",
    "geotiff_export",
    "lidar_validation",
]


def run_pipeline(data: bytes, filename: str | None, content_type: str | None) -> dict:
    """Run the full Stage 1 pipeline and return the API response payload."""
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    # --- 1. decode + validate -------------------------------------------
    step = time.perf_counter()
    image_io.validate_upload(filename, content_type, data)
    original = image_io.load_rgb_image(data)
    timings["load_image_ms"] = (time.perf_counter() - step) * 1000.0

    job_id = storage.new_job_id()
    logger.info(
        "[%s] %s (%dx%d, %.1f KB)",
        job_id, filename or "upload", original.width, original.height, len(data) / 1024,
    )

    # --- 2. monocular depth ---------------------------------------------
    inference_image = image_io.limit_side(original, settings.MAX_INFERENCE_SIDE)
    estimator = get_depth_estimator()
    depth_result = estimator.predict(inference_image)
    timings["depth_inference_ms"] = depth_result.inference_ms

    # --- 3. relative depth -> relative height (rDSM) ---------------------
    step = time.perf_counter()
    relative_depth = depth_processor.normalize(depth_result.depth)
    relative_height = depth_processor.to_relative_height(
        depth_result.depth, is_inverse_depth=depth_result.is_inverse_depth
    )
    # [stage-2] relative_height = semantic_refiner.refine(relative_height, original)
    # [stage-3] absolute_dsm = scale_calibrator.calibrate(relative_height, dem_or_gcps)
    stats = depth_processor.compute_stats(relative_height)
    timings["height_processing_ms"] = (time.perf_counter() - step) * 1000.0

    # --- 4. terrain grid -------------------------------------------------
    step = time.perf_counter()
    terrain = terrain_generator.generate_terrain(
        relative_height, resolution=settings.TERRAIN_RESOLUTION
    )
    timings["terrain_generation_ms"] = (time.perf_counter() - step) * 1000.0

    # --- 5. artefacts ----------------------------------------------------
    step = time.perf_counter()
    texture = image_io.limit_side(original, settings.MAX_TEXTURE_SIDE)
    assets = {
        "original": storage.save_image(job_id, storage.ORIGINAL_PNG, original),
        "texture": storage.save_image(job_id, storage.TEXTURE_JPG, texture),
        "depth_map": storage.save_image(
            job_id, storage.DEPTH_PNG, depth_processor.colorize(relative_depth, "inferno")
        ),
        "relative_dsm": storage.save_image(
            job_id, storage.DSM_PNG, depth_processor.hillshaded_dsm(relative_height)
        ),
        "height_array": storage.save_array(job_id, storage.HEIGHT_NPY, relative_height),
    }
    timings["artefact_export_ms"] = (time.perf_counter() - step) * 1000.0
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0

    # --- 6. payload + metadata ------------------------------------------
    payload = {
        "job_id": job_id,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": STAGE,
        "stage_label": settings.APP_STAGE,
        "source": {
            "filename": filename or "upload",
            "bytes": len(data),
            "width": original.width,
            "height": original.height,
            "inference_width": inference_image.width,
            "inference_height": inference_image.height,
        },
        "model": {
            "name": depth_result.model_label,
            "checkpoint": depth_result.model_name,
            "type": "relative monocular depth",
            "precision": depth_result.dtype,
            "device": depth_result.device,
            "device_label": depth_result.device_label,
        },
        "statistics": {
            "image_width": original.width,
            "image_height": original.height,
            "min_relative_height": stats.min,
            "max_relative_height": stats.max,
            "mean_relative_height": stats.mean,
            "median_relative_height": stats.median,
            "std_relative_height": stats.std,
            "p05_relative_height": stats.p05,
            "p95_relative_height": stats.p95,
            "height_units": stats.units,
            "model_name": depth_result.model_label,
            "processing_device": depth_result.device_label,
            "is_metric": False,
            "georeferenced": False,
        },
        "assets": assets,
        "terrain": terrain.to_dict(),
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "disclaimer": "Relative Height - Not Metric",
        "metadata": {
            # The height field is produced at the inference resolution, which is
            # capped at DW_MAX_INFERENCE_SIDE. For a large source image the .npy
            # is therefore smaller than the original raster - the model's native
            # prediction is only 518 px anyway, so upsampling would add no
            # information. These two fields state the array's true shape.
            "height_array_width": int(relative_height.shape[1]),
            "height_array_height": int(relative_height.shape[0]),
            "model": depth_result.model_label,
            "checkpoint": depth_result.model_name,
            "type": "relative monocular depth",
            "georeferenced": False,
            "height_units": "relative",
            "crs": None,
            "stage": STAGE,
            "pipeline": PIPELINE_STEPS,
            "planned_future_steps": FUTURE_STEPS,
            "app_version": settings.VERSION,
        },
    }

    storage.save_metadata(job_id, payload)
    logger.info("[%s] done in %.0f ms", job_id, timings["total_ms"])
    return payload
