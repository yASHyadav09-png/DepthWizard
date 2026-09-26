"""HTTP surface for the DepthWizard Phase 3 pipeline."""

from __future__ import annotations

import logging

from anyio import to_thread
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from app.api.schemas import (
    HealthResponse,
    JobListResponse,
    ProcessResponse,
)
from app.config import settings
from app.services import storage
from app.services.height_estimator import get_height_estimator
from app.services.pipeline import run_pipeline
from app.utils.errors import InvalidImageError, JobNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe that also reports which device inference will run on."""
    info = get_height_estimator().info()
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        version=settings.VERSION,
        stage=settings.APP_STAGE,
        model_name=info["model_label"],
        model_checkpoint=info["run"],
        model_loaded=info["loaded"],
        device=info["device_label"],
        cuda_available=info["cuda_available"],
        torch_version=info["torch_version"],
        terrain_resolution=settings.TERRAIN_RESOLUTION,
        max_upload_mb=settings.MAX_UPLOAD_BYTES // (1024 * 1024),
    )


@router.post("/process", response_model=ProcessResponse, tags=["pipeline"])
async def process(image: UploadFile = File(...), gsd_m: float | None = Form(None)) -> dict:
    """Run RGB -> nDSM (metres) -> terrain for one uploaded image.

    `gsd_m` (optional): ground resolution in metres per pixel, if known."""
    try:
        data = await image.read()
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Could not read the uploaded file: {exc}") from exc
    finally:
        await image.close()

    # Inference is CPU/GPU-bound and blocking; keep the event loop responsive.
    return await to_thread.run_sync(
        lambda: run_pipeline(data, image.filename, image.content_type, gsd_m)
    )


@router.get("/results/{job_id}", response_model=ProcessResponse, tags=["pipeline"])
def results(job_id: str) -> dict:
    """Re-fetch a completed job's payload (statistics, assets, terrain)."""
    return storage.load_result(job_id)


@router.get("/results/{job_id}/height-array", tags=["pipeline"])
def height_array(job_id: str) -> FileResponse:
    """Download the full-resolution nDSM (float32 metres) as .npy."""
    path = storage.job_dir(job_id) / storage.HEIGHT_NPY
    if not path.is_file():
        raise JobNotFoundError(f"No height array stored for job '{job_id}'.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"depthwizard_{job_id}_ndsm_m.npy",
    )


@router.get("/jobs", response_model=JobListResponse, tags=["pipeline"])
def jobs() -> JobListResponse:
    return JobListResponse(jobs=storage.list_jobs())


__all__ = ["router"]
