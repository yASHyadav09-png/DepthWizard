"""HTTP surface for the DepthWizard Phase 3 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

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
async def process(image: UploadFile = File(...), gsd_m: float | None = Form(None),
                  gcps: UploadFile | None = File(None), gcp_model: str = Form("offset")) -> dict:
    """JPG/PNG: RGB -> nDSM (metres) -> terrain. `gsd_m` (optional): metres per pixel, if known.
    GeoTIFF: CRS/transform/GSD from the file -> nDSM + GLO-30 DTM -> absolute DSM (EGM2008);
    optional `gcps` CSV (x,y,z in the image CRS or lon,lat,z; z in m EGM2008), `gcp_model`
    offset|plane."""
    try:
        data = await image.read()
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"Could not read the uploaded file: {exc}") from exc
    finally:
        await image.close()

    suffix = Path(image.filename or "").suffix.lower()
    if suffix in settings.GEO_EXTENSIONS:
        from app.services.geo_pipeline import run_geo_pipeline
        from app.utils import image_io

        image_io.validate_upload(image.filename, image.content_type, data)
        if gcp_model not in ("offset", "plane"):
            raise InvalidImageError("gcp_model must be 'offset' or 'plane'.")
        gcp_bytes = await gcps.read() if gcps is not None else None
        return await to_thread.run_sync(lambda: run_geo_pipeline(data, image.filename, gcp_bytes, gcp_model))
    if gcps is not None:
        raise InvalidImageError("GCPs need a georeferenced GeoTIFF input.")
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


@router.post("/results/{job_id}/validate", tags=["validation"])
async def validate_job(job_id: str, reference: UploadFile = File(...), target: str = Form("auto")) -> dict:
    """Compare the job's heights with a reference: a .npy on the same pixel grid (m, NaN = no data)
    or, for georeferenced jobs, a GeoTIFF that is reprojected onto the job grid. `target`:
    auto | dsm | ndsm (auto = the job's product: DSM for georeferenced jobs with a DEM)."""
    from app.services.validation import validate

    storage.validate_job_id(job_id)
    try:
        data = await reference.read()
    finally:
        await reference.close()
    if len(data) > settings.MAX_UPLOAD_BYTES * 4:
        raise InvalidImageError("Reference file is too large.")
    if target not in ("auto", "dsm", "ndsm"):
        raise InvalidImageError("target must be auto, dsm or ndsm.")
    return await to_thread.run_sync(lambda: validate(job_id, data, reference.filename, target))


@router.get("/jobs", response_model=JobListResponse, tags=["pipeline"])
def jobs() -> JobListResponse:
    return JobListResponse(jobs=storage.list_jobs())


__all__ = ["router"]
