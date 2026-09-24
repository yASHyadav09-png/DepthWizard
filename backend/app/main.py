"""DepthWizard backend entrypoint.

SIH 2026 - PS 26175 (ISRO): Single-View Height Estimation and 3D Flythrough.
Stage 1: relative height / relative DSM from non-georeferenced JPG/PNG.

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.services.depth_estimator import get_depth_estimator
from app.utils.errors import DepthWizardError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("depthwizard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    estimator = get_depth_estimator()
    logger.info("%s %s - %s", settings.APP_NAME, settings.VERSION, settings.APP_STAGE)
    logger.info("Inference device: %s", estimator.device_label)
    if settings.PRELOAD_MODEL:
        try:
            estimator.load()
        except DepthWizardError as exc:
            # Don't kill the server: /api/health still reports the failure and
            # the next request retries the download.
            logger.error("Model preload failed: %s", exc.message)
    yield
    logger.info("DepthWizard shutting down.")


app = FastAPI(
    title="DepthWizard API",
    description=(
        "Single-view relative height estimation (SIH 2026, PS 26175 / ISRO). "
        "Stage 1 produces RELATIVE, NON-METRIC height only."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DepthWizardError)
async def depthwizard_error_handler(_: Request, exc: DepthWizardError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.exception("Pipeline failure: %s", exc.message)
    else:
        logger.warning("%s: %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "detail": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_request",
            "detail": "Expected a multipart form with an 'image' file field.",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc) or "Unexpected failure."},
    )


# Generated PNG / NPY / JSON artefacts are served straight from disk.
app.mount(
    settings.STATIC_URL_PREFIX,
    StaticFiles(directory=str(settings.OUTPUT_DIR)),
    name="static",
)
app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "stage": settings.APP_STAGE,
        "problem_statement": "SIH 2026 - 26175 (ISRO)",
        "output": "Relative Height / Relative DSM - NOT metric, NOT georeferenced",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/process", "/api/results/{job_id}"],
    }
