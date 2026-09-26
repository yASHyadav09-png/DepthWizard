"""Response models. Kept loose on purpose: the pipeline payload is the source
of truth and new stages may add fields without breaking older clients."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    # "model_*" collides with pydantic's protected namespace; these are ours.
    model_config = {"protected_namespaces": ()}

    status: str
    app: str
    version: str
    stage: str
    model_name: str
    model_checkpoint: str
    model_loaded: bool
    device: str
    cuda_available: bool
    torch_version: str
    terrain_resolution: int
    max_upload_mb: int


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine readable error code")
    detail: str = Field(description="Human readable message")


class TerrainGridModel(BaseModel):
    width: int
    height: int
    heights_b64: str
    encoding: str
    min_height: float
    max_height: float
    mean_height: float
    aspect_ratio: float
    plane_width: float
    plane_depth: float
    source_width: int
    source_height: int
    gsd_m: float
    display_min: float
    display_max: float
    height_units: str


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    stage: int
    stage_label: str
    source: dict[str, Any]
    model: dict[str, Any]
    height_product: dict[str, Any]
    statistics: dict[str, Any]
    assets: dict[str, str]
    terrain: TerrainGridModel
    timings_ms: dict[str, float]
    disclaimer: str
    metadata: dict[str, Any]

    model_config = {"protected_namespaces": ()}


class JobListResponse(BaseModel):
    jobs: list[str]
