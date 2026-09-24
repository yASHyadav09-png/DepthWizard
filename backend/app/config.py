"""Central configuration for the DepthWizard backend.

Every tunable lives here so later stages (GAMUS semantics, SRTM/GCP scaling,
GeoTIFF IO) can register their own settings in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings:
    """Runtime settings, overridable through environment variables."""

    APP_NAME = "DepthWizard"
    APP_STAGE = "Stage 1 - Relative Height / Relative DSM"
    VERSION = "0.1.0"

    # --- Model -----------------------------------------------------------
    # Hugging Face Transformers port of Depth Anything V2 Small.
    MODEL_NAME = _env("DW_MODEL_NAME", "depth-anything/Depth-Anything-V2-Small-hf")
    MODEL_LABEL = "Depth Anything V2 Small"
    # "auto" -> cuda when available, otherwise cpu. Can be forced to cpu/cuda.
    DEVICE = _env("DW_DEVICE", "auto")
    # Load the model at server startup instead of on first request.
    PRELOAD_MODEL = _env("DW_PRELOAD_MODEL", "1") == "1"

    # --- Storage ---------------------------------------------------------
    OUTPUT_DIR = Path(_env("DW_OUTPUT_DIR", str(BASE_DIR / "outputs")))
    STATIC_URL_PREFIX = "/static"

    # --- Upload limits ---------------------------------------------------
    MAX_UPLOAD_BYTES = int(_env("DW_MAX_UPLOAD_MB", "25")) * 1024 * 1024
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    # Longest side fed to the network. Depth Anything internally resizes to
    # 518px, so anything beyond this only costs memory.
    MAX_INFERENCE_SIDE = int(_env("DW_MAX_INFERENCE_SIDE", "1536"))
    # Longest side of the RGB texture handed to the 3D viewer.
    MAX_TEXTURE_SIDE = int(_env("DW_MAX_TEXTURE_SIDE", "2048"))

    # --- Terrain ---------------------------------------------------------
    TERRAIN_RESOLUTION = int(_env("DW_TERRAIN_RESOLUTION", "256"))

    # --- CORS ------------------------------------------------------------
    CORS_ORIGINS = _env(
        "DW_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
    ).split(",")


settings = Settings()
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
