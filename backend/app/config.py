"""Central configuration for the DepthWizard backend.

Every tunable lives here so later phases (resolution handling, DEM/GCP
calibration, GeoTIFF IO) can register their own settings in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings:
    """Runtime settings, overridable through environment variables."""

    APP_NAME = "DepthWizard"
    APP_STAGE = "nDSM (JPG/PNG) and absolute DSM (GeoTIFF + GLO-30)"
    VERSION = "1.0.0"

    # --- Model -----------------------------------------------------------
    # A Phase 2 training run; its checkpoints/best.pt is loaded. Default: the
    # kept Phase 2b model (see docs/phase2b_results.md).
    RUN_DIR = Path(_env("DW_RUN_DIR", str(REPO_ROOT / "runs" / "20260926-003630_phase2b_partial")))
    MODEL_LABEL = "DepthWizard nDSM (Depth Anything V2 Small, fine-tuned on GAMUS)"
    # "auto" -> cuda when available, otherwise cpu.
    DEVICE = _env("DW_DEVICE", "auto")
    # Load the model at server startup instead of on first request.
    PRELOAD_MODEL = _env("DW_PRELOAD_MODEL", "1") == "1"

    # --- Storage ---------------------------------------------------------
    OUTPUT_DIR = Path(_env("DW_OUTPUT_DIR", str(BASE_DIR / "outputs")))
    # "/static" locally; overridden on the HF Space (DW_STATIC_URL_PREFIX), where Gradio
    # itself already reserves "/static/{path:path}" for its own UI assets and would
    # otherwise shadow ours (see scripts/build_space.py).
    STATIC_URL_PREFIX = _env("DW_STATIC_URL_PREFIX", "/static")

    # --- Upload limits ---------------------------------------------------
    MAX_UPLOAD_BYTES = int(_env("DW_MAX_UPLOAD_MB", "40")) * 1024 * 1024
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif",
                             "application/octet-stream"}
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    GEO_EXTENSIONS = {".tif", ".tiff"}

    # --- DEM (Phase 5) ---------------------------------------------------
    # Cached Copernicus GLO-30 under data/dem and data/geo/*; if none covers an upload,
    # fetch a clip from Microsoft Planetary Computer (set DW_DEM_ALLOW_FETCH=0 for an
    # offline demo; uncovered uploads then fall back to a georeferenced nDSM).
    DEM_ALLOW_FETCH = _env("DW_DEM_ALLOW_FETCH", "1") == "1"
    # Inference runs at native resolution (never downscaled: the model is
    # trained at a fixed ground resolution), in overlapping 1024 px windows.
    # This caps the work per request.
    MAX_INFERENCE_PIXELS = int(_env("DW_MAX_INFERENCE_PIXELS", str(40_000_000)))
    # Longest side of the RGB texture handed to the 3D viewer.
    MAX_TEXTURE_SIDE = int(_env("DW_MAX_TEXTURE_SIDE", "2048"))

    # --- Terrain ---------------------------------------------------------
    # Mesh grid cells on the longest side (area-averaged). Measured on a laptop integrated GPU
    # (Radeon 780M, 1920x1080): 512 -> 144 fps, 0.9 s to open the Explorer; 768 -> 143 fps, 2.1 s;
    # 1024 -> 2.5-68 fps, 5-6 s. 768 keeps 2.25x the detail of 512 and stays smooth.
    TERRAIN_RESOLUTION = int(_env("DW_TERRAIN_RESOLUTION", "768"))

    # --- CORS ------------------------------------------------------------
    CORS_ORIGINS = _env(
        "DW_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
    ).split(",")


settings = Settings()
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
