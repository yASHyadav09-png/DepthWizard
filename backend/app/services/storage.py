"""Per-job artefact storage: PNGs, the .npy height array and metadata JSON."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.utils.errors import JobNotFoundError

JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

ORIGINAL_PNG = "original.png"
TEXTURE_JPG = "texture.jpg"
HEIGHT_PNG = "ndsm_colour.png"
HILLSHADE_PNG = "ndsm_hillshade.png"
HEIGHT_NPY = "ndsm_m.npy"
METADATA_JSON = "metadata.json"


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_RE.match(job_id or ""):
        raise JobNotFoundError(f"'{job_id}' is not a valid job id.")
    return job_id


def job_dir(job_id: str, create: bool = False) -> Path:
    """Resolve a job directory, refusing anything that escapes OUTPUT_DIR."""
    validate_job_id(job_id)
    path = (settings.OUTPUT_DIR / job_id).resolve()
    root = settings.OUTPUT_DIR.resolve()
    if root != path and root not in path.parents:
        raise JobNotFoundError(f"'{job_id}' is not a valid job id.")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def asset_url(job_id: str, filename: str) -> str:
    return f"{settings.STATIC_URL_PREFIX}/{job_id}/{filename}"


def save_image(job_id: str, filename: str, image: Image.Image, quality: int = 92) -> str:
    path = job_dir(job_id, create=True) / filename
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, format="JPEG", quality=quality)
    else:
        # compress_level 6 is Pillow's default; `optimize=True` costs several
        # hundred ms per image for a few percent of size.
        image.save(path, format="PNG")
    return asset_url(job_id, filename)


def save_array(job_id: str, filename: str, array: np.ndarray) -> str:
    path = job_dir(job_id, create=True) / filename
    np.save(path, np.asarray(array, dtype=np.float32))
    return asset_url(job_id, filename)


def save_metadata_file(job_id: str, filename: str, payload: dict) -> str:
    path = job_dir(job_id, create=True) / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return asset_url(job_id, filename)


def save_metadata(job_id: str, metadata: dict) -> str:
    path = job_dir(job_id, create=True) / METADATA_JSON
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return asset_url(job_id, METADATA_JSON)


def load_result(job_id: str) -> dict:
    """Re-read a finished job's response payload."""
    path = job_dir(job_id) / METADATA_JSON
    if not path.is_file():
        raise JobNotFoundError(f"No results stored for job '{job_id}'.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobNotFoundError(f"Results for job '{job_id}' are unreadable.") from exc


def load_height_array(job_id: str) -> np.ndarray:
    path = job_dir(job_id) / HEIGHT_NPY
    if not path.is_file():
        raise JobNotFoundError(f"No height array stored for job '{job_id}'.")
    return np.load(path)


def list_jobs(limit: int = 50) -> list[str]:
    root = settings.OUTPUT_DIR
    if not root.is_dir():
        return []
    jobs = [
        p for p in root.iterdir() if p.is_dir() and JOB_ID_RE.match(p.name)
        and (p / METADATA_JSON).is_file()
    ]
    jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in jobs[:limit]]
