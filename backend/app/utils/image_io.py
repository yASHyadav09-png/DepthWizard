"""Upload validation and image loading helpers."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.utils.errors import InvalidImageError, PayloadTooLargeError

# Guard against decompression-bomb style uploads while still allowing large
# aerial frames.
Image.MAX_IMAGE_PIXELS = 200_000_000

MIN_SIDE = 32


def validate_upload(filename: str | None, content_type: str | None, data: bytes) -> None:
    """Cheap checks performed before we ever hand bytes to Pillow."""
    if not data:
        raise InvalidImageError("Uploaded file is empty.")

    if len(data) > settings.MAX_UPLOAD_BYTES:
        limit_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise PayloadTooLargeError(f"Image exceeds the {limit_mb} MB upload limit.")

    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in settings.ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file type '{suffix}'. Stage 1 accepts JPG and PNG only."
        )

    if content_type and content_type.lower() not in settings.ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"Unsupported content type '{content_type}'. Stage 1 accepts JPG and PNG only."
        )


def load_rgb_image(data: bytes) -> Image.Image:
    """Decode upload bytes into an upright RGB image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The file could not be decoded as a JPG or PNG image."
        ) from exc

    if image.format not in {"JPEG", "PNG"}:
        raise InvalidImageError(
            f"Decoded format '{image.format}' is not supported. Use JPG or PNG."
        )

    # Honour EXIF orientation so the terrain matches what the user sees.
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    if min(image.size) < MIN_SIDE:
        raise InvalidImageError(
            f"Image is too small ({image.width}x{image.height}). "
            f"Both sides must be at least {MIN_SIDE} px."
        )
    return image


def limit_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side is at most ``max_side`` (aspect preserved)."""
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def to_float_array(image: Image.Image) -> np.ndarray:
    """RGB image -> float32 array in [0, 1] with shape (H, W, 3)."""
    return np.asarray(image, dtype=np.float32) / 255.0
