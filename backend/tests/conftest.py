import io
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point artefact storage at a temp dir before app.config is imported anywhere.
os.environ.setdefault("DW_OUTPUT_DIR", tempfile.mkdtemp(prefix="dw_test_"))
# Keep the suite fast: no weights are downloaded at import time.
os.environ.setdefault("DW_PRELOAD_MODEL", "0")


def make_image_bytes(width=160, height=120, fmt="PNG", mode="RGB") -> bytes:
    """A small synthetic scene with a bright blob so depth is not uniform."""
    img = Image.new(mode, (width, height), color=(30, 60, 90) if mode == "RGB" else 0)
    if mode == "RGB":
        for y in range(height // 3, 2 * height // 3):
            for x in range(width // 3, 2 * width // 3):
                img.putpixel((x, y), (230, 200, 120))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def jpg_bytes() -> bytes:
    return make_image_bytes(fmt="JPEG")
