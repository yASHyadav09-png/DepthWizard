"""Unit tests for the pure-numpy half of the pipeline (no model)."""

import base64

import numpy as np
import pytest

from app.services import height_processor, terrain_generator
from app.utils import errors, image_io
from tests.conftest import make_image_bytes


# --- height_processor ----------------------------------------------------
def building_scene(h=120, w=160, height_m=15.0):
    f = np.zeros((h, w), np.float32)
    f[40:80, 50:110] = height_m
    return f


def test_stats_are_in_metres():
    s = height_processor.compute_stats(building_scene())
    assert s.units == "m" and s.min == 0.0 and s.max == 15.0
    assert s.frac_above_2m == pytest.approx(40 * 60 / (120 * 160))


def test_stats_and_colours_survive_nonfinite_values():
    f = building_scene(); f[0, 0] = np.nan; f[1, 1] = np.inf
    assert np.isfinite(list(height_processor.compute_stats(f).to_dict().values())[:-1]).all()
    assert height_processor.colorize(f, 0, 15).size == (160, 120)


def test_display_range_has_a_floor():
    assert height_processor.display_range(np.zeros((10, 10))) == (0.0, 5.0)
    lo, hi = height_processor.display_range(building_scene(height_m=40.0))
    assert lo == 0.0 and hi == pytest.approx(40.0)


def test_visualisations_render_rgb_images():
    f = building_scene()
    for img in (height_processor.colorize(f, 0, 15), height_processor.hillshade(f, 0.33, 0, 15)):
        assert img.mode == "RGB" and img.size == (160, 120)


# --- terrain_generator ---------------------------------------------------
def test_terrain_keeps_metres_and_true_footprint():
    t = terrain_generator.generate_terrain(building_scene(), 0.5, (0.0, 15.0), resolution=512)
    heights = np.frombuffer(base64.b64decode(t.heights_b64), dtype="<f4").reshape(t.height, t.width)
    assert (t.width, t.height) == (160, 120) and heights.max() == pytest.approx(15.0)
    assert t.plane_width == pytest.approx(80.0) and t.plane_depth == pytest.approx(60.0)
    assert t.height_units == "m" and t.gsd_m == 0.5


def test_terrain_downsamples_and_preserves_aspect():
    t = terrain_generator.generate_terrain(np.zeros((1600, 2400), np.float32), 0.33, (0, 5), resolution=512)
    assert (t.width, t.height) == (512, 341)
    assert np.frombuffer(base64.b64decode(t.heights_b64), dtype="<f4").size == 512 * 341


def test_terrain_rejects_bad_input():
    with pytest.raises(ValueError):
        terrain_generator.generate_terrain(np.zeros((3, 4, 5)), 0.33, (0, 5))
    with pytest.raises(ValueError):
        terrain_generator.generate_terrain(np.zeros((30, 40)), 0.0, (0, 5))


# --- image_io --------------------------------------------------------------
def test_rejects_empty_and_corrupt_uploads():
    with pytest.raises(errors.InvalidImageError):
        image_io.validate_upload("a.png", "image/png", b"")
    with pytest.raises(errors.InvalidImageError):
        image_io.load_rgb_image(b"this is definitely not an image")


def test_rejects_disallowed_extension_and_content_type():
    with pytest.raises(errors.InvalidImageError):
        image_io.validate_upload("terrain.bmp", "image/bmp", b"x" * 100)
    with pytest.raises(errors.InvalidImageError):
        image_io.validate_upload("terrain.png", "application/pdf", b"x" * 100)


def test_rejects_oversized_upload():
    from app.config import settings
    with pytest.raises(errors.PayloadTooLargeError):
        image_io.validate_upload("big.png", "image/png", b"x" * (settings.MAX_UPLOAD_BYTES + 1))


def test_rejects_tiny_image():
    with pytest.raises(errors.InvalidImageError):
        image_io.load_rgb_image(make_image_bytes(8, 8))


def test_loads_png_and_jpeg_as_rgb():
    for fmt in ("PNG", "JPEG"):
        img = image_io.load_rgb_image(make_image_bytes(fmt=fmt))
        assert img.mode == "RGB" and img.size == (160, 120)


def test_limit_side_preserves_aspect():
    img = image_io.load_rgb_image(make_image_bytes(800, 400))
    small = image_io.limit_side(img, 200)
    assert small.size == (200, 100)
    assert image_io.limit_side(small, 4000).size == (200, 100)
