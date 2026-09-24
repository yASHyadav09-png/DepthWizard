"""Unit tests for the pure-numpy half of the pipeline (no model download)."""

import base64

import numpy as np
import pytest

from app.services import depth_processor, terrain_generator
from app.utils import errors, image_io
from tests.conftest import make_image_bytes


# --- depth_processor -----------------------------------------------------
def test_normalize_maps_to_unit_range():
    depth = np.linspace(-5.0, 12.0, 400, dtype=np.float32).reshape(20, 20)
    out = depth_processor.normalize(depth)
    assert out.dtype == np.float32
    assert out.min() == pytest.approx(0.0, abs=1e-6)
    assert out.max() == pytest.approx(1.0, abs=1e-6)


def test_normalize_handles_flat_and_nonfinite_input():
    flat = np.full((8, 8), 3.0, dtype=np.float32)
    assert depth_processor.normalize(flat).max() == 0.0

    noisy = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)
    out = depth_processor.normalize(noisy)
    assert np.isfinite(out).all()


def test_inverse_depth_is_used_as_height_directly():
    depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    inverse = depth_processor.to_relative_height(depth, is_inverse_depth=True)
    metric = depth_processor.to_relative_height(depth, is_inverse_depth=False)
    # Near the camera (large inverse depth) must read as tall.
    assert inverse[1, 1] > inverse[0, 0]
    assert metric[1, 1] < metric[0, 0]


def test_stats_are_relative_and_consistent():
    field = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    stats = depth_processor.compute_stats(field)
    assert stats.min == pytest.approx(0.0)
    assert stats.max == pytest.approx(1.0)
    assert 0.0 < stats.mean < 1.0
    assert "relative" in stats.units
    assert "metre" not in stats.units and "meter" not in stats.units


def test_visualisations_render_rgb_images():
    field = np.random.default_rng(0).random((32, 48)).astype(np.float32)
    depth_png = depth_processor.colorize(field, "inferno")
    dsm_png = depth_processor.hillshaded_dsm(field)
    assert depth_png.size == (48, 32) and depth_png.mode == "RGB"
    assert dsm_png.size == (48, 32) and dsm_png.mode == "RGB"
    # A colour map must not collapse to a single flat colour.
    assert len(set(depth_png.getdata())) > 10


# --- terrain_generator ---------------------------------------------------
def test_terrain_downsamples_and_preserves_aspect():
    field = np.random.default_rng(1).random((600, 900)).astype(np.float32)
    grid = terrain_generator.generate_terrain(field, resolution=256)
    assert grid.width == 256
    assert grid.height == 171          # 256 * 600/900
    assert grid.aspect_ratio == pytest.approx(1.5)
    assert grid.plane_width == pytest.approx(1.0)
    assert grid.plane_depth == pytest.approx(1 / 1.5)


def test_terrain_buffer_decodes_to_the_declared_shape():
    field = np.random.default_rng(2).random((200, 200)).astype(np.float32)
    grid = terrain_generator.generate_terrain(field, resolution=64)
    raw = base64.b64decode(grid.heights_b64)
    heights = np.frombuffer(raw, dtype="<f4")
    assert heights.size == grid.width * grid.height
    assert np.isfinite(heights).all()
    assert heights.min() >= 0.0 and heights.max() <= 1.0


def test_terrain_keeps_real_relief():
    """A ramp must survive downsampling - the mesh has to actually be 3D."""
    ramp = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (256, 1))
    grid = terrain_generator.generate_terrain(ramp, resolution=128)
    assert grid.max_height - grid.min_height > 0.8


def test_terrain_rejects_non_2d_input():
    with pytest.raises(ValueError):
        terrain_generator.generate_terrain(np.zeros((4, 4, 3), dtype=np.float32))


# --- image_io ------------------------------------------------------------
def test_rejects_empty_and_corrupt_uploads():
    with pytest.raises(errors.InvalidImageError):
        image_io.validate_upload("a.png", "image/png", b"")
    with pytest.raises(errors.InvalidImageError):
        image_io.load_rgb_image(b"this is definitely not an image")


def test_rejects_disallowed_extension_and_content_type():
    with pytest.raises(errors.InvalidImageError):
        image_io.validate_upload("terrain.tif", "image/tiff", b"x" * 100)
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
