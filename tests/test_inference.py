from pathlib import Path

import numpy as np
import pytest

from depthwizard.inference import DEFAULT_RUN, feather, metric_validity, window_starts

HAS_MODEL = (DEFAULT_RUN / "checkpoints" / "best.pt").exists()
HAS_VAL = (Path(__file__).resolve().parents[1] / "data" / "gamus" / "images" / "val").exists()


def test_window_starts_cover_everything_with_overlap():
    for n in (1, 500, 1024, 1025, 1500, 2048, 3000, 4097):
        s = window_starts(n, 1024, 128)
        t = min(1024, n)
        covered = np.zeros(n, bool)
        for a in s:
            covered[a:a + t] = True
        assert covered.all(), n
        assert s[0] == 0 and s[-1] + t == n
        assert all(b - a <= t - 128 for a, b in zip(s, s[1:])), n   # overlap >= 128


def test_feather_is_positive_and_flat_at_image_borders():
    w = feather(1024, 128, fade_start=True, fade_end=True)
    assert (w > 0).all() and w.max() == 1.0 and w[0] < 0.01 and w[-1] < 0.01
    edge = feather(1024, 128, fade_start=False, fade_end=True)
    assert edge[0] == 1.0 and edge[-1] < 0.01


def test_blend_weights_reconstruct_a_constant_field():
    """Blending identical window predictions must return exactly that value."""
    H, W, T, O = 1700, 2300, 1024, 128
    acc = np.zeros((H, W)); ws = np.zeros((H, W))
    for y in window_starts(H, T, O):
        wy = feather(min(T, H), O, y > 0, y + min(T, H) < H)
        for x in window_starts(W, T, O):
            wx = feather(min(T, W), O, x > 0, x + min(T, W) < W)
            w = wy[:, None] * wx[None, :]
            acc[y:y + T, x:x + T] += 7.5 * w; ws[y:y + T, x:x + T] += w
    assert (ws > 0).all() and np.allclose(acc / ws, 7.5)


def test_metric_validity_labels():
    assert metric_validity(None)[0] == "uncertain"
    assert metric_validity(0.33)[0] == "valid"
    assert metric_validity(0.30)[0] == "valid"
    assert metric_validity(1.0)[0] == "uncertain"


@pytest.mark.skipif(not (HAS_MODEL and HAS_VAL), reason="needs the Phase 2b checkpoint and GAMUS val data")
def test_tiling_has_no_seams_and_handles_large_images():
    """Predict one real 1024 tile whole, and again with forced 640 px windows
    (overlap 128). Seams would appear as extra deviation from the whole-tile
    prediction near the window boundaries."""
    from depthwizard.data.gamus import load_subset, read_rgb
    from depthwizard.inference import NDSMPredictor
    p = NDSMPredictor(DEFAULT_RUN)
    rgb = read_rgb("val", load_subset("full")["val"][0])
    whole = p.predict(rgb)
    tiled = p.predict(rgb, tile=640, overlap=128)
    assert tiled.shape == whole.shape and np.isfinite(tiled).all() and (tiled >= 0).all()
    dev = np.abs(tiled - whole)
    # window starts along each axis: 0 and 384 -> overlap band 384..640
    band = dev[:, 384:640].mean()
    away = np.concatenate([dev[:, 64:320], dev[:, 704:960]], 1).mean()
    assert band < 1.5 * away + 0.05, (band, away)
    # NOTE (measured, not asserted): 640 px windows differ from whole-tile prediction
    # by ~1.0 m mean absolute deviation on this tile. The model is context-sensitive,
    # which is why production tiling uses 1024 px windows (the validation context).

    # a large non-square image goes through the 1024 tiling path
    big = np.concatenate([rgb, rgb[:, :500]], 1)                     # 1024 x 1524
    h = p.predict(big)
    assert h.shape == (1024, 1524) and np.isfinite(h).all()
