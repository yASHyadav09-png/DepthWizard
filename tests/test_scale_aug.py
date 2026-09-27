"""Phase 6 scale augmentation: same degradation as the Phase 4 study, correct geometry,
modalities stay aligned, and it is off outside training."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from depthwizard.data import coarsen, gamus

ROOT = Path(__file__).resolve().parents[1]


def _phase4():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("phase4_gsd_study", ROOT / "scripts" / "phase4_gsd_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_coarsening_equals_phase4_study():
    p4 = _phase4()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    nd = rng.uniform(0, 40, (200, 200)).astype(np.float32)
    nd[rng.random(nd.shape) < 0.01] = np.nan
    valid = np.isfinite(nd) & (rng.random(nd.shape) > 0.2)
    cls = rng.integers(0, 7, (200, 200), dtype=np.uint8)
    for size in (200, 133, 101):
        assert np.array_equal(coarsen.coarsen_rgb(rgb, size), p4.coarsen_rgb(rgb, size))
        a, va = coarsen.coarsen_gt(nd, valid, size)
        b, vb = p4.coarsen_gt(nd, valid, size)
        assert np.array_equal(va, vb) and np.allclose(a[va], b[vb]) and np.isfinite(a[va]).all()
        assert np.array_equal(coarsen.coarsen_cls(cls, size), p4.coarsen_cls(cls, size))


def _fake(monkeypatch, H=64):
    """nDSM = native column index (a ramp, metres), RGB/cls/valid consistent with it."""
    cols = np.tile(np.arange(H, dtype=np.float32), (H, 1))

    def fake_read(split, tid, root=None, window=None):
        y, x, c = window
        w = cols[y:y + c, x:x + c]
        rgb = np.stack([w, w, w], -1).astype(np.uint8)
        return {"rgb": rgb, "ndsm": w.copy(), "cls": np.zeros_like(w, np.uint8), "valid": np.ones_like(w, bool)}

    monkeypatch.setattr(gamus, "read_tile", fake_read)
    monkeypatch.setattr(gamus, "load_subset", lambda name: {"train": ["X"], "val": ["X"]})


def test_scaled_sample_has_crop_size_and_coarser_ground_resolution(monkeypatch):
    _fake(monkeypatch)
    ds = gamus.GAMUSDataset("fake", "train", crop=32, train=True, tile_size=64,
                            scale_aug={"p": 1.0, "min": 1.5, "max": 1.5})
    for seed in range(10):
        np.random.seed(seed)
        s = ds[0]
        assert s["scale"] == pytest.approx(1.5)
        assert s["image"].shape == (3, 32, 32) and s["ndsm"].shape == (1, 32, 32)
        nd = s["ndsm"][0].numpy()
        # the ramp rises 1 m per native pixel -> 1.5 m per output pixel along one axis (after rot/flip)
        step = max(np.abs(np.diff(nd, axis=0)).mean(), np.abs(np.diff(nd, axis=1)).mean())
        assert step == pytest.approx(1.5, abs=0.05)
        assert s["valid"].all()
        # RGB was coarsened in the same way as the nDSM (aligned)
        img = s["image"].numpy().transpose(1, 2, 0) * gamus.IMAGENET_STD + gamus.IMAGENET_MEAN
        assert np.abs(img[..., 0] * 255 - nd).max() < 1.01


def test_scale_aug_off_for_validation_and_probability_zero(monkeypatch):
    _fake(monkeypatch)
    ev = gamus.GAMUSDataset("fake", "val", crop=32, train=False, tile_size=64,
                            scale_aug={"p": 1.0, "min": 1.9, "max": 1.9})
    assert ev.scale_aug is None and ev[0]["scale"] == 1.0
    tr = gamus.GAMUSDataset("fake", "train", crop=32, train=True, tile_size=64,
                            scale_aug={"p": 0.0, "min": 1.5, "max": 1.9})
    np.random.seed(0)
    assert all(tr[0]["scale"] == 1.0 for _ in range(10))


def test_window_must_fit_the_tile(monkeypatch):
    _fake(monkeypatch)
    with pytest.raises(AssertionError):
        gamus.GAMUSDataset("fake", "train", crop=40, train=True, tile_size=64,
                           scale_aug={"p": 1.0, "min": 1.0, "max": 1.9})
