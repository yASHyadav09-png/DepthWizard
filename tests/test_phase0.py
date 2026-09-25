"""Phase 0 checks: metrics are correct, augmentations keep modalities aligned,
NoData rule behaves, and the real subset loads through a DataLoader."""
from pathlib import Path

import numpy as np
import pytest
import torch

from depthwizard.data import gamus
from depthwizard.eval.metrics import HeightMetrics, tile_metrics

ROOT = Path(__file__).resolve().parents[1]
HAS_SUBSET = (ROOT / "configs" / "subsets" / "phase0.yaml").exists() and \
    (ROOT / "data" / "gamus" / "images").exists()


def test_metrics_match_numpy():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 30, (64, 64)).astype(np.float32)
    pred = gt + rng.normal(0, 2, gt.shape).astype(np.float32)
    valid = rng.random(gt.shape) > 0.1
    cls = rng.integers(0, 7, gt.shape)
    m = HeightMetrics()
    # split into two "tiles" to check pooling
    m.update(pred[:32], gt[:32], valid[:32], cls[:32])
    m.update(pred[32:], gt[32:], valid[32:], cls[32:])
    r = m.result()["overall"]
    e = (pred - gt)[valid]
    assert r["mae"] == pytest.approx(np.abs(e).mean(), rel=1e-6)
    assert r["rmse"] == pytest.approx(np.sqrt((e ** 2).mean()), rel=1e-6)
    assert r["pearson_r"] == pytest.approx(np.corrcoef(pred[valid], gt[valid])[0, 1], rel=1e-6)
    b = cls == 3
    rb = m.result()["per_class"]["building"]
    assert rb["mae"] == pytest.approx(np.abs((pred - gt)[valid & b]).mean(), rel=1e-6)


def test_perfect_prediction():
    gt = np.linspace(0, 10, 100).reshape(10, 10)
    r = tile_metrics(gt, gt, np.ones_like(gt, bool))
    assert r["mae"] == 0 and r["rmse"] == 0 and r["pearson_r"] == pytest.approx(1.0)


def test_valid_mask_rules():
    nd = np.full((32, 32), 5.0, np.float32)
    nd[0, 0] = -5.0          # DC clip sentinel -> invalid
    nd[1, 1] = -0.8          # small DSM-DTM noise -> valid
    nd[10, 10] = 180.0       # isolated spike -> invalid
    nd[20:30, 20:30] = 90.0  # real tall block -> valid (interior)
    nd[5, 5] = np.nan
    v = gamus.valid_mask(nd)
    assert not v[0, 0] and v[1, 1] and not v[10, 10] and not v[5, 5]
    assert v[25, 25]


def test_augmentation_keeps_alignment(monkeypatch):
    """Encode pixel index into every modality; after random crop/rot/flip they must still agree."""
    H = W = 64
    idx = np.arange(H * W).reshape(H, W)

    def fake_read(split, tid, root=None):
        rgb = np.stack([idx % 256, idx // 256, np.zeros_like(idx)], -1).astype(np.uint8)
        return {"rgb": rgb, "ndsm": idx.astype(np.float32), "cls": (idx % 7).astype(np.uint8),
                "valid": (idx % 2 == 0)}

    monkeypatch.setattr(gamus, "read_tile", fake_read)
    monkeypatch.setattr(gamus, "load_subset", lambda name: {"train": ["X"]})
    ds = gamus.GAMUSDataset("fake", "train", crop=32, train=True)
    for seed in range(20):
        np.random.seed(seed)
        s = ds[0]
        img = s["image"].numpy().transpose(1, 2, 0) * gamus.IMAGENET_STD + gamus.IMAGENET_MEAN
        rgb = np.rint(img * 255).astype(int)
        decoded = rgb[..., 0] + 256 * rgb[..., 1]
        nd = s["ndsm"][0].numpy().astype(int)
        assert s["image"].shape == (3, 32, 32)
        assert np.array_equal(decoded, nd)
        assert np.array_equal(s["cls"].numpy(), nd % 7)
        assert np.array_equal(s["valid"][0].numpy(), nd % 2 == 0)


@pytest.mark.skipif(not HAS_SUBSET, reason="phase0 subset not downloaded")
def test_real_subset_dataloader():
    ds = gamus.GAMUSDataset("phase0", "train", crop=518, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    b = next(iter(dl))
    assert b["image"].shape == (4, 3, 518, 518) and b["image"].dtype == torch.float32
    assert b["ndsm"].shape == (4, 1, 518, 518)
    assert b["valid"].dtype == torch.bool and b["cls"].dtype == torch.int64
    assert b["cls"].max() <= 6
    val = gamus.GAMUSDataset("phase0", "val")
    s = val[0]
    assert s["image"].shape == (3, 1024, 1024)


def test_constant_prediction_has_undefined_r():
    from depthwizard.eval.metrics import tile_metrics
    gt = np.random.default_rng(0).uniform(0, 30, (256, 256)).astype(np.float32)
    r = tile_metrics(np.full_like(gt, 4.709597), gt, np.ones_like(gt, bool))
    assert r["pearson_r"] is None
