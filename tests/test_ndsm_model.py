import torch

from depthwizard.models.ndsm import NDSMModel

REV = "5426e4f0f36572d16453bbda7a8389317b1bef99"


def test_freezing_and_counts():
    m = NDSMModel(REV)
    m.set_trainable("frozen_encoder")
    c = m.param_counts()
    assert c["trainable"] == 2_728_513 and c["frozen"] == 22_056_576
    assert not any(p.requires_grad for p in m.net.backbone.parameters())
    assert not m.s0.requires_grad and "s0" not in dict(m.named_parameters())
    m.train()
    assert not m.net.backbone.training and m.net.head.training
    m.set_trainable("partial", unfreeze_last_blocks=4)
    assert m.param_counts()["trainable"] > 2_728_513


def test_predict_tile_pads_and_crops_back():
    m = NDSMModel(REV, s0=2.0).eval()
    x = torch.randn(1, 3, 60, 50)
    out = m.predict_tile(x, amp=False)
    assert out.shape == (1, 1, 60, 50)
    assert (out >= 0).all()
    # s0 scales the raw network output
    # 60 -> 70 (pad 5 top/bottom), 50 -> 56 (pad 3 left/right): next multiples of 14
    raw = m.raw(torch.nn.functional.pad(x, (3, 3, 5, 5), mode="reflect"))[..., 5:65, 3:53]
    assert torch.allclose(out, 2.0 * raw, atol=1e-5)
