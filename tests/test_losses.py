import torch

from depthwizard.losses import gradient_loss, masked_l1, total_loss


def _t(a):
    return torch.as_tensor(a, dtype=torch.float32)[None, None]


def test_l1_ignores_invalid():
    gt = _t([[1.0, 2.0], [3.0, 4.0]])
    pred = _t([[1.0, 2.0], [3.0, 400.0]])
    valid = _t([[1, 1], [1, 0]]).bool()
    assert masked_l1(pred, gt, valid).item() == 0.0


def test_constant_offset_has_zero_gradient_loss():
    gt = torch.rand(1, 1, 32, 32) * 20
    assert gradient_loss(gt + 5.0, gt, torch.ones_like(gt, dtype=torch.bool)).item() < 1e-6


def test_nodata_boundary_creates_no_gradient_error():
    # perfect prediction on valid pixels, garbage (e.g. -5 m clip) on invalid ones
    gt = torch.full((1, 1, 16, 16), 10.0)
    pred = gt.clone()
    valid = torch.ones_like(gt, dtype=torch.bool)
    valid[..., :, 8:] = False
    pred[..., :, 8:] = -5.0
    gt[..., :, 8:] = 300.0
    assert gradient_loss(pred, gt, valid).item() == 0.0
    assert masked_l1(pred, gt, valid).item() == 0.0


def test_gradient_loss_value_single_scale():
    # R has one horizontal step of 2 m between columns 1 and 2 of a 3x3 grid
    gt = torch.zeros(1, 1, 3, 3)
    pred = _t([[0, 0, 2], [0, 0, 2], [0, 0, 2]])
    v = torch.ones_like(gt, dtype=torch.bool)
    # horizontal pairs: 6, of which 3 have |dR| = 2; vertical pairs: 6, all 0 -> 6 / 12
    assert abs(gradient_loss(pred, gt, v, scales=1).item() - 0.5) < 1e-6


def test_total_reports_components():
    gt = torch.rand(2, 1, 32, 32) * 10
    pred = gt + torch.randn_like(gt)
    out = total_loss(pred, gt, torch.ones_like(gt, dtype=torch.bool), lambda_grad=0.5)
    assert torch.isclose(out["total"], out["l1"] + 0.5 * out["grad"])
    assert torch.isclose(out["grad_weighted"], 0.5 * out["grad"])
