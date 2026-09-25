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


def test_nan_target_at_invalid_pixel_gives_finite_loss_and_same_gradient():
    gt = torch.rand(1, 1, 16, 16) * 10
    gt_nan = gt.clone(); gt_nan[..., 3, 4] = float("nan"); gt_nan[..., 10, :] = float("nan")
    valid = torch.isfinite(gt_nan)
    p1 = (gt + torch.randn_like(gt)).requires_grad_(True)
    p2 = p1.detach().clone().requires_grad_(True)
    a, b = total_loss(p1, gt_nan, valid), total_loss(p2, gt, valid)
    assert torch.isfinite(a["total"]) and torch.isclose(a["total"], b["total"])
    a["total"].backward(); b["total"].backward()
    assert torch.equal(p1.grad, p2.grad)


def test_nan_targets_regression_full_training_steps():
    """Regression test for the Phase 2a finding (14 PHL tiles with NaN nDSM pixels).

    With NaN (and inf) targets at pixels excluded by the validity mask, a few real
    optimisation steps (forward, loss, backward, grad clipping, AdamW) must keep:
      - the loss and every loss component finite,
      - all gradients finite,
      - all parameters and AdamW state (exp_avg, exp_avg_sq) finite,
    and the invalid pixels must be fully excluded from the effective loss
    (arbitrary values there give an identical loss and identical gradients)."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.Conv2d(8, 1, 3, padding=1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=0.01)
    x = torch.randn(2, 3, 24, 24)
    gt = torch.rand(2, 1, 24, 24) * 20
    gt[0, 0, :5, :] = float("nan")          # a NaN strip, like the PHL tiles
    gt[1, 0, 7, 9] = float("nan")
    gt[1, 0, 12, 3] = float("inf")
    valid = torch.isfinite(gt)
    assert (~valid).sum() == 5 * 24 + 2

    for _ in range(5):
        out = total_loss(net(x), gt, valid, lambda_grad=0.5)
        for k, v in out.items():
            assert torch.isfinite(v), k
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        for p in net.parameters():
            assert torch.isfinite(p.grad).all()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step()
        for p in net.parameters():
            assert torch.isfinite(p).all()
            st = opt.state[p]
            assert torch.isfinite(st["exp_avg"]).all() and torch.isfinite(st["exp_avg_sq"]).all()

    # exclusion: the values stored at invalid pixels do not matter at all
    pred = net(x).detach().requires_grad_(True)
    pred2 = pred.detach().clone().requires_grad_(True)
    other = gt.clone(); other[~valid] = 1e6
    a, b = total_loss(pred, gt, valid), total_loss(pred2, other, valid)
    assert all(torch.equal(a[k], b[k]) for k in a)
    a["total"].backward(); b["total"].backward()
    assert torch.equal(pred.grad, pred2.grad)
    # and L1 equals the plain mean absolute error over valid pixels only
    assert torch.isclose(a["l1"], (pred - gt).abs()[valid].mean())
