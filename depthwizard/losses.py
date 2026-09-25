"""Phase 2 training losses. All inputs are (B, 1, H, W) tensors in metres plus a
boolean validity mask of the same shape. Only valid pixels ever contribute.

L1 (masked):
    L_1 = (1 / N) * sum_{i in V} |p_i - g_i|,    N = |V| (valid pixels in the batch)

Gradient-matching loss on the residual R = p - g (multi-scale, MiDaS-style):
    for scale k = 0..K-1 with stride s = 2^k, on the grid of pixels whose row and
    column indices are multiples of s:
        horizontal pairs: (i, j), (i, j+s)    vertical pairs: (i, j), (i+s, j)
        a pair counts only if BOTH pixels are valid (m = v_a * v_b)
        G_k = sum_pairs m * |R_a - R_b|  /  sum_pairs m
    L_grad = (1 / K') * sum_k G_k   over the K' scales that have at least one valid pair

    Because R is the residual, a prediction that is off by a constant has zero
    gradient loss; the term only penalises wrong *relative* structure (edges,
    slopes). Requiring both pixels valid means a NoData boundary never creates
    an artificial gradient error. Each scale is normalised by its own number of
    valid pairs, so every scale is a mean absolute difference in metres, and G is
    on the same scale as L_1.

Total:
    L = L_1 + lambda_grad * L_grad
"""
from __future__ import annotations

import torch


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.to(pred.dtype)
    n = v.sum()
    return (torch.abs(pred - gt) * v).sum() / n.clamp_min(1)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, scales: int = 4) -> torch.Tensor:
    R = (pred - gt).float()
    v = valid.float()
    terms = []
    for k in range(scales):
        s = 2 ** k
        Rk, vk = R[..., ::s, ::s], v[..., ::s, ::s]
        mx = vk[..., :, 1:] * vk[..., :, :-1]
        my = vk[..., 1:, :] * vk[..., :-1, :]
        num = (torch.abs(Rk[..., :, 1:] - Rk[..., :, :-1]) * mx).sum() + \
              (torch.abs(Rk[..., 1:, :] - Rk[..., :-1, :]) * my).sum()
        den = mx.sum() + my.sum()
        if den > 0:
            terms.append(num / den)
    if not terms:
        return R.sum() * 0.0
    return torch.stack(terms).mean()


def total_loss(pred, gt, valid, lambda_grad: float = 0.5, scales: int = 4) -> dict[str, torch.Tensor]:
    l1 = masked_l1(pred, gt, valid)
    g = gradient_loss(pred, gt, valid, scales)
    return {"l1": l1, "grad": g, "grad_weighted": lambda_grad * g, "total": l1 + lambda_grad * g}
