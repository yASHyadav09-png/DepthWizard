"""Relative depth -> nDSM (metres) calibration used by the Phase 1 baselines.

Deployable path (no ground truth at inference):
    d~ = normalise(d)          per image, from d alone
    h  = max(0, f(d~))         f fitted once on TRAIN pixels, then frozen

Rationale for fitting in inverse-depth space: from altitude H a point of height h
is at range H - h, and 1/(H-h) ~ (1/H)(1 + h/H) for h << H, i.e. approximately
linear in h. The per-image scale/shift of the relative model is unknown, hence
the per-image normalisation.
"""
from __future__ import annotations

import numpy as np


def normalise(d: np.ndarray, method: str, valid: np.ndarray | None = None) -> np.ndarray:
    """Per-image normalisation of relative inverse depth. Uses only d (never GT).

    `valid` may restrict which pixels define the statistics; at inference time it is
    None (all pixels), and the baselines also pass None so train and deploy match.
    """
    x = d[valid] if valid is not None else d.ravel()
    if method == "pct":
        lo, hi = np.percentile(x, [2, 98])
        return (d - lo) / max(hi - lo, 1e-6)
    if method == "medmad":
        med = np.median(x)
        mad = np.mean(np.abs(x - med))
        return (d - med) / max(mad, 1e-6)
    raise ValueError(method)


class LinearMap:
    kind = "lin"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearMap":
        A = np.stack([x, np.ones_like(x)], 1).astype(np.float64)
        (self.a, self.b), *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.a * x + self.b).astype(np.float32)

    def to_dict(self) -> dict:
        return {"kind": "lin", "a": float(self.a), "b": float(self.b)}


class IsotonicMap:
    """Global monotone non-decreasing map, clipped outside the fitted range."""
    kind = "iso"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "IsotonicMap":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        self.xt, self.yt = ir.X_thresholds_.astype(np.float64), ir.y_thresholds_.astype(np.float64)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # np.interp clips outside [xt[0], xt[-1]] exactly like out_of_bounds="clip"
        return np.maximum(0.0, np.interp(x, self.xt, self.yt)).astype(np.float32)

    def to_dict(self) -> dict:
        return {"kind": "iso", "x": self.xt.tolist(), "y": self.yt.tolist()}


def oracle_affine(d: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
    """ORACLE per-image calibration: least-squares a, b against this image's own GT.
    Diagnostic only: needs ground truth, so it is not deployable."""
    m = LinearMap().fit(d[valid], gt[valid])
    return m(d), m.a, m.b
