"""Height-estimation metrics (MAE, RMSE, Pearson r, bias), global and per label.

All functions take arrays in metres and a boolean validity mask. Invalid pixels
(NoData) never enter any statistic.

Metrics are accumulated as sufficient statistics (n, sum|e|, sum e^2, and the
moments needed for Pearson r), so split-level numbers are pooled over all valid
pixels, not averaged over tiles.

    MAE  = mean |p - g|          RMSE = sqrt(mean (p - g)^2)
    bias = mean (p - g)          r    = cov(p, g) / (std p * std g)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# GAMUS label ids (from the dataset README).
GAMUS_CLASSES = {
    0: "others",
    1: "ground",
    2: "low_vegetation",
    3: "building",
    4: "water",
    5: "road",
    6: "tree",
}

# order of the sufficient statistics
_N, _ABS, _SQ, _SP, _SG, _SPP, _SGG, _SPG = range(8)


def pixel_stats(pred_v: np.ndarray, gt_v: np.ndarray) -> np.ndarray:
    """(8, N) per-pixel contributions for already-masked 1-D arrays."""
    p = pred_v.astype(np.float64).ravel()
    g = gt_v.astype(np.float64).ravel()
    e = p - g
    return np.stack([np.ones_like(p), np.abs(e), e * e, p, g, p * p, g * g, p * g])


def label_stats(W: np.ndarray, lab: np.ndarray, K: int) -> np.ndarray:
    """(K, 8) per-label sums of pixel_stats. Pass a joint label (e.g. class*B + bin)
    to get several breakdowns from one pass, then marginalise by summing."""
    return np.stack([np.bincount(lab, W[i], minlength=K) for i in range(8)], 1)


@dataclass
class _Acc:
    s: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def add_vec(self, v: np.ndarray) -> None:
        self.s += v

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        self.add_vec(pixel_stats(pred, gt).sum(1))

    def result(self) -> dict:
        s = self.s
        n = s[_N]
        if n == 0:
            return {"n": 0, "mae": None, "rmse": None, "pearson_r": None, "bias": None}
        cov = s[_SPG] / n - (s[_SP] / n) * (s[_SG] / n)
        vp = s[_SPP] / n - (s[_SP] / n) ** 2
        vg = s[_SGG] / n - (s[_SG] / n) ** 2
        # relative tolerance: a constant prediction leaves only float rounding in vp,
        # and its correlation is undefined, not ~0
        tol = 1e-9
        defined = vp > tol * max(s[_SPP] / n, 1.0) and vg > tol * max(s[_SGG] / n, 1.0)
        r = cov / np.sqrt(vp * vg) if defined else None
        return {
            "n": int(n),
            "mae": float(s[_ABS] / n),
            "rmse": float(np.sqrt(s[_SQ] / n)),
            "pearson_r": None if r is None else float(r),
            "bias": float((s[_SP] - s[_SG]) / n),
        }


@dataclass
class HeightMetrics:
    """Pooled metrics over a whole split, with a breakdown by an integer label map
    (GAMUS classes by default; pass label_names for e.g. height bins)."""

    overall: _Acc = field(default_factory=_Acc)
    per_class: dict = field(default_factory=dict)
    label_names: dict = field(default_factory=lambda: dict(GAMUS_CLASSES))

    def add_table(self, table: np.ndarray) -> None:
        """table: (K, 8) sufficient statistics per label (see label_stats)."""
        self.overall.add_vec(table.sum(0))
        for cid in np.flatnonzero(table[:, _N]):
            self.per_class.setdefault(int(cid), _Acc()).add_vec(table[cid])

    def add_stats(self, W: np.ndarray, labels_v: np.ndarray | None = None) -> None:
        """W from pixel_stats on the valid pixels; labels_v the matching 1-D labels."""
        if labels_v is None or labels_v.size == 0:
            self.overall.add_vec(W.sum(1))
            return
        lab = labels_v.astype(np.int64).ravel()
        self.add_table(label_stats(W, lab, int(lab.max()) + 1))

    def update(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
               cls: np.ndarray | None = None) -> None:
        valid = valid.astype(bool)
        self.add_stats(pixel_stats(pred[valid], gt[valid]), None if cls is None else cls[valid])

    def result(self) -> dict:
        out = {"overall": self.overall.result(), "per_class": {}}
        for cid in sorted(self.per_class):
            name = self.label_names.get(cid, f"class_{cid}")
            out["per_class"][name] = self.per_class[cid].result()
        return out


def tile_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    acc = _Acc()
    acc.add(pred[valid], gt[valid])
    return acc.result()
