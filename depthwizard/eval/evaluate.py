"""Split evaluation shared by Phase 2+ (same metric definitions as Phase 1):
pooled overall metrics, per class, per true-height band, per city, and per tile."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import HeightMetrics, _Acc, label_stats, pixel_stats

HEIGHT_EDGES = [0, 2, 5, 10, 20, 1000]
BIN_NAMES = {i: (f"{HEIGHT_EDGES[i]}-{HEIGHT_EDGES[i + 1]}m" if HEIGHT_EDGES[i + 1] < 1000
                 else f">{HEIGHT_EDGES[i]}m") for i in range(len(HEIGHT_EDGES) - 1)}


class SplitEvaluator:
    def __init__(self):
        self.cls = HeightMetrics()
        self.bin = HeightMetrics(label_names=BIN_NAMES)
        self.city: dict[str, HeightMetrics] = {}
        self.rows = []

    def add(self, tid: str, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, cls: np.ndarray) -> dict:
        nb = len(HEIGHT_EDGES) - 1
        hb = np.digitize(gt, HEIGHT_EDGES[1:-1])
        joint = cls[valid].astype(np.int64) * nb + hb[valid]
        T = label_stats(pixel_stats(pred[valid], gt[valid]), joint, 7 * nb).reshape(7, nb, 8)
        self.cls.add_table(T.sum(1))
        self.bin.add_table(T.sum(0))
        tot = T.sum((0, 1))
        city = tid.split("_")[0]
        self.city.setdefault(city, HeightMetrics()).overall.add_vec(tot)
        a = _Acc(); a.add_vec(tot); tm = a.result()
        row = {"id": tid, "city": city, "valid_frac": float(valid.mean()), "n_valid": int(valid.sum()),
               "rmse": tm["rmse"], "mae": tm["mae"], "r": tm["pearson_r"], "bias": tm["bias"]}
        self.rows.append(row)
        return row

    def result(self) -> dict:
        res = self.cls.result()
        res["per_height_bin"] = self.bin.result()["per_class"]
        res["per_city"] = {c: h.result()["overall"] for c, h in sorted(self.city.items())}
        return res

    def per_tile(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
