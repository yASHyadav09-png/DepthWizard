"""Ground control point (GCP) correction of the DTM (Phase 5).

GCPs are ground points with known elevation: (x, y) in the IMAGE CRS and z in metres in
the SAME vertical datum as the output (EGM2008). Residuals r_i = z_i - DTM(x_i, y_i)
(DTM bilinearly sampled). Models:
    offset : DTM' = DTM + median(r)                         (>= 1 GCP; robust)
    plane  : DTM' = DTM + a + b*(x - x0) + c*(y - y0)       (>= 3 GCPs; least squares after
                                                              dropping residuals > 3 MAD)
The correction is applied to the DTM and therefore to the DSM (= DTM + nDSM).
"""
from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates


def sample(grid: np.ndarray, transform: Affine, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cols, rows = ~transform * (np.asarray(x, float), np.asarray(y, float))
    # pixel (0,0) centre is at (0.5, 0.5) in transform space
    return map_coordinates(grid, [np.asarray(rows) - 0.5, np.asarray(cols) - 0.5], order=1, mode="nearest")


def fit_correction(dtm: np.ndarray, transform: Affine, gcps: np.ndarray, model: str = "offset") -> dict:
    gcps = np.asarray(gcps, float)
    if gcps.ndim != 2 or gcps.shape[1] != 3 or len(gcps) == 0:
        raise ValueError("GCPs must be an N x 3 array of (x, y, z) in the image CRS / metres.")
    h, w = dtm.shape
    cols, rows = ~transform * (gcps[:, 0], gcps[:, 1])
    inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not inside.any():
        raise ValueError("None of the GCPs lies inside the image.")
    g = gcps[inside]
    r = g[:, 2] - sample(dtm, transform, g[:, 0], g[:, 1])
    x0, y0 = float(g[:, 0].mean()), float(g[:, 1].mean())
    if model == "offset":
        params = {"a": float(np.median(r)), "b": 0.0, "c": 0.0}
        used = np.ones(len(g), bool)
    elif model == "plane":
        if len(g) < 3:
            raise ValueError("A plane correction needs at least 3 GCPs inside the image.")
        med = np.median(r)
        mad = np.median(np.abs(r - med)) * 1.4826 + 1e-9
        used = np.abs(r - med) <= 3 * mad
        if used.sum() < 3:
            used = np.ones(len(g), bool)
        A = np.column_stack([np.ones(used.sum()), g[used, 0] - x0, g[used, 1] - y0])
        (a, b, c), *_ = np.linalg.lstsq(A, r[used], rcond=None)
        params = {"a": float(a), "b": float(b), "c": float(c)}
    else:
        raise ValueError(model)
    pred = params["a"] + params["b"] * (g[:, 0] - x0) + params["c"] * (g[:, 1] - y0)
    return {"model": model, "params": params, "x0": x0, "y0": y0, "n_inside": int(inside.sum()),
            "n_used": int(used.sum()), "n_outside": int((~inside).sum()),
            "rmse_before": float(np.sqrt(np.mean(r ** 2))), "rmse_after": float(np.sqrt(np.mean((r - pred) ** 2))),
            "residuals_before": r.tolist()}


def correction_surface(fit: dict, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = transform * (cols, rows)
    p = fit["params"]
    return (p["a"] + p["b"] * (x - fit["x0"]) + p["c"] * (y - fit["y0"])).astype(np.float32)
