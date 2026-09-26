"""Phase 5 evaluation on a real georeferenced area (default: Pittsburgh, NAIP 2019 + 3DEP 2019).

Runs the georeferenced pipeline (NAIP GeoTIFF -> nDSM -> GLO-30 DTM -> absolute DSM) and
compares it with USGS 3DEP LiDAR (2 m) after converting the LiDAR's NAVD88 heights to
EGM2008 with PROJ (constant offset at the AOI centre; the geoid difference varies by
millimetres over the AOI). Comparison grid: the LiDAR 2 m grid (same horizontal CRS as
NAIP, EPSG:26917); our 0.6 m products are AREA-averaged onto it.

Also: an error decomposition (DTM-only / nDSM-only / combined), a horizontal alignment
check (cross-correlation of nDSM vs LiDAR nDSM), a SIMULATED GCP experiment and a
DIAGNOSTIC ground-filter window sensitivity (the production window stays 150 m).

Usage:  python scripts/phase5_eval_aoi.py --aoi pittsburgh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from matplotlib.colors import LightSource
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.eval.metrics import _Acc  # noqa: E402
from depthwizard.geo.dem import OPEN_WINDOW_M, dtm_on_grid  # noqa: E402
from depthwizard.geo.gcp import correction_surface, fit_correction  # noqa: E402
from depthwizard.geo.pipeline import run_geo, write_outputs  # noqa: E402
from depthwizard.inference import DEFAULT_RUN, NDSMPredictor  # noqa: E402
from depthwizard.utils.runs import new_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def metrics(pred, ref, mask=None) -> dict:
    v = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        v &= mask
    a = _Acc(); a.add(pred[v], ref[v])
    return a.result()


def navd88_to_egm2008_offset(lon: float, lat: float) -> dict:
    pyproj.network.set_network_enabled(True)
    t = pyproj.Transformer.from_crs("EPSG:4269+5703", "EPSG:4326+3855", always_xy=True)
    offs = [t.transform(lon + dx, lat + dy, 300.0)[2] - 300.0 for dx in (-0.01, 0, 0.01) for dy in (-0.01, 0, 0.01)]
    t.transform(lon, lat, 300.0)
    return {"offset_m": float(np.mean(offs)), "spread_m": float(np.ptp(offs)), "pipeline": t.definition[:600]}


def to_grid(arr, src_t, src_crs, dst_t, dst_crs, shape, resampling=Resampling.average):
    out = np.full(shape, np.nan, np.float32)
    reproject(arr.astype(np.float32), out, src_transform=src_t, src_crs=src_crs, dst_transform=dst_t, dst_crs=dst_crs,
              resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def xcorr_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 10) -> tuple[float, float, float]:
    """Sub-pixel (dy, dx) shift of b relative to a from the normalised cross-correlation peak."""
    v = np.isfinite(a) & np.isfinite(b)
    a = np.where(v, a - a[v].mean(), 0); b = np.where(v, b - b[v].mean(), 0)
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    c = np.fft.fftshift(np.real(np.fft.ifft2(F)))
    cy, cx = np.array(c.shape) // 2
    win = c[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)

    def sub(m1, m0, p1):
        d = m1 - 2 * m0 + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d
    dy = iy - max_shift + (sub(win[iy - 1, ix], win[iy, ix], win[iy + 1, ix]) if 0 < iy < win.shape[0] - 1 else 0)
    dx = ix - max_shift + (sub(win[iy, ix - 1], win[iy, ix], win[iy, ix + 1]) if 0 < ix < win.shape[1] - 1 else 0)
    peak = float(win.max() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))
    return float(dy), float(dx), peak


def hill(z, px):
    z = np.nan_to_num(z, nan=np.nanmin(z))
    return LightSource(315, 45).hillshade(z, vert_exag=1, dx=px, dy=px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="pittsburgh")
    ap.add_argument("--gcps", type=int, default=5)
    a = ap.parse_args()
    d = ROOT / "data" / "geo" / a.aoi
    man = json.loads((d / "manifest.json").read_text())
    run = new_run(f"phase5_{a.aoi}", {"aoi": man["aoi"], "naip": man["naip"]["item"], "lidar_dsm": man["lidar_dsm"]["item"],
                                       "lidar_dtm": man["lidar_dtm"]["item"], "glo30": man["glo30"].get("items"),
                                       "model_run": DEFAULT_RUN.name, "ground_filter_window_m": OPEN_WINDOW_M,
                                       "comparison_grid": "LiDAR 2 m grid, our products area-averaged"})
    print("run:", run)
    predictor = NDSMPredictor(DEFAULT_RUN)
    res = run_geo(d / "naip_rgb.tif", predictor.predict, d / "glo30_dem.tif")
    files = write_outputs(res, run / "geotiff", a.aoi, {"MODEL_RUN": DEFAULT_RUN.name,
                                                        "MODEL_SHA256": predictor.info.sha256})
    img = res.image

    # ---- reference LiDAR in EGM2008
    datum = navd88_to_egm2008_offset(man["aoi"]["lon"], man["aoi"]["lat"])
    with rasterio.open(d / "lidar_dsm.tif") as s:
        ref_dsm = s.read(1, masked=True).filled(np.nan).astype(np.float32) + datum["offset_m"]
        rt, rcrs_full = s.transform, s.crs
    with rasterio.open(d / "lidar_dtm.tif") as s:
        ref_dtm = s.read(1, masked=True).filled(np.nan).astype(np.float32) + datum["offset_m"]
        assert s.transform == rt
    horiz = pyproj.CRS(rcrs_full.to_wkt()).sub_crs_list[0] if pyproj.CRS(rcrs_full.to_wkt()).is_compound else pyproj.CRS(rcrs_full.to_wkt())
    rcrs = rasterio.crs.CRS.from_wkt(horiz.to_wkt())
    same_h = horiz.equals(pyproj.CRS(img.crs.to_wkt()))
    ref_ndsm = ref_dsm - ref_dtm
    shp = ref_dsm.shape

    ours = {k: to_grid(getattr(res, k), img.transform, img.crs, rt, rcrs, shp) for k in ("dsm", "dtm", "ndsm")}
    raw = dtm_on_grid(d / "glo30_dem.tif", rcrs, rt, shp, window_m=0).dtm

    m = {
        "DTM: ours (filtered GLO-30) vs LiDAR DTM": metrics(ours["dtm"], ref_dtm),
        "DTM: raw GLO-30 vs LiDAR DTM (baseline)": metrics(raw, ref_dtm),
        "DSM: ours vs LiDAR DSM": metrics(ours["dsm"], ref_dsm),
        "DSM: raw GLO-30 vs LiDAR DSM (baseline)": metrics(raw, ref_dsm),
        "nDSM: model vs LiDAR nDSM": metrics(ours["ndsm"], ref_ndsm),
        "DSM: LiDAR DTM + model nDSM (nDSM error only)": metrics(ref_dtm + ours["ndsm"], ref_dsm),
        "DSM: our DTM + LiDAR nDSM (DTM error only)": metrics(ours["dtm"] + ref_ndsm, ref_dsm),
    }
    tall = np.isfinite(ref_ndsm) & (ref_ndsm > 2)
    m["nDSM: model vs LiDAR nDSM, LiDAR nDSM > 2 m"] = metrics(ours["ndsm"], ref_ndsm, tall)

    dy, dx, peak = xcorr_shift(ref_ndsm, ours["ndsm"])
    align = {"shift_px_2m": [dy, dx], "shift_m": [dy * abs(rt.e), dx * abs(rt.a)], "ncc_peak": peak,
             "method": "FFT cross-correlation of model nDSM vs LiDAR nDSM on the 2 m grid, sub-pixel parabola"}

    # ---- simulated GCPs: points on LiDAR ground (flat-ish, outside objects), fixed seed
    rng = np.random.default_rng(0)
    cand = np.argwhere(np.isfinite(ref_dtm) & np.isfinite(ref_ndsm) & (np.abs(ref_ndsm) < 0.3))
    pick = cand[rng.choice(len(cand), a.gcps, replace=False)]
    xs, ys = rt * (pick[:, 1] + 0.5, pick[:, 0] + 0.5)
    gcps = np.column_stack([xs, ys, ref_dtm[pick[:, 0], pick[:, 1]]])
    far = np.ones(shp, bool)
    for r, c in pick:
        far[max(r - 25, 0):r + 26, max(c - 25, 0):c + 26] = False     # exclude 50 m around each GCP
    gcp_res = {}
    for model in ("offset", "plane"):
        fit = fit_correction(res.dtm, img.transform, gcps, model)
        dtm_c = res.dtm + correction_surface(fit, img.transform, img.shape)
        dtm_c2 = to_grid(dtm_c, img.transform, img.crs, rt, rcrs, shp)
        gcp_res[model] = {"fit": {k: v for k, v in fit.items() if k != "residuals_before"},
                          "DTM vs LiDAR (pixels > 50 m from GCPs)": metrics(dtm_c2, ref_dtm, far),
                          "DSM vs LiDAR (pixels > 50 m from GCPs)": metrics(dtm_c2 + ours["ndsm"], ref_dsm, far)}
    gcp_res["no_gcp_same_pixels"] = {"DTM": metrics(ours["dtm"], ref_dtm, far), "DSM": metrics(ours["dsm"], ref_dsm, far)}

    # ---- diagnostic only: ground-filter window
    sens = {str(w): metrics(dtm_on_grid(d / "glo30_dem.tif", rcrs, rt, shp, window_m=w).dtm, ref_dtm)
            for w in (0, 90, 150, 300, 600)}

    out = {"datum": datum, "horizontal_crs_match": bool(same_h), "image": {"crs": img.crs.to_string(), "gsd_m": img.gsd_m,
           "shape": list(img.shape), "notes": img.notes}, "dem": res.dem_info, "metrics": m, "alignment": align,
           "gcp_simulation": gcp_res, "filter_window_sensitivity_DIAGNOSTIC": sens,
           "outputs": {k: str(v.relative_to(ROOT)) for k, v in files.items()}}
    (run / "phase5_eval.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- figures
    f = run / "figures"; f.mkdir(exist_ok=True)
    rgb = img.rgb[::3, ::3]
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title("NAIP 2019 (0.6 m)")
    ax[0, 1].imshow(hill(ours["dsm"], 2.0), cmap="gray"); ax[0, 1].set_title("our DSM (hillshade)")
    ax[0, 2].imshow(hill(ref_dsm, 2.0), cmap="gray"); ax[0, 2].set_title("LiDAR DSM 2019 (hillshade)")
    im = ax[0, 3].imshow(ours["dsm"] - ref_dsm, cmap="RdBu_r", vmin=-15, vmax=15); ax[0, 3].set_title("DSM error (ours - LiDAR)")
    fig.colorbar(im, ax=ax[0, 3], fraction=0.046)
    lo, hi = np.nanpercentile(ref_dtm, [1, 99])
    for k, (arr, t) in enumerate(((raw, "raw GLO-30"), (ours["dtm"], "our DTM (filtered GLO-30)"), (ref_dtm, "LiDAR DTM"))):
        im = ax[1, k].imshow(arr, cmap="terrain", vmin=lo, vmax=hi); ax[1, k].set_title(f"{t} (m, EGM2008)")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
    im = ax[1, 3].imshow(ours["dtm"] - ref_dtm, cmap="RdBu_r", vmin=-15, vmax=15); ax[1, 3].set_title("DTM error (ours - LiDAR)")
    fig.colorbar(im, ax=ax[1, 3], fraction=0.046)
    for x in ax.ravel(): x.axis("off")
    fig.tight_layout(); fig.savefig(f / "phase5_overview.png", dpi=70); plt.close(fig)

    r = shp[0] // 2
    fig, ax = plt.subplots(figsize=(14, 4))
    xs_m = np.arange(shp[1]) * abs(rt.a)
    for arr, lab, st in ((ref_dsm, "LiDAR DSM", "k-"), (ours["dsm"], "our DSM", "C3-"), (ref_dtm, "LiDAR DTM", "k--"),
                         (ours["dtm"], "our DTM", "C0--"), (raw, "raw GLO-30", "C2:")):
        ax.plot(xs_m, arr[r], st, lw=1, label=lab)
    ax.set_xlabel("distance west->east along the AOI centre row (m)"); ax.set_ylabel("elevation (m, EGM2008)")
    ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f / "profile_centre_row.png", dpi=80)
    plt.close(fig)

    print(json.dumps({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in m.items()},
                     indent=1))
    print("datum offset NAVD88->EGM2008:", round(datum["offset_m"], 3), "m; horizontal CRS match:", same_h)
    print("alignment:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3) if isinstance(v, float) else v)
                         for k, v in align.items() if k != "method"})
    for k, v in gcp_res.items():
        print("GCP", k, {kk: (round(vv["rmse"], 3) if isinstance(vv, dict) and "rmse" in vv else "") for kk, vv in v.items()})
    print("filter window (DIAGNOSTIC) DTM RMSE:", {w: round(v["rmse"], 3) for w, v in sens.items()})
    print("wrote", run)


if __name__ == "__main__":
    main()
