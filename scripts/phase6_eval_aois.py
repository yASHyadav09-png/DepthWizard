"""Phase 6 out-of-domain evaluation on real NAIP imagery vs USGS 3DEP LiDAR (configs/phase6_aois.yaml).

For one model run and every non-dropped AOI:
    NAIP GeoTIFF -> nDSM (model, native 0.6 m, direct inference) -> + GLO-30 DTM (150 m filter) -> DSM,
compared on the LiDAR 2 m grid (our products AREA-averaged onto it), LiDAR NAVD88 -> EGM2008 (PROJ).
Reference nDSM = LiDAR DSM - LiDAR DTM.

Metrics (depthwizard.eval.metrics, pooled over valid pixels): nDSM overall / per height band /
LiDAR nDSM > 2 m; DTM, DSM (ours and raw GLO-30 baselines); horizontal alignment; per-AOI and
POOLED over all AOIs. DIAGNOSTIC only (never used to choose anything): ground-filter windows
0/90/150/300 m and raw GLO-30 + model nDSM.

Reference QC (model-independent, reported for every AOI): `dtm_cliff_frac` = fraction of LiDAR DTM
pixels steeper than 63 deg between 2 m neighbours (building walls left in the "bare-earth" model).
(A first QC measure, pixels > 5 m above a 200 m grey opening, did not separate buildings from steep
hills and was replaced; see docs/phase6_results.md.) AOIs listed with `reference_exclude` in the config are
excluded from the named pooled metrics (and blocks) but still reported per AOI.

For paired model comparisons, per-block error sums (blocks of 64 x 64 px = 128 x 128 m on the 2 m grid)
are written to per_block_ndsm.csv (id = "<aoi>:<by>_<bx>", n, sse, sae) -> spatial block bootstrap.

Usage:  python scripts/phase6_eval_aois.py [--run runs/<model_run>] [--tag phase2b]
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
import pandas as pd
import pyproj
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depthwizard.eval.evaluate import BIN_NAMES, HEIGHT_EDGES  # noqa: E402
from depthwizard.eval.metrics import HeightMetrics, _Acc  # noqa: E402
from depthwizard.geo.dem import OPEN_WINDOW_M, dtm_on_grid  # noqa: E402
from depthwizard.geo.pipeline import run_geo  # noqa: E402
from depthwizard.inference import DEFAULT_RUN, NDSMPredictor  # noqa: E402
from depthwizard.utils.runs import append_results, new_run  # noqa: E402
from phase5_eval_aoi import metrics, navd88_to_egm2008_offset, to_grid, xcorr_shift  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 64                       # 2 m px -> 128 m blocks
FILTER_WINDOWS_DIAG = (0, 90, 150, 300)


def load_refs(d: Path, lon: float, lat: float):
    datum = navd88_to_egm2008_offset(lon, lat)
    with rasterio.open(d / "lidar_dsm.tif") as s:
        dsm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        rt, crs_full = s.transform, s.crs
    with rasterio.open(d / "lidar_dtm.tif") as s:
        dtm = s.read(1, masked=True).filled(np.nan).astype(np.float32)
        assert s.transform == rt and dtm.shape == dsm.shape
    pc = pyproj.CRS(crs_full.to_wkt())
    horiz = pc.sub_crs_list[0] if pc.is_compound else pc
    vert = pc.sub_crs_list[1].name if pc.is_compound else None
    assert vert is None or "NAVD88" in vert, f"unexpected vertical CRS {vert}"
    off = datum["offset_m"] if vert else 0.0
    return dsm + off, dtm + off, rt, rasterio.crs.CRS.from_wkt(horiz.to_wkt()), datum, vert


def dtm_cliff_frac(dtm: np.ndarray, px_m: float, max_grad: float = 2.0) -> float:
    """Fraction of LiDAR DTM pixels steeper than atan(2) = 63 deg between 2 m neighbours: building walls
    left in a 'bare-earth' model show up as cliffs (natural slopes rarely are)."""
    gy, gx = np.gradient(dtm, px_m)
    g = np.hypot(gx, gy)
    v = np.isfinite(g)
    return float((g[v] > max_grad).mean())


def band_metrics(pred, ref):
    v = np.isfinite(pred) & np.isfinite(ref)
    hm = HeightMetrics(label_names=BIN_NAMES)
    r0 = np.where(v, ref, 0).astype(np.float32)
    hm.update(np.where(v, pred, 0).astype(np.float32), r0, v, np.digitize(r0, HEIGHT_EDGES[1:-1]).astype(np.uint8))
    return hm.result()


def block_rows(aoi: str, pred, ref) -> list[dict]:
    rows = []
    H, W = ref.shape
    for by in range(0, H - BLOCK + 1, BLOCK):
        for bx in range(0, W - BLOCK + 1, BLOCK):
            p, g = pred[by:by + BLOCK, bx:bx + BLOCK], ref[by:by + BLOCK, bx:bx + BLOCK]
            v = np.isfinite(p) & np.isfinite(g)
            n = int(v.sum())
            if n < 0.5 * BLOCK * BLOCK:
                continue
            e = (p - g)[v].astype(np.float64)
            rows.append({"id": f"{aoi}:{by // BLOCK}_{bx // BLOCK}", "aoi": aoi, "n": n,
                         "sse": float((e ** 2).sum()), "sae": float(np.abs(e).sum()), "se": float(e.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="model run dir (checkpoints/best.pt)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--aois", default="configs/phase6_aois.yaml")
    a = ap.parse_args()
    model_run = Path(a.run)
    cfg = yaml.safe_load(open(ROOT / a.aois))
    aois = [x for x in cfg["aois"] if not x.get("dropped")]
    tag = a.tag or model_run.name
    run = new_run(f"phase6_aois_{tag}", {"model_run": model_run.name, "aois": aois, "aoi_config": a.aois,
                                         "ground_filter_window_m": OPEN_WINDOW_M, "block_px_2m": BLOCK,
                                         "comparison_grid": "LiDAR 2 m grid, our products area-averaged",
                                         "reference_ndsm": "LiDAR DSM - LiDAR DTM"})
    print("run:", run)
    predictor = NDSMPredictor(model_run)
    out, blocks, pooled = {}, [], {k: [] for k in ("pred", "ref")}
    figdir = run / "figures"; figdir.mkdir(exist_ok=True)
    for x in aois:
        d = ROOT / "data" / "geo" / x["name"]
        man = json.loads((d / "manifest.json").read_text())
        res = run_geo(d / "naip_rgb.tif", predictor.predict, d / "glo30_dem.tif")
        img = res.image
        ref_dsm, ref_dtm, rt, rcrs, datum, vert = load_refs(d, x["lon"], x["lat"])
        ref_ndsm = ref_dsm - ref_dtm
        shp = ref_dsm.shape
        ours = {k: to_grid(getattr(res, k), img.transform, img.crs, rt, rcrs, shp) for k in ("dsm", "dtm", "ndsm")}
        raw = dtm_on_grid(d / "glo30_dem.tif", rcrs, rt, shp, window_m=0).dtm
        tall = np.isfinite(ref_ndsm) & (ref_ndsm > 2)
        bm = band_metrics(ours["ndsm"], ref_ndsm)
        dy, dx, peak = xcorr_shift(ref_ndsm, ours["ndsm"])
        m = {
            "landscape": x["landscape"], "naip_gsd_m": img.gsd_m, "naip_date": man["naip"].get("datetime"),
            "lidar_project": x["lidar_project"], "lidar_vertical": vert, "datum_offset_m": datum["offset_m"],
            "coverage": man.get("coverage"),
            "ndsm": bm["overall"], "ndsm_per_band": bm["per_class"],
            "ndsm_ref_gt_2m": metrics(ours["ndsm"], ref_ndsm, tall),
            "ref_ndsm_stats": {"frac_gt_2m": float(tall.sum() / np.isfinite(ref_ndsm).sum()),
                               "p95": float(np.nanpercentile(ref_ndsm, 95)), "max": float(np.nanmax(ref_ndsm))},
            "dtm_ours": metrics(ours["dtm"], ref_dtm), "dtm_raw_glo30": metrics(raw, ref_dtm),
            "dsm_ours": metrics(ours["dsm"], ref_dsm), "dsm_raw_glo30": metrics(raw, ref_dsm),
            "DIAGNOSTIC_dsm_raw_glo30_plus_ndsm": metrics(raw + ours["ndsm"], ref_dsm),
            "DIAGNOSTIC_dtm_filter_window": {str(w): metrics(dtm_on_grid(d / "glo30_dem.tif", rcrs, rt, shp,
                                                                         window_m=w).dtm, ref_dtm)
                                             for w in FILTER_WINDOWS_DIAG},
            "alignment_m": {"dy": dy * abs(rt.e), "dx": dx * abs(rt.a), "ncc_peak": peak},
            "reference_qc": {"dtm_cliff_frac": dtm_cliff_frac(ref_dtm, abs(rt.a)),
                             "excluded_from": x.get("reference_exclude", []),
                             "reason": x.get("reference_exclude_reason")},
        }
        out[x["name"]] = m
        excl = "ndsm" in x.get("reference_exclude", [])
        blocks += [{**r, "excluded": excl} for r in block_rows(x["name"], ours["ndsm"], ref_ndsm)]
        v = np.isfinite(ours["ndsm"]) & np.isfinite(ref_ndsm)
        pooled["pred"].append(ours["ndsm"][v]); pooled["ref"].append(ref_ndsm[v]); pooled.setdefault("excl", []).append(excl)
        print(f"{x['name']:18s} nDSM RMSE {m['ndsm']['rmse']:.2f} MAE {m['ndsm']['mae']:.2f} bias {m['ndsm']['bias']:+.2f}"
              f" | >2m RMSE {m['ndsm_ref_gt_2m']['rmse']:.2f} bias {m['ndsm_ref_gt_2m']['bias']:+.2f}"
              f" | DSM {m['dsm_ours']['rmse']:.2f} (raw {m['dsm_raw_glo30']['rmse']:.2f})"
              f" DTM {m['dtm_ours']['rmse']:.2f} (raw {m['dtm_raw_glo30']['rmse']:.2f})", flush=True)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
        vmax = max(5.0, float(np.nanpercentile(ref_ndsm, 99)))
        ax[0].imshow(img.rgb[::3, ::3]); ax[0].set_title(f"{x['name']}: NAIP {img.gsd_m:g} m")
        ax[1].imshow(ref_ndsm, cmap="magma", vmin=0, vmax=vmax); ax[1].set_title("LiDAR nDSM (m)")
        im = ax[2].imshow(ours["ndsm"], cmap="magma", vmin=0, vmax=vmax); ax[2].set_title(f"model nDSM ({tag})")
        fig.colorbar(im, ax=ax[2], fraction=0.046)
        im = ax[3].imshow(ours["ndsm"] - ref_ndsm, cmap="RdBu_r", vmin=-10, vmax=10); ax[3].set_title("error (model - LiDAR)")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        for z in ax: z.axis("off")
        fig.tight_layout(); fig.savefig(figdir / f"{x['name']}.png", dpi=60); plt.close(fig)

    keep = [k for k, e in enumerate(pooled["excl"]) if not e]
    pr = np.concatenate([pooled["pred"][k] for k in keep]); rf = np.concatenate([pooled["ref"][k] for k in keep])
    acc = _Acc(); acc.add(pr, rf)
    pooled_m = acc.result()
    pooled_bands = band_metrics(pr[None], rf[None])
    acc_all = _Acc(); acc_all.add(np.concatenate(pooled["pred"]), np.concatenate(pooled["ref"]))
    used = [aois[k]["name"] for k in keep]
    summary = {"model_run": model_run.name, "pooled_ndsm": pooled_m, "pooled_ndsm_aois": used,
               "pooled_ndsm_per_band": pooled_bands["per_class"],
               "pooled_ndsm_ALL_incl_reference_failures": acc_all.result(), "aois": out}
    (run / "phase6_aois.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.DataFrame(blocks).to_csv(run / "per_block_ndsm.csv", index=False)
    rows = [{"aoi": k, "landscape": v["landscape"], "ndsm_rmse": v["ndsm"]["rmse"], "ndsm_mae": v["ndsm"]["mae"],
             "ndsm_bias": v["ndsm"]["bias"], "ndsm_r": v["ndsm"]["pearson_r"],
             "gt2m_rmse": v["ndsm_ref_gt_2m"]["rmse"], "gt2m_bias": v["ndsm_ref_gt_2m"]["bias"],
             "dsm_rmse": v["dsm_ours"]["rmse"], "dsm_raw_rmse": v["dsm_raw_glo30"]["rmse"],
             "dtm_rmse": v["dtm_ours"]["rmse"], "dtm_raw_rmse": v["dtm_raw_glo30"]["rmse"]} for k, v in out.items()]
    pd.DataFrame(rows).to_csv(run / "summary.csv", index=False)
    for k, v in out.items():
        append_results(run, f"aoi:{k}", {"overall": v["ndsm"], "per_class": {}}, method=f"phase6_aoi_{tag}",
                       deployable=True, extra={"target": "nDSM vs 3DEP LiDAR", "gsd_m": v["naip_gsd_m"]})
    append_results(run, "aoi:pooled", {"overall": pooled_m, "per_class": {}}, method=f"phase6_aoi_{tag}",
                   deployable=True, extra={"target": "nDSM vs 3DEP LiDAR", "aois": used})
    for k, v in out.items():
        print(f"  QC {k:18s} dtm_cliff_frac {v['reference_qc']['dtm_cliff_frac']:.4f} excluded {v['reference_qc']['excluded_from']}")
    print(f"ALL incl. reference failures: RMSE {summary['pooled_ndsm_ALL_incl_reference_failures']['rmse']:.3f}")
    print(f"POOLED nDSM ({len(used)} AOIs: {used}): RMSE {pooled_m['rmse']:.3f} MAE {pooled_m['mae']:.3f} "
          f"bias {pooled_m['bias']:+.3f} r {pooled_m['pearson_r']:.3f}")
    print("wrote", run)


if __name__ == "__main__":
    main()
