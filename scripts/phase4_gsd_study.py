"""Phase 4 resolution (GSD) study, VAL ONLY: how does the Phase 2b model behave on
imagery coarser than its 0.33 m/px training resolution?

Simulation (per val tile, per target GSD g):
    size    = round(1024 * 0.33 / g)                           (0.50 -> 676, 0.66 -> 512, 1.00 -> 338, 2.00 -> 169 px)
    RGB_g   = area-average (PIL BOX) of the 0.33 m RGB           (a coarser sensor integrates radiance)
    GT_g    = area-average of VALID LiDAR nDSM pixels only; a coarse pixel is valid if >= 50% of its area is valid
    CLS_g   = nearest-neighbour resize of the class mask         (labels are categorical)

Strategies:
    A  direct:     predict(RGB_g)                                        -> heights on the g grid
    B  resampled:  predict(bicubic-upsample RGB_g to 1024 px = ~0.33 m)  -> area-average back to the g grid
Both are scored on the g grid against GT_g with the same metrics and breakdowns as
Phases 1-2. At g = 0.33, A = B = the Phase 2b val evaluation (sanity check).

Usage:  python scripts/phase4_gsd_study.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depthwizard.data.gamus import GAMUS_GSD_M, load_subset, read_tile  # noqa: E402
from depthwizard.eval.evaluate import SplitEvaluator  # noqa: E402
from depthwizard.inference import DEFAULT_RUN, NDSMPredictor  # noqa: E402
from depthwizard.utils.runs import append_results, new_run, save_metrics, write_manifest  # noqa: E402
from paired_bootstrap import paired  # noqa: E402

TILE = 1024


def box(a: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BOX),
                      dtype=np.float32)


def coarsen_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    if size == rgb.shape[0]:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def coarsen_gt(nd: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if size == nd.shape[0]:
        return nd, valid
    num = box(np.where(valid, nd, 0.0), size)
    den = box(valid.astype(np.float32), size)
    v = den >= 0.5
    return np.where(v, num / np.maximum(den, 1e-6), 0.0).astype(np.float32), v


def coarsen_cls(cls: np.ndarray, size: int) -> np.ndarray:
    if size == cls.shape[0]:
        return cls
    return np.asarray(Image.fromarray(cls).resize((size, size), Image.Resampling.NEAREST))


def upsample_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)


def to_tile_frame(ev: SplitEvaluator) -> pd.DataFrame:
    return ev.per_tile()[["id", "n_valid", "rmse", "mae"]].rename(columns={"n_valid": "n"}).assign(
        sse=lambda d: d.rmse ** 2 * d.n, sae=lambda d: d.mae * d.n)[["id", "n", "sse", "sae"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsds", type=float, nargs="+", default=[0.33, 0.50, 0.66, 1.00, 2.00])
    ap.add_argument("--limit", type=int, help="first N val tiles only (smoke test)")
    ap.add_argument("--smoke", action="store_true", help="no results.csv rows")
    a = ap.parse_args()

    ids = load_subset("full")["val"][: a.limit]
    cfg = {"name": "phase4_gsd_study", "split": "val", "model_run": DEFAULT_RUN.name, "gsds": a.gsds,
           "base_gsd": GAMUS_GSD_M, "tile": TILE, "n_tiles": len(ids),
           "rgb_downsample": "PIL BOX (area average)", "gt_downsample": "area average of valid pixels, valid if >=50%",
           "cls_downsample": "nearest", "strategy_B_upsample": "PIL BICUBIC to 1024 px",
           "strategy_B_back_to_grid": "area average (BOX)"}
    run = new_run(cfg["name"] + ("_smoke" if a.smoke else ""), cfg)
    write_manifest(run, {"val": ids})
    pred = NDSMPredictor(DEFAULT_RUN)
    sizes = {g: int(round(TILE * GAMUS_GSD_M / g)) for g in a.gsds}
    print("grid sizes:", sizes)
    evs = {(g, s): SplitEvaluator() for g in a.gsds for s in "AB"}
    t0 = time.time()
    for tid in tqdm(ids, desc="gsd study"):
        t = read_tile("val", tid)
        for g in a.gsds:
            n = sizes[g]
            rgb_c = coarsen_rgb(t["rgb"], n)
            gt_c, v_c = coarsen_gt(t["ndsm"], t["valid"], n)
            cls_c = coarsen_cls(t["cls"], n)
            if n == TILE:
                p = pred.predict(rgb_c)
                pa = pb = p
            else:
                pa = pred.predict(rgb_c)
                pb = box(pred.predict(upsample_rgb(rgb_c, TILE)), n)
            evs[(g, "A")].add(tid, pa, gt_c, v_c, cls_c)
            evs[(g, "B")].add(tid, pb, gt_c, v_c, cls_c)

    results, rows = {}, []
    for (g, s), ev in evs.items():
        key = f"gsd{g:.2f}_{s}"
        res = ev.result()
        results[key] = res
        ev.per_tile().to_csv(run / f"per_tile_{key}.csv", index=False)
        o, pc = res["overall"], res["per_class"]
        rows.append({"gsd_m": g, "strategy": s, "grid_px": sizes[g], "rmse": o["rmse"], "mae": o["mae"],
                     "r": o["pearson_r"], "bias": o["bias"], "rmse_building": pc["building"]["rmse"],
                     "rmse_tree": pc["tree"]["rmse"], "n_pixels": o["n"]})
        if not a.smoke:
            append_results(run, "val", res, method=f"phase4_{key}", deployable=True,
                           extra={"study": "gsd", "gsd_m": g, "strategy": s})
    save_metrics(run, "val", results)
    table = pd.DataFrame(rows).sort_values(["gsd_m", "strategy"])
    table.to_csv(run / "summary.csv", index=False)

    # paired tile bootstrap B vs A at each coarse GSD (same tiles, same grid)
    boots = {}
    for g in a.gsds:
        if sizes[g] == TILE:
            continue
        boots[f"{g:.2f}"] = paired(to_tile_frame(evs[(g, "B")]), to_tile_frame(evs[(g, "A")]), 2000, 0)
    (run / "bootstrap_B_minus_A.json").write_text(json.dumps(
        {"definition": "delta = metric_B - metric_A on the same tiles and grid; paired tile bootstrap, 2000 resamples",
         "results": boots}, indent=2))

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for s, style in (("A", "o-"), ("B", "s--")):
        d = table[table.strategy == s]
        lab = "A direct" if s == "A" else "B resample to 0.33"
        ax[0].plot(d.gsd_m, d.rmse, style, label=lab); ax[1].plot(d.gsd_m, d.mae, style, label=lab)
        ax[2].plot(d.gsd_m, d.bias, style, label=lab)
    for x, t in zip(ax, ("RMSE (m)", "MAE (m)", "bias (m)")):
        x.set_xlabel("ground resolution (m/px)"); x.set_title(t); x.set_xscale("log")
        x.set_xticks(a.gsds); x.set_xticklabels([f"{g:g}" for g in a.gsds]); x.legend(); x.grid(alpha=.3)
        x.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())   # no overlapping log minor labels
    ax[2].axhline(0, c="k", lw=.8)
    fig.suptitle(f"Phase 2b model vs ground resolution (GAMUS val, {len(ids)} tiles, simulated coarsening)")
    (run / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(run / "figures" / "gsd_curves.png", dpi=85); plt.close(fig)

    meta = json.loads((run / "meta.json").read_text())
    meta["wall_min"] = (time.time() - t0) / 60
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for g, b in boots.items():
        print(f"B-A @ {g}: dRMSE {b['RMSE']['delta']:+.3f} {np.round(b['RMSE']['ci95'], 3).tolist()} | "
              f"dMAE {b['MAE']['delta']:+.3f} {np.round(b['MAE']['ci95'], 3).tolist()}")
    print("wrote", run)


if __name__ == "__main__":
    main()
