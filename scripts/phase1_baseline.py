"""Phase 1 baselines on GAMUS val: B0 (trivial), B1 (DA-V2-S + global calibration),
B2 (oracle per-image calibration baseline; diagnostic only, not deployable).

Usage:
    python scripts/phase1_baseline.py --config configs/phase1_baseline.yaml
    python scripts/phase1_baseline.py --config ... --subset phase0 --name smoke --smoke   # smoke test

Stages (all in one call):
  1. cache   DA-V2-S raw outputs for every train/val tile and input resolution
  2. verify  recompute a few tiles and compare with the cache (determinism)
  3. fit     global calibration maps on TRAIN pixels only
  4. eval    every method on VAL in one pass, pooled metrics + breakdowns
  5. report  calibration.json, metrics_val.json, per_tile_val.csv, results.csv rows, figures
The test split is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.calibration import IsotonicMap, LinearMap, normalise, oracle_affine  # noqa: E402
from depthwizard.data.gamus import ROOT, load_subset, read_rgb, read_tile  # noqa: E402
from depthwizard.eval import viz  # noqa: E402
from depthwizard.eval.metrics import HeightMetrics, _Acc, label_stats, pixel_stats  # noqa: E402
from depthwizard.models.dav2 import DAV2Relative, set_deterministic, upsample  # noqa: E402
from depthwizard.utils.runs import (append_results, new_run, save_metrics,  # noqa: E402
                                    set_seed, write_manifest)

TILE = (1024, 1024)


def cache_dir(cfg, res: int) -> Path:
    return ROOT / "data" / "cache" / f"dav2s_{cfg['model']['revision'][:8]}_r{res}"


def load_depth(cfg, res, split, tid) -> np.ndarray:
    return upsample(np.load(cache_dir(cfg, res) / split / f"{tid}.npy"), TILE)


def b1_name(res, norm, mp):
    return f"B1_r{res}_{norm}_{mp}"


# --------------------------------------------------------------------------- stages
def stage_cache(cfg, splits, model_holder) -> dict:
    info = {}
    for res in cfg["grid"]["input_res"]:
        todo = [(s, t) for s, ids in splits.items() for t in ids
                if not (cache_dir(cfg, res) / s / f"{t}.npy").exists()]
        if not todo:
            print(f"[cache] r{res}: all cached"); continue
        model = model_holder()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for s, t in tqdm(todo, desc=f"DA-V2-S r{res}"):
            d = model.predict(read_rgb(s, t), res)
            p = cache_dir(cfg, res) / s / f"{t}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, d.astype(np.float32))
        info[f"r{res}"] = {"tiles": len(todo), "sec_per_tile": (time.time() - t0) / len(todo),
                           "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
                           "model_output_shape": list(d.shape)}
        print(f"[cache] r{res}: {info[f'r{res}']}")
    return info


def stage_verify(cfg, splits, model_holder, n: int) -> dict:
    model = model_holder()
    ids = splits["val"][:n]
    out = {}
    for res in cfg["grid"]["input_res"]:
        diffs = [float(np.abs(model.predict(read_rgb("val", t), res)
                              - np.load(cache_dir(cfg, res) / "val" / f"{t}.npy")).max()) for t in ids]
        out[f"r{res}_max_abs_diff"] = max(diffs)
    print(f"[verify] recomputed {len(ids)} val tiles: {out}")
    return out


def stage_fit(cfg, splits):
    g, seed, k = cfg["grid"], cfg["seed"], cfg["calibration"]["pixels_per_tile"]
    X = {(r, n): [] for r in g["input_res"] for n in g["normalise"]}
    raw = {r: [] for r in g["input_res"]}
    Y = []
    for i, tid in enumerate(tqdm(splits["train"], desc="fit: sample train pixels")):
        t = read_tile("train", tid)
        vidx = np.flatnonzero(t["valid"].ravel())
        idx = np.random.default_rng([seed, i]).choice(vidx, min(k, vidx.size), replace=False)
        Y.append(t["ndsm"].ravel()[idx])
        for r in g["input_res"]:
            d = load_depth(cfg, r, "train", tid)
            raw[r].append(d.ravel()[idx])
            for n in g["normalise"]:
                X[(r, n)].append(normalise(d, n).ravel()[idx])
    Y = np.concatenate(Y)
    X = {key: np.concatenate(v) for key, v in X.items()}
    raw = {r: np.concatenate(v) for r, v in raw.items()}

    sign = {f"r{r}": float(np.corrcoef(raw[r], Y)[0, 1]) for r in raw}
    print(f"[fit] sign check, Pearson(raw DA output, GT nDSM) on train pixels: {sign}")
    maps = {}
    for (r, n), x in X.items():
        for mp in g["mapping"]:
            maps[b1_name(r, n, mp)] = (LinearMap() if mp == "lin" else IsotonicMap()).fit(x, Y)
    b0_mean = float(Y.mean())
    return maps, b0_mean, sign, X, Y


def stage_eval(cfg, splits, maps, b0_mean):
    g, ev = cfg["grid"], cfg["eval"]
    edges = ev["height_bins"]
    bin_names = {i: f"{edges[i]}-{edges[i+1]}m" if edges[i + 1] < 1000 else f">{edges[i]}m"
                 for i in range(len(edges) - 1)}
    methods = ["B0_zero", "B0_mean", *maps, *[f"B2_oracle_r{r}" for r in g["input_res"]]]
    acc = {m: {"cls": HeightMetrics(), "bin": HeightMetrics(label_names=bin_names), "city": {}} for m in methods}
    per_tile, scatter = [], {m: ([], []) for m in methods}

    for i, tid in enumerate(tqdm(splits["val"], desc="eval val")):
        t = read_tile("val", tid)
        gt, v, cls = t["ndsm"], t["valid"], t["cls"]
        city = tid.split("_")[0]
        hb = np.digitize(gt, edges[1:-1]).astype(np.uint8)
        preds = {"B0_zero": np.zeros_like(gt), "B0_mean": np.full_like(gt, b0_mean)}
        oracle_ab = {}
        for r in g["input_res"]:
            d = load_depth(cfg, r, "val", tid)
            preds[f"B2_oracle_r{r}"], a, b = oracle_affine(d, gt, v)
            oracle_ab[r] = (a, b)
            for n in g["normalise"]:
                dn = normalise(d, n)
                for mp in g["mapping"]:
                    preds[b1_name(r, n, mp)] = maps[b1_name(r, n, mp)](dn)
        sidx = np.random.default_rng([cfg["seed"], 10_000 + i]).choice(
            np.flatnonzero(v.ravel()), min(ev["scatter_pixels_per_tile"], int(v.sum())), replace=False)
        row = {"id": tid, "city": city, "valid_frac": float(v.mean()),
               "gt_p99": float(np.percentile(gt[v], 99)), "frac_building": float((cls[v] == 3).mean()),
               **{f"oracle_a_r{r}": ab[0] for r, ab in oracle_ab.items()}}
        gt_v = gt[v]
        n_bins = len(edges) - 1
        joint = cls[v].astype(np.int64) * n_bins + hb[v]          # one pass -> class x height-bin
        for m, p in preds.items():
            T = label_stats(pixel_stats(p[v], gt_v), joint, 7 * n_bins).reshape(7, n_bins, 8)
            acc[m]["cls"].add_table(T.sum(1))
            acc[m]["bin"].add_table(T.sum(0))
            tot = T.sum((0, 1))
            acc[m]["city"].setdefault(city, HeightMetrics()).overall.add_vec(tot)
            ta = _Acc(); ta.add_vec(tot); tm = ta.result()
            row[f"{m}_rmse"], row[f"{m}_mae"], row[f"{m}_r"] = tm["rmse"], tm["mae"], tm["pearson_r"]
            scatter[m][0].append(gt.ravel()[sidx]); scatter[m][1].append(p.ravel()[sidx])
        per_tile.append(row)

    metrics = {}
    for m in methods:
        res = acc[m]["cls"].result()
        res["per_height_bin"] = acc[m]["bin"].result()["per_class"]
        res["per_city"] = {c: h.result()["overall"] for c, h in sorted(acc[m]["city"].items())}
        metrics[m] = res
    scatter = {m: (np.concatenate(a), np.concatenate(b)) for m, (a, b) in scatter.items()}
    return metrics, pd.DataFrame(per_tile), scatter


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", help="override subset (e.g. phase0 for the smoke test)")
    ap.add_argument("--name", help="override run name")
    ap.add_argument("--verify-tiles", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="pipeline check only: no results.csv rows")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.subset: cfg["subset"] = args.subset
    if args.name: cfg["name"] = args.name
    set_seed(cfg["seed"]); set_deterministic()

    splits = load_subset(cfg["subset"])
    splits = {s: splits[s] for s in ("train", "val")}           # test is never loaded
    assert "test" not in cfg["eval"]["splits"], "Phase 1 must not touch the test split"

    run_dir = new_run(cfg["name"], cfg)
    write_manifest(run_dir, splits)
    print(f"run dir: {run_dir}")

    _model = {}
    def model_holder():
        if "m" not in _model:
            _model["m"] = DAV2Relative(cfg["model"]["revision"], "cuda")
        return _model["m"]

    cache_info = stage_cache(cfg, splits, model_holder)
    verify = stage_verify(cfg, splits, model_holder, args.verify_tiles)
    maps, b0_mean, sign, X, Y = stage_fit(cfg, splits)
    metrics, per_tile, scatter = stage_eval(cfg, splits, maps, b0_mean)

    # ---- selection: B1 only, by val RMSE. B2 oracle is never a candidate.
    b1 = {m: metrics[m]["overall"]["rmse"] for m in maps}
    best = min(b1, key=b1.get)
    print(f"[select] B1 by val RMSE -> {best} ({b1[best]:.3f} m)")

    # ---- save
    calib = {
        "model": cfg["model"], "train_subset": cfg["subset"], "seed": cfg["seed"],
        "pixels_per_tile": cfg["calibration"]["pixels_per_tile"], "n_fit_pixels": int(Y.size),
        "pipeline": "h = max(0, f(normalise(d))), d = DA-V2-S output bilinearly upsampled to 1024x1024",
        "b0_mean_m": b0_mean, "sign_check_pearson_raw_vs_gt": sign,
        "selected_b1": best, "selected_by": "val_rmse",
        "maps": {k: m.to_dict() for k, m in maps.items()},
    }
    (run_dir / "calibration.json").write_text(json.dumps(calib, indent=1))
    save_metrics(run_dir, "val", metrics)
    per_tile.to_csv(run_dir / "per_tile_val.csv", index=False)
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update({"cache": cache_info, "verify_determinism": verify, "selected_b1": best})
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for m, res in (metrics.items() if not args.smoke else []):
        deployable = not m.startswith("B2")
        append_results(run_dir, "val", res, method=m, deployable=deployable,
                       extra={"selected": m == best, "subset": cfg["subset"],
                              **({"role": "oracle per-image calibration baseline (diagnostic, not deployable)"}
                                 if not deployable else {})})

    # ---- figures
    fig = run_dir / "figures"
    r_best = int(best.split("_")[1][1:]); n_best = best.split("_")[2]
    oracle = f"B2_oracle_r{r_best}"
    pt = per_tile.sort_values(f"{best}_rmse").reset_index(drop=True)
    picks = {"best": pt.id.iloc[0], "median": pt.id.iloc[len(pt) // 2], "worst": pt.id.iloc[-1]}
    def pick(tag, ordered):                      # first tile not already shown
        tid = next((t for t in ordered if t not in picks.values()), None)
        if tid: picks[tag] = tid
    pick("tallest", per_tile.sort_values("gt_p99", ascending=False).id)
    for c in sorted(per_tile.city.unique()):
        cands = pt[pt.city == c].id.tolist()
        pick(f"{c}_median", cands[len(cands) // 2:] + cands[:len(cands) // 2])
    extra = [t for t in pt.id if t not in picks.values()]
    while len(picks) < cfg["eval"]["n_panel_tiles"] and extra:
        picks[f"extra{len(picks)}"] = extra.pop(len(extra) // 2)
    for tag, tid in picks.items():
        t = read_tile("val", tid)
        d = load_depth(cfg, r_best, "val", tid)
        preds = {f"{best} (deployable)": maps[best](normalise(d, n_best)),
                 f"{oracle} (oracle, not deployable)": oracle_affine(d, t["ndsm"], t["valid"])[0]}
        row = per_tile.set_index("id").loc[tid]
        viz.tile_panel(fig / f"panel_{tag}_{tid}.png", tid, t["rgb"], t["ndsm"], t["valid"], d, preds,
                       err_key=f"{best} (deployable)",
                       title_extra=f"{tag}: B1 RMSE {row[f'{best}_rmse']:.2f} m, "
                                   f"oracle RMSE {row[f'{oracle}_rmse']:.2f} m, B0-zero RMSE {row['B0_zero_rmse']:.2f} m")
    viz.density_scatter(fig / "scatter_val.png",
                        {f"{best} (deployable)": scatter[best],
                         f"{oracle} (oracle, not deployable)": scatter[oracle],
                         "B0_mean": scatter["B0_mean"]})
    show = ["B0_zero", "B0_mean", best, oracle]
    cls_names = ["ground", "low_vegetation", "road", "building", "tree", "water"]
    viz.grouped_bars(fig / "rmse_per_class.png",
                     {m: {c: metrics[m]["per_class"].get(c, {}).get("rmse") for c in cls_names} for m in show},
                     "RMSE (m)", "Val RMSE per class")
    viz.grouped_bars(fig / "mae_per_height_bin.png",
                     {m: {b: v["mae"] for b, v in metrics[m]["per_height_bin"].items()} for m in show},
                     "MAE (m)", "Val MAE by true height band")
    viz.grouped_bars(fig / "rmse_all_b1.png",
                     {"val RMSE": {m.replace("B1_", ""): metrics[m]["overall"]["rmse"] for m in maps}},
                     "RMSE (m)", "All 8 B1 settings (val)")
    for r in cfg["grid"]["input_res"]:
        for n in cfg["grid"]["normalise"]:
            x = X[(r, n)]
            viz.calibration_curve(fig / f"calib_curve_r{r}_{n}.png", x, Y,
                                  {"F-lin": (maps[b1_name(r, n, 'lin')], "r-"),
                                   "F-iso": (maps[b1_name(r, n, 'iso')], "c-")},
                                  f"train pixels, r{r} {n} (n={Y.size:,})")

    # ---- console summary
    print("\nVAL (pooled)          RMSE    MAE     r      bias   | RMSE bldg  RMSE tree")
    for m in ["B0_zero", "B0_mean", *sorted(maps, key=b1.get), *[k for k in metrics if k.startswith("B2")]]:
        o, pc = metrics[m]["overall"], metrics[m]["per_class"]
        r_ = "  n/a " if o["pearson_r"] is None else f"{o['pearson_r']:6.3f}"
        tag = " <- selected" if m == best else (" (oracle)" if m.startswith("B2") else "")
        print(f"{m:22s}{o['rmse']:6.3f}  {o['mae']:6.3f}  {r_}  {o['bias']:+6.3f} | "
              f"{pc['building']['rmse']:8.3f}  {pc['tree']['rmse']:8.3f}{tag}")
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
