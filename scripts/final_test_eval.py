"""ONE-TIME final evaluation on the untouched GAMUS test split (approved 2026-09-26).

Methods (all frozen, nothing is fitted or selected here):
    B0_zero   predict 0 m everywhere
    B1        Phase 1 deployable baseline: DA-V2-S @ r1022, per-image pct normalisation,
              global linear map from runs/<phase1 official>/calibration.json (fitted on train)
    phase2b   the kept Phase 2 model, runs/20260926-003630_phase2b_partial/checkpoints/best.pt

Safety:
  * `--split val` is a dry run that must reproduce the published val numbers.
  * `--split test` refuses to run if a completed final-test run already exists
    (runs/*_final_test/FROZEN). After it finishes it writes FROZEN; the report is final.
  * No threshold, calibration, checkpoint or hyperparameter is derived from test.

Usage:
    python scripts/final_test_eval.py --split val     # dry run (verification)
    python scripts/final_test_eval.py --split test    # the one-time evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depthwizard.calibration import LinearMap, normalise  # noqa: E402
from depthwizard.data.gamus import ROOT, load_subset, read_tile  # noqa: E402
from depthwizard.eval import viz  # noqa: E402
from depthwizard.eval.evaluate import SplitEvaluator  # noqa: E402
from depthwizard.inference import DEFAULT_RUN, NDSMPredictor  # noqa: E402
from depthwizard.models.dav2 import DAV2Relative, upsample  # noqa: E402
from depthwizard.utils.runs import RUNS_DIR, append_results, new_run, save_metrics, write_manifest  # noqa: E402
from paired_bootstrap import paired  # noqa: E402

PHASE1_RUN = ROOT / "runs" / "20260925-052422_phase1_baseline_repro"
EXPECTED_VAL = {"B0_zero": (7.845, 3.942), "B1": (6.602, 4.663), "phase2b": (3.148, 1.556)}  # published (RMSE, MAE)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frame(ev: SplitEvaluator) -> pd.DataFrame:
    d = ev.per_tile()
    return pd.DataFrame({"id": d.id, "n": d.n_valid, "sse": d.rmse ** 2 * d.n_valid, "sae": d.mae * d.n_valid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    a = ap.parse_args()

    frozen = sorted(RUNS_DIR.glob("*_final_test/FROZEN"))
    if a.split == "test" and frozen:
        sys.exit(f"REFUSED: the one-time test evaluation already exists ({frozen[0].parent.name}).")

    subset = "test_final" if a.split == "test" else "full"
    ids = load_subset(subset)[a.split]
    cal_path = PHASE1_RUN / "calibration.json"
    cal = json.loads(cal_path.read_text())
    assert cal["selected_b1"] == "B1_r1022_pct_lin"
    lin = LinearMap(); lin.a, lin.b = cal["maps"]["B1_r1022_pct_lin"]["a"], cal["maps"]["B1_r1022_pct_lin"]["b"]
    ckpt = DEFAULT_RUN / "checkpoints" / "best.pt"
    cfg = {"name": "final_test" if a.split == "test" else "final_test_dryrun_val_smoke", "split": a.split,
           "subset": subset, "n_tiles": len(ids),
           "methods": {"B0_zero": "0 m everywhere",
                       "B1": {"calibration": str(cal_path.relative_to(ROOT)), "sha256": sha256(cal_path),
                              "a": lin.a, "b": lin.b, "input_res": 1022, "normalise": "pct",
                              "model_revision": cal["model"]["revision"]},
                       "phase2b": {"run": DEFAULT_RUN.name, "checkpoint_sha256": sha256(ckpt)}},
           "policy": "one-time; nothing fitted, tuned or selected on these results"}
    run = new_run(cfg["name"], cfg)
    write_manifest(run, {a.split: ids})
    print(f"run dir: {run}  ({len(ids)} {a.split} tiles)")

    dav2 = DAV2Relative(cal["model"]["revision"], "cuda")
    model = NDSMPredictor(DEFAULT_RUN)        # also sets deterministic numerics
    evs = {m: SplitEvaluator() for m in ("B0_zero", "B1", "phase2b")}
    t0 = time.time()
    for tid in tqdm(ids, desc=f"final eval ({a.split})"):
        t = read_tile(a.split, tid)
        gt, v, cls = t["ndsm"], t["valid"], t["cls"]
        d = upsample(dav2.predict(t["rgb"], 1022), gt.shape)
        preds = {"B0_zero": np.zeros_like(gt), "B1": lin(normalise(d, "pct")), "phase2b": model.predict(t["rgb"])}
        for m, p in preds.items():
            evs[m].add(tid, p, gt, v, cls)

    results = {m: ev.result() for m, ev in evs.items()}
    save_metrics(run, a.split, results)
    for m, ev in evs.items():
        ev.per_tile().to_csv(run / f"per_tile_{m}.csv", index=False)
    boots = {f"{x}_minus_{y}": paired(frame(evs[x]), frame(evs[y]), 2000, 0)
             for x, y in (("phase2b", "B1"), ("phase2b", "B0_zero"), ("B1", "B0_zero"))}
    (run / "paired_bootstrap.json").write_text(json.dumps(
        {"definition": "delta = metric_x - metric_y, paired tile bootstrap, 2000 resamples, seed 0", "results": boots},
        indent=2))

    fig = run / "figures"; fig.mkdir(exist_ok=True)
    cities = sorted(results["phase2b"]["per_city"])
    viz.grouped_bars(fig / "rmse_per_city.png", {m: {c: r["per_city"][c]["rmse"] for c in cities}
                                                 for m, r in results.items()}, "RMSE (m)", f"{a.split}: RMSE per city")
    cls_names = ["ground", "low_vegetation", "road", "building", "tree", "water"]
    viz.grouped_bars(fig / "rmse_per_class.png", {m: {c: r["per_class"].get(c, {}).get("rmse") for c in cls_names}
                                                  for m, r in results.items()}, "RMSE (m)", f"{a.split}: RMSE per class")
    viz.grouped_bars(fig / "mae_per_height_bin.png", {m: {b: x["mae"] for b, x in r["per_height_bin"].items()}
                                                      for m, r in results.items()}, "MAE (m)",
                     f"{a.split}: MAE by true height band")

    print(f"\n{a.split.upper()} ({len(ids)} tiles, pooled)   RMSE    MAE     r      bias  | bldg   tree")
    ok = True
    for m, r in results.items():
        o, pc = r["overall"], r["per_class"]
        rr = "  n/a " if o["pearson_r"] is None else f"{o['pearson_r']:6.3f}"
        print(f"  {m:10s}               {o['rmse']:6.3f} {o['mae']:6.3f} {rr} {o['bias']:+6.3f} | "
              f"{pc['building']['rmse']:5.2f} {pc['tree']['rmse']:5.2f}")
        if a.split == "val":
            er, em = EXPECTED_VAL[m]
            ok &= abs(o["rmse"] - er) < 5e-4 and abs(o["mae"] - em) < 5e-4
        else:
            append_results(run, "test", r, method=m, deployable=True, extra={"final_test": True})
    for c in cities:
        print("  " + c + ": " + " | ".join(f"{m} {results[m]['per_city'][c]['rmse']:.3f}/{results[m]['per_city'][c]['mae']:.3f}"
                                           for m in results))
    for k, b in boots.items():
        print(f"  {k}: dRMSE {b['RMSE']['delta']:+.3f} {np.round(b['RMSE']['ci95'], 3).tolist()}  "
              f"dMAE {b['MAE']['delta']:+.3f} {np.round(b['MAE']['ci95'], 3).tolist()}")
    meta = json.loads((run / "meta.json").read_text()); meta["wall_min"] = (time.time() - t0) / 60
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    if a.split == "val":
        print("DRY RUN on val:", "REPRODUCES published numbers -> PASS" if ok else "MISMATCH -> FAIL")
        sys.exit(0 if ok else 1)
    (run / "FROZEN").write_text(f"one-time final test evaluation completed {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"FROZEN: {run}")


if __name__ == "__main__":
    main()
