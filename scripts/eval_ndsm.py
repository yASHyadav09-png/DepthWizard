"""Final validation analysis of a Phase 2 run (val only; the test split is refused).

  1. load checkpoints/best.pt, evaluate all 859 val tiles TWICE -> must be bit-identical
  2. metrics_val_final.json + per_tile_val_final.csv
  3. paired bootstrap vs Phase 1 references (B1, B0-zero, B0-mean, oracle R518/R1022)
  4. figures: training curves, the same 8 panel tiles as Phase 1 (B1 vs this model),
     scatter, per-class and per-height-band bars

Usage:  python scripts/eval_ndsm.py --run runs/<phase2 run> --phase1 runs/<phase1 official run>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.calibration import LinearMap, normalise  # noqa: E402
from depthwizard.data.gamus import ROOT, load_subset, read_tile  # noqa: E402
from depthwizard.eval import viz  # noqa: E402
from depthwizard.eval.evaluate import SplitEvaluator  # noqa: E402
from depthwizard.models.dav2 import set_deterministic, upsample  # noqa: E402
from depthwizard.models.ndsm import NDSMModel  # noqa: E402
from depthwizard.data.gamus import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

DEVICE = "cuda"


def load_model(run: Path, cfg: dict) -> tuple[NDSMModel, dict]:
    ck = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    m = NDSMModel(cfg["model"]["revision"], cfg["model"]["id"])
    m.set_trainable(cfg["trainable"]["mode"], cfg["trainable"]["unfreeze_last_blocks"])
    missing, unexpected = m.load_state_dict(ck["trainable"], strict=False)
    assert not unexpected, unexpected
    assert all(not k.startswith(("net.neck", "net.head")) for k in missing), "trainable weights missing"
    return m.to(DEVICE).eval(), ck


def to_input(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


@torch.no_grad()
def run_eval(model, ids, amp):
    ev = SplitEvaluator()
    for tid in ids:
        t = read_tile("val", tid)
        p = model.predict_tile(to_input(t["rgb"]).to(DEVICE), amp=amp)[0, 0].cpu().numpy()
        ev.add(tid, p, t["ndsm"], t["valid"], t["cls"])
    return ev.result(), ev.per_tile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--name", help="label for figures/tables (default: config name)")
    ap.add_argument("--compare-run", type=Path, help="earlier Phase 2 run (e.g. 2a) for delta = this - that")
    ap.add_argument("--compare-name", default="phase2a")
    a = ap.parse_args()
    run, p1 = a.run, a.phase1
    cfg = yaml.safe_load(open(run / "config.yaml"))
    name = a.name or cfg["name"]
    splits = load_subset(cfg["subset"])
    assert "test" not in splits
    ids = splits["val"]
    set_deterministic()
    model, ck = load_model(run, cfg)
    cmp_model = None
    if a.compare_run:
        cmp_cfg = yaml.safe_load(open(a.compare_run / "config.yaml"))
        cmp_model, _ = load_model(a.compare_run, cmp_cfg)
        cmp_metrics = json.loads((a.compare_run / "metrics_val_final.json").read_text())
    amp = cfg["eval"]["amp"] == "bf16"
    print(f"best.pt: epoch {ck['epoch']}, step {ck['global_step']}, s0 {ck['s0']:.6f}")

    # 1. evaluate twice
    r1, t1 = run_eval(model, ids, amp)
    r2, t2 = run_eval(model, ids, amp)
    identical = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True) and t1.equals(t2)
    train_time = json.loads((run / "metrics_val.json").read_text())
    same_as_training = abs(train_time["overall"]["rmse"] - r1["overall"]["rmse"]) < 1e-9
    print(f"re-evaluation bit-identical: {identical}; matches in-training best eval: {same_as_training}")
    (run / "metrics_val_final.json").write_text(json.dumps(r1, indent=2))
    t1.to_csv(run / "per_tile_val_final.csv", index=False)
    repro = {"evaluated_twice_identical": identical, "matches_in_training_eval": same_as_training,
             "best_epoch": ck["epoch"], "best_step": ck["global_step"]}
    (run / "eval_reproducibility.json").write_text(json.dumps(repro, indent=2))

    # 2. paired bootstrap vs Phase 1 references
    subprocess.run([sys.executable, str(ROOT / "scripts" / "paired_bootstrap.py"),
                    "--model", str(run / "per_tile_val_final.csv"), "--model-name", name,
                    "--reference", str(p1 / "per_tile_val.csv"),
                    "--ref-methods", "B1_r1022_pct_lin", "B0_zero", "B0_mean", "B2_oracle_r518", "B2_oracle_r1022",
                    *(["--ref-model", str(a.compare_run / "per_tile_val_final.csv")] if a.compare_run else []),
                    "--out", str(run / "paired_bootstrap.json")], check=True)

    # 3. figures
    fig_dir = run / "figures"; fig_dir.mkdir(exist_ok=True)
    p1m = json.loads((p1 / "metrics_val.json").read_text())
    cal = json.loads((p1 / "calibration.json").read_text())
    b1 = cal["selected_b1"]                                  # B1_r1022_pct_lin
    lin = LinearMap(); lin.a, lin.b = cal["maps"][b1]["a"], cal["maps"][b1]["b"]
    res_b1, norm_b1 = int(b1.split("_")[1][1:]), b1.split("_")[2]
    cache = ROOT / "data" / "cache" / f"dav2s_{cal['model']['revision'][:8]}_r{res_b1}" / "val"

    # training curves
    tl = pd.read_csv(run / "train_log.csv"); vl = pd.read_csv(run / "val_log.csv")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ax[0].plot(tl.step, tl.l1, label="L1"); ax[0].plot(tl.step, tl.grad_weighted, label="0.5 x L_grad")
    ax[0].plot(tl.step, tl.total, label="total", lw=1, alpha=.6); ax[0].set_xlabel("optimizer step")
    ax[0].set_ylabel("train loss (m)"); ax[0].legend(); ax[0].set_title("training losses (train crops)")
    ax[1].plot(tl.step, tl.grad_to_l1_ratio); ax[1].set_title("0.5 x L_grad / L1"); ax[1].set_xlabel("step")
    ax[2].plot(vl.epoch, vl.rmse, "o-", label="val RMSE"); ax[2].plot(vl.epoch, vl.mae, "o-", label="val MAE")
    for v, lab, c in ((p1m[b1]["overall"]["rmse"], "B1 RMSE", "C0"), (p1m["B0_zero"]["overall"]["mae"], "B0-zero MAE", "C1"),
                      (p1m["B2_oracle_r518"]["overall"]["rmse"], "oracle RMSE", "C2")):
        ax[2].axhline(v, ls="--", c=c, lw=1, label=lab)
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("m"); ax[2].legend(fontsize=8); ax[2].set_title("validation (859 tiles)")
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=80); plt.close(fig)

    # same 8 tiles as Phase 1
    p1_ids = set(pd.read_csv(p1 / "per_tile_val.csv").id)
    panels = sorted(p for p in (p1 / "figures").glob("panel_*.png"))
    t1i = t1.set_index("id"); p1t = pd.read_csv(p1 / "per_tile_val.csv").set_index("id")
    sc_g, sc_m, sc_b, sc_c = [], [], [], []
    rng = np.random.default_rng(0)
    for pth in panels:
        stem = pth.stem[len("panel_"):]
        tid = next(t for t in p1_ids if stem.endswith(t))
        tag = stem[: -len(tid) - 1]
        t = read_tile("val", tid)
        with torch.no_grad():
            pm = model.predict_tile(to_input(t["rgb"]).to(DEVICE), amp=amp)[0, 0].cpu().numpy()
        d = upsample(np.load(cache / f"{tid}.npy"), (1024, 1024))
        pb = lin(normalise(d, norm_b1))
        preds = {"B1 (Phase 1)": pb}
        if cmp_model is not None:
            with torch.no_grad():
                preds[a.compare_name] = cmp_model.predict_tile(to_input(t["rgb"]).to(DEVICE), amp=amp)[0, 0].cpu().numpy()
        preds[name] = pm
        viz.tile_panel(fig_dir / f"panel_{tag}_{tid}.png", tid, t["rgb"], t["ndsm"], t["valid"], d,
                       preds, err_key=f"{name}",
                       title_extra=f"{tag}: {name} RMSE {t1i.loc[tid, 'rmse']:.2f} m | "
                                   f"B1 RMSE {p1t.loc[tid, b1 + '_rmse']:.2f} m | B0-zero {p1t.loc[tid, 'B0_zero_rmse']:.2f} m")

    # scatter on a fixed pixel sample (2000 per tile) from all val tiles
    for i, tid in enumerate(ids):
        t = read_tile("val", tid)
        v = np.flatnonzero(t["valid"].ravel())
        s = np.random.default_rng([0, 10_000 + i]).choice(v, min(2000, v.size), replace=False)
        with torch.no_grad():
            pm = model.predict_tile(to_input(t["rgb"]).to(DEVICE), amp=amp)[0, 0].cpu().numpy()
        pb = lin(normalise(upsample(np.load(cache / f"{tid}.npy"), (1024, 1024)), norm_b1))
        sc_g.append(t["ndsm"].ravel()[s]); sc_m.append(pm.ravel()[s]); sc_b.append(pb.ravel()[s])
        if cmp_model is not None:
            with torch.no_grad():
                pc = cmp_model.predict_tile(to_input(t["rgb"]).to(DEVICE), amp=amp)[0, 0].cpu().numpy()
            sc_c.append(pc.ravel()[s])
    g = np.concatenate(sc_g)
    pairs = {"B1 (Phase 1)": (g, np.concatenate(sc_b))}
    if sc_c:
        pairs[a.compare_name] = (g, np.concatenate(sc_c))
    pairs[name] = (g, np.concatenate(sc_m))
    viz.density_scatter(fig_dir / "scatter_val.png", pairs)

    tables = {"B0_zero": p1m["B0_zero"], "B1": p1m[b1], "oracle_r518": p1m["B2_oracle_r518"],
              **({a.compare_name: cmp_metrics} if a.compare_run else {}), name: r1}
    cls_names = ["ground", "low_vegetation", "road", "building", "tree", "water"]
    viz.grouped_bars(fig_dir / "rmse_per_class.png",
                     {k: {c: v["per_class"].get(c, {}).get("rmse") for c in cls_names} for k, v in tables.items()},
                     "RMSE (m)", "Val RMSE per class")
    viz.grouped_bars(fig_dir / "mae_per_height_bin.png",
                     {k: {b: x["mae"] for b, x in v["per_height_bin"].items()} for k, v in tables.items()},
                     "MAE (m)", "Val MAE by true height band")

    o = r1["overall"]; pc = r1["per_class"]
    print(f"\n{name} (val, pooled): RMSE {o['rmse']:.3f}  MAE {o['mae']:.3f}  r {o['pearson_r']:.3f}  "
          f"bias {o['bias']:+.3f} | bldg RMSE {pc['building']['rmse']:.3f}  tree RMSE {pc['tree']['rmse']:.3f}")
    for b, x in r1["per_height_bin"].items():
        print(f"   {b:8s} MAE {x['mae']:6.2f}  bias {x['bias']:+6.2f}")
    for c, x in r1["per_city"].items():
        print(f"   {c:4s} RMSE {x['rmse']:.3f}  MAE {x['mae']:.3f}  r {x['pearson_r']:.3f}")
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
