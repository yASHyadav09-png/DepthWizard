"""Phase 2: supervised RGB -> nDSM training of Depth Anything V2 Small.

Usage:
    python scripts/train_ndsm.py --config configs/phase2a_frozen.yaml --speed-test   # 200 steps
    python scripts/train_ndsm.py --config configs/phase2a_frozen.yaml                # full run
    python scripts/train_ndsm.py --config ... --resume runs/<run>/checkpoints/last.pt

Only the train and val splits are ever loaded; the test split is refused.
Model selection: lowest pooled val RMSE, evaluated every `eval.every_epochs`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.gamus import GAMUSDataset, load_subset, worker_init_fn  # noqa: E402
from depthwizard.eval.evaluate import SplitEvaluator  # noqa: E402
from depthwizard.losses import total_loss  # noqa: E402
from depthwizard.models.dav2 import set_deterministic  # noqa: E402
from depthwizard.models.ndsm import NDSMModel, fit_initial_scale  # noqa: E402
from depthwizard.utils.runs import (append_results, git_info, new_run, save_metrics,  # noqa: E402
                                    set_seed, write_manifest)

DEVICE = "cuda"


# ----------------------------------------------------------------------------- helpers
class GPUSampler(threading.Thread):
    """Samples GPU utilisation / memory with nvidia-smi once per second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.mem, self._stop = [], [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True, timeout=5).strip().splitlines()[0]
                u, m = (float(x) for x in out.split(","))
                self.util.append(u); self.mem.append(m)
            except Exception:
                pass
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        return {"gpu_util_mean_pct": float(np.mean(self.util)) if self.util else None,
                "gpu_util_median_pct": float(np.median(self.util)) if self.util else None,
                "nvidia_smi_mem_max_mb": float(max(self.mem)) if self.mem else None,
                "samples": len(self.util)}


def lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def make_loaders(cfg, bs, seed):
    tr = GAMUSDataset(cfg["subset"], "train", crop=cfg["data"]["crop"], train=True, jitter=cfg["data"]["jitter"])
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True,
                                     num_workers=cfg["data"]["workers"], pin_memory=True,
                                     persistent_workers=True, worker_init_fn=worker_init_fn,
                                     generator=g, prefetch_factor=4)
    return tr, dl


@torch.no_grad()
def evaluate(model, cfg, desc="val"):
    ds = GAMUSDataset(cfg["subset"], "val", crop=None, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["eval"]["workers"])
    model.eval()
    ev = SplitEvaluator()
    for b in tqdm(dl, desc=desc, leave=False):
        pred = model.predict_tile(b["image"].to(DEVICE, non_blocking=True),
                                  amp=cfg["eval"]["amp"] == "bf16")[0, 0].cpu().numpy()
        ev.add(b["id"][0], pred, b["ndsm"][0, 0].numpy(), b["valid"][0, 0].numpy(), b["cls"][0].numpy())
    return ev.result(), ev.per_tile()


def save_ckpt(path, model, opt, sched, epoch, step, best, cfg, bs, accum):
    torch.save({"trainable": model.trainable_state_dict(), "s0": float(model.s0),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epoch, "global_step": step, "best": best, "config": cfg,
                "batch": [bs, accum],
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(), "python": random.getstate()}}, path)


def build_model(cfg):
    model = NDSMModel(cfg["model"]["revision"], cfg["model"]["id"])
    model.set_trainable(cfg["trainable"]["mode"], cfg["trainable"]["unfreeze_last_blocks"])
    return model.to(DEVICE)


# ----------------------------------------------------------------------------- train
def train_steps(cfg, model, dl, bs, accum, max_opt_steps, opt, sched, log_path, start_step=0,
                epoch=0, speed=None):
    """Runs one epoch (or until max_opt_steps). Returns (global_step, stats)."""
    model.train()
    lam, scales = cfg["loss"]["lambda_grad"], cfg["loss"]["grad_scales"]
    step, micro = start_step, 0
    agg = {k: 0.0 for k in ("l1", "grad", "grad_weighted", "total")}; n_agg = 0
    t_data = t_comp = 0.0
    t_last = time.perf_counter(); samples = 0
    it = iter(dl)
    opt.zero_grad(set_to_none=True)
    while True:
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        x = b["image"].to(DEVICE, non_blocking=True)
        y = b["ndsm"].to(DEVICE, non_blocking=True)
        v = b["valid"].to(DEVICE, non_blocking=True)
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"] == "bf16"):
            pred = model(x)
        L = total_loss(pred.float(), y, v, lam, scales)
        (L["total"] / accum).backward()
        micro += 1; samples += x.shape[0]
        for k in agg:
            agg[k] = agg[k] + L[k].detach()          # stays on GPU: no per-step sync
        n_agg += 1
        if speed is not None:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_data += t1 - t0; t_comp += t2 - t1
        if micro % accum:
            continue
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       cfg["optim"]["grad_clip"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg["log_every"] == 0 or (speed is not None and step == max_opt_steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            agg = {k: float(v) for k, v in agg.items()}
            row = {"step": step, "epoch": epoch, "lr": sched.get_last_lr()[0],
                   **{k: agg[k] / n_agg for k in agg},
                   "grad_to_l1_ratio": (agg["grad_weighted"] / max(agg["l1"], 1e-12)),
                   "samples_per_s": samples / (now - t_last),
                   "data_wait_frac": t_data / max(t_data + t_comp, 1e-9),
                   "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
            new = not log_path.exists()
            with open(log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new: w.writeheader()
                w.writerow(row)
            tqdm.write(f"step {step} ep {epoch} lr {row['lr']:.2e} L1 {row['l1']:.3f} "
                       f"grad {row['grad']:.3f} (x{lam} = {row['grad_weighted']:.3f}) total {row['total']:.3f} "
                       f"| {row['samples_per_s']:.1f} samp/s, data wait {row['data_wait_frac']:.0%}")
            if speed is not None:
                speed.append(row)
            agg = {k: 0.0 for k in agg}; n_agg = 0; t_data = t_comp = 0.0
            t_last = time.perf_counter(); samples = 0
        if max_opt_steps and step >= max_opt_steps:
            break
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--speed-test", action="store_true", help="200 optimizer steps, no eval, no checkpoints")
    ap.add_argument("--speed-steps", type=int, default=200)
    ap.add_argument("--resume")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    splits = load_subset(cfg["subset"])
    assert "test" not in splits, "Phase 2 must not load the test split"
    splits = {s: splits[s] for s in ("train", "val")}
    gi = git_info()
    if gi["dirty"] and not args.allow_dirty:
        sys.exit("working tree is dirty; commit first (or --allow-dirty for debugging)")

    set_seed(cfg["seed"]); set_deterministic()
    name = cfg["name"] + ("_speedtest" if args.speed_test else "")
    run_dir = Path(args.resume).parents[1] if args.resume else new_run(name, cfg)
    if not args.resume:
        write_manifest(run_dir, splits)
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["phase1_commit"] = cfg["phase1_commit"]
    print(f"run dir: {run_dir}")

    # ---- batch size with the approved fallback order (effective batch unchanged)
    options = [[cfg["batch"]["size"], cfg["batch"]["accum"]], *cfg["batch"]["fallbacks"]]
    model = build_model(cfg)
    meta["params"] = model.param_counts()
    print("params:", meta["params"])

    # ---- fixed initial scale s0 (train tiles only, once)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.s0.fill_(ck["s0"])
    else:
        tr_plain = GAMUSDataset(cfg["subset"], "train", crop=cfg["data"]["crop"], train=False)
        init = fit_initial_scale(model, tr_plain, cfg["init_scale"]["n_tiles"], cfg["init_scale"]["seed"], DEVICE)
        model.s0.fill_(init["s0"])
        meta["init_scale"] = init
        print(f"initial scale s0 = {init['s0']:.6f}  ({init['equation']})")

    chosen = None
    for bs, accum in options:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            opt_probe = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
            x = torch.randn(bs, 3, cfg["data"]["crop"], cfg["data"]["crop"], device=DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"] == "bf16"):
                p = model(x)
            L = total_loss(p.float(), torch.zeros_like(p.float()), torch.ones_like(p, dtype=torch.bool))
            L["total"].backward(); opt_probe.step(); opt_probe.zero_grad(set_to_none=True)
            del x, p, L, opt_probe
            chosen = (bs, accum)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs} x accum {accum}: out of memory, trying next fallback")
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    if chosen is None:
        sys.exit("no batch configuration fits in GPU memory")
    bs, accum = chosen
    meta["batch"] = {"size": bs, "accum": accum, "effective": bs * accum}
    print(f"batch {bs} x accum {accum} = effective {bs * accum}")
    set_seed(cfg["seed"])

    tr, dl = make_loaders(cfg, bs, cfg["seed"])
    steps_per_epoch = len(dl) // accum
    total_steps = steps_per_epoch * cfg["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"],
                            betas=tuple(cfg["optim"]["betas"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg["optim"]["warmup_steps"], total_steps))
    meta["schedule"] = {"steps_per_epoch": steps_per_epoch, "total_opt_steps": total_steps}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- speed test
    if args.speed_test:
        torch.cuda.reset_peak_memory_stats()
        sampler = GPUSampler(); sampler.start()
        rows = []
        t0 = time.perf_counter()
        steps = train_steps(cfg, model, dl, bs, accum, args.speed_steps, opt, sched,
                            run_dir / "train_log.csv", speed=rows)
        el = time.perf_counter() - t0
        gpu = sampler.stop()
        steady = rows[1:] if len(rows) > 1 else rows          # drop the warm-up interval
        sps = float(np.mean([r["samples_per_s"] for r in steady]))
        # measure one full val pass on 40 tiles to extrapolate eval time
        tv = time.perf_counter()
        ds = GAMUSDataset(cfg["subset"], "val", crop=None, train=False)
        model.eval()
        for i in range(40):
            s = ds[i]
            model.predict_tile(s["image"][None].to(DEVICE))
        torch.cuda.synchronize()
        val_sec_per_tile = (time.perf_counter() - tv) / 40
        n_evals = cfg["epochs"] // cfg["eval"]["every_epochs"]
        est_train_h = cfg["epochs"] * len(tr) / sps / 3600
        est_eval_h = n_evals * len(splits["val"]) * val_sec_per_tile / 3600
        res = {"optimizer_steps": steps, "wall_s": el, "batch": bs, "accum": accum,
               "samples_per_s_steady": sps, "steps_per_s_steady": sps / (bs * accum),
               "data_wait_frac_steady": float(np.mean([r["data_wait_frac"] for r in steady])),
               "peak_vram_torch_mb": torch.cuda.max_memory_allocated() / 2**20,
               "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
               **gpu, "val_sec_per_tile": val_sec_per_tile,
               "loss_first_interval": {k: rows[0][k] for k in ("l1", "grad", "grad_weighted", "total", "grad_to_l1_ratio")},
               "loss_last_interval": {k: rows[-1][k] for k in ("l1", "grad", "grad_weighted", "total", "grad_to_l1_ratio")},
               "estimate_full_run_h": {"train": est_train_h, "eval": est_eval_h, "total": est_train_h + est_eval_h}}
        (run_dir / "speed.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    # ---- full training
    start_epoch, step, best = 0, 0, {"rmse": float("inf"), "epoch": None}
    if args.resume:
        model.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["global_step"], ck["best"]
        torch.set_rng_state(ck["rng"]["torch"]); torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"]); random.setstate(ck["rng"]["python"])
        print(f"resumed at epoch {start_epoch}, step {step}, best {best}")
    ckdir = run_dir / "checkpoints"; ckdir.mkdir(exist_ok=True)
    val_log = run_dir / "val_log.csv"
    for epoch in range(start_epoch, cfg["epochs"]):
        t_ep = time.perf_counter()
        step = train_steps(cfg, model, dl, bs, accum, 0, opt, sched, run_dir / "train_log.csv",
                           start_step=step, epoch=epoch)
        print(f"epoch {epoch} done in {(time.perf_counter() - t_ep) / 60:.1f} min")
        last_epoch = epoch == cfg["epochs"] - 1
        if (epoch + 1) % cfg["eval"]["every_epochs"] == 0 or last_epoch:
            te = time.perf_counter()
            res, per_tile = evaluate(model, cfg, desc=f"val ep{epoch}")
            o, pc = res["overall"], res["per_class"]
            row = {"epoch": epoch, "step": step, "rmse": o["rmse"], "mae": o["mae"], "pearson_r": o["pearson_r"],
                   "bias": o["bias"], "rmse_building": pc["building"]["rmse"], "rmse_tree": pc["tree"]["rmse"],
                   "eval_min": (time.perf_counter() - te) / 60}
            new = not val_log.exists()
            with open(val_log, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new: w.writeheader()
                w.writerow(row)
            print(f"[val] epoch {epoch}: RMSE {o['rmse']:.3f} MAE {o['mae']:.3f} r {o['pearson_r']:.3f} "
                  f"bias {o['bias']:+.3f} | bldg {pc['building']['rmse']:.3f} tree {pc['tree']['rmse']:.3f}")
            if o["rmse"] < best["rmse"]:
                best = {"rmse": o["rmse"], "epoch": epoch, "step": step}
                save_metrics(run_dir, "val", res)
                per_tile.to_csv(run_dir / "per_tile_val.csv", index=False)
                save_ckpt(ckdir / "best.pt", model, opt, sched, epoch, step, best, cfg, bs, accum)
                print(f"  new best -> checkpoints/best.pt")
        save_ckpt(ckdir / "last.pt", model, opt, sched, epoch, step, best, cfg, bs, accum)

    meta = json.loads((run_dir / "meta.json").read_text())
    meta["best"] = best
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    res = json.loads((run_dir / "metrics_val.json").read_text())
    append_results(run_dir, "val", res, method=cfg["name"], deployable=True,
                   extra={"best_epoch": best["epoch"], "selected_by": "val_rmse",
                          "phase1_commit": cfg["phase1_commit"][:10]})
    print(f"done. best val RMSE {best['rmse']:.3f} at epoch {best['epoch']}  ->  {run_dir}")


if __name__ == "__main__":
    main()
