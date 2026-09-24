"""Experiment bookkeeping: one folder per run + one global results.csv.

Every run folder contains:
  config.yaml          the exact config used (after CLI overrides)
  meta.json            seed, git commit, dirty flag, python/torch versions, time
  split_manifest.txt   the tile ids used, per split (the dataset "version")
  metrics_<split>.json pooled metrics
  figures/             qualitative figures
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
RESULTS_CSV = RUNS_DIR / "results.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_info() -> dict:
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def new_run(name: str, config: dict) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{name}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    meta = {
        "name": name,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": config.get("seed"),
        "git": git_info(),
        "python": platform.python_version(),
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def write_manifest(run_dir: Path, splits: dict[str, list[str]]) -> None:
    lines = []
    for split, ids in splits.items():
        lines += [f"# {split} ({len(ids)})", *ids, ""]
    (run_dir / "split_manifest.txt").write_text("\n".join(lines))


def save_metrics(run_dir: Path, split: str, metrics: dict) -> None:
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))


def append_results(run_dir: Path, split: str, metrics: dict, extra: dict | None = None,
                   method: str = "", deployable: bool | None = None) -> None:
    """One row per (run, method, split) in runs/results.csv — the table for the presentation."""
    ov = metrics["overall"]
    pc = metrics.get("per_class", {})
    meta = json.loads((run_dir / "meta.json").read_text())
    row = {
        "run": run_dir.name,
        "method": method,
        "deployable": deployable,
        "split": split,
        "commit": (meta["git"]["commit"] or "")[:10],
        "dirty": meta["git"]["dirty"],
        "seed": meta.get("seed"),
        "n_pixels": ov["n"],
        "mae": ov["mae"],
        "rmse": ov["rmse"],
        "pearson_r": ov["pearson_r"],
        "bias": ov.get("bias"),
        "mae_building": pc.get("building", {}).get("mae"),
        "rmse_building": pc.get("building", {}).get("rmse"),
        "mae_tree": pc.get("tree", {}).get("mae"),
        "rmse_tree": pc.get("tree", {}).get("rmse"),
        # free-form extras go in one JSON column so the CSV header never changes
        "notes": json.dumps(extra or {}),
    }
    RUNS_DIR.mkdir(exist_ok=True)
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
