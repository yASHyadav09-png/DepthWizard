"""Tile-level bootstrap confidence intervals for pooled RMSE and RMSE differences.

Resamples val tiles with replacement (pixels within a tile are not independent,
so tiles are the resampling unit) and recomputes pooled RMSE from per-tile SSE.

Usage:
    python scripts/bootstrap_ci.py runs/<run> --methods B0_mean B1_r1022_pct_lin \
        --pairs B1_r1022_pct_lin:B0_mean
Writes <run>/bootstrap_ci.txt.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--methods", nargs="+", required=True)
ap.add_argument("--pairs", nargs="*", default=[], help="a:b -> CI of RMSE(a) - RMSE(b)")
ap.add_argument("--resamples", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tile-pixels", type=int, default=1024 * 1024)
args = ap.parse_args()

pt = pd.read_csv(args.run / "per_tile_val.csv")
n = pt.valid_frac.values * args.tile_pixels
names = set(args.methods) | {x for p in args.pairs for x in p.split(":")}
S = {m: pt[f"{m}_rmse"].values ** 2 * n for m in names}
idx = np.random.default_rng(args.seed).integers(0, len(pt), (args.resamples, len(pt)))
point = {m: float(np.sqrt(S[m].sum() / n.sum())) for m in names}
boot = {m: np.sqrt(S[m][idx].sum(1) / n[idx].sum(1)) for m in names}

lines = [f"tile-level bootstrap: {len(pt)} val tiles, {args.resamples} resamples, seed {args.seed}; 95% CI"]
for m in args.methods:
    lo, hi = np.percentile(boot[m], [2.5, 97.5])
    lines.append(f"  RMSE {m:24s} {point[m]:.3f}  [{lo:.3f}, {hi:.3f}]")
for p in args.pairs:
    a, b = p.split(":")
    d = boot[a] - boot[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    lines.append(f"  RMSE({a}) - RMSE({b}) = {point[a] - point[b]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                 f"P(<0) = {np.mean(d < 0):.3f}")
out = "\n".join(lines)
print(out)
(args.run / "bootstrap_ci.txt").write_text(out + "\n")
