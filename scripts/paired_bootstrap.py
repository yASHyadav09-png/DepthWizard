"""Paired tile-level bootstrap: Delta = metric_model - metric_reference (pooled over pixels).

Definitions (docs/phase2_plan.md):
    delta_RMSE = RMSE_model - RMSE_reference
    delta_MAE  = MAE_model  - MAE_reference
Both runs are evaluated on the same val tiles with the same validity masks. Each
bootstrap resample draws tiles with replacement and recomputes BOTH pooled metrics
on that same resample (paired). "Significantly better" <=> upper 95% bound < 0.

Usage:
    python scripts/paired_bootstrap.py --model runs/<p2>/per_tile_val.csv --model-name phase2a \
        --reference runs/<p1>/per_tile_val.csv --ref-methods B1_r1022_pct_lin B0_zero B2_oracle_r518 \
        --out runs/<p2>/paired_bootstrap.json
    (a reference may also be another Phase 2 per-tile csv: --ref-model runs/<p2a>/per_tile_val.csv)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TILE_PX = 1024 * 1024


def load_model_tiles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({"id": df.id, "n": df.n_valid.astype(float),
                         "sse": df.rmse ** 2 * df.n_valid, "sae": df.mae * df.n_valid})


def load_phase1_tiles(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n = df.valid_frac * TILE_PX
    return pd.DataFrame({"id": df.id, "n": n, "sse": df[f"{method}_rmse"] ** 2 * n,
                         "sae": df[f"{method}_mae"] * n})


def paired(model: pd.DataFrame, ref: pd.DataFrame, B: int, seed: int) -> dict:
    m = model.merge(ref, on="id", suffixes=("_m", "_r"), validate="one_to_one")
    assert len(m) == len(model) == len(ref), "model and reference must cover the same tiles"
    # same tiles + same NoData rule => identical valid-pixel counts
    assert np.allclose(m.n_m, m.n_r, rtol=0, atol=0.5), "valid-pixel counts differ between runs"
    n = m.n_m.values
    idx = np.random.default_rng(seed).integers(0, len(m), (B, len(m)))
    N = n[idx].sum(1)
    out = {}
    for name, col, f in (("RMSE", "sse", lambda s, N: np.sqrt(s / N)), ("MAE", "sae", lambda s, N: s / N)):
        pm, pr = f(m[f"{col}_m"].sum(), n.sum()), f(m[f"{col}_r"].sum(), n.sum())
        d = f(m[f"{col}_m"].values[idx].sum(1), N) - f(m[f"{col}_r"].values[idx].sum(1), N)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = {"model": float(pm), "reference": float(pr), "delta": float(pm - pr),
                     "ci95": [float(lo), float(hi)], "significantly_better": bool(hi < 0),
                     "significantly_worse": bool(lo > 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--reference", type=Path, help="Phase 1 per_tile_val.csv")
    ap.add_argument("--ref-methods", nargs="*", default=[])
    ap.add_argument("--ref-model", type=Path, nargs="*", default=[], help="other Phase 2 per_tile_val.csv")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    model = load_model_tiles(a.model)
    refs = {m: load_phase1_tiles(a.reference, m) for m in a.ref_methods}
    refs.update({p.parent.name: load_model_tiles(p) for p in a.ref_model})
    res = {"definition": "delta = metric_model - metric_reference; paired tile bootstrap; "
                         "significantly better <=> upper 95% bound < 0",
           "model": a.model_name, "n_tiles": len(model), "resamples": a.resamples, "seed": a.seed,
           "comparisons": {}}
    print(f"{a.model_name}: paired tile bootstrap over {len(model)} tiles, {a.resamples} resamples")
    for name, ref in refs.items():
        r = paired(model, ref, a.resamples, a.seed)
        res["comparisons"][name] = r
        for k in ("RMSE", "MAE"):
            x = r[k]
            verdict = "better (sig.)" if x["significantly_better"] else \
                      "worse (sig.)" if x["significantly_worse"] else "not significant"
            print(f"  vs {name:22s} d{k:4s} = {x['delta']:+.3f} m  95% CI [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
                  f"  ({x['model']:.3f} vs {x['reference']:.3f})  -> {verdict}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
