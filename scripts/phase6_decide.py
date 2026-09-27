"""Phase 6b decision rule (pre-registered in docs/phase6_plan.md). Delta = candidate - reference (2b).

    R1  GAMUS val, simulated 0.66 m (direct inference), paired TILE bootstrap: Delta RMSE upper 95% < 0
    R2  GAMUS val, native 0.33 m, paired TILE bootstrap:                       Delta RMSE upper 95% < +0.10 m
    R3  NAIP AOIs pooled nDSM vs LiDAR (reference-QC exclusions applied), paired 128 m BLOCK bootstrap:
                                                                               Delta RMSE upper 95% < 0
Recommend the candidate only if R1, R2 and R3 all hold (the switch still needs the user's approval).
2,000 resamples, seed 0 (scripts/paired_bootstrap.paired).

Usage:
  python scripts/phase6_decide.py --ref-gsd runs/<2b phase4 study> --cand-gsd runs/<6b gsd study> \
      --ref-aoi runs/<2b phase6_aois> --cand-aoi runs/<6b phase6_aois> --out runs/<6b train run>/decision.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_bootstrap import load_model_tiles, paired  # noqa: E402

B, SEED, NONINF_M = 2000, 0, 0.10


def blocks(run: Path) -> pd.DataFrame:
    df = pd.read_csv(run / "per_block_ndsm.csv")
    df = df[~df.excluded.astype(bool)]
    return df[["id", "n", "sse", "sae"]].astype({"n": float})


def main():
    ap = argparse.ArgumentParser()
    for k in ("ref-gsd", "cand-gsd", "ref-aoi", "cand-aoi"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    r1 = paired(load_model_tiles(a.cand_gsd / "per_tile_gsd0.66_A.csv"),
                load_model_tiles(a.ref_gsd / "per_tile_gsd0.66_A.csv"), B, SEED)
    r2 = paired(load_model_tiles(a.cand_gsd / "per_tile_gsd0.33_A.csv"),
                load_model_tiles(a.ref_gsd / "per_tile_gsd0.33_A.csv"), B, SEED)
    rb, cb = blocks(a.ref_aoi), blocks(a.cand_aoi)
    assert set(rb.id) == set(cb.id), "block sets differ between runs"
    r3 = paired(cb, rb, B, SEED)
    res = {
        "definition": "delta = candidate - reference(2b); paired bootstrap, 2000 resamples, seed 0",
        "R1_val_0.66m": {**r1, "pass": r1["RMSE"]["ci95"][1] < 0, "criterion": "upper95 < 0"},
        "R2_val_0.33m": {**r2, "pass": r2["RMSE"]["ci95"][1] < NONINF_M, "criterion": f"upper95 < +{NONINF_M}"},
        "R3_naip_pooled": {**r3, "pass": r3["RMSE"]["ci95"][1] < 0, "criterion": "upper95 < 0",
                           "n_blocks": len(cb), "aois": sorted(cb.id.str.split(":").str[0].unique())},
        "inputs": {k: str(v) for k, v in vars(a).items()},
    }
    res["recommend_candidate"] = all(res[k]["pass"] for k in ("R1_val_0.66m", "R2_val_0.33m", "R3_naip_pooled"))
    a.out.write_text(json.dumps(res, indent=2))
    for k in ("R1_val_0.66m", "R2_val_0.33m", "R3_naip_pooled"):
        d = res[k]["RMSE"]
        print(f"{k:16s} RMSE {d['reference']:.3f} -> {d['model']:.3f}  delta {d['delta']:+.3f} "
              f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  {res[k]['criterion']}: {'PASS' if res[k]['pass'] else 'FAIL'}")
    print("recommend candidate:", res["recommend_candidate"], "->", a.out)


if __name__ == "__main__":
    main()
