"""Phase 3 gate 1: the demo inference path must reproduce the evaluated model.

For N val tiles, run depthwizard.inference.NDSMPredictor (the code the web app
uses) and compare each tile's RMSE/MAE with the Phase 2 evaluation's
per_tile_val_final.csv. Val tiles only; the test split is never touched.

Usage:  python scripts/check_demo_consistency.py --tiles 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.gamus import load_subset, read_tile  # noqa: E402
from depthwizard.eval.metrics import tile_metrics  # noqa: E402
from depthwizard.inference import DEFAULT_RUN, NDSMPredictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
ap.add_argument("--tiles", type=int, default=5)
a = ap.parse_args()

pred = NDSMPredictor(a.run)
ref = pd.read_csv(a.run / "per_tile_val_final.csv").set_index("id")
ids = load_subset("full")["val"]
ids = [ids[i] for i in np.linspace(0, len(ids) - 1, a.tiles).astype(int)]
worst = 0.0
for tid in ids:
    t = read_tile("val", tid)
    m = tile_metrics(pred.predict(t["rgb"]), t["ndsm"], t["valid"])
    d = max(abs(m["rmse"] - ref.loc[tid, "rmse"]), abs(m["mae"] - ref.loc[tid, "mae"]))
    worst = max(worst, d)
    print(f"{tid:10s} demo RMSE {m['rmse']:.6f}  eval RMSE {ref.loc[tid, 'rmse']:.6f}  |diff| {d:.2e}")
print(f"model {pred.info.run} sha256 {pred.info.sha256[:12]}  max |diff| = {worst:.2e}")
print("GATE 1:", "PASS" if worst < 1e-6 else "FAIL")
sys.exit(0 if worst < 1e-6 else 1)
