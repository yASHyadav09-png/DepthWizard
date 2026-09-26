"""Export a few GAMUS VAL tiles as demo images (the test split is never used).

For each tile writes to sample_data/gamus_val/:
    <id>_rgb.png           the aerial image to upload in the demo (0.33 m/px)
    <id>_lidar_ndsm.png    LiDAR reference height, coloured with a fixed 0-30 m scale
    <id>_lidar_ndsm.npy    LiDAR reference height (float32 m, NaN = no data)
plus README.md with the tile list and each tile's Phase 2b val RMSE.

Usage:  python scripts/export_demo_samples.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.gamus import GAMUS_GSD_M, ROOT, load_subset, read_tile  # noqa: E402
from depthwizard.inference import DEFAULT_RUN  # noqa: E402

OUT = ROOT / "sample_data" / "gamus_val"
VMAX = 30.0


def pick(per_tile: pd.DataFrame) -> dict[str, str]:
    """A spread of scenes: tall buildings, forest, flat ground, typical, hardest."""
    val = set(load_subset("full")["val"])
    pt = per_tile[per_tile.id.isin(val)].sort_values("rmse").reset_index(drop=True)
    picks = {"easy": pt.id.iloc[len(pt) // 10], "typical_1": pt.id.iloc[len(pt) // 2],
             "typical_2": pt.id.iloc[len(pt) // 2 + 7], "hard": pt.id.iloc[int(len(pt) * 0.95)]}
    extra = {"PHL_6183": "flat_roofs", "DC_48_31": "tall_buildings", "DC_18_24": "tall_forest",
             "DC_37_10": "dc_median", "PHL_6470": "phl_median"}
    for tid, tag in extra.items():
        if tid in val and tid not in picks.values():
            picks[tag] = tid
    return picks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_tile = pd.read_csv(DEFAULT_RUN / "per_tile_val_final.csv")
    rmse = per_tile.set_index("id").rmse
    rows = []
    for tag, tid in pick(per_tile).items():
        t = read_tile("val", tid)
        Image.fromarray(t["rgb"]).save(OUT / f"{tid}_rgb.png")
        gt = np.where(t["valid"], t["ndsm"], np.nan).astype(np.float32)
        np.save(OUT / f"{tid}_lidar_ndsm.npy", gt)
        col = matplotlib.colormaps["magma"](np.clip(np.nan_to_num(gt, nan=0.0) / VMAX, 0, 1))[..., :3]
        col[~t["valid"]] = 0.35
        Image.fromarray((col * 255).astype(np.uint8)).save(OUT / f"{tid}_lidar_ndsm.png")
        rows.append((tag, tid, float(rmse[tid]), float(np.nanpercentile(gt, 99))))
        print(f"{tag:15s} {tid:10s} 2b RMSE {rmse[tid]:.2f} m")

    lines = ["# GAMUS validation tiles for the Phase 3 demo", "",
             f"All tiles are from the GAMUS **val** split (never the test split), {GAMUS_GSD_M} m/px, "
             "1024 x 1024 px (about 338 m x 338 m).",
             f"Upload `*_rgb.png` in the web app and enter **{GAMUS_GSD_M}** as ground resolution. "
             f"Compare with `*_lidar_ndsm.png` (LiDAR reference, fixed 0-{VMAX:.0f} m magma scale; grey = no data).",
             "", "| scene | tile | Phase 2b tile RMSE (m) | LiDAR p99 height (m) |", "|---|---|---|---|"]
    lines += [f"| {tag} | {tid} | {r:.2f} | {p:.1f} |" for tag, tid, r, p in rows]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tiles to {OUT}")


if __name__ == "__main__":
    main()
