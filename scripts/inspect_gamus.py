"""Phase 0 survey of a GAMUS subset: formats, ranges, NoData, per-class heights, alignment figures.

Usage:  python scripts/inspect_gamus.py --subset phase0
Outputs: runs/phase0_inspection/{survey.json, tiles.csv, figures/*.png}
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.gamus import load_subset, read_tile, tile_paths  # noqa: E402
from depthwizard.eval.metrics import GAMUS_CLASSES  # noqa: E402

CLASS_COLORS = ["#000000", "#a0826d", "#b5e08a", "#e03c31", "#2f6fdf", "#9e9e9e", "#1f7a1f"]
CMAP_CLS = ListedColormap(CLASS_COLORS)


def raw_info(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        d = f["image"]
        a = d[()]
        return {"keys": list(f.keys()), "dtype": str(d.dtype), "shape": list(d.shape),
                "attrs": {**{k: str(v) for k, v in f.attrs.items()},
                          **{k: str(v) for k, v in d.attrs.items()}},
                "nonfinite": int((~np.isfinite(a.astype(np.float64))).sum())}


def figure(tid, t, out: Path):
    rgb, ndsm, cls = t["rgb"], t["ndsm"], t["cls"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ax[0].imshow(rgb); ax[0].set_title(f"{tid}  RGB")
    im = ax[1].imshow(ndsm, cmap="magma", vmin=0, vmax=max(1, np.percentile(ndsm, 99.5)))
    ax[1].set_title("nDSM (m above ground)"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(cls, cmap=CMAP_CLS, vmin=0, vmax=6, interpolation="nearest"); ax[2].set_title("classes")
    # alignment check: RGB with building (red) and tree (green) edges + nDSM>3 m contour
    ax[3].imshow(rgb)
    ax[3].contour(ndsm > 3, levels=[0.5], colors="cyan", linewidths=0.6)
    ax[3].contour(cls == 3, levels=[0.5], colors="red", linewidths=0.6)
    ax[3].set_title("overlay: nDSM>3m (cyan), building mask (red)")
    for a in ax: a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS]
    fig.legend(handles, [GAMUS_CLASSES[i] for i in range(7)], loc="lower center", ncol=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=90); plt.close(fig)


def zoom_figure(tid, t, out: Path, size=256):
    """Pixel-level alignment: crop around the tallest building region."""
    ndsm, cls = t["ndsm"], t["cls"]
    b = np.where(cls == 3, ndsm, 0)
    y, x = np.unravel_index(np.argmax(b if b.max() > 0 else ndsm), ndsm.shape)
    y0 = int(np.clip(y - size // 2, 0, ndsm.shape[0] - size)); x0 = int(np.clip(x - size // 2, 0, ndsm.shape[1] - size))
    sl = (slice(y0, y0 + size), slice(x0, x0 + size))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    ax[0].imshow(t["rgb"][sl]); ax[0].set_title(f"{tid} zoom RGB")
    ax[1].imshow(t["rgb"][sl]); ax[1].contour(ndsm[sl] > 3, levels=[0.5], colors="cyan", linewidths=1)
    ax[1].set_title("RGB + nDSM>3m edge")
    ax[2].imshow(t["rgb"][sl]); ax[2].contour(cls[sl] == 3, levels=[0.5], colors="red", linewidths=1)
    ax[2].contour(cls[sl] == 6, levels=[0.5], colors="lime", linewidths=1)
    ax[2].set_title("RGB + building (red) / tree (green) edges")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="phase0")
    ap.add_argument("--figs-per-city", type=int, default=2)
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / "runs" / f"{args.subset}_inspection"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    splits = load_subset(args.subset)

    rows, formats = [], defaultdict(set)
    cls_heights = defaultdict(list)             # class -> sampled heights
    hist_edges = np.arange(0, 201, 1.0)
    hist = np.zeros(len(hist_edges) - 1)
    figs_done = defaultdict(int)
    rng = np.random.default_rng(0)

    for split, ids in splits.items():
        for tid in ids:
            city = tid.split("_")[0]
            for mod, p in tile_paths(split, tid).items():
                info = raw_info(p)
                formats[(city, mod)].add((p.name.rsplit("_", 1)[1], info["dtype"], tuple(info["shape"]),
                                          tuple(info["keys"]), json.dumps(info["attrs"]), info["nonfinite"]))
            t = read_tile(split, tid)
            nd, c = t["ndsm"], t["cls"]
            cls_raw = h5py.File(tile_paths(split, tid)["cls"], "r")["image"][()]
            rows.append({
                "split": split, "id": tid, "city": city,
                "rgb_mean": float(t["rgb"].mean()), "rgb_min": int(t["rgb"].min()), "rgb_max": int(t["rgb"].max()),
                "ndsm_min": float(nd.min()), "ndsm_max": float(nd.max()), "ndsm_mean": float(nd.mean()),
                "ndsm_p99": float(np.percentile(nd, 99)),
                "frac_zero": float((nd == 0).mean()), "frac_lt_0.5m": float((nd < 0.5).mean()),
                "frac_negative": float((nd < 0).mean()), "frac_invalid": float((~t["valid"]).mean()),
                "cls_non_integer": bool(np.any(cls_raw != np.rint(cls_raw))),
                **{f"frac_{GAMUS_CLASSES[k]}": float((c == k).mean()) for k in range(7)},
            })
            hist += np.histogram(np.clip(nd, 0, 199.9), hist_edges)[0]
            idx = rng.choice(nd.size, 20000, replace=False)
            for k in range(7):
                sel = c.ravel()[idx] == k
                cls_heights[k].extend(nd.ravel()[idx][sel].tolist())
            if figs_done[city] < args.figs_per_city:
                figure(tid, t, out / "figures" / f"{split}_{tid}.png")
                zoom_figure(tid, t, out / "figures" / f"{split}_{tid}_zoom.png")
                figs_done[city] += 1

    df = pd.DataFrame(rows)
    df.to_csv(out / "tiles.csv", index=False)

    per_class = {}
    for k in range(7):
        h = np.array(cls_heights[k])
        per_class[GAMUS_CLASSES[k]] = None if h.size == 0 else {
            "n_sampled": int(h.size), "mean": float(h.mean()), "median": float(np.median(h)),
            "p5": float(np.percentile(h, 5)), "p95": float(np.percentile(h, 95)),
            "frac_gt_2m": float((h > 2).mean())}

    survey = {
        "subset": args.subset,
        "n_tiles": {s: len(v) for s, v in splits.items()},
        "formats": {f"{c}/{m}": [dict(zip(["suffix", "dtype", "shape", "keys", "attrs", "nonfinite"], x))
                                 for x in sorted(v)] for (c, m), v in formats.items()},
        "ndsm_global": {"min": float(df.ndsm_min.min()), "max": float(df.ndsm_max.max()),
                        "any_negative": bool((df.frac_negative > 0).any()),
                        "any_invalid": bool((df.frac_invalid > 0).any()),
                        "mean_frac_exact_zero": float(df.frac_zero.mean())},
        "cls_non_integer_tiles": int(df.cls_non_integer.sum()),
        "per_class_height_m": per_class,
        "per_city": df.groupby("city")[["ndsm_mean", "ndsm_p99", "ndsm_max", "frac_zero", "rgb_mean",
                                          "frac_building", "frac_tree"]].mean().round(3).to_dict(orient="index"),
    }
    (out / "survey.json").write_text(json.dumps(survey, indent=2, default=str))

    # height histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hist_edges[:-1], hist / hist.sum(), width=1.0, color="#4a6fa5")
    ax.set_yscale("log"); ax.set_xlabel("nDSM (m)"); ax.set_ylabel("pixel fraction (log)")
    ax.set_title(f"GAMUS {args.subset}: nDSM distribution"); fig.tight_layout()
    fig.savefig(out / "figures" / "ndsm_histogram.png", dpi=90); plt.close(fig)

    print(json.dumps(survey, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
