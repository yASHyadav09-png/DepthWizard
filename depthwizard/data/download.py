"""Download a reproducible, city-stratified subset of GAMUS from Hugging Face.

GAMUS (earthflow/GAMUS) stores one .h5 per tile and modality:
    images/<split>/<CITY>_<r>_<c>_RGB.h5
    heights/<split>/<CITY>_<r>_<c>_AGL.h5
    classes/<split>/<CITY>_<r>_<c>_CLS.h5
so we can fetch exactly the tiles we want instead of the full ~80 GB.

Usage:
    python -m depthwizard.data.download --name phase0 --train 40 --val 10 --seed 0
    python -m depthwizard.data.download --name phase1 --train 200 --val -1   # -1 = whole split
    python -m depthwizard.data.download --name full --all          # later phases

Writes the chosen tile ids to configs/subsets/<name>.yaml (the dataset "version"
referenced by every experiment). The test split is never downloaded unless
--include-test is given.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

REPO = "earthflow/GAMUS"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "gamus"
SUBSET_DIR = ROOT / "configs" / "subsets"
# File suffix per modality. NOTE: the image suffix is not uniform across cities
# (DC/PHL use _RGB, NYC uses _IMG), so images accept either.
MODALITIES = {"images": ("RGB", "IMG"), "heights": ("AGL",), "classes": ("CLS",)}


def list_tiles() -> dict[str, dict[str, list[str]]]:
    """{split: {tile_id: [image_file, height_file, class_file]}}. Only complete triplets."""
    files = set(list_repo_files(REPO, repo_type="dataset"))
    out = {}
    for split in ("train", "val", "test"):
        tiles = {}
        for f in files:
            if not f.startswith(f"images/{split}/"):
                continue
            stem = f.split("/")[-1][:-len(".h5")]
            tid, suffix = stem.rsplit("_", 1)
            if suffix not in MODALITIES["images"]:
                continue
            paths = [f] + [f"{m}/{split}/{tid}_{s[0]}.h5"
                           for m, s in MODALITIES.items() if m != "images"]
            if all(p in files for p in paths):
                tiles[tid] = paths
        out[split] = dict(sorted(tiles.items()))
    return out


def stratified_sample(ids: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample n ids, spread across cities in proportion to their share (min 1 each)."""
    by_city = defaultdict(list)
    for t in ids:
        by_city[t.split("_")[0]].append(t)
    total = len(ids)
    quota = {c: max(1, round(n * len(v) / total)) for c, v in by_city.items()}
    while sum(quota.values()) > n:          # fix rounding overshoot
        quota[max(quota, key=quota.get)] -= 1
    picked = []
    for c in sorted(by_city):
        picked += rng.sample(sorted(by_city[c]), min(quota[c], len(by_city[c])))
    return sorted(picked)


def _fetch_one(path: str, tries: int = 8) -> None:
    """Download one file unless present. Backs off on HTTP 429 (anonymous rate limit)."""
    if (DATA_DIR / path).exists():
        return
    for k in range(tries):
        try:
            hf_hub_download(REPO, path, repo_type="dataset", local_dir=DATA_DIR)
            return
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code != 429 and k >= 2:
                raise
            wait = 60 * (k + 1) if code == 429 else 5
            tqdm.write(f"{path}: HTTP {code}, retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {path}")


def fetch(paths: list[str], workers: int) -> None:
    """File-by-file download (resumable: existing files are skipped). Replaces
    snapshot_download, which hung on stale locks after a rate-limited run."""
    todo = [p for p in paths if not (DATA_DIR / p).exists()]
    print(f"{len(paths) - len(todo)} already present, {len(todo)} to fetch")
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(_fetch_one, todo), total=len(todo), desc="download"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="whole train+val splits")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--only-test", action="store_true",
                    help="download ONLY the test split (for the one-time final evaluation)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tiles = list_tiles()
    rng = random.Random(args.seed)
    subset = {}
    if args.only_test:
        subset = {}
    elif args.all:
        subset = {"train": list(tiles["train"]), "val": list(tiles["val"])}
    else:
        # a negative count means "the whole split"
        for split, n in (("train", args.train), ("val", args.val)):
            ids = list(tiles[split])
            subset[split] = ids if n < 0 else stratified_sample(ids, n, rng)
    if args.include_test or args.only_test:
        subset["test"] = list(tiles["test"])

    patterns = [p for split, ids in subset.items() for t in ids for p in tiles[split][t]]
    print(f"Downloading {len(patterns)} files "
          f"({', '.join(f'{k}={len(v)}' for k, v in subset.items())}) -> {DATA_DIR}")
    fetch(patterns, args.workers)

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": REPO, "seed": args.seed, "splits": subset}
    with open(SUBSET_DIR / f"{args.name}.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Wrote {SUBSET_DIR / f'{args.name}.yaml'}")


if __name__ == "__main__":
    main()
