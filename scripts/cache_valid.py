"""Precompute validity masks (NoData rule) for every tile of a subset, in parallel.

Usage:  python scripts/cache_valid.py --subset full --workers 6
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.gamus import VALID_CACHE_DIR, load_subset, read_tile  # noqa: E402


def work(args):
    split, tid = args
    if not (VALID_CACHE_DIR / split / f"{tid}.npy").exists():
        read_tile(split, tid)          # computes + caches the mask
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="full")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    splits = load_subset(a.subset)
    jobs = [(s, t) for s in ("train", "val") for t in splits.get(s, [])]
    from tqdm import tqdm
    with Pool(a.workers) as pool:
        for _ in tqdm(pool.imap_unordered(work, jobs, chunksize=8), total=len(jobs), desc="valid masks"):
            pass
    print("cached:", sum(1 for _ in VALID_CACHE_DIR.rglob("*.npy")))
