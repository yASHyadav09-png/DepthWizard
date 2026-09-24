"""Dump the raw structure of GAMUS .h5 files: keys, attrs, dtype, shape, value stats.

Usage:  python scripts/inspect_h5.py data/gamus/images/train/DC_01_25_RGB.h5 [...]
"""
import sys

import h5py
import numpy as np


def describe(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  dataset '{name}': shape={obj.shape} dtype={obj.dtype} "
              f"chunks={obj.chunks} compression={obj.compression}")
        a = obj[()]
        if a.dtype.kind in "fiub":
            af = a.astype(np.float64)
            finite = np.isfinite(af)
            print(f"    min={np.nanmin(af):.4f} max={np.nanmax(af):.4f} "
                  f"mean={np.nanmean(af):.4f} nonfinite={(~finite).sum()}")
            q = np.percentile(af[finite], [0.1, 1, 50, 99, 99.9])
            print(f"    p0.1/1/50/99/99.9 = {np.round(q, 3).tolist()}")
            if a.dtype.kind in "iu" or len(np.unique(a[:64, :64])) < 50:
                u, c = np.unique(a, return_counts=True)
                if len(u) <= 30:
                    print(f"    unique={dict(zip(u.tolist(), c.tolist()))}")
    else:
        print(f"  group '{name}'")
    for k, v in obj.attrs.items():
        print(f"    attr {k} = {v!r}")


for path in sys.argv[1:]:
    print(path)
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            print(f"  file attr {k} = {v!r}")
        f.visititems(describe)
