"""Visual + speed check of GAMUSDataset exactly as training will see it.

Usage:  python scripts/check_dataloader.py --subset phase0
Writes runs/<subset>_inspection/figures/dataloader_batch.png and prints samples/s.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depthwizard.data.gamus import IMAGENET_MEAN, IMAGENET_STD, GAMUSDataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="phase0")
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    ds = GAMUSDataset(args.subset, "train", crop=args.crop, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=args.workers,
                                     persistent_workers=args.workers > 0)
    b = next(iter(dl))

    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        img = b["image"][i].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        nd, v, c = b["ndsm"][i, 0].numpy(), b["valid"][i, 0].numpy(), b["cls"][i].numpy()
        ax[i, 0].imshow(np.clip(img, 0, 1)); ax[i, 0].set_title(b["id"][i])
        ax[i, 1].imshow(np.where(v, nd, np.nan), cmap="magma", vmin=0, vmax=max(1, np.percentile(nd[v], 99)))
        ax[i, 1].set_title(f"nDSM  max={nd[v].max():.1f} m")
        ax[i, 2].imshow(c, cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[i, 2].set_title("classes")
        ax[i, 3].imshow(~v, cmap="gray"); ax[i, 3].set_title(f"invalid px: {(~v).mean():.2%}")
        for a in ax[i]: a.axis("off")
    out = Path(__file__).resolve().parents[1] / "runs" / f"{args.subset}_inspection" / "figures" / "dataloader_batch.png"
    fig.tight_layout(); fig.savefig(out, dpi=70); plt.close(fig)
    print("batch:", {k: (tuple(v.shape), str(v.dtype)) for k, v in b.items() if torch.is_tensor(v)})

    t0, n = time.time(), 0
    for _ in range(3):
        for b in dl:
            n += b["image"].shape[0]
    print(f"{n / (time.time() - t0):.1f} samples/s with {args.workers} workers (crop {args.crop})")
    print("wrote", out)
