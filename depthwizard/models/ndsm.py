"""Depth Anything V2 Small adapted to predict metric nDSM directly.

    h = s0 * DA-V2-S(x)        [metres, >= 0 because the DPT head ends in ReLU]

s0 is a FIXED constant (a registered buffer, never trained) chosen once before
training so that step-0 outputs are in the metre range (see fit_initial_scale).
Everything that learns the metric mapping lives inside the network (DPT neck +
head, and optionally the last encoder blocks), so the trained model maps
RGB -> nDSM with no calibration step afterwards.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dav2 import MODEL_ID


class NDSMModel(nn.Module):
    def __init__(self, revision: str, model_id: str = MODEL_ID, s0: float = 1.0):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.net = AutoModelForDepthEstimation.from_pretrained(model_id, revision=revision)
        self.register_buffer("s0", torch.tensor(float(s0)))
        self.trainable_mode = None

    # ------------------------------------------------------------------ params
    def set_trainable(self, mode: str, unfreeze_last_blocks: int = 0) -> None:
        """mode 'frozen_encoder': train DPT neck + head only.
        mode 'partial': additionally train the last `unfreeze_last_blocks` encoder blocks."""
        for p in self.net.parameters():
            p.requires_grad_(False)
        for m in (self.net.neck, self.net.head):
            for p in m.parameters():
                p.requires_grad_(True)
        if mode == "partial":
            layers = self.net.backbone.encoder.layer
            for blk in layers[len(layers) - unfreeze_last_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            # the final backbone LayerNorm feeds the neck from the last block
            for p in self.net.backbone.layernorm.parameters():
                p.requires_grad_(True)
        elif mode != "frozen_encoder":
            raise ValueError(mode)
        self.trainable_mode = mode

    def param_counts(self) -> dict:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        return {"trainable": tr, "frozen": tot - tr, "total": tot}

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen encoder: keep it in eval mode (no dropout / stochastic depth changes)
        if mode and self.trainable_mode == "frozen_encoder":
            self.net.backbone.eval()
        return self

    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names or k == "s0"}

    # ------------------------------------------------------------------ forward
    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=x).predicted_depth[:, None]      # (B,1,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.s0 * self.raw(x)

    @torch.no_grad()
    def predict_tile(self, image: torch.Tensor, multiple: int = 14, amp: bool = True) -> torch.Tensor:
        """Full-tile inference at native resolution: reflect-pad H and W up to a
        multiple of 14 (1024 -> 1036), run once, crop back. No resampling."""
        B, C, H, W = image.shape
        Hp, Wp = -(-H // multiple) * multiple, -(-W // multiple) * multiple
        top, left = (Hp - H) // 2, (Wp - W) // 2
        x = F.pad(image, (left, Wp - W - left, top, Hp - H - top), mode="reflect")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = self(x)
        return out.float()[..., top:top + H, left:left + W]


@torch.no_grad()
def fit_initial_scale(model: NDSMModel, dataset, n_tiles: int, seed: int, device: str) -> dict:
    """Least-squares scale through the origin, s0 = sum(d*h) / sum(d^2), over valid
    pixels of one 518 crop from each of `n_tiles` TRAIN tiles (d = raw network output).
    Done once, before training; never uses val/test."""
    assert dataset.split == "train", "initial scale must be fitted on training tiles only"
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), n_tiles, replace=False)
    model.eval()
    sdh = sdd = 0.0
    ids = []
    state = np.random.get_state()
    np.random.seed(seed)                       # deterministic crop positions / aug
    try:
        for i in idx:
            s = dataset[int(i)]
            ids.append(s["id"])
            d = model.raw(s["image"][None].to(device)).float().cpu()[0, 0].numpy().astype(np.float64)
            h = s["ndsm"][0].numpy().astype(np.float64)
            v = s["valid"][0].numpy()
            sdh += float((d[v] * h[v]).sum())
            sdd += float((d[v] ** 2).sum())
    finally:
        np.random.set_state(state)
    s0 = sdh / sdd
    return {"s0": s0, "equation": "h = s0 * DA-V2-S(x); s0 = sum(d*h)/sum(d^2) over valid pixels",
            "n_tiles": n_tiles, "seed": seed, "tile_ids": ids}
