"""Frozen Depth Anything V2 Small: RGB tile -> relative inverse depth.

The output is affine-ambiguous relative inverse depth ("larger = closer"), NOT
metres. Converting it to nDSM is the job of depthwizard.calibration.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def set_deterministic() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class DAV2Relative:
    def __init__(self, revision: str, device: str = "cuda", model_id: str = MODEL_ID):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        # use_fast=False pinned: the default is changing upstream and the fast
        # processor gives slightly different pixels, which would break reproducibility.
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, revision=revision)
        self.model.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, input_res: int) -> np.ndarray:
        """rgb: HxWx3 uint8. Returns the raw model output at model resolution (float32).

        Preprocessing is the model's own DPTImageProcessor (bicubic resize with kept
        aspect ratio and sides a multiple of 14, rescale to [0,1], ImageNet normalisation);
        only the target size is changed.
        """
        inputs = self.processor(images=rgb, return_tensors="pt",
                                size={"height": input_res, "width": input_res})
        pv = inputs["pixel_values"].to(self.device, torch.float32)
        out = self.model(pixel_values=pv).predicted_depth  # (1, h, w)
        return out[0].float().cpu().numpy()


def upsample(d: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a continuous field to `size` (H, W)."""
    t = torch.from_numpy(d)[None, None]
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
