"""Monocular depth inference with Depth Anything V2 Small.

The model is loaded exactly once per process and reused across requests.
This module deliberately knows nothing about DSMs, colour maps or meshes --
it only turns an RGB image into a raw relative (inverse) depth array.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from app.config import settings
from app.utils.errors import ModelError

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Pick the inference device, honouring an explicit override."""
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu"
    if preference.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable - falling back to CPU.")
            return "cpu"
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> str:
    """Human readable device label for the UI, e.g. 'CUDA - RTX 5050'."""
    if device.startswith("cuda") and torch.cuda.is_available():
        index = 0
        if ":" in device:
            try:
                index = int(device.split(":", 1)[1])
            except ValueError:
                index = 0
        return f"CUDA ({torch.cuda.get_device_name(index)})"
    return "CPU"


@dataclass(frozen=True)
class DepthResult:
    """Raw model output plus the bookkeeping the rest of the pipeline needs."""

    depth: np.ndarray            # (H, W) float32, larger value = closer to camera
    model_name: str
    model_label: str
    device: str
    device_label: str
    inference_ms: float
    dtype: str
    # Depth Anything V2 predicts inverse depth (disparity), so a large value
    # means "near the sensor". For nadir imagery that maps to "tall".
    is_inverse_depth: bool = True


class DepthEstimator:
    """Thread-safe, lazily-loaded wrapper around Depth Anything V2 Small."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.model_label = settings.MODEL_LABEL
        self.device = resolve_device(device or settings.DEVICE)
        self.device_label = describe_device(self.device)
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        # Inference is serialised: one CUDA context, one model instance.
        self._infer_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (first run) and initialise the network. Idempotent."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            logger.info("Loading %s on %s ...", self.model_name, self.device_label)
            started = time.perf_counter()
            try:
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
                model.to(self.device)
                model.eval()
                self._model = model
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                self._model = None
                self._processor = None
                raise ModelError(
                    f"Could not load '{self.model_name}'. The first run needs "
                    f"internet access to download the weights. Details: {exc}"
                ) from exc
            logger.info(
                "Model ready in %.1fs on %s",
                time.perf_counter() - started,
                self.device_label,
            )

    # -- inference --------------------------------------------------------
    def predict(self, image: Image.Image) -> DepthResult:
        """Run depth estimation and return a depth map at the image's resolution."""
        self.load()
        assert self._model is not None and self._processor is not None

        target_h, target_w = image.height, image.width
        use_half = self.device.startswith("cuda")

        started = time.perf_counter()
        try:
            with self._infer_lock:
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    if use_half:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self._model(**inputs)
                    else:
                        outputs = self._model(**inputs)

                depth = outputs.predicted_depth.float()
                if depth.ndim == 3:
                    depth = depth.unsqueeze(1)          # (B, 1, h, w)
                depth = torch.nn.functional.interpolate(
                    depth,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )
                depth_np = depth.squeeze().detach().cpu().numpy().astype(np.float32)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ModelError(
                "GPU ran out of memory during inference. Try a smaller image "
                "or set DW_DEVICE=cpu."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Depth inference failed: {exc}") from exc

        if not np.isfinite(depth_np).any():
            raise ModelError("Depth inference produced an all-invalid depth map.")

        return DepthResult(
            depth=depth_np,
            model_name=self.model_name,
            model_label=self.model_label,
            device=self.device,
            device_label=self.device_label,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            dtype="float16-autocast" if use_half else "float32",
        )

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_label": self.model_label,
            "device": self.device,
            "device_label": self.device_label,
            "loaded": self.is_loaded,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        }


_estimator: DepthEstimator | None = None
_estimator_lock = threading.Lock()


def get_depth_estimator() -> DepthEstimator:
    """Process-wide singleton so the weights are only ever loaded once."""
    global _estimator
    if _estimator is None:
        with _estimator_lock:
            if _estimator is None:
                _estimator = DepthEstimator()
    return _estimator
