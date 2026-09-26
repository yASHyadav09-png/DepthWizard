"""The trained nDSM model behind the API.

A thin, lazily loaded wrapper around `depthwizard.inference.NDSMPredictor` (the
same code path as the validation evaluation). This module knows nothing about
meshes or PNGs: it turns an RGB array into heights above ground in metres.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch

from app.config import settings
from app.utils.errors import ModelError

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu"
    if preference.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable - falling back to CPU.")
        return "cpu"
    if preference.startswith("cuda"):
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"


class HeightEstimator:
    def __init__(self):
        self.device = resolve_device(settings.DEVICE)
        self.device_label = describe_device(self.device)
        self._predictor = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self) -> None:
        with self._lock:
            if self._predictor is not None:
                return
            try:
                from depthwizard.inference import NDSMPredictor
                t0 = time.perf_counter()
                self._predictor = NDSMPredictor(settings.RUN_DIR, device=self.device)
                logger.info("Loaded %s (sha256 %s...) in %.1f s", self._predictor.info.checkpoint,
                            self._predictor.info.sha256[:12], time.perf_counter() - t0)
            except Exception as exc:  # noqa: BLE001
                raise ModelError(f"Could not load the trained model from {settings.RUN_DIR}: {exc}") from exc

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """HxWx3 uint8 -> (HxW float32 nDSM in metres, inference ms)."""
        self.load()
        t0 = time.perf_counter()
        try:
            h = self._predictor.predict(rgb)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ModelError("GPU ran out of memory for this image.") from exc
        return h, (time.perf_counter() - t0) * 1000.0

    def model_info(self) -> dict:
        self.load()
        return {**self._predictor.info.to_dict(), "label": settings.MODEL_LABEL,
                "device_label": self.device_label}

    def info(self) -> dict:
        return {"model_label": settings.MODEL_LABEL, "run": settings.RUN_DIR.name,
                "loaded": self.is_loaded, "device": self.device, "device_label": self.device_label,
                "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__}


_estimator: HeightEstimator | None = None
_elock = threading.Lock()


def get_height_estimator() -> HeightEstimator:
    global _estimator
    with _elock:
        if _estimator is None:
            _estimator = HeightEstimator()
        return _estimator
