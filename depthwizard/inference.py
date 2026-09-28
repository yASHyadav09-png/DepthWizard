"""Inference with a trained Phase 2 checkpoint: RGB image of any size -> nDSM (metres).

Identical to the validation path for a 1024x1024 GAMUS tile (same normalisation,
reflect-pad to a multiple of 14, bf16 autocast). Larger images are processed in
overlapping 1024 windows whose predictions are blended with linear feathering,
so there are no seams. Nothing is resampled: the network always sees pixels at
their native size (the model was trained at ~0.33 m/px; see `metric_validity`).
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.gamus import GAMUS_GSD_M, IMAGENET_MEAN, IMAGENET_STD, ROOT
from .models.ndsm import NDSMModel

DEFAULT_RUN = ROOT / "runs" / "20260926-003630_phase2b_partial"   # the kept Phase 2 model
TILE = 1024
OVERLAP = 128
# tolerance for calling a user-supplied GSD "the training GSD"
GSD_TOLERANCE = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window_starts(n: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering [0, n) with windows of min(tile, n) and >= `overlap` overlap."""
    if n <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, n - tile, step))
    starts.append(n - tile)
    return starts


def feather(n: int, overlap: int, fade_start: bool, fade_end: bool) -> np.ndarray:
    """1-D blend weights: linear ramps of length `overlap` on the faded sides, 1 elsewhere.
    Ramps never reach 0, so every pixel keeps a positive total weight."""
    w = np.ones(n, np.float32)
    r = min(overlap, n // 2)
    if r > 0:
        ramp = (np.arange(r, dtype=np.float32) + 0.5) / r
        if fade_start:
            w[:r] = np.minimum(w[:r], ramp)
        if fade_end:
            w[n - r:] = np.minimum(w[n - r:], ramp[::-1])
    return w


def metric_validity(gsd_m: float | None) -> tuple[str, str]:
    if gsd_m is None:
        return "uncertain", (f"Ground resolution unknown: heights are estimates that assume "
                             f"~{GAMUS_GSD_M} m/px imagery (the model's training resolution).")
    rel = abs(gsd_m - GAMUS_GSD_M) / GAMUS_GSD_M
    if rel <= GSD_TOLERANCE:
        return "valid", f"Ground resolution {gsd_m:g} m/px is close to the training resolution ({GAMUS_GSD_M} m/px)."
    return "uncertain", (f"Ground resolution {gsd_m:g} m/px differs from the training resolution "
                         f"({GAMUS_GSD_M} m/px). Direct inference at this resolution is less accurate "
                         f"(Phase 4 study, val RMSE 3.15 m at 0.33 m vs 5.19 m at 1 m); heights may be biased.")


@dataclass(frozen=True)
class ModelInfo:
    run: str
    checkpoint: str
    sha256: str
    epoch: int
    val_rmse: float
    git_commit: str | None
    trainable_mode: str
    s0: float
    device: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class NDSMPredictor:
    """Loads a Phase 2 run's best.pt once; thread-safe predict()."""

    def __init__(self, run_dir: Path = DEFAULT_RUN, device: str | None = None, amp: bool = True,
                 deterministic: bool = True):
        if deterministic:
            # the same numerics as the validation evaluation (cuDNN deterministic,
            # TF32 off); otherwise predictions differ from the reported metrics ~1e-6 m
            from .models.dav2 import set_deterministic
            set_deterministic()
        self.run_dir = Path(run_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp and self.device.startswith("cuda")
        self._lock = threading.Lock()
        cfg = yaml.safe_load(open(self.run_dir / "config.yaml"))
        ck_path = self.run_dir / "checkpoints" / "best.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = NDSMModel(cfg["model"]["revision"], cfg["model"]["id"])
        m.set_trainable(cfg["trainable"]["mode"], cfg["trainable"]["unfreeze_last_blocks"])
        missing, unexpected = m.load_state_dict(ck["trainable"], strict=False)
        assert not unexpected, unexpected
        assert all(not k.startswith(("net.neck", "net.head")) for k in missing), "decoder weights missing"
        self.model = m.to(self.device).eval()
        meta = json.loads((self.run_dir / "meta.json").read_text())
        try:
            # display-only path, relative to the repo when the run lives under it (the normal
            # case); falls back to an absolute path when it doesn't (e.g. a checkpoint fetched
            # into an external cache dir, as on the HF Space deployment) -- never affects the
            # loaded weights, sha256 (computed on the file bytes below) or any prediction.
            checkpoint_str = str(ck_path.relative_to(ROOT))
        except ValueError:
            checkpoint_str = str(ck_path)
        self.info = ModelInfo(run=self.run_dir.name, checkpoint=checkpoint_str,
                              sha256=sha256_file(ck_path), epoch=int(ck["epoch"]) + 1,
                              val_rmse=float(ck["best"]["rmse"]), git_commit=meta["git"]["commit"],
                              trainable_mode=cfg["trainable"]["mode"], s0=float(ck["s0"]),
                              device=self.device)

    @staticmethod
    def to_input(rgb: np.ndarray) -> torch.Tensor:
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

    @torch.no_grad()
    def _predict_window(self, rgb: np.ndarray) -> np.ndarray:
        x = self.to_input(rgb).to(self.device)
        return self.model.predict_tile(x, amp=self.amp)[0, 0].cpu().numpy()

    def predict(self, rgb: np.ndarray, tile: int = TILE, overlap: int = OVERLAP) -> np.ndarray:
        """rgb: HxWx3 uint8 -> HxW float32 nDSM in metres (>= 0)."""
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
        H, W = rgb.shape[:2]
        with self._lock:
            if H <= tile and W <= tile:
                return self._predict_window(rgb)
            acc = np.zeros((H, W), np.float64)
            wsum = np.zeros((H, W), np.float64)
            ys, xs = window_starts(H, tile, overlap), window_starts(W, tile, overlap)
            th, tw = min(tile, H), min(tile, W)
            for y in ys:
                wy = feather(th, overlap, y > 0, y + th < H)
                for x in xs:
                    wx = feather(tw, overlap, x > 0, x + tw < W)
                    p = self._predict_window(rgb[y:y + th, x:x + tw])
                    w = wy[:, None] * wx[None, :]
                    acc[y:y + th, x:x + tw] += p * w
                    wsum[y:y + th, x:x + tw] += w
            return (acc / wsum).astype(np.float32)


_predictor: NDSMPredictor | None = None
_plock = threading.Lock()


def get_predictor(run_dir: Path | None = None) -> NDSMPredictor:
    global _predictor
    with _plock:
        if _predictor is None:
            _predictor = NDSMPredictor(run_dir or DEFAULT_RUN)
        return _predictor
