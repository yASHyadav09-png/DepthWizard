"""Real inference with the trained Phase 2b model. Skipped unless DW_RUN_MODEL_TESTS=1
(needs the checkpoint under runs/ and, ideally, a GPU).

    DW_RUN_MODEL_TESTS=1 pytest tests/test_end_to_end.py -v
"""

import base64
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DW_RUN_MODEL_TESTS") != "1",
    reason="Set DW_RUN_MODEL_TESTS=1 to run real model inference.",
)


def test_real_pipeline_produces_metric_heights(jpg_bytes):
    from app.services.pipeline import run_pipeline

    payload = run_pipeline(jpg_bytes, "scene.jpg", "image/jpeg")
    assert payload["status"] == "completed"
    assert payload["height_product"]["kind"] == "ndsm"
    assert payload["height_product"]["height_units"] == "m"
    assert len(payload["model"]["checkpoint_sha256"]) == 64
    heights = np.frombuffer(base64.b64decode(payload["terrain"]["heights_b64"]), dtype="<f4")
    assert np.isfinite(heights).all() and (heights >= 0).all()
