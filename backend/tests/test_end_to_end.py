"""Real Depth Anything V2 inference. Skipped unless DW_RUN_MODEL_TESTS=1
(the first run downloads ~100 MB of weights).

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


def test_real_pipeline_produces_relief(jpg_bytes):
    from app.services.pipeline import run_pipeline

    payload = run_pipeline(jpg_bytes, "scene.jpg", "image/jpeg")

    assert payload["status"] == "completed"
    assert payload["model"]["checkpoint"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert payload["statistics"]["is_metric"] is False

    heights = np.frombuffer(
        base64.b64decode(payload["terrain"]["heights_b64"]), dtype="<f4"
    )
    assert np.isfinite(heights).all()
    # A real depth map is never a constant plane.
    assert heights.std() > 0.01
