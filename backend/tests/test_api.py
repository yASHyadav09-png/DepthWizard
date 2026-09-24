"""API-level tests. The depth model is stubbed so CI needs no GPU or download;
`test_end_to_end.py` exercises the real network."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import depth_estimator
from app.services.depth_estimator import DepthResult
from tests.conftest import make_image_bytes


class FakeEstimator:
    """Deterministic stand-in: a diagonal ramp with a raised block."""

    model_name = "stub/depth-model"
    model_label = "Depth Anything V2 Small"
    device = "cpu"
    device_label = "CPU"
    is_loaded = True

    def load(self):
        pass

    def predict(self, image):
        h, w = image.height, image.width
        yy, xx = np.mgrid[0:h, 0:w]
        depth = (xx / max(w - 1, 1) + yy / max(h - 1, 1)).astype(np.float32)
        depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] += 1.5
        return DepthResult(
            depth=depth,
            model_name=self.model_name,
            model_label=self.model_label,
            device=self.device,
            device_label=self.device_label,
            inference_ms=1.0,
            dtype="float32",
        )

    def info(self):
        return {
            "model_name": self.model_name,
            "model_label": self.model_label,
            "device": self.device,
            "device_label": self.device_label,
            "loaded": True,
            "cuda_available": False,
            "torch_version": "stub",
        }


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    fake = FakeEstimator()
    monkeypatch.setattr(depth_estimator, "_estimator", fake)
    monkeypatch.setattr("app.services.pipeline.get_depth_estimator", lambda: fake)
    monkeypatch.setattr("app.api.routes.get_depth_estimator", lambda: fake)
    return fake


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "Depth Anything V2 Small"
    assert body["terrain_resolution"] == 256


def test_root_advertises_relative_output(client):
    body = client.get("/").json()
    assert "NOT metric" in body["output"]


def test_process_returns_full_payload(client, png_bytes):
    r = client.post("/api/process", files={"image": ("scene.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["status"] == "completed"
    assert len(body["job_id"]) == 12
    assert body["stage"] == 1
    assert body["disclaimer"] == "Relative Height - Not Metric"

    stats = body["statistics"]
    assert stats["image_width"] == 160 and stats["image_height"] == 120
    assert stats["min_relative_height"] == pytest.approx(0.0, abs=1e-5)
    assert stats["max_relative_height"] == pytest.approx(1.0, abs=1e-5)
    assert 0.0 < stats["mean_relative_height"] < 1.0
    assert stats["model_name"] == "Depth Anything V2 Small"
    assert stats["processing_device"] == "CPU"
    assert stats["is_metric"] is False and stats["georeferenced"] is False

    meta = body["metadata"]
    assert meta["type"] == "relative monocular depth"
    assert meta["georeferenced"] is False
    assert meta["height_units"] == "relative"

    for key in ("original", "texture", "depth_map", "relative_dsm", "height_array"):
        assert body["assets"][key].startswith("/static/")


def test_process_terrain_is_a_real_3d_grid(client, png_bytes):
    body = client.post(
        "/api/process", files={"image": ("scene.png", png_bytes, "image/png")}
    ).json()
    terrain = body["terrain"]

    assert terrain["width"] == 160 and terrain["height"] == 120
    heights = np.frombuffer(base64.b64decode(terrain["heights_b64"]), dtype="<f4")
    assert heights.size == terrain["width"] * terrain["height"]
    assert np.isfinite(heights).all()
    # Actual relief, not a flat plane.
    assert heights.max() - heights.min() > 0.5
    assert terrain["height_units"].startswith("relative")


def test_artifacts_are_served_and_results_are_replayable(client, jpg_bytes):
    body = client.post(
        "/api/process", files={"image": ("scene.jpg", jpg_bytes, "image/jpeg")}
    ).json()
    job_id = body["job_id"]

    for url in body["assets"].values():
        assert client.get(url).status_code == 200, url

    again = client.get(f"/api/results/{job_id}")
    assert again.status_code == 200
    assert again.json()["job_id"] == job_id
    assert again.json()["statistics"] == body["statistics"]

    npy = client.get(f"/api/results/{job_id}/height-array")
    assert npy.status_code == 200
    assert npy.content[:6] == b"\x93NUMPY"

    assert job_id in client.get("/api/jobs").json()["jobs"]


def test_downsamples_large_image_to_terrain_resolution(client):
    big = make_image_bytes(900, 600)
    terrain = client.post(
        "/api/process", files={"image": ("big.png", big, "image/png")}
    ).json()["terrain"]
    assert terrain["width"] == 256
    assert terrain["height"] == 171
    assert terrain["aspect_ratio"] == pytest.approx(1.5)


def test_height_array_shape_is_declared_in_metadata(client):
    """The exported .npy is at inference resolution, not source resolution."""
    import io

    big = make_image_bytes(2400, 1600)
    body = client.post(
        "/api/process", files={"image": ("big.png", big, "image/png")}
    ).json()

    meta = body["metadata"]
    assert (meta["height_array_width"], meta["height_array_height"]) == (
        body["source"]["inference_width"],
        body["source"]["inference_height"],
    )
    assert body["statistics"]["image_width"] == 2400  # source dims stay honest

    npy = client.get(f"/api/results/{body['job_id']}/height-array")
    array = np.load(io.BytesIO(npy.content))
    assert array.shape == (meta["height_array_height"], meta["height_array_width"])


# --- error handling ------------------------------------------------------
def test_rejects_corrupt_image(client):
    r = client.post("/api/process", files={"image": ("x.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


def test_rejects_unsupported_format(client):
    r = client.post("/api/process", files={"image": ("x.tif", b"II*\x00data", "image/tiff")})
    assert r.status_code == 400


def test_missing_file_field_is_422(client):
    r = client.post("/api/process", files={"wrong": ("x.png", b"abc", "image/png")})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_unknown_job_is_404(client):
    assert client.get("/api/results/ffffffffffff").status_code == 404
    # Path traversal must not be treated as a job id.
    assert client.get("/api/results/..%2F..%2Fetc").status_code in (400, 404)


def test_model_failure_surfaces_as_503(client, monkeypatch, png_bytes):
    from app.utils.errors import ModelError

    class Broken(FakeEstimator):
        def predict(self, image):
            raise ModelError("weights unavailable")

    monkeypatch.setattr("app.services.pipeline.get_depth_estimator", lambda: Broken())
    r = client.post("/api/process", files={"image": ("s.png", png_bytes, "image/png")})
    assert r.status_code == 503
    assert r.json()["error"] == "model_error"
