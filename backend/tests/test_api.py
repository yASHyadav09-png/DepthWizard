"""API-level tests. The nDSM model is stubbed so these need no GPU or checkpoint;
`test_end_to_end.py` exercises the real trained model."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import height_estimator
from tests.conftest import make_image_bytes

FAKE_MODEL = {"run": "stub-run", "checkpoint": "runs/stub/checkpoints/best.pt", "sha256": "0" * 64,
              "epoch": 20, "val_rmse": 3.148, "git_commit": "abc", "trainable_mode": "partial",
              "s0": 3.66, "device": "cpu", "label": "stub", "device_label": "CPU"}


class FakeEstimator:
    """Deterministic stand-in: ground at 0 m with a 12 m 'building' in the middle."""

    device = "cpu"
    device_label = "CPU"
    is_loaded = True

    def load(self):
        pass

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        field = np.zeros((h, w), np.float32)
        field[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 12.0
        return field, 1.0

    def model_info(self):
        return dict(FAKE_MODEL)

    def info(self):
        return {"model_label": "stub", "run": "stub-run", "loaded": True, "device": "cpu",
                "device_label": "CPU", "cuda_available": False, "torch_version": "stub"}


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    fake = FakeEstimator()
    monkeypatch.setattr(height_estimator, "_estimator", fake)
    monkeypatch.setattr("app.services.pipeline.get_height_estimator", lambda: fake)
    monkeypatch.setattr("app.api.routes.get_height_estimator", lambda: fake)
    monkeypatch.setattr("app.main.get_height_estimator", lambda: fake)
    return fake


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def post(client, data, name="scene.png", ctype="image/png", **form):
    return client.post("/api/process", files={"image": (name, data, ctype)},
                       data={k: str(v) for k, v in form.items()})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_checkpoint"] == "stub-run"
    assert body["terrain_resolution"] == 768


def test_root_advertises_metric_ndsm(client):
    assert "metres" in client.get("/").json()["output"]


def test_process_returns_height_product_contract(client, png_bytes):
    r = post(client, png_bytes)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed" and body["stage"] == 3 and len(body["job_id"]) == 12

    hp = body["height_product"]
    assert hp["kind"] == "ndsm" and hp["height_units"] == "m"
    assert (hp["width"], hp["height"]) == (160, 120)
    assert hp["gsd_m"] == pytest.approx(0.33) and hp["gsd_source"] == "assumed_training_gsd"
    assert hp["metric_validity"] == "uncertain" and "assume" in hp["validity_note"]
    assert hp["crs"] is None and hp["transform"] is None and hp["vertical_datum"] is None
    assert hp["model"]["checkpoint_sha256"] == "0" * 64

    s = body["statistics"]
    assert s["units"] == "m" and s["is_metric"] is True
    assert s["min"] == pytest.approx(0.0) and s["max"] == pytest.approx(12.0)
    assert body["disclaimer"].endswith("resolution uncertain")
    for key in ("original", "texture", "height_map", "hillshade", "height_array"):
        assert body["assets"][key].startswith("/static/")


def test_user_gsd_is_used_and_validity_follows_it(client, png_bytes):
    hp = post(client, png_bytes, gsd_m=0.33).json()["height_product"]
    assert hp["gsd_source"] == "user" and hp["metric_validity"] == "valid"
    body = post(client, png_bytes, gsd_m=1.0).json()
    assert body["height_product"]["metric_validity"] == "uncertain"
    # footprint in metres follows the given resolution
    assert body["terrain"]["plane_width"] == pytest.approx(160 * 1.0)


def test_rejects_absurd_gsd(client, png_bytes):
    assert post(client, png_bytes, gsd_m=0).status_code == 400
    assert post(client, png_bytes, gsd_m=500).status_code == 400


def test_terrain_is_in_metres_with_true_footprint(client, png_bytes):
    t = post(client, png_bytes).json()["terrain"]
    assert (t["width"], t["height"]) == (160, 120)          # not upsampled
    heights = np.frombuffer(base64.b64decode(t["heights_b64"]), dtype="<f4")
    assert heights.size == t["width"] * t["height"] and np.isfinite(heights).all()
    assert heights.max() == pytest.approx(12.0) and heights.min() == pytest.approx(0.0)
    assert t["height_units"] == "m"
    assert t["plane_width"] == pytest.approx(160 * 0.33) and t["plane_depth"] == pytest.approx(120 * 0.33)
    assert t["display_min"] == 0.0 and t["display_max"] >= 5.0


def test_large_image_is_gridded_but_array_is_full_resolution(client):
    body = post(client, make_image_bytes(2400, 1600), name="big.png").json()
    t = body["terrain"]
    assert (t["width"], t["height"]) == (768, 512) and t["aspect_ratio"] == pytest.approx(1.5)
    arr = np.load(io.BytesIO(client.get(f"/api/results/{body['job_id']}/height-array").content))
    assert arr.shape == (1600, 2400) and arr.dtype == np.float32   # native resolution, never downscaled
    assert (body["height_product"]["width"], body["height_product"]["height"]) == (2400, 1600)


def test_artifacts_are_served_and_results_are_replayable(client, jpg_bytes):
    body = post(client, jpg_bytes, name="scene.jpg", ctype="image/jpeg").json()
    for url in body["assets"].values():
        assert client.get(url).status_code == 200, url
    again = client.get(f"/api/results/{body['job_id']}").json()
    assert again["statistics"] == body["statistics"] and again["height_product"] == body["height_product"]
    assert client.get(f"/api/results/{body['job_id']}/height-array").content[:6] == b"\x93NUMPY"
    assert body["job_id"] in client.get("/api/jobs").json()["jobs"]


# --- error handling ------------------------------------------------------
def test_rejects_corrupt_image(client):
    r = post(client, b"not-an-image", name="x.png")
    assert r.status_code == 400 and r.json()["error"] == "invalid_image"


def test_rejects_unsupported_format(client):
    assert post(client, b"II*\x00data", name="x.tif", ctype="image/tiff").status_code == 400


def test_missing_file_field_is_422(client):
    r = client.post("/api/process", files={"wrong": ("x.png", b"abc", "image/png")})
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_unknown_job_is_404(client):
    assert client.get("/api/results/ffffffffffff").status_code == 404
    assert client.get("/api/results/..%2F..%2Fetc").status_code in (400, 404)


def test_model_failure_surfaces_as_503(client, monkeypatch, png_bytes):
    from app.utils.errors import ModelError

    class Broken(FakeEstimator):
        def predict(self, rgb):
            raise ModelError("weights unavailable")

    monkeypatch.setattr("app.services.pipeline.get_height_estimator", lambda: Broken())
    r = post(client, png_bytes)
    assert r.status_code == 503 and r.json()["error"] == "model_error"


# --- validation (Phase 7d) -------------------------------------------------
def _npy(a) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(a, dtype=np.float32))
    return buf.getvalue()


def test_validation_against_identical_reference_is_zero_error(client, png_bytes):
    job = post(client, png_bytes).json()["job_id"]
    pred = np.load(io.BytesIO(client.get(f"/api/results/{job}/height-array").content))
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("ref.npy", _npy(pred), "application/octet-stream")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall"]["rmse"] == pytest.approx(0.0) and body["overall"]["mae"] == pytest.approx(0.0)
    assert body["reference"]["valid_fraction"] == pytest.approx(1.0)
    assert client.get(body["error_map"]).status_code == 200
    grid = np.frombuffer(base64.b64decode(body["error_grid"]["values_b64"]), dtype="<f4")
    assert grid.size == body["error_grid"]["width"] * body["error_grid"]["height"]


def test_validation_metrics_and_nodata(client, png_bytes):
    job = post(client, png_bytes).json()["job_id"]
    pred = np.load(io.BytesIO(client.get(f"/api/results/{job}/height-array").content))
    ref = pred + 2.0                      # prediction is 2 m too low everywhere
    ref[:10] = np.nan                     # no-data rows are excluded
    body = client.post(f"/api/results/{job}/validate",
                       files={"reference": ("ref.npy", _npy(ref), "application/octet-stream")}).json()
    assert body["overall"]["bias"] == pytest.approx(-2.0, abs=1e-5)
    assert body["overall"]["mae"] == pytest.approx(2.0, abs=1e-5)
    assert body["reference"]["valid_pixels"] == pred.size - 10 * pred.shape[1]


def test_validation_rejects_wrong_grid_and_garbage(client, png_bytes):
    job = post(client, png_bytes).json()["job_id"]
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("r.npy", _npy(np.zeros((5, 5))), "x")})
    assert r.status_code == 400 and "grid" in r.json()["detail"]
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("r.npy", b"not numpy", "x")})
    assert r.status_code == 400
    assert client.post("/api/results/ffffffffffff/validate",
                       files={"reference": ("r.npy", _npy(np.zeros((5, 5))), "x")}).status_code == 404


def test_phase4_decision_inference_runs_on_native_pixels(client, stub_model, monkeypatch):
    """Phase 4 closure: production uses DIRECT inference. The model must receive the
    uploaded pixels unchanged (never resampled to 0.33 m/px), whatever gsd_m says."""
    seen = []
    original = stub_model.predict

    def spy(rgb):
        seen.append(rgb.shape)
        return original(rgb)

    monkeypatch.setattr(stub_model, "predict", spy)
    for gsd in (None, 0.33, 0.6, 1.0):
        form = {} if gsd is None else {"gsd_m": gsd}
        assert post(client, make_image_bytes(300, 200), name="s.png", **form).status_code == 200
    assert seen == [(200, 300, 3)] * 4
