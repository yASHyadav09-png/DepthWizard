"""Phase 5 API tests: GeoTIFF upload -> absolute DSM, GCP correction, nDSM fallback and
validation against a GeoTIFF reference. The model is stubbed (tests.test_api.FakeEstimator)
and the DEM is a synthetic EPSG:4326 raster, so these need no GPU, network or data files."""

import io

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from tests.test_api import client, post, stub_model  # noqa: F401  (fixtures)

UTM = CRS.from_epsg(32617)
GSD = 0.33
W, H = 300, 240
X0, Y0 = 585_000.0, 4_477_000.0          # near Pittsburgh, UTM 17N
TRANSFORM = from_origin(X0, Y0, GSD, GSD)


def rgb_geotiff(crs=UTM, transform=TRANSFORM) -> bytes:
    rgb = np.zeros((3, H, W), np.uint8)
    rgb[:] = np.array([30, 60, 90], np.uint8)[:, None, None]
    rgb[:, H // 3: 2 * H // 3, W // 3: 2 * W // 3] = 200
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=crs,
                      transform=transform) as dst:
            dst.write(rgb)
        return mem.read()


def float_geotiff(arr, crs, transform, nodata=-9999.0) -> bytes:
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
                      crs=crs, transform=transform, nodata=nodata) as dst:
            dst.write(np.where(np.isfinite(arr), arr, nodata).astype(np.float32), 1)
        return mem.read()


@pytest.fixture
def dem_path(tmp_path):
    """Constant 250 m ground in EPSG:4326 (1 arc-second pixels) covering the image + margin."""
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(UTM, "EPSG:4326", X0, Y0 - H * GSD, X0 + W * GSD, Y0)
    pad, px = 0.02, 1 / 3600
    t = from_origin(w - pad, n + pad, px, px)
    nx, ny = int((e - w + 2 * pad) / px) + 1, int((n - s + 2 * pad) / px) + 1
    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32",
                       crs="EPSG:4326", transform=t, nodata=-32767.0) as dst:
        dst.write(np.full((ny, nx), 250.0, np.float32), 1)
    return p


@pytest.fixture
def with_dem(monkeypatch, dem_path):
    monkeypatch.setattr("depthwizard.geo.dem_cache.get_dem", lambda *a, **k: (dem_path, "cache"))
    return dem_path


@pytest.fixture
def no_dem(monkeypatch):
    monkeypatch.setattr("depthwizard.geo.dem_cache.get_dem",
                        lambda *a, **k: (None, "unavailable: fetching is disabled"))


def post_tif(client, data, gcps=None, **form):
    files = {"image": ("scene.tif", data, "image/tiff")}
    if gcps is not None:
        files["gcps"] = ("gcps.csv", gcps, "text/csv")
    return client.post("/api/process", files=files, data={k: str(v) for k, v in form.items()})


def read_url(client, url):
    r = client.get(url)
    assert r.status_code == 200, url
    return r.content


def test_geotiff_upload_produces_georeferenced_dsm(client, with_dem):
    r = post_tif(client, rgb_geotiff())
    assert r.status_code == 200, r.text
    b = r.json()
    hp = b["height_product"]
    assert b["stage"] == 5 and hp["kind"] == "dsm" and hp["gsd_source"] == "geotiff"
    assert hp["crs_epsg"] == 32617 and hp["gsd_m"] == pytest.approx(GSD)
    assert hp["transform"] == pytest.approx([GSD, 0, X0, 0, -GSD, Y0])
    assert "EGM2008" in hp["vertical_datum"] and hp["metric_validity"] == "valid"
    lon, lat = hp["corners_lonlat"][0]
    assert -81 < lon < -79 and 40 < lat < 41
    assert b["terrain"]["surface_kind"] == "dsm" and "ndsm_b64" in b["terrain"]
    assert set(hp["geotiffs"]) >= {"dsm", "dtm", "ndsm"}

    with MemoryFile(read_url(client, hp["geotiffs"]["dsm"])) as m, m.open() as src:
        assert src.crs.to_epsg() == 32617 and src.transform.almost_equals(TRANSFORM)
        dsm = src.read(1, masked=True).filled(np.nan)
    with MemoryFile(read_url(client, hp["geotiffs"]["ndsm"])) as m, m.open() as src:
        ndsm = src.read(1, masked=True).filled(np.nan)
    # stub nDSM: 0 m ground with a 12 m block; DTM = 250 m constant  ->  DSM = 250 / 262 m
    assert np.nanmax(ndsm) == pytest.approx(12.0) and np.nanmin(ndsm) == pytest.approx(0.0)
    np.testing.assert_allclose(dsm - ndsm, 250.0, atol=0.01)
    assert b["statistics"]["elevation"]["dtm_min"] == pytest.approx(250.0, abs=0.01)
    dsm_npy = np.load(io.BytesIO(read_url(client, hp["height_array"])))
    np.testing.assert_allclose(dsm_npy, dsm, atol=1e-4)


def test_gcp_offset_shifts_the_dtm(client, with_dem):
    x, y = X0 + 20, Y0 - 20
    csv = f"x,y,z\n{x},{y},253.0\n{x + 40},{y - 30},253.0\n".encode()
    r = post_tif(client, rgb_geotiff(), gcps=csv, gcp_model="offset")
    assert r.status_code == 200, r.text
    b = r.json()
    g = b["height_product"]["gcp"]
    assert g["model"] == "offset" and g["params"]["a"] == pytest.approx(3.0, abs=0.01)
    assert b["statistics"]["elevation"]["dtm_min"] == pytest.approx(253.0, abs=0.01)


def test_gcp_lonlat_columns_and_bad_input(client, with_dem):
    lon, lat = pyproj.Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(X0 + 30, Y0 - 30)
    r = post_tif(client, rgb_geotiff(), gcps=f"lon,lat,z\n{lon},{lat},248\n".encode())
    assert r.status_code == 200, r.text
    assert r.json()["height_product"]["gcp"]["params"]["a"] == pytest.approx(-2.0, abs=0.01)
    assert post_tif(client, rgb_geotiff(), gcps=b"x,y,z\n1,2,3\n", gcp_model="spline").status_code == 400
    assert post_tif(client, rgb_geotiff(), gcps=b"a,b\n1,2\n").status_code == 400
    assert post_tif(client, rgb_geotiff(), gcps=b"x,y,z\n0,0,1\n").status_code == 400  # outside the image


def test_gcps_need_a_geotiff(client, png_bytes):
    r = client.post("/api/process", files={"image": ("s.png", png_bytes, "image/png"),
                                           "gcps": ("g.csv", b"x,y,z\n1,2,3\n", "text/csv")})
    assert r.status_code == 400


def test_no_dem_falls_back_to_georeferenced_ndsm(client, no_dem):
    r = post_tif(client, rgb_geotiff())
    assert r.status_code == 200, r.text
    b = r.json()
    hp = b["height_product"]
    assert hp["kind"] == "ndsm" and hp["vertical_datum"] is None and hp["crs_epsg"] == 32617
    assert any("No DEM" in n for n in hp["notes"]) and set(hp["geotiffs"]) == {"ndsm"}
    assert b["terrain"]["surface_kind"] == "ndsm"


def test_geotiff_without_crs_is_rejected(client, with_dem):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=64, height=64, count=3, dtype="uint8") as dst:
            dst.write(np.zeros((3, 64, 64), np.uint8))
        data = mem.read()
    r = post_tif(client, data)
    assert r.status_code == 400 and "CRS" in r.text


def test_validate_against_geotiff_reference(client, with_dem):
    b = post_tif(client, rgb_geotiff()).json()
    job = b["job_id"]
    # reference DSM on a coarser 1 m grid, same CRS, no vertical CRS declared, 251 m everywhere
    ref_tif = float_geotiff(np.full((H // 3, W // 3), 251.0, np.float32), UTM, from_origin(X0, Y0, 1.0, 1.0))
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("ref.tif", ref_tif, "image/tiff")})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["target"] == "dsm" and "ASSUMED EGM2008" in v["reference"]["datum_conversion"]
    assert v["reference"]["valid_fraction"] > 0.9
    # predicted DSM = 250 on the ground (-1 m) and 262 on the block (+11 m, 1/9 of the area)
    assert v["overall"]["bias"] == pytest.approx(-1 + 12 / 9, abs=0.2)

    # nDSM target against a 0 m reference: error = the stub's heights (>= 0)
    zero = float_geotiff(np.zeros((H, W), np.float32), UTM, TRANSFORM)
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("ref.tif", zero, "image/tiff")},
                    data={"target": "ndsm"})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["target"] == "ndsm" and v["overall"]["bias"] > 0


def test_validate_navd88_reference_is_converted(client, with_dem):
    job = post_tif(client, rgb_geotiff()).json()["job_id"]
    compound = CRS.from_wkt(pyproj.CRS("EPSG:26917+5703").to_wkt())
    ref_tif = float_geotiff(np.full((H, W), 250.0, np.float32), compound, TRANSFORM)
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("ref.tif", ref_tif, "image/tiff")})
    assert r.status_code == 200, r.text
    assert "NAVD88 -> EGM2008" in r.json()["reference"]["datum_conversion"]


def test_validate_dsm_target_refused_for_ndsm_jobs(client, png_bytes):
    job = post(client, png_bytes).json()["job_id"]
    r = client.post(f"/api/results/{job}/validate",
                    files={"reference": ("r.npy", b"x", "application/octet-stream")}, data={"target": "dsm"})
    assert r.status_code == 400
    r = client.post(f"/api/results/{job}/validate", files={"reference": ("r.tif", b"x", "image/tiff")})
    assert r.status_code == 400
