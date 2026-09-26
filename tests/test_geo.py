"""Phase 5 geospatial correctness on synthetic rasters (no network, no model)."""
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from depthwizard.geo.dem import dtm_on_grid, ground_filter
from depthwizard.geo.gcp import correction_surface, fit_correction, sample
from depthwizard.geo.pipeline import run_geo, write_outputs
from depthwizard.geo.raster import GeoInputError, read_geo_image, write_geotiff

UTM = CRS.from_epsg(32617)                       # WGS 84 / UTM 17N
ORIGIN = (585_000.0, 4_476_000.0)                 # near Pittsburgh


def make_rgb_tif(path, crs=UTM, transform=None, w=200, h=150, dtype="uint8", count=3):
    transform = transform or from_origin(*ORIGIN, 0.5, 0.5)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255 if dtype == "uint8" else 4000, (count, h, w)).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=count, dtype=dtype, crs=crs,
                       transform=transform) as dst:
        dst.write(data)
    return path


def plane_dem_tif(path, f, bump=None):
    """GLO-30-like DEM in EPSG:4326 around the test area; heights = f(lon, lat) (+ optional bump)."""
    t = from_origin(-80.02, 40.45, 1 / 3600, 1 / 3600)
    h, w = 180, 200
    rows, cols = np.mgrid[0:h, 0:w]
    lon, lat = t * (cols + 0.5, rows + 0.5)
    z = f(lon, lat).astype(np.float32)
    if bump is not None:
        z = z + bump(lon, lat)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
                       crs="EPSG:4326", transform=t, nodata=-32767.0) as dst:
        dst.write(z, 1)
    return path


def plane(lon, lat):
    return 300.0 + 2000.0 * (lon + 80.0) + 1000.0 * (lat - 40.4)


def test_read_preserves_crs_transform_and_gsd(tmp_path):
    g = read_geo_image(make_rgb_tif(tmp_path / "a.tif"))
    assert g.crs == UTM and g.transform == from_origin(*ORIGIN, 0.5, 0.5)
    assert g.gsd_m == 0.5 and g.rgb.shape == (150, 200, 3) and g.rgb.dtype == np.uint8 and g.notes == []


def test_feet_crs_gsd_is_converted_to_metres(tmp_path):
    crs = CRS.from_epsg(2272)                     # NAD83 / Pennsylvania South (ftUS)
    g = read_geo_image(make_rgb_tif(tmp_path / "f.tif", crs=crs, transform=from_origin(1_300_000, 400_000, 1.0, 1.0)))
    assert g.gsd_m == pytest.approx(0.3048006, rel=1e-6)
    assert any("unit" in n for n in g.notes)


def test_geographic_crs_is_reprojected_to_utm(tmp_path):
    t = from_origin(-79.99, 40.43, 5e-6, 5e-6)    # ~0.4-0.55 m pixels
    g = read_geo_image(make_rgb_tif(tmp_path / "g.tif", crs="EPSG:4326", transform=t))
    assert g.crs.to_epsg() == 32617 and 0.3 < g.gsd_m < 0.6
    assert any("reprojected" in n for n in g.notes)


def test_uint16_is_stretched_and_extra_bands_ignored(tmp_path):
    g = read_geo_image(make_rgb_tif(tmp_path / "u.tif", dtype="uint16", count=4))
    assert g.rgb.dtype == np.uint8 and g.rgb.shape[2] == 3
    assert any("stretched" in n for n in g.notes) and any("bands 1-3" in n for n in g.notes)


def test_rejects_rotation_and_missing_crs(tmp_path):
    rot = Affine(0.5, 0.1, ORIGIN[0], 0.1, -0.5, ORIGIN[1])
    with pytest.raises(GeoInputError, match="Rotated"):
        read_geo_image(make_rgb_tif(tmp_path / "r.tif", transform=rot))
    p = tmp_path / "n.tif"
    with rasterio.open(p, "w", driver="GTiff", width=10, height=10, count=3, dtype="uint8") as dst:
        dst.write(np.zeros((3, 10, 10), np.uint8))
    with pytest.raises(GeoInputError, match="no CRS"):
        read_geo_image(p)


def test_dem_reprojection_lands_on_the_exact_grid(tmp_path):
    """A plane defined in lon/lat must appear on the UTM grid at its analytic value
    for every pixel centre (checks CRS handling, bilinear resampling and registration)."""
    dem = plane_dem_tif(tmp_path / "dem.tif", plane)
    t = from_origin(*ORIGIN, 0.5, 0.5)
    out = dtm_on_grid(dem, UTM, t, (150, 200), window_m=0)          # no filter: pure reprojection
    rows, cols = np.mgrid[0:150, 0:200]
    x, y = t * (cols + 0.5, rows + 0.5)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(x, y)
    assert out.covered.all()
    assert np.abs(out.dtm - plane(lon, lat)).max() < 0.02           # < 2 cm over a 60 m/km slope
    assert out.info["vertical_datum"].startswith("EGM2008")


def test_ground_filter_removes_narrow_objects_keeps_wide_terrain():
    dem = np.full((60, 60), 100.0, np.float32)
    dem[:, :30] += 50                               # wide terrain step (a 900 m wide plateau at 30 m px)
    dem[40:43, 45:48] += 20                         # a 90 m "building"
    f = ground_filter(dem, 30.0, 30.0, window_m=150, sigma_px=0)
    assert f[41, 46] == pytest.approx(100.0)        # building removed
    assert f[30, 10] == pytest.approx(150.0)        # plateau kept


def test_gcp_offset_and_plane_recovery():
    t = from_origin(*ORIGIN, 1.0, 1.0)
    dtm = np.full((100, 100), 200.0, np.float32)
    rng = np.random.default_rng(1)
    xs = ORIGIN[0] + rng.uniform(5, 95, 8)
    ys = ORIGIN[1] - rng.uniform(5, 95, 8)
    off = fit_correction(dtm, t, np.column_stack([xs, ys, np.full(8, 203.5)]), "offset")
    assert off["params"]["a"] == pytest.approx(3.5) and off["rmse_after"] == pytest.approx(0, abs=1e-5)
    zs = 200 + 1.0 + 0.02 * (xs - xs.mean()) - 0.01 * (ys - ys.mean())
    zs[0] += 50                                                        # one bad GCP (outlier)
    pl = fit_correction(dtm, t, np.column_stack([xs, ys, zs]), "plane")
    assert pl["n_used"] == 7
    assert pl["params"]["b"] == pytest.approx(0.02, abs=1e-6) and pl["params"]["c"] == pytest.approx(-0.01, abs=1e-6)
    corr = dtm + correction_surface(pl, t, dtm.shape)
    assert np.allclose(sample(corr, t, xs[1:], ys[1:]), zs[1:], atol=1e-4)
    with pytest.raises(ValueError):
        fit_correction(dtm, t, np.array([[0.0, 0.0, 1.0]]))          # outside the image


def test_pipeline_outputs_share_crs_transform_and_add_up(tmp_path):
    img = make_rgb_tif(tmp_path / "img.tif")
    dem = plane_dem_tif(tmp_path / "dem.tif", plane)
    res = run_geo(img, lambda rgb: np.full(rgb.shape[:2], 7.0, np.float32), dem)
    assert np.allclose(res.dsm, res.dtm + 7.0, equal_nan=True) and res.valid.all()
    files = write_outputs(res, tmp_path / "out", "t")
    for name, p in files.items():
        with rasterio.open(p) as s:
            assert s.crs == UTM and s.transform == from_origin(*ORIGIN, 0.5, 0.5)
            assert (s.width, s.height) == (200, 150) and s.nodata == -9999.0
            tags = s.tags()
            assert tags["PRODUCT"] == name.upper() and tags["UNITS"] == "metre"
            assert ("EGM2008" in tags["VERTICAL_DATUM"]) == (name != "ndsm")
            assert s.tags(1)["units"] == "metre"
    with rasterio.open(files["ndsm"]) as s:
        assert np.allclose(s.read(1), 7.0)


def test_write_geotiff_maps_nan_to_nodata(tmp_path):
    a = np.ones((4, 5), np.float32); a[1, 1] = np.nan
    write_geotiff(tmp_path / "x.tif", a, UTM, from_origin(*ORIGIN, 1, 1))
    with rasterio.open(tmp_path / "x.tif") as s:
        m = s.read(1, masked=True)
        assert m.mask[1, 1] and m.count() == 19
