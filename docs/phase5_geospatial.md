# Phase 5: Georeferenced input, absolute DSM, GCP correction

Status: implemented and evaluated on one US area (Pittsburgh). One decision is open
(the DTM ground-filter window, see "Open decision").

## Pipeline

```
GeoTIFF RGB --(CRS, affine transform, GSD read from the file)--> nDSM (Phase 2b model, native grid, tiled)
            --(Copernicus GLO-30, cached; ground filter; bilinear onto the image grid)--> DTM
            --(optional GCP correction: offset or plane)--> DTM'
DSM = DTM' + nDSM        metres, EGM2008 geoid heights (EPSG:3855), same CRS/transform as the input
```

Code: `depthwizard/geo/` (`raster.py`, `dem.py`, `gcp.py`, `pipeline.py`, `dem_cache.py`),
backend `backend/app/services/geo_pipeline.py`, tests `tests/test_geo.py` (10) and
`backend/tests/test_geo_api.py` (9).

### Input rules (read from the data, never assumed)
- CRS and transform from the file. No CRS: rejected. Rotated/sheared transforms: rejected.
  Non-square pixels (>1%): rejected.
- GSD = |pixel size| x the CRS linear unit (feet are converted to metres, and the conversion is reported).
- Geographic CRS (degrees): reprojected to the local UTM zone (bilinear RGB). This is the only case where
  user imagery is resampled, and it is reported in `height_product.notes`.
- uint16 bands: stretched to uint8 (0.2-99.8 percentile per band, reported). Bands 1-3 = R, G, B.
- The model runs at the file's native resolution (Phase 4 decision: direct inference, no resampling).
  Outside 0.33 m +-25% the product is flagged "resolution uncertain".

### DEM -> DTM
GLO-30 is an X-band radar **surface** model (30 m, EGM2008): it partly contains buildings and canopy.
A ground approximation is made by (1) filling no-data (tile seams) with the nearest value, (2) a grey
morphological opening with a window in metres (default **150 m**, fixed a priori; configurable with
`DW_DEM_OPEN_WINDOW_M`, 0 = off), (3) a Gaussian smoothing of 1 DEM pixel, and (4) bilinear
reprojection onto the image grid.

DEM source for the app (`dem_cache.py`): cached GeoTIFFs in `data/dem/` and `data/geo/*/glo30_dem.tif`.
If none covers the image plus the margin, a clip is fetched once from Microsoft Planetary Computer
(anonymous) and cached, unless `DW_DEM_ALLOW_FETCH=0`. Without a DEM, the product falls back to a
**georeferenced nDSM** (kind "ndsm") and says so.

### Vertical datum
Output: EGM2008 geoid heights (inherited from GLO-30). GCP z values must be EGM2008 metres.
Validation references: a compound CRS declaring NAVD88 is converted to EGM2008 with PROJ
(NADCON5 + GEOID + EGM2008 grids) as a constant offset at the image centre; at Pittsburgh this is
**-0.478 m** (spread over the area 7 mm). A reference without a vertical CRS is ASSUMED to be EGM2008,
and the response says so.

### GCP correction
Residuals r = z_GCP - DTM(x, y) (bilinear). `offset`: DTM + median(r) (>= 1 point).
`plane`: least-squares a + b(x - x0) + c(y - y0) after dropping residuals > 3 MAD (>= 3 points).
CSV columns `x,y,z` (image CRS) or `lon,lat,z` (WGS 84). Points outside the image are ignored;
none inside = error.

### Outputs
- GeoTIFFs `depthwizard_{dsm,dtm,ndsm}.tif`: float32, deflate, nodata -9999, the input CRS and transform,
  tags PRODUCT / VERTICAL_DATUM / UNITS / MODEL_RUN / MODEL_SHA256 / DEM source / GSD.
- `height_product` (API contract) adds: crs, crs_epsg, crs_wkt, transform, bounds, corners_lonlat,
  vertical_datum, dem (source, how, filter, coverage), gcp (model, params, residuals), geotiffs, notes.
- Terrain grid: `surface_kind` ("dsm" = elevations) and `ndsm_b64` (height above ground, same grid).

## Evaluation area: Pittsburgh, PA (not used for any fitting)

Data (`scripts/fetch_geo_aoi.py`, manifest `data/geo/pittsburgh/manifest.json`):
NAIP 2019-09-18 RGB 0.6 m (EPSG:26917, 2194 x 2185 px, 1.3 x 1.3 km, leaf-on);
USGS 3DEP LiDAR `PA_WesternPA_2_2019` DSM and DTM 2 m (NAD83 / UTM 17N + NAVD88);
GLO-30 tiles N40 W080 + N40 W081 (merged). The area mixes dense urban blocks, steep wooded hillsides and
~150 m of relief.

Run: `runs/20260926-161419_phase5_pittsburgh` (`scripts/phase5_eval_aoi.py`). Comparison on the LiDAR
2 m grid (our products area-averaged), LiDAR converted to EGM2008. Metrics use `depthwizard.eval.metrics`.

| Product vs LiDAR | RMSE (m) | MAE (m) | bias (m) | r |
|---|---|---|---|---|
| **DSM: ours** (filtered GLO-30 + model nDSM) | **5.40** | 3.98 | -1.60 | 0.994 |
| DSM baseline: raw GLO-30 | 4.76 | 3.58 | -2.27 | 0.996 |
| DTM: ours (150 m filter) | 4.45 | 2.98 | -1.02 | 0.996 |
| DTM baseline: raw GLO-30 | 3.27 | 2.37 | +0.89 | 0.998 |
| nDSM: model | 3.05 | 2.02 | -0.56 | 0.55 |
| nDSM: model, LiDAR nDSM > 2 m | 3.94 | 3.04 | -2.03 | 0.26 |

Error decomposition: LiDAR DTM + model nDSM gives 3.05 m (nDSM error only); our DTM + LiDAR nDSM gives
4.45 m (DTM error only). **The DTM dominates the DSM error.**

The live app path gives the same result: the uploaded NAIP GeoTIFF validated against `lidar_dsm.tif`
in the web app (NAVD88 converted automatically, bilinear onto the 0.6 m grid) gives
DSM RMSE 5.36 m, bias -1.60, r 0.995.

### Findings
1. **The 150 m ground filter hurts on this hilly site.** It lowers ridges and hilltops: DTM RMSE 4.45 m
   against 3.27 m for unfiltered GLO-30. Filter-window sensitivity (DIAGNOSTIC ONLY, computed on the
   evaluation area, not used to choose anything): 0 m 3.27, 90 m 3.61, 150 m 4.45, 300 m 8.92,
   600 m 11.62 m. Also diagnostic: raw GLO-30 + model nDSM gives a DSM RMSE of 4.17 m (bias +0.33),
   better than raw GLO-30 alone (4.76 m). Note that raw GLO-30 double-counts objects that are both in
   the radar DEM and the nDSM, so the "right" filter depends on land cover: more evidence is needed.
2. **The nDSM underestimates tall objects** here (bias -2.0 m above 2 m; r 0.26 on objects). Causes:
   0.6 m GSD (outside the valid band), leaf-on canopy (GAMUS training is leaf-off DC and summer
   PHL/NYC), different sensor. This is the Phase 6 scale-augmentation question.
3. **Horizontal alignment:** cross-correlation of the model nDSM with the LiDAR nDSM peaks at a shift of
   about 4 m along the image columns (east-west) and 0.2 m north-south (NCC 0.58). This is consistent
   with the NAIP horizontal accuracy spec (up to 6 m) plus relief displacement of tall objects; the
   pipeline itself keeps the input transform exactly (tested).
4. **Simulated GCPs** (5 points on LiDAR ground, fixed seed; metrics > 50 m from any GCP):
   offset: DTM 4.33 m, DSM 5.13 m (without GCPs, on all pixels: 4.45 / 5.40 m). Plane: DTM 7.48 m,
   DSM 7.95 m (worse): 5 points over 1.3 km
   of steep terrain over-fit a tilt. **Recommend "offset" unless GCPs are many and well spread.**

## Tests
- `tests/test_geo.py`: CRS/transform/GSD preserved, feet -> m, geographic -> UTM, uint16 stretch,
  rotation and missing CRS rejected, a lon/lat DEM plane lands on the UTM grid (within 2 cm), the ground
  filter removes narrow objects and keeps wide terrain, GCP offset/plane recovery, pipeline outputs share
  CRS/transform and DSM = DTM + nDSM, NaN -> nodata in GeoTIFFs.
- `backend/tests/test_geo_api.py`: GeoTIFF upload -> DSM GeoTIFFs with the input CRS/transform;
  DSM - nDSM = DTM; GCP offset (x,y,z and lon,lat,z); bad GCP inputs; no-DEM fallback; missing CRS
  rejected; GeoTIFF validation on a different grid (reprojected); NAVD88 conversion; DSM target refused
  for nDSM jobs.
- `frontend/src/services/geo.test.ts`: scene offset is a shift (1 m = 1 m), height above ground readout,
  walk physics on a shifted surface, easting/northing and lon/lat mapping.

## Viewer (Phase 7 integration)
- Upload accepts `.tif/.tiff`; GSD is read from the file; optional GCP CSV + model choice.
- DSM jobs: scene Y = elevation - floor(min elevation), a rigid vertical shift (nothing scaled). The
  surrounding ground plane sits at that base elevation. The HUD shows elevation (EGM2008), height
  above ground (nDSM), easting/northing in the image CRS and lon/lat (interpolated between the exactly
  projected corners). Layers: RGB, Elevation, Height above ground, Slope, Error.
  Profile/measure report elevations.
- Validation accepts GeoTIFF references in any CRS, with a DSM/nDSM target choice.
- Exports: DSM/DTM/nDSM GeoTIFF, DSM and nDSM .npy, metadata.

## Limitations
- One evaluation area (US, hilly urban, leaf-on). No Indian imagery or reference yet.
- GLO-30 is 30 m radar: steep terrain, canopy and wide buildings are not bare earth; narrow valleys are
  smoothed. Any DEM error goes straight into the DSM.
- Datum conversion of NAVD88 references is a constant offset at the image centre (fine for areas up to
  a few km; 7 mm spread here).
- Artefact export takes ~7 s for a 4.8 MP image (PNG + GeoTIFF encoding); inference ~2 s on the RTX 5050.
- Not yet checked by opening the GeoTIFFs in QGIS by hand (the automated tests read them back with
  rasterio and check CRS, transform and values).

## Open decision (for the user)
Ground-filter default: keep 150 m (the a-priori plan), switch to 0 (raw GLO-30), or keep 150 m until the
Phase 6 multi-area evaluation decides. Choosing from the Pittsburgh numbers would tune on the evaluation
area, so the setting has not been changed.
