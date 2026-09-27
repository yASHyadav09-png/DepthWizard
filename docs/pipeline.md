# Pipeline reference: from uploaded pixels to metres and 3D

What happens to an upload, with the code that does it. (This replaces the Stage-1 description of a
unitless relative-depth prototype; the current system outputs metres.)

## 1. Upload and validation (`backend/app/api/routes.py`, `app/utils/image_io.py`)
- Size (default 40 MB), extension and MIME type are checked before decoding; decompression-bomb limit.
- `.jpg/.jpeg/.png` → section 2. `.tif/.tiff` → section 4. A GCP CSV is accepted only with a GeoTIFF.

## 2. JPG/PNG: height above ground (`app/services/pipeline.py`)
- EXIF orientation applied; RGB uint8.
- Ground resolution: user value (0.01-100 m/px) or ASSUMED 0.33 m/px (the training GSD), reported as
  `gsd_source` = `user` / `assumed_training_gsd`. `metric_validity` = "valid" only within 0.33 m ±25%.
- No resampling of user imagery (Phase 4 decision).

## 3. Model (`depthwizard/inference.py`, `depthwizard/models/ndsm.py`)
- Depth Anything V2 Small (pinned revision) with a DPT head trained to regress GAMUS nDSM in metres:
  h = s0 · net(x), s0 = 3.659840 fixed; Phase 2b checkpoint (last 4 encoder blocks + decoder fine-tuned).
- ImageNet normalisation; reflect-pad to a multiple of 14; images larger than 1024 px are processed in
  1024 px windows with 128 px overlap and linear feathering. bf16 on CUDA, fp32 on CPU;
  deterministic numerics (the web path equals the evaluation within 3e-13 m).
- Output: nDSM, float32 metres, same pixel grid as the input.

## 4. GeoTIFF: absolute DSM (`app/services/geo_pipeline.py`, `depthwizard/geo/`)
1. `raster.read_geo_image`: CRS and transform from the file (no CRS → rejected; rotated/sheared or
   non-square pixels → rejected); GSD = pixel size × CRS unit (feet converted); geographic CRSs are
   reprojected to UTM (reported); uint16 stretched to uint8 (reported).
2. nDSM as in section 3, on the native grid.
3. `dem_cache.get_dem`: cached Copernicus GLO-30 covering the image + margin (`data/dem/`,
   `data/geo/*/glo30_dem.tif`), else fetched once from Planetary Computer (unless `DW_DEM_ALLOW_FETCH=0`).
   No DEM → a georeferenced nDSM only, reported in `notes`.
4. `dem.dtm_on_grid`: fill DEM gaps, grey opening with a 150 m window + Gaussian (removes buildings and
   canopy from the radar DEM), bilinear onto the image grid → DTM (m, EGM2008).
5. `gcp.fit_correction` (optional): offset = median residual, or plane after 3·MAD outlier removal.
6. DSM = DTM + nDSM. `pipeline.write_outputs`: DSM/DTM/nDSM GeoTIFFs, float32, nodata -9999, input CRS
   and transform, tags (product, datum, units, model run and hash, DEM source, GSD).

## 5. Products for the viewer (`app/services/height_processor.py`, `terrain_generator.py`)
- Statistics (median, p95, share above 2 m, ...), colour height map, hillshade at true scale.
- Terrain grid: the surface (nDSM, or DSM for GeoTIFFs) area-averaged to at most 512 px per side,
  float32 base64; for DSM jobs also the nDSM on the same grid (`ndsm_b64`).
- `height_product` contract: kind (ndsm/dsm), units, GSD + source, validity, CRS/transform/bounds/
  corners (lon/lat), vertical datum, DEM and GCP details, GeoTIFF links, model provenance.
- Everything is stored under `backend/outputs/<job_id>/` and served at `/static/<job_id>/`.

## 6. Viewer (`frontend/src/`)
- Dashboard (`components/`): 3D preview (scaled to fit, true proportions), rasters, statistics.
- Explorer (`explorer/`): scene in real metres (X = col·GSD, Z = row·GSD, Y = height; for DSM jobs
  Y = elevation - floor(min elevation), a shift only). Terrain-constrained Orbit/Fly/Walk (`physics.ts`,
  ≤ 0.25 m sub-steps checked before they are applied), layers, profile/measure, minimap, HUD with map
  coordinates (`services/geo.ts`).

## 7. Validation (`app/services/validation.py`)
- Reference `.npy` on the job grid, or a GeoTIFF (reprojected bilinearly onto the job grid; NAVD88
  converted to EGM2008 with PROJ; other vertical datums ASSUMED EGM2008 and reported).
- Target: the job's DSM or nDSM. Metrics with the same code as the model evaluation
  (`depthwizard/eval/metrics.py`), per height band for nDSM; error map and 3D error layer (±10 m).
