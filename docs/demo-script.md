# 5-minute demo script (SIH judges)

Goal: show that the heights are real metres, checked against LiDAR, and that the 3D view is built on them.

## Before the judges arrive (5 min)
1. `.\start_demo.ps1 -Offline` (or double-click `start_demo.bat`). Wait for "DepthWizard is running".
   Offline mode needs no internet: DEMs for the demo areas are cached in `data/geo/`, the model in the
   Hugging Face cache.
2. Do one throwaway run (any PNG) so the GPU is warm.
3. Have Explorer open in a second tab on a finished GeoTIFF job (`?job=<id>` in the URL) as a fallback.
4. Inputs: `sample_data/gamus_val/` (PNGs + `.npy` LiDAR references) and `sample_data/geo/`
   (NAIP GeoTIFFs + `*_lidar_dsm.tif` references + `*_gcps_example.csv`).

## 0:00 Problem (30 s)
"Height maps normally need stereo pairs, LiDAR or radar. We estimate height above ground in metres
from one ordinary top-down image, turn it into an absolute elevation model when the image is
georeferenced, and let you explore it in 3D."

## 0:30 One image → metres (1 min)
1. Upload `sample_data/gamus_val/PHL_6325_rgb.png` (typical tile, RMSE 2.3 m), ground resolution **0.33**.
   Generate (~2 s). For tall buildings use `DC_48_31_rgb.png`.
2. Point at: height map, statistics in metres, the green "valid resolution" badge, model provenance
   (run, checkpoint hash, GPU).
3. Hover a building: height in metres.
4. Say: "Trained on GAMUS LiDAR heights. On 2,861 unseen test tiles, evaluated once: RMSE 4.7 m, MAE 2.0 m,
   versus 7.3 m for the best calibration of a generic depth model."

## 1:30 Proof against LiDAR (45 s)
1. Enter 3D Explorer → Validation → upload the tile's `*_lidar_ndsm.npy`.
2. Show RMSE/MAE/bias and switch the layer to **Error vs reference** (blue = too low, red = too high).
3. Say honestly: "Tall buildings are underestimated; that is our main known limitation."

## 2:15 GeoTIFF → absolute DSM (1 min 15 s)
1. Back on the dashboard, upload `sample_data/geo/pittsburgh_naip.tif`, add
   `pittsburgh_gcps_example.csv` (offset). Generate (~10 s).
2. Point at: CRS EPSG:26917 read from the file, 0.6 m resolution (flagged: outside the trained band),
   elevation 218-371 m above EGM2008, DEM = Copernicus GLO-30 (cached), GCP offset applied.
3. Download buttons: DSM / DTM / nDSM GeoTIFFs keep the input CRS (open in QGIS).
4. Explorer → Validation → upload `pittsburgh_lidar_dsm.tif`: the NAVD88 datum is converted
   automatically. DSM RMSE ~5 m on a hilly city.

## 3:30 Fly through (1 min)
1. Elevation layer, then **Fly** (key 2): click to capture the mouse, W A S D, Space/C, wheel = speed.
   "The camera can't enter the surface: every step is checked against the terrain, and it stays ≥ 2 m up."
2. **Walk** (key 3): eye height 1.7 m, follows the ground; walls block you.
3. HUD: elevation, height above ground, easting/northing, lat/lon under the cursor.
4. **Profile** tool across the valley: elevation profile in metres. **Shot** saves a PNG.

## 4:30 Honesty and next steps (30 s)
"We validated on five real US landscapes against USGS LiDAR (RMSE 2.2 m). We also rejected our own
fine-tune because it looked better on simulated data but was worse on real imagery. Next: Indian imagery
with Indian reference heights, and better handling of tall buildings."

## If something goes wrong
- Backend window shows an error: close both windows, run `start_demo.ps1` again.
- A GeoTIFF outside the cached areas without internet: the app still works but returns a georeferenced
  **nDSM** (no DEM) and says so.
- Reload a finished result: `http://127.0.0.1:5173/?job=<id>`.
