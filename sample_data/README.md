# sample_data: demo inputs

| Folder | What | Created by |
|---|---|---|
| `gamus_val/` | GAMUS **validation** tiles (never test): `*_rgb.png` to upload (0.33 m/px), `*_lidar_ndsm.npy` / `.png` LiDAR references, README with each tile's error | `scripts/export_demo_samples.py` |
| `geo/` | NAIP GeoTIFFs (0.6 m, native CRS) `*_naip.tif`, LiDAR DSM references `*_lidar_dsm.tif` (NAVD88; the app converts to EGM2008), example GCPs `*_gcps_example.csv` | `scripts/prepare_demo_data.py` (from `data/geo/`) |
| `sample_street_scene.jpg` | a street-level photo: NOT a valid input (the model expects top-down imagery); useful to show the resolution/validity warning | - |

Large files are not in git (`.gitignore`: `*.tif`, `gamus_val/`); run the two scripts to recreate them.
The GeoTIFF demos work offline because their Copernicus GLO-30 DEMs are cached in `data/geo/<area>/`.

GCP CSV format: header `x,y,z` (image CRS) or `lon,lat,z` (WGS 84); z = ground elevation in metres above
EGM2008. The example files take z from the LiDAR DTM (converted to EGM2008) at open-ground points.
