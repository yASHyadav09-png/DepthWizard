# Phase 3: end-to-end demo (upload → trained model → nDSM in metres → 3D)

Status: **PASSED** (2026-09-26). Model: the kept Phase 2b checkpoint
(`runs/20260926-003630_phase2b_partial/checkpoints/best.pt`, sha256 `e55ec4ca7bda…`,
val RMSE 3.148 m). The test split was not used; demo samples come from **val**.

## How to run (Windows, project `.venv`)

Simplest (Phase 8): `start_demo.ps1` (or `start_demo.bat`, `-Offline` without internet), see README.md.
Manual equivalent:

```bash
# terminal 1: backend (loads the model on the GPU at startup, ~8 s)
E:\SIH_Project\DepthWizard\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir E:\SIH_Project\DepthWizard\backend --host 127.0.0.1 --port 8000
```
```bash
# terminal 2: frontend
npm --prefix E:\SIH_Project\DepthWizard\frontend run dev
```
Open http://localhost:5173, drop an image, optionally enter the ground resolution
(m/pixel), and click **Generate Terrain**. Sample images:
`python scripts/export_demo_samples.py` writes 9 GAMUS val tiles (with LiDAR reference) to
`sample_data/gamus_val/`. Upload `*_rgb.png` and enter `0.33`.

The old `backend/.venv` is no longer used. Everything runs from the project `.venv`
(FastAPI, uvicorn and python-multipart were added to it).

## What happens to an upload
1. Decode and validate (JPG/PNG, ≤ 40 MB, ≤ 40 MP). **Never resampled**: the model runs at
   the pixels' native size.
2. `depthwizard.inference.NDSMPredictor` (the same code path as the validation
   evaluation):
   - ≤ 1024 px: one pass, reflect-padded to a multiple of 14, bf16.
   - Larger: 1024 px windows with 128 px overlap, blended with linear feathering.
   - Deterministic numerics (cuDNN deterministic, TF32 off), as in evaluation.
3. Statistics in metres; 2D products (height map and hillshade at true vertical scale); a
   512-cell mesh grid in metres.
4. The response carries the **height product** (the hand-off contract below) and the
   provenance: run, checkpoint sha256, git commit, val RMSE.

## Hand-off contract (`height_product`)
| field | Phase 3 | later |
|---|---|---|
| `kind` | `"ndsm"` (height above ground) | `"dsm"` (Phase 5) |
| `height_units` | `"m"` | |
| `height_array` | full-resolution float32 `.npy` URL | + GeoTIFF |
| `gsd_m`, `gsd_source` | user value → `"user"`; else 0.33 → `"assumed_training_gsd"` | `"geotiff"` |
| `metric_validity` | `"valid"` if GSD known and within ±25% of 0.33 m, else `"uncertain"` + `validity_note` | Phase 4 resampling |
| `nodata` | `null` (every pixel predicted) | |
| `crs`, `transform`, `vertical_datum` | `null` | Phase 5 |
| `model` | run, checkpoint_sha256, git_commit, val_rmse_m | |

## Viewer
- The mesh is built in **metres**: footprint = pixels × GSD, height = predicted metres. The scene
  is scaled uniformly, so **1× exaggeration is true scale** (the slider goes to 5×, as a visual aid only).
- Hovering reads the **true (un-exaggerated) height** of the nearest grid cell and its ground
  position in metres.
- Elevation mode colours by metres, with a legend over the same range as the 2D height map
  (0 m to the 99th percentile, at least 5 m).
- Badges and the statistics panel show the metric validity: green "Height above ground · m"
  or amber "Estimated m · resolution uncertain", with the reason.

## Gate results
| gate | result |
|---|---|
| 1. Demo path = evaluated model | ✅ `scripts/check_demo_consistency.py`: 6 val tiles, max \|ΔRMSE\| 3e-13 m. **Through the web API** (PNG upload of DC_48_31 → downloaded `.npy`): RMSE 5.137524 = eval 5.137524, MAE identical (Δ 1.4e-13) |
| 2. Large image, no seams | ✅ 2048×1524 JPG in 2.9 s (tiled inference 1.2 s); no visible seams. `tests/test_inference.py`: window coverage and overlap, exact reconstruction of a constant field, no extra deviation in the overlap band |
| 3. Upload and fly around | ✅ checked in the browser: upload flow, orbit view, elevation mode + legend, hover readout (cross-checked against the grid values), unknown-GSD warning |

Tests: 26 ML tests + 26 backend tests pass (+1 real-model end-to-end test, run with
`DW_RUN_MODEL_TESTS=1`). The frontend passes the type-check and production build.

Timing on the RTX 5050: a 1024² tile takes 1.35 s end to end (0.65 s inference); 2048×1524 takes 2.9 s.

## Bugs found and fixed during Phase 3
1. **Predictor numerics.** Without the evaluation's deterministic settings (TF32 on in cuDNN),
   predictions differed from the reported metrics by up to 5e-6 m. The predictor now applies the
   same settings: exact match.
2. **3D view blank (React 19).** drei's `<Html>` as the Suspense fallback mounts a separate React
   root. Unmounting it when the texture resolves happened mid-render, which aborted the commit
   (`removeChild` error), so the terrain never appeared. The loader is now a plain DOM overlay
   outside the canvas (`useProgress`).
3. **Elevation mode stayed textured.** three.js doesn't recompile a material's shader when
   `vertexColors`/`map` change. The material is now keyed by view mode.

## Limitations and findings
- **Context sensitivity (measured):** predicting a tile with 640 px windows instead of whole
  differs by about 1.0 m mean absolute deviation. Production uses 1024 px windows, the context
  the model was validated with. Tiles of large images still see less context than a 1024 GAMUS
  tile would at their borders.
- **Non-0.33 m imagery isn't resampled yet** (Phase 4). Heights are labelled "uncertain".
- Not georeferenced (Phase 5). No profile, slope or validation tools yet (Phase 7).
- The production JS bundle is 1.17 MB (three.js); code-splitting is a Phase 8 item.
