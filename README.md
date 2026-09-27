# DepthWizard: single-view height estimation and 3D flythrough

**Smart India Hackathon 2026 · Problem Statement 26175 (ISRO)**

One top-down RGB image → **height above ground in metres** (nDSM) → with a GeoTIFF: **absolute surface
elevation** (DSM, metres above the EGM2008 geoid) → interactive **3D terrain** you can orbit, fly and
walk through, with profile, measurement, slope and validation tools.

```
JPG / PNG ─┐                                  ┌─> nDSM (m) ─────────────────────────────┐
           ├─> Depth Anything V2 Small,       │                                          ├─> 3D Explorer
GeoTIFF  ──┘   fine-tuned on GAMUS (nDSM, m) ─┤                                          │   + GeoTIFF / .npy export
  (CRS, transform, GSD read from the file)    └─> + Copernicus GLO-30 ground (DTM)       │   + validation vs LiDAR
                                                  + optional GCP correction ─> DSM (m) ──┘
```

## Results (details in [docs/results_summary.md](docs/results_summary.md))

| Evaluation | RMSE | MAE | Notes |
|---|---|---|---|
| GAMUS **test** (2,861 tiles, one-time, frozen) | **4.69 m** | **2.01 m** | global-calibration baseline: 7.29 / 4.83 m |
| GAMUS val (859 tiles) | 3.15 m | 1.56 m | r 0.89; checkpoint selection on val only |
| Real **NAIP 0.6 m vs USGS LiDAR**, 5 US areas (out of domain) | 2.17 m | 1.10 m | urban, suburban, farmland, arid, forest |
| Absolute DSM, Pittsburgh (hilly) | 5.40 m | 3.98 m | vs LiDAR DSM; the ground model (GLO-30) dominates the error |

Honest limits: tall objects are underestimated (skyscrapers most), accuracy drops for imagery coarser
than ~0.4 m/px, and no Indian imagery has been evaluated yet. See [Limitations](#limitations).

## Quick start (Windows)

Prerequisites: Python 3.11, Node.js 20+ (tested with 24), an NVIDIA GPU is optional (tested on an RTX 5050
Laptop; CPU works but is slower).

```powershell
# 1. Python environment (repository root)
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt      # exact versions: requirements-ml.lock.txt

# 2. Model weights (not in git, 118 MB): put the Phase 2b checkpoint at
#    runs\20260926-003630_phase2b_partial\checkpoints\best.pt
#    (sha256 starts with e55ec4ca7bda; reproduce with scripts/train_ndsm.py, see below)

# 3. Start (installs the frontend packages on first run, waits for the model, opens the browser)
.\start_demo.ps1            # or double-click start_demo.bat
.\start_demo.ps1 -Offline   # no internet: cached DEMs + cached Hugging Face backbone only
```

The first online start downloads the Depth Anything V2 Small backbone (pinned revision) into the
Hugging Face cache; after that, `-Offline` works. Backend: http://127.0.0.1:8000 (API docs at `/docs`),
app: http://127.0.0.1:5173. Close the two windows to stop.

### Demo inputs
- `sample_data/gamus_val/*_rgb.png`: GAMUS validation tiles (0.33 m/px) with LiDAR references
  (`*_lidar_ndsm.npy`) for the in-app validation. Created by `scripts/export_demo_samples.py`.
- `sample_data/geo/*_naip.tif`: georeferenced NAIP GeoTIFFs (Pittsburgh, Carmel IN, Del Rio TX) with
  LiDAR DSM references (`*_lidar_dsm.tif`, NAVD88, converted automatically) and example GCP files.
  Created by `scripts/prepare_demo_data.py` from `data/geo/` (which also holds their cached DEMs).
- A judges' walkthrough: [docs/demo-script.md](docs/demo-script.md).

## Using the app
1. **Upload** a top-down image.
   - JPG/PNG: height above ground. Enter the ground resolution (m/px) if you know it; otherwise
     0.33 m/px (the training resolution) is assumed and flagged.
   - GeoTIFF: CRS, transform and resolution are read from the file; the app adds the Copernicus GLO-30
     ground surface and returns an absolute DSM. Optional **GCP CSV** (`x,y,z` in the image CRS or
     `lon,lat,z`; z in metres above EGM2008) with an `offset` (recommended) or `plane` correction.
2. **Dashboard**: 3D preview at true scale, height map, shaded relief, statistics and provenance
   (model run, checkpoint hash, device).
3. **Enter 3D Explorer**: Orbit / Fly (≥ 2 m above the surface) / Walk (eye height 1.7 m), minimap,
   layers (RGB, elevation, height above ground, slope, error), height profile, two-point measurement,
   screenshots. For GeoTIFFs the HUD shows elevation, height above ground, easting/northing and lat/lon.
4. **Validate** against a reference (`.npy` on the same grid, or a GeoTIFF in any CRS): MAE, RMSE, bias,
   r, per height band, and an error layer in 3D.
5. **Export**: DSM / DTM / nDSM GeoTIFFs (input CRS, float32 metres), `.npy` arrays, PNGs, metadata JSON.

A result is shareable with `?job=<id>` in the URL.

## Repository layout
| Path | Contents |
|---|---|
| `depthwizard/` | Python package: `data/` (GAMUS reader, validity mask, coarsening), `models/` (DA-V2 + nDSM head), `eval/` (metrics), `geo/` (GeoTIFF IO, DEM → DTM, GCP, pipeline), `inference.py` (tiled predictor used by the app) |
| `scripts/` | Entry points: data download/inspection, baselines, training, evaluation, bootstrap, GSD study, geo evaluations, demo data |
| `configs/` | Experiment configs and dataset manifests (`subsets/`), Phase 6 evaluation areas |
| `backend/` | FastAPI app (`app/`), tests |
| `frontend/` | React + TypeScript + Vite + React Three Fiber viewer and Explorer |
| `runs/` | One folder per experiment (config, git hash, metrics, figures; checkpoints are not in git); `runs/results.csv` = all results |
| `docs/` | Per-phase reports, plans and results |

## Reproducing the results
All experiments go through `depthwizard.utils.runs` (config, seed, git commit, dataset manifest,
metrics, a row in `runs/results.csv`).

```powershell
.venv\Scripts\python.exe -m depthwizard.data.download --name full --all   # GAMUS train+val (~52 GB), no test
.venv\Scripts\python.exe scripts\phase1_baseline.py --config configs\phase1_baseline.yaml
.venv\Scripts\python.exe scripts\train_ndsm.py --config configs\phase2a_frozen.yaml
.venv\Scripts\python.exe scripts\train_ndsm.py --config configs\phase2b_partial.yaml
.venv\Scripts\python.exe scripts\phase4_gsd_study.py
.venv\Scripts\python.exe scripts\fetch_geo_aoi.py --name pittsburgh --lon -79.985 --lat 40.425 --naip-year 2019
.venv\Scripts\python.exe scripts\phase5_eval_aoi.py --aoi pittsburgh
.venv\Scripts\python.exe scripts\phase6_eval_aois.py
```
Tests: `.venv\Scripts\python.exe -m pytest -q` (ML), `cd backend; ..\.venv\Scripts\python.exe -m pytest -q`,
`cd frontend; npm test`.

Rules followed throughout: the supervised target is the GAMUS nDSM in metres (relative depth is never
treated as height); the test split was used exactly once for the final model and never for fitting,
calibration, thresholds or checkpoint selection; GSD/CRS/datum/units are read from data, never assumed
(JPG/PNG resolution is flagged as assumed); decision rules for model changes were written down before
the results were seen (e.g. Phase 6: a fine-tune that looked better on simulated data was rejected
because it was worse on real imagery).

## Limitations
- **Tall objects are underestimated**: bias -6 m for 10-20 m objects on NAIP; on GAMUS test the
  Philadelphia skyscrapers (up to 305 m) cause half of the squared error.
- **Resolution**: trained at 0.33 m/px. Direct inference at 0.5-1 m works but degrades (val RMSE
  3.4 m at 0.5 m, 5.2 m at 1 m); outside 0.33 m ±25% results are flagged "resolution uncertain".
  A scale-augmented fine-tune improved simulated coarse imagery but not real NAIP and was not adopted.
- **Ground model**: GLO-30 is a 30 m radar surface model; the 150 m ground filter helps in flat,
  built-up and forested areas and hurts on steep terrain. DEM errors go straight into the DSM.
- **Geography**: training data are three US cities (GAMUS); validation areas are in the US. No Indian
  imagery or reference data has been evaluated.
- Imagery must be near-nadir; relief displacement and image registration errors (NAIP up to ~6 m)
  shift objects relative to reference data.

## Documentation
[progress](docs/progress.md) · [data report](docs/phase0_data_report.md) · [baselines](docs/phase1_baseline.md) ·
[training plan](docs/phase2_plan.md) · [2a](docs/phase2a_results.md) · [2b](docs/phase2b_results.md) ·
[final test](docs/final_test_report.md) · [demo app](docs/phase3_demo.md) · [GSD study](docs/phase4_gsd_study.md) ·
[geospatial / DSM](docs/phase5_geospatial.md) · [out-of-domain + fine-tune](docs/phase6_results.md) ·
[Explorer](docs/phase7_explorer.md) · [pipeline](docs/pipeline.md) · [results summary](docs/results_summary.md) ·
[demo script](docs/demo-script.md)

## Data and model credits
- **GAMUS** (Xiong et al.; Hugging Face `earthflow/GAMUS`): training/validation/test data.
- **Depth Anything V2 Small** (Yang et al.; `depth-anything/Depth-Anything-V2-Small-hf`, Apache-2.0): backbone.
- **Copernicus DEM GLO-30**: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus
  Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
- **USDA NAIP** imagery and **USGS 3DEP** LiDAR (public domain), accessed via Microsoft Planetary Computer.
- Datum conversion: PROJ with NOAA/NGS and EGM2008 grids.
