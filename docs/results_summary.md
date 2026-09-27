# DepthWizard results summary (for the presentation)

All numbers come from logged runs (`runs/<stamp>_<name>/`, one row each in `runs/results.csv`).
Metrics are pooled over valid pixels: MAE = mean |pred - ref|, RMSE = sqrt(mean (pred - ref)^2),
bias = mean(pred - ref), Pearson r. "Significant" = paired bootstrap (2,000 resamples, seed 0) with the
95% interval entirely below 0.

## 1. Height above ground from a single image (GAMUS, 0.33 m/px)

| Method | val RMSE / MAE (m) | **test** RMSE / MAE (m) | test r |
|---|---|---|---|
| B0: predict 0 m everywhere | 7.85 / 3.94 | 8.72 / 4.42 | n/a |
| B1: frozen Depth Anything V2 + best global calibration | 6.60 / 4.66 | 7.29 / 4.83 | 0.25 |
| Oracle per-image calibration (diagnostic; uses the answer, not deployable) | 4.77 / 3.03 | n/a | |
| 2a: supervised, frozen encoder | 3.55 / 1.73 | n/a | |
| **2b: supervised, last 4 encoder blocks fine-tuned (production)** | **3.15 / 1.56** | **4.69 / 2.01** | **0.79** |

- Test = 2,861 tiles incl. 1,000 NYC tiles never seen in validation; evaluated **once**
  (`docs/final_test_report.md`, run `20260926-141407_final_test`, frozen).
- 2b vs B1 on test: dRMSE -2.61 m [-2.72, -2.50]. Per city (test RMSE / MAE): DC 4.63 / 2.73,
  NYC 4.61 / 2.55, PHL 4.75 / 1.49.
- Per class (test RMSE; val in brackets): building 7.43 (3.84), tree 6.10 (4.88), ground 2.51 (1.38) m.
- Test > val mainly because of Center City Philadelphia skyscrapers (LiDAR up to 305 m; the worst 5% of
  tiles hold 51% of the squared error). Median test tile RMSE 2.71 m.
- Relative depth is never used as height: B1 shows why (frozen depth needs a per-image scale that
  varies ~25x between tiles).

Figures: `runs/20260926-003630_phase2b_partial/figures/` (training curves, scatter, per-class,
example panels), `runs/20260926-141407_final_test/figures/` (per city, per class, per height band).

## 2. Ground resolution (Phase 4, val, simulated coarser imagery)

| GSD (m/px) | 0.33 | 0.50 | 0.66 | 1.00 | 2.00 |
|---|---|---|---|---|---|
| val RMSE, direct inference (production) | 3.15 | 3.41 | 3.87 | 5.19 | 7.46 |
| val RMSE, resample to 0.33 m first | 3.15 | worse at every GSD, significantly from 0.66 m | | | |

Decision: direct inference at the image's own resolution; results outside 0.33 m +-25% are flagged
"resolution uncertain". Figure: `runs/20260926-124248_phase4_gsd_study/figures/gsd_curves.png`.

## 3. Absolute DSM from a GeoTIFF (Phase 5)
Pipeline: CRS/transform/GSD from the file → nDSM → + Copernicus GLO-30 ground (150 m filter, EGM2008)
→ optional GCP correction → DSM / DTM / nDSM GeoTIFFs in the input CRS. LiDAR references in NAVD88 are
converted to EGM2008 with PROJ (-0.478 m at Pittsburgh).

| Pittsburgh (NAIP 0.6 m vs 3DEP LiDAR, 2 m grid) | RMSE (m) |
|---|---|
| our DSM | 5.40 |
| raw GLO-30 as a DSM (baseline) | 4.76 |
| our DTM / raw GLO-30 DTM | 4.45 / 3.27 |
| model nDSM | 3.05 |
| LiDAR DTM + model nDSM (nDSM error only) | 3.05 |

The ground model dominates the DSM error on hilly terrain. GCP offset correction (5 simulated points):
DTM 4.33 m, DSM 5.13 m on pixels > 50 m from the points. Figures:
`runs/20260926-161419_phase5_pittsburgh/figures/`.

## 4. Real imagery, other landscapes (Phase 6, out of domain)
Production model on real NAIP 0.6 m vs USGS 3DEP LiDAR (areas fixed before evaluation):

| Area | nDSM RMSE / MAE (m) | DSM ours / raw GLO-30 (m) |
|---|---|---|
| Pittsburgh, hilly urban | 3.05 / 2.02 | 5.40 / 4.76 |
| Carmel IN, suburban | 2.82 / 1.64 | 3.09 / 3.66 |
| Champaign IL, farmland | 0.37 / 0.04 | 0.50 / 0.49 |
| Del Rio TX, arid town | 1.41 / 0.70 | 1.43 / 1.79 |
| Grayling MI, forest | 2.08 / 1.10 | 3.20 / 2.46 |
| **pooled nDSM** | **2.17 / 1.10** (r 0.70) | |

- Indianapolis was excluded from nDSM/DTM metrics: its LiDAR "bare-earth" DTM contains the buildings.
- Tall objects are underestimated at 0.6 m (10-20 m objects: bias -6.0 m).
- The 150 m ground filter beats raw GLO-30 for the DTM in 4 of 5 areas (loses on steep terrain).

**Scale-augmentation fine-tune (6b), rejected by a pre-registered rule:** on simulated coarse val it
was much better (0.66 m: 3.87 → 3.17 m) but on real NAIP it was worse (2.14 → 2.52 m, +0.38
[+0.32, +0.44]). Simulated degradation did not transfer to a real sensor, so 2b stays in production.
Figures: `runs/20260927-131340_phase6_aois_phase2b/figures/` (2b),
`runs/20260927-155111_phase6_aois_phase6_scaleaug/figures/` (6b).

## 5. System
- Inference: RTX 5050 Laptop, bf16; a 1024 x 1024 tile ~0.3 s; a 2194 x 2185 GeoTIFF: 2 s inference,
  ~10 s end to end including DEM, GeoTIFF and PNG export. Large images are tiled (1024 px windows,
  128 px overlap, feathered) at native resolution.
- The web app reproduces the evaluated model exactly (max difference 3e-13 m on val tiles;
  `scripts/check_demo_consistency.py`).
- Tests: 41 ML, 40 backend (1 opt-in real-model test), 34 frontend (physics, slope, measurement,
  geo mapping).
- Explorer: real-metre scene, orbit / fly (≥ 2 m clearance) / walk (1.7 m eye height) with movement
  checked against the terrain before it is applied; profile, measure, slope, error layers; GeoTIFF export.

## 6. What we would do next
1. Indian imagery: evaluate on Indian aerial/satellite scenes with any available reference heights
   (e.g. Cartosat stereo DSMs or local LiDAR), then fine-tune with Indian data.
2. Tall buildings: more training data with high-rise districts; a height-aware loss.
3. Coarser imagery: train on real coarse imagery paired with LiDAR (not simulated degradation).
4. Ground model: a terrain-adaptive DEM filter, or a better DEM where available.
