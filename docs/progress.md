# Progress

| Phase | Status | Gate |
|---|---|---|
| 0 Data inspection + dataloader | **done** 2026-09-25 | RGB/nDSM/mask aligned; units, NoData verified; loader tested |
| 1 Baselines (B0 trivial, B1 DA-V2-S + global calibration, B2 oracle per-image calibration) | **done** 2026-09-25 | reproducible val metrics + figures (bit-identical rerun) |
| 2 Supervised RGB → nDSM (frozen encoder, then partial fine-tune) | **done** 2026-09-26: 2b kept (pre-registered rule) | beats B1 on val; one test run |
| 3 Thin end-to-end demo (predict.py + FastAPI + minimal R3F viewer) | **done** 2026-09-26 | web path = evaluated model (Δ 1e-13 m); large images tiled; 3D in metres |
| 4 Resolution/GSD handling | **closed** 2026-09-26 (study done; GCP -> Phase 5, scale-aug fine-tune -> Phase 6) | GSD study in results.csv |
| 5 Absolute DSM (DEM, CRS, vertical datum) + GCP correction | **done** 2026-09-27 (ground-filter default: decision open) | valid GeoTIFF out (tested with rasterio; manual QGIS check pending) |
| 6 Out-of-domain validation (3DEP LiDAR + NAIP: urban/sparse/hilly/forest) + scale-aug fine-tune test on real ~0.6 m NAIP | | per-landscape metrics |
| 7 Viewer features (fly mode, probe, profile, slope, validation panel) | **7a-7e done** 2026-09-26 | see docs/phase7_explorer.md |
| 8 Polish, docs, packaging | | |

## Phase 0 summary
See `docs/phase0_data_report.md`. Key facts: 8,724 tiles in 3 cities (DC/NYC/PHL); val has no NYC;
no georeferencing in the tiles; nDSM in metres (cathedral check); DC -5 m clip = NoData;
LiDAR spikes filtered by 5×5 median; leaf-off imagery; relief displacement on tall buildings.

## Phase 1 summary
See `docs/phase1_baseline.md`. Official run `runs/20260925-052422_phase1_baseline_repro`.
Val (859 tiles): B0-zero RMSE 7.85 / MAE 3.94; B0-mean 6.83 / 5.10; **B1 (r1022_pct_lin) 6.60 / 4.66, r 0.25**;
oracle per-image calibration baseline 4.77-4.83 / 3.03-3.10 (not deployable).
B1 beats B0-mean by only 0.22 m and loses to B0-zero on MAE. Frozen DA-V2-S lacks a consistent metric scale
(oracle slope varies ~25x between tiles, 6% of tiles inverted); predictions collapse into a 2-10 m band.

## Phase 2 targets (fixed before training)
val RMSE < 6.60 (bootstrap CI excluding 0) and building RMSE < 7.10; MAE < 3.94 (beat B0-zero);
aim for RMSE < 4.77 / MAE < 3.03 (outperform the oracle per-image calibration baseline).

## Phase 2a summary
See `docs/phase2a_results.md`. Run `runs/20260925-163139_phase2a_frozen`, best epoch 27/30.
Val: RMSE 3.548, MAE 1.729, r 0.861, bias -0.62; building RMSE 4.61, tree 5.51.
Significantly better than B1 and B0, and outperformed the Phase 1 oracle per-image calibration baseline
on the reported val metrics (all paired-bootstrap upper bounds < 0; the oracle baseline is a diagnostic
reference, not a theoretical upper bound). Trained before the NaN-target fix (logging-only issue). Remaining error: tall objects underestimated (-9 m above 20 m), DC leaf-off forest.
Test split untouched. 2b not started.

## Phase 2b summary
See `docs/phase2b_results.md`. Run `runs/20260926-003630_phase2b_partial`, init from 2a best, best epoch 20/20.
Val: RMSE 3.148, MAE 1.556, r 0.889, bias -0.42; building RMSE 3.84, tree 4.88.
Decision: dRMSE(2b-2a) = -0.400 [-0.480, -0.326] -> upper bound < 0 -> **2b is the Phase 2 model**.
Still underestimates tall objects (-6.3 m bias above 20 m). NYC untested (no NYC val tiles).
Test split untouched; awaiting explicit approval for any test-set action.

## Phase 3 summary
See `docs/phase3_demo.md`. Upload JPG/PNG -> 2b model (tiled, native resolution) -> nDSM in metres ->
3D terrain at true scale, hover height readout, elevation legend, metric-validity warning.
Hand-off contract `height_product` (kind, units, gsd + source, metric_validity, crs/transform/datum null
until Phase 5). Run: see docs/phase3_demo.md. Old backend/.venv retired (project .venv only).

## Phase 4 summary (CLOSED 2026-09-26)
See `docs/phase4_gsd_study.md`. Direct inference (A) RMSE 3.15 (0.33 m) / 3.41 (0.5) / 3.87 (0.66) / 5.19 (1.0) /
7.46 (2.0); tall objects and bias degrade most. Resample-to-0.33 (B) is never better and significantly worse
from 0.66 m.
Decisions: (1) production = direct inference, never resample user imagery (test-locked);
(2) the validity threshold is NOT widened: "valid" stays at 0.33 m +-25% (test-locked);
(3) scale-augmentation fine-tuning moves to Phase 6, tested on real ~0.6 m NAIP imagery;
(4) GCP correction stays in Phase 5. The trained 2b model is unchanged.

## Phase 5 summary
See `docs/phase5_geospatial.md`. GeoTIFF input -> CRS/transform/GSD from the file -> nDSM (2b, native grid)
+ Copernicus GLO-30 ground (150 m opening filter, EGM2008) + optional GCP offset/plane -> DSM/DTM/nDSM
GeoTIFFs in the input CRS. Falls back to a georeferenced nDSM without a DEM. Viewer: elevation, height above
ground, easting/northing, lon/lat, GeoTIFF export, GeoTIFF validation (NAVD88 -> EGM2008 via PROJ).
Pittsburgh (NAIP 0.6 m vs 3DEP LiDAR, run `runs/20260926-161419_phase5_pittsburgh`): DSM RMSE 5.40 m
(raw GLO-30 DSM 4.76), DTM 4.45 (raw GLO-30 3.27), model nDSM 3.05 (bias -2.0 m on objects).
The 150 m filter hurts on steep terrain; the default is left unchanged pending a decision
(choosing it from Pittsburgh would tune on the evaluation area). GCP offset helps slightly; plane over-fits.

## Phase 7 summary
See `docs/phase7_explorer.md`. Full-screen 3D Explorer in real metres: Orbit/Fly/Walk with terrain-constrained
movement (fly >= 2 m, walk eye 1.7 m), minimap, HUD, slope layer, profile + measure, validation panel
(same metric code as evaluation), screenshot + export.

## Final test evaluation (FROZEN)
See `docs/final_test_report.md`, run `runs/20260926-141407_final_test` (2,861 tiles incl. 1,000 NYC).
Test RMSE / MAE: B0-zero 8.72 / 4.42, B1 7.29 / 4.83, **Phase 2b 4.69 / 2.01** (r 0.79).
2b - B1: dRMSE -2.61 [-2.72, -2.50], dMAE -2.82. NYC 4.61 / 2.55. Test > val mainly because of Center City
Philadelphia skyscrapers (LiDAR up to 305 m; worst 5% of tiles = 51% of squared error). No further test use.

## Data status
- `phase1` subset: 200 train + 859 val, complete.
- `full` subset: all 5,004 train (DC 1439, NYC 1167, PHL 2398) + 859 val, complete and verified (52 GB). Test not downloaded.
- 14 PHL tiles (7 train, 7 val) contain NaN nDSM pixels (24,235 px); excluded by valid_mask.
