# Progress

| Phase | Status | Gate |
|---|---|---|
| 0 Data inspection + dataloader | **done** 2026-09-25 | RGB/nDSM/mask aligned; units, NoData verified; loader tested |
| 1 Baselines (B0 trivial, B1 DA-V2-S + global calibration, B2 oracle per-image calibration) | **done** 2026-09-25 | reproducible val metrics + figures (bit-identical rerun) |
| 2 Supervised RGB → nDSM (frozen encoder, then partial fine-tune) | **done** 2026-09-26: 2b kept (pre-registered rule) | beats B1 on val; one test run |
| 3 Thin end-to-end demo (predict.py + FastAPI + minimal R3F viewer) | | GAMUS image viewable as 3D |
| 4 Resolution/GSD handling + GCP correction | | GSD study in results.csv |
| 5 Absolute DSM (DEM, CRS, vertical datum) | | valid GeoTIFF out, correct in QGIS |
| 6 Out-of-domain validation (3DEP LiDAR + NAIP: urban/sparse/hilly/forest) | | per-landscape metrics |
| 7 Viewer features (fly mode, probe, profile, slope, validation panel) | | |
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

## Data status
- `phase1` subset: 200 train + 859 val, complete.
- `full` subset: all 5,004 train (DC 1439, NYC 1167, PHL 2398) + 859 val, complete and verified (52 GB). Test not downloaded.
- 14 PHL tiles (7 train, 7 val) contain NaN nDSM pixels (24,235 px); excluded by valid_mask.
