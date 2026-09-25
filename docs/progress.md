# Progress

| Phase | Status | Gate |
|---|---|---|
| 0 Data inspection + dataloader | **done** 2026-09-25 | RGB/nDSM/mask aligned; units, NoData verified; loader tested |
| 1 Baselines (B0 trivial, B1 DA-V2-S + global calibration, B2 oracle per-image calibration) | **done** 2026-09-25 | reproducible val metrics + figures (bit-identical rerun) |
| 2 Supervised RGB → nDSM (frozen encoder, then partial fine-tune) | next: awaiting plan approval | beats B1 on val; one test run |
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
aim for RMSE < 4.77 / MAE < 3.03 (beat the oracle per-image calibration baseline).

## Data status
- `phase1` subset: 200 train + 859 val, complete.
- `full` subset: all 5,004 train (DC 1439, NYC 1167, PHL 2398) + 859 val, complete and verified (52 GB). Test not downloaded.
