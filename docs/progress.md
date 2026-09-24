# Progress

| Phase | Status | Gate |
|---|---|---|
| 0 Data inspection + dataloader | **done** 2026-09-25 | RGB/nDSM/mask aligned; units, NoData verified; loader tested |
| 1 Baselines (B0 trivial, B1 global affine DA-V2-S, B2 per-image oracle) | next | reproducible val metrics + figures |
| 2 Supervised RGB → nDSM (frozen encoder, then partial fine-tune) | | beats B1 on val; one test run |
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

## Next: Phase 1 plan
1. `depthwizard/models/dav2.py`: load `depth-anything/Depth-Anything-V2-Small-hf` (already cached in the HF cache), run on full 1024 tiles, upsample output to 1024.
2. `scripts/baseline.py`:
   - B0: predict 0 and the train-mean nDSM
   - B1: fit one global (a, b) on **train** tiles, pred = max(0, a·d + b) where d is DA-V2 relative inverse depth; apply to val
   - B2: per-image least squares on val GT (**oracle**, labelled as such, not deployable)
3. Log via `depthwizard.utils.runs`; figures RGB | GT | pred | error.
4. Cache valid masks before scaling the subset up.
