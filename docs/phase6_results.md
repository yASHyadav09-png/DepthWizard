# Phase 6 results

Plan and pre-registered decision rule: `docs/phase6_plan.md`. Areas: `configs/phase6_aois.yaml`.

## 6a: Phase 2b on real NAIP 0.6 m imagery vs USGS 3DEP LiDAR

Run `runs/20260927-131340_phase6_aois_phase2b` (`scripts/phase6_eval_aois.py`, model
`20260926-003630_phase2b_partial`, direct inference at 0.6 m, ground filter 150 m).
Superseded runs from the same day (identical model outputs, earlier QC fields):
`20260927-130944_phase6_aois_phase2b` (no reference QC), `20260927-131209_phase6_aois_phase2b`
(first QC measure). Their results.csv rows are kept for the record; use the 131340 run.

Comparison on the LiDAR 2 m grid; reference nDSM = LiDAR DSM - DTM; heights EGM2008.

| Area | Landscape | NAIP date | nDSM RMSE / MAE / bias (m) | r | objects > 2 m: RMSE / bias | share > 2 m |
|---|---|---|---|---|---|---|
| pittsburgh | hilly urban | 2019-09 | 3.05 / 2.02 / -0.56 | 0.55 | 3.94 / -2.03 | 48% |
| carmel_suburb | suburban | 2016-09 | 2.82 / 1.64 / +0.45 | 0.62 | 3.69 / -0.51 | 34% |
| champaign_farm | farmland | 2019-09 | 0.37 / 0.04 / -0.01 | 0.73 | 3.85 / -2.42 | 1% |
| del_rio_arid | arid town | 2019-03 | 1.41 / 0.70 / -0.14 | 0.60 | 2.38 / -1.53 | 20% |
| grayling_forest | forest | 2018-10 | 2.08 / 1.10 / -0.20 | 0.80 | 3.04 / -1.02 | 37% |
| **pooled (5 areas)** | | | **2.17 / 1.10 / -0.10** | **0.70** | | |
| indianapolis_core | dense urban | 2016-09 | *reference invalid (see below)* | | | |

Pooled by LiDAR height band (5 areas): 0-2 m RMSE 1.39 (bias +0.38); 2-5 m 2.43 (-0.04);
5-10 m 3.33 (-1.84); 10-20 m 6.83 (-5.95); > 20 m 19.4 (-18.1, only 1,018 px).
**Tall objects are strongly underestimated at 0.6 m**, as in the Phase 4 simulation. Low vegetation and
ground are slightly over-predicted.

### Reference failure: Indianapolis
The LiDAR "DTM" of `USGS_LPC_IN_Central_MarionCo_2016` contains most downtown buildings: on flat
ground (~218 m) the DTM reaches 258 m at its 99th percentile, and building outlines are visible in it.
DSM - DTM is therefore not a height-above-ground reference there (roofs come out ~0 m; only edges remain),
and the DTM is not bare earth (even raw GLO-30 scores 11 m "error"). The area is excluded from pooled
nDSM and DTM metrics (amendment in the config, made after this evaluation and before any 6b model
existed, so it cannot favour either model). Its DSM comparison remains valid: our DSM RMSE 15.8 m against
raw GLO-30 19.4 m (skyscrapers).

Reference QC reported for all areas: `dtm_cliff_frac` (share of DTM pixels steeper than 63 deg between
2 m neighbours): Indianapolis 0.040; the others 0.000-0.0017. A first QC measure (pixels > 5 m above a 200 m
grey opening) did not separate buildings from steep hills (Pittsburgh 0.27 vs Indianapolis 0.26) and was
replaced. The exclusion rests on the visual evidence; the metric documents it.

### DSM / DTM (GLO-30 150 m filter vs raw GLO-30)
| Area | DSM ours | DSM raw GLO-30 | DTM ours | DTM raw |
|---|---|---|---|---|
| pittsburgh | 5.40 | 4.76 | 4.45 | 3.27 |
| carmel_suburb | 3.09 | 3.66 | 1.80 | 2.32 |
| champaign_farm | 0.50 | 0.49 | 0.31 | 0.38 |
| del_rio_arid | 1.43 | 1.79 | 1.13 | 1.56 |
| grayling_forest | 3.20 | 2.46 | 3.14 | 4.24 |
| indianapolis_core (DSM only) | 15.77 | 19.35 | n/a | n/a |

DIAGNOSTIC, ground-filter window (DTM RMSE, m; not used to choose anything; the default stays 150 m
by the user's decision):

| Area | 0 (raw) | 90 | 150 | 300 |
|---|---|---|---|---|
| pittsburgh (hilly) | **3.27** | 3.61 | 4.45 | 8.92 |
| carmel_suburb | 2.32 | **1.79** | 1.80 | 2.02 |
| champaign_farm | 0.38 | **0.28** | 0.31 | 0.43 |
| del_rio_arid | 1.56 | 1.23 | 1.13 | **1.08** |
| grayling_forest | 4.24 | 3.49 | 3.14 | **2.07** |

The filter helps in 4 of 5 valid areas (it removes canopy/buildings from the radar DEM) and hurts only on
steep terrain. This supports keeping 150 m as the default; a terrain-adaptive window is a possible
future improvement (it would need its own held-out evaluation).

### Alignment
NAIP vs LiDAR nDSM cross-correlation shifts are 0.2-4.5 m (e.g. Champaign +4.5 m, Grayling -3.8 m in x):
consistent with NAIP's registration spec (<= 6 m). They inflate per-pixel errors at object edges for any
model equally; not corrected.

## 6b: scale-augmentation fine-tune (NOT adopted; 2b stays)

Training run `runs/20260927-133353_phase6_scaleaug` (commit 97d4354, clean tree): from 2b best, 10 epochs,
scale augmentation 0.33-0.64 m/px (p = 0.5). Selected epoch 10 by the pre-registered val criterion
(mean val RMSE at 0.33 and 0.66 m: 3.135). Evaluations: `runs/20260927-153217_phase6_gsd_scaleaug`
(val, Phase 4 protocol) and `runs/20260927-155111_phase6_aois_phase6_scaleaug` (NAIP areas, once).
(A first launch of the val study failed on a relative model path before producing results; its empty
run folder was deleted.)

### Decision rule (`runs/20260927-133353_phase6_scaleaug/decision.json`)
| Rule | 2b RMSE | 6b RMSE | Delta [95% CI] | Criterion | Result |
|---|---|---|---|---|---|
| R1 val, simulated 0.66 m | 3.869 | 3.166 | -0.703 [-0.781, -0.633] | upper < 0 | PASS |
| R2 val, native 0.33 m | 3.148 | 3.105 | -0.042 [-0.068, -0.017] | upper < +0.10 | PASS |
| R3 real NAIP, 5 areas (128 m blocks) | 2.141 | 2.521 | **+0.380 [+0.319, +0.442]** | upper < 0 | **FAIL** |

**Outcome: 6b is not recommended; Phase 2b remains the production model.** MAE agrees (R3 +0.17 m
[+0.14, +0.20]).

### What happened
- On SIMULATED coarse imagery 6b is much better at every resolution (val RMSE, direct inference):

| GSD | 2b | 6b |
|---|---|---|
| 0.33 m | 3.15 | 3.11 |
| 0.50 m | 3.41 | 3.02 |
| 0.66 m | 3.87 | 3.17 |
| 1.00 m (outside the training range) | 5.19 | 4.04 |

- On REAL NAIP it is worse in every valid area (nDSM RMSE 2b -> 6b): Pittsburgh 3.05 -> 3.24,
  Carmel 2.82 -> 3.69, Champaign 0.37 -> 0.39, Del Rio 1.41 -> 1.60, Grayling 2.08 -> 2.45.
- Per LiDAR height band (pooled; RMSE / bias): 0-2 m 1.39/+0.38 -> 1.72/+0.46; 2-5 m 2.43/-0.04 ->
  3.38/+0.83; 5-10 m 3.33/-1.84 -> 3.88/-0.67; 10-20 m 6.83/-5.95 -> 6.34/-4.52; > 20 m 19.4/-18.1 ->
  18.3/-16.6. 6b does reduce the underestimation of tall objects, but it adds false height to low
  vegetation and small objects (suburban lawns/shrubs, forest understorey), which dominate the area.
- Interpretation: the simulation (area-averaging 0.33 m aerial imagery) does not reproduce real 0.6 m
  NAIP (different sensor, optics/MTF, sharpening, radiometry, season). The model learned "blurrier =
  coarser scale" on GAMUS, which does not transfer. This is a **sim-to-real gap**, and it is exactly what
  the real-data check (R3) was pre-registered to catch.

### Consequences
- Production model unchanged (2b); the "valid" resolution band is unchanged (0.33 m +-25%).
- Simulated-GSD val results alone must not be used to claim coarse-resolution performance.
- Possible next steps (not started; each needs its own plan and held-out areas): train or calibrate
  on REAL coarse imagery paired with LiDAR (e.g. NAIP + 3DEP from other areas than these 5), more
  realistic degradation (sensor blur/noise/colour models), or report 0.6 m results as "uncertain" (current
  behaviour).
