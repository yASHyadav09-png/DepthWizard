# Phase 4: resolution (GSD) study of the Phase 2b model

Status: **Phase 4 CLOSED** (2026-09-26). Study complete (validation only); decisions in §Decision below.
No training was started; the trained 2b model is unchanged. The test split was not used by this study.

Run: `runs/20260926-124248_phase4_gsd_study/` (`summary.csv`, `metrics_val.json`, `per_tile_*.csv`,
`bootstrap_B_minus_A.json`, `figures/gsd_curves.png`). Script: `scripts/phase4_gsd_study.py`.

## Method
All 859 GAMUS val tiles, model = Phase 2b `best.pt`. For each target ground resolution g:

| step | how |
|---|---|
| simulated imagery | 0.33 m RGB area-averaged (PIL BOX) to `round(1024·0.33/g)` px: 676 (0.50 m), 512 (0.66 m), 338 (1.00 m), 169 (2.00 m) |
| reference | LiDAR nDSM area-averaged over **valid pixels only**; a coarse pixel is valid if ≥ 50% of its area is |
| classes | nearest-neighbour |
| **A direct** | the model runs on the coarse image as it is |
| **B resampled** | coarse image → bicubic upsample to 1024 px (≈ 0.33 m) → model → area-average back to the coarse grid |

Both strategies are scored on the coarse grid with the same metric code as Phases 1–2.
**Sanity anchor:** at 0.33 m, A = B = the Phase 2b val evaluation exactly (every tile ΔRMSE = 0.0).

## Results (val, pooled)

| GSD (m) | grid | A RMSE | A MAE | A r | A bias | B RMSE | B MAE | B r | B bias |
|---|---|---|---|---|---|---|---|---|---|
| 0.33 | 1024 | **3.148** | **1.556** | 0.889 | −0.42 | = A | = A | = A | = A |
| 0.50 | 676 | 3.413 | 1.710 | 0.884 | −0.92 | 3.429 | 1.702 | 0.873 | −0.67 |
| 0.66 | 512 | 3.869 | 1.978 | 0.860 | −1.26 | 4.229 | 2.044 | 0.817 | −1.28 |
| 1.00 | 338 | 5.189 | 2.659 | 0.749 | −2.12 | 6.610 | 3.064 | 0.459 | −2.79 |
| 2.00 | 169 | 7.455 | 3.809 | 0.301 | −3.74 | 7.709 | 3.899 | 0.062 | −3.89 |

**B − A, paired tile bootstrap** (859 tiles, 2,000 resamples; Δ = B − A):

| GSD | ΔRMSE [95% CI] | ΔMAE [95% CI] | verdict |
|---|---|---|---|
| 0.50 | +0.017 [−0.018, +0.054] | −0.008 [−0.024, +0.009] | no significant difference |
| 0.66 | +0.360 [+0.295, +0.425] | +0.066 [+0.034, +0.097] | **B significantly worse** |
| 1.00 | +1.421 [+1.229, +1.606] | +0.405 [+0.323, +0.492] | **B significantly worse** |
| 2.00 | +0.254 [+0.198, +0.314] | +0.091 [+0.067, +0.117] | **B significantly worse** |

### MAE by true height band (m)
| config | 0–2 m | 2–5 m | 5–10 m | 10–20 m | > 20 m |
|---|---|---|---|---|---|
| 0.33 | 0.53 | 2.04 | 2.24 | 4.57 | 6.89 |
| 0.50 A / B | 0.46 / 0.51 | 1.86 / 2.20 | 2.50 / 2.32 | 5.63 / 5.24 | 8.83 / 8.64 |
| 0.66 A / B | 0.48 / 0.43 | 1.87 / 2.20 | 2.99 / 2.71 | 6.73 / 7.21 | 10.84 / 12.49 |
| 1.00 A / B | 0.45 / 0.27 | 2.24 / 2.50 | 4.44 / 4.31 | 9.15 / 11.63 | 16.57 / 23.80 |
| 2.00 A / B | 0.25 / 0.22 | 3.31 / 3.44 | 7.01 / 7.05 | 13.29 / 13.73 | 26.57 / 27.90 |

### RMSE by class (m), strategy A
| GSD | ground | low veg. | road | building | tree | water |
|---|---|---|---|---|---|---|
| 0.33 | 1.38 | 1.63 | 2.57 | 3.84 | 4.88 | 0.74 |
| 0.50 | 1.28 | 1.35 | 2.70 | 3.83 | 5.69 | 0.72 |
| 0.66 | 1.29 | 1.27 | 3.00 | 4.49 | 6.49 | 0.73 |
| 1.00 | 1.32 | 1.15 | 3.67 | 6.00 | 9.00 | 0.78 |
| 2.00 | 0.92 | 0.94 | 4.14 | 10.29 | 12.44 | 0.94 |

(Strategy B per class: `metrics_val.json`.)

### By city, RMSE / MAE (m), A and B
| GSD | DC (A) | DC (B) | PHL (A) | PHL (B) |
|---|---|---|---|---|
| 0.33 | 4.30 / 2.48 | = A | 1.98 / 0.91 | = A |
| 0.50 | 4.76 / 2.84 | 4.71 / 2.72 | 1.99 / 0.92 | 2.11 / 0.99 |
| 0.66 | 5.50 / 3.39 | 6.07 / 3.52 | 2.06 / 0.99 | 2.15 / 1.01 |
| 1.00 | 7.49 / 4.65 | 9.86 / 5.80 | 2.56 / 1.26 | 2.48 / 1.14 |
| 2.00 | 10.80 / 6.60 | 11.25 / 6.86 | 3.57 / 1.86 | 3.52 / 1.83 |

NYC isn't included: the GAMUS val split has no NYC tiles.

## Interpretation
1. **Accuracy degrades steadily with coarser imagery.** RMSE rises by +8% at 0.50 m, +23% at 0.66 m,
   +65% at 1.00 m and +137% at 2.00 m (r falls to 0.30). Up to about 0.5 m the model is nearly
   unaffected; from 1 m on it degrades substantially.
2. **The loss is concentrated in tall objects, and the bias turns increasingly negative**
   (−0.4 m at 0.33 m → −3.7 m at 2 m). Above 20 m, MAE grows from 6.9 m to 26.6 m, while ground and
   low vegetation stay flat or even improve (the coarse reference is smoother). The model's height
   prior is tied to the object sizes it saw at 0.33 m: at coarser resolution, buildings and crowns
   span fewer pixels and are predicted lower.
3. **Resampling to the training resolution (B) is not a fix.** It is no better at 0.50 m and
   significantly worse at 0.66, 1.00 and 2.00 m. Bicubic upsampling creates blurred imagery unlike
   anything in training, and the model reads the lost detail as less height (tall objects especially:
   at 1 m, B's > 20 m MAE is 23.8 vs 16.6 m for A).
4. **DC degrades much more than PHL** (tall buildings and forest). PHL stays within 0.6 m RMSE of
   full resolution up to 1 m.

## Decision (Phase 4 closure, 2026-09-26)
| # | decision | status / enforcement |
|---|---|---|
| 1 | **Production strategy = direct inference (A).** User imagery is never resampled to 0.33 m/px. | Already the app behaviour. Locked by `backend/tests/test_api.py::test_phase4_decision_inference_runs_on_native_pixels`. |
| 2 | **The metric-validity threshold is NOT widened.** "valid" stays within ±25% of 0.33 m (≈ 0.25–0.41 m/px); 0.5 m stays "uncertain". This study shows a moderate degradation at 0.5 m (+8% RMSE, larger for tall objects) but doesn't show that ≤ 0.5 m meets a project accuracy criterion, so the label isn't relaxed. | Locked by `tests/test_inference.py::test_phase4_decision_validity_threshold_unchanged`. |
| 3 | **Scale-augmentation fine-tuning moves to Phase 6**, where it will be tested on real ≈ 0.6 m NAIP imagery instead of this simulated coarsening. | Not started. |
| 4 | **GCP correction stays in Phase 5** with georeferenced input handling. | Not started. |

The analysis below is kept as written; the "Implications" were proposals, and the table above records what
was adopted.

## Implications (proposals at the time of the study; see Decision above)
- **App policy that follows from the data:** run **direct inference (A)**, not resampling. Keep the
  "uncertain" label for resolutions coarser than about 0.5 m (the current ±25% band around 0.33 m
  labels only ≈ 0.25–0.41 m as "valid"; the data would support widening it to 0.5 m, if you agree).
- **Candidate remedy for ≥ 0.66 m imagery:** scale-augmentation fine-tuning (random down-sampling of
  training crops, starting from 2b). The study shows a real gap it could target (tall-object bias).
  Not started, as instructed.

## Limitations
- The coarsening is **simulated** (area-averaging of the same aerial image). Real coarser sensors
  also differ in optics (MTF), noise, compression, viewing geometry and acquisition date, so real
  imagery may perform worse than these numbers.
- Metrics at each GSD are on **that GSD's grid** against an equally coarsened reference. The coarser
  reference is smoother (extreme heights averaged out), so absolute RMSE values across GSDs aren't
  strictly like-for-like; the A-vs-B comparison at each GSD is exact (same tiles and grid).
- Only resolutions coarser than 0.33 m can be simulated from GAMUS. Finer imagery (e.g. 0.15 m) is
  untested.
