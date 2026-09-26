# Final test evaluation (one-time, FROZEN)

Run: `runs/20260926-141407_final_test/` (contains `FROZEN`). Approved 2026-09-26. Evaluated **once**;
nothing was fitted, tuned, re-trained or selected on these results, and no experiment follows from
them. `scripts/final_test_eval.py` refuses a second test run.

| | |
|---|---|
| Split | GAMUS **test**, all 2,861 tiles (DC 361, **NYC 1,000**, PHL 1,500); integrity check: 0 bad files |
| Methods (all frozen) | **B0-zero** (0 m) · **B1** (DA-V2-S r1022, pct normalisation, Phase 1 train-fitted linear map, `calibration.json` sha256 recorded) · **Phase 2b** (`best.pt` sha256 `e55ec4ca7bda…`) |
| Script verification | a dry run on **val** reproduced every published val number exactly (B0-zero 7.845, B1 6.602, 2b 3.148, per-city, bootstrap CIs) before test was touched |
| Metrics | same code as all phases: pooled over valid pixels (5.49 G), per class / true-height band / city; paired tile bootstrap (2,000 resamples, seed 0) |

## Results (test, pooled)

| method | RMSE | MAE | Pearson r | bias | RMSE building | RMSE tree |
|---|---|---|---|---|---|---|
| B0-zero | 8.724 | 4.423 | n/a | −4.411 | 13.40 | 13.15 |
| B1 (Phase 1) | 7.293 | 4.827 | 0.251 | +0.314 | 10.07 | 9.89 |
| **Phase 2b** | **4.687** | **2.012** | **0.793** | −0.763 | **7.43** | **6.10** |

**Paired tile bootstrap, Δ = x − y (95% CI):**
| comparison | ΔRMSE | ΔMAE |
|---|---|---|
| 2b − B1 | **−2.606** [−2.716, −2.497] | **−2.815** [−2.868, −2.760] |
| 2b − B0-zero | **−4.037** [−4.189, −3.886] | **−2.411** [−2.495, −2.327] |
| B1 − B0-zero | −1.431 [−1.502, −1.358] | +0.404 [+0.333, +0.474] |

On test, Phase 2b is significantly better than B1 and B0-zero on both RMSE and MAE (all upper bounds
< 0). As on val, B1 is **worse than predicting 0 m on MAE**. (The oracle per-image calibration
baseline wasn't part of the approved test set of methods.)

### By city (RMSE / MAE)
| city | B0-zero | B1 | Phase 2b |
|---|---|---|---|
| DC | 12.19 / 7.58 | 9.76 / 6.78 | **4.63 / 2.73** |
| **NYC** (first evaluation, never in val) | 8.91 / 5.14 | 7.21 / 5.55 | **4.61 / 2.55** |
| PHL | 7.55 / 3.21 | 6.64 / 3.89 | **4.75 / 1.49** |

### Phase 2b by true height band: test vs val (MAE / bias, m)
| band | test | val |
|---|---|---|
| 0–2 m | 0.71 / +0.52 | 0.53 / +0.39 |
| 2–5 m | 2.30 / +0.08 | 2.04 / +0.21 |
| 5–10 m | 2.19 / −1.03 | 2.24 / −0.91 |
| 10–20 m | 4.95 / −4.02 | 4.57 / −3.19 |
| > 20 m | **11.10 / −10.85** | 6.89 / −6.33 |

### Phase 2b by class: test vs val (RMSE, m)
ground 2.51 vs 1.38 · low vegetation 1.84 vs 1.63 · road 2.35 vs 2.57 · **building 7.43 vs 3.84** ·
tree 6.10 vs 4.88 · water 1.15 vs 0.74.

## Interpretation
1. **The Phase 2 result holds on unseen data.** 2b beats both deployable baselines on test by large,
   significant margins, and it beats B1 on 99.0% of test tiles (supplementary).
2. **NYC generalisation.** NYC was never used for model selection (no NYC val tiles). Its test result
   (RMSE 4.61 / MAE 2.55) is in line with DC (4.63 / 2.73).
3. **Test is harder than val because of Philadelphia's skyscrapers.** Test RMSE is 4.69 vs 3.15 on
   val, while MAE is only 2.01 vs 1.56. That pattern points to a few large errors:
   - The **worst 5% of test tiles contribute 51% of all squared error.**
   - The worst tiles are all a Center City Philadelphia cluster (PHL_3510, 3596, 3511, 3680, 3595,
     3679, …) with **LiDAR heights up to 305 m** and up to 21% of pixels above 100 m. The model
     underestimates them by 13–31 m on average (tile RMSE up to 70 m).
   - The tallest structures in training/val are about 67 m (PHL, sampled) and about 94 m (DC). These
     towers are **outside the height range the model learned**, and the regression-to-the-mean
     weakness seen on val (above 20 m) becomes severe there.
   - Typical tiles are fine: median test tile RMSE is 2.71 m (IQR 1.82–4.09), and PHL's median tile
     RMSE is 2.07 m.
4. **The known weakness is confirmed:** the > 20 m bias is −10.9 m on test (−6.3 m on val), and
   building RMSE is 7.4 m. Tall-structure heights are the main limitation to state in the presentation.

## Limitations
- One evaluation of one model on GAMUS test (US East Coast, leaf-off aerial imagery, 0.33 m). It says
  nothing about Indian imagery, other resolutions (see `docs/phase4_gsd_study.md`) or hilly and forest
  terrain (Phase 6).
- The breakdown of the worst tiles (LiDAR heights above) is descriptive only. The frozen metrics are
  unchanged, and no threshold or model was chosen from it.
