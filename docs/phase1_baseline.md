# Phase 1: Depth Anything V2 Small baselines on GAMUS

Status: **PASSED** (2026-09-25). Nothing was trained: the network is frozen, and only
calibration parameters were fitted, on train pixels only. The test split was never
downloaded or read.

Reproduce:
```bash
python scripts/phase1_baseline.py --config configs/phase1_baseline.yaml
python scripts/bootstrap_ci.py runs/<run> --methods B0_zero B0_mean B1_r1022_pct_lin B2_oracle_r1022 \
       --pairs B1_r1022_pct_lin:B0_mean B1_r1022_pct_lin:B1_r518_pct_lin
```
Official run: `runs/20260925-052422_phase1_baseline_repro/` (reproduction of `runs/20260925-045450_phase1_baseline/`, see §7).

## 1. Setup (as approved)

| item | value |
|---|---|
| Model | `depth-anything/Depth-Anything-V2-Small-hf` @ `5426e4f0f36572d16453bbda7a8389317b1bef99`, frozen, fp32, `eval()` |
| Preprocessing | the model's own `DPTImageProcessor` (`use_fast=False` pinned): bicubic resize to R×R (sides a multiple of 14), rescale to [0, 1], ImageNet mean/std |
| Output | relative inverse depth at R×R, bilinearly upsampled to 1024×1024 |
| Data | subset `phase1`: 200 train tiles (DC 58, NYC 47, PHL 95; seed 0, city-stratified); **all 859 val tiles** (DC 359, PHL 500; val has no NYC) |
| Fitting | 20,000 random valid pixels per train tile (4,000,000 in total), seed 0 |
| Evaluation | pooled over every valid val pixel at 1024², NoData via `valid_mask` (891.3M pixels) |
| Seed / determinism | seed 0; cuDNN deterministic, TF32 off; re-inference matches the cache exactly (max abs diff 0.0) |
| GPU | RTX 5050 Laptop: peak 385 MB (R518), 1,196 MB (R1022); 0.11 s and 0.46 s per tile |

## 2. How relative depth becomes nDSM (exact)

```
d  = DA-V2-S(rgb)                               # relative inverse depth, 1024×1024 after upsampling
d~ = (d - P2(d)) / (P98(d) - P2(d))             # "pct": per-image, uses d only, all pixels
h  = max(0, 6.4234 · d~ + 2.1303)   [metres]    # "lin": global a, b by least squares on TRAIN pixels
```
This is the selected B1 (`B1_r1022_pct_lin`, R = 1022). The alternatives in the grid were
`medmad` normalisation, d~ = (d − median) / mean|d − median|, and an isotonic (monotone)
global map instead of the line. All parameters are in `calibration.json`.

Why fit in inverse-depth space: from altitude H a point of height h is at range H − h, and
1/(H − h) ≈ (1/H)(1 + h/H), which is linear in h.

**Sign check** (train pixels): Pearson(raw d, GT nDSM) = **+0.21** (R518) / **+0.27** (R1022).
Positive ("closer" means taller), but weak.

## 3. Results (val, 859 tiles, pooled)

| method | deployable | RMSE (m) | MAE (m) | Pearson r | bias (m) | RMSE bldg | RMSE tree |
|---|---|---|---|---|---|---|---|
| B0-zero | ✔ | 7.845 | **3.942** | n/a | −3.942 | 10.58 | 13.26 |
| B0-mean (4.71 m) | ✔ | 6.826 | 5.103 | n/a | +0.768 | 7.56 | 10.24 |
| **B1 = `r1022_pct_lin`** | ✔ | **6.602** | 4.663 | 0.246 | +0.586 | 7.10 | 10.23 |
| B2 oracle per-image calibration baseline (R518) | ✘ | 4.766 | 3.026 | 0.712 | +0.020 | 4.94 | 7.05 |
| B2 oracle per-image calibration baseline (R1022) | ✘ | 4.830 | 3.101 | 0.702 | +0.019 | 5.33 | 6.94 |

All 8 B1 settings fall between 6.602 and 6.641 m RMSE (`figures/rmse_all_b1.png`).

**Tile-level bootstrap** (2000 resamples, 95% CI, `bootstrap_ci.txt`):
- B1 RMSE 6.602 [6.21, 7.02]
- B1 − B0-mean = **−0.224 m [−0.263, −0.187]**: a real improvement, but small (3%)
- B1 − B0-zero = −1.243 m [−1.36, −1.12]
- R1022 vs R518 (both pct_lin) = −0.021 m [−0.049, +0.005]: **not significant**
- pct vs medmad (R1022, lin) = −0.038 m [−0.076, −0.004]; lin vs iso = −0.004 m: negligible

### By true height band (MAE / bias, m)
| band | pixel share | B0-zero | B0-mean | B1 | B2 oracle R1022 |
|---|---|---|---|---|---|
| 0–2 m | 62% | 0.17 / −0.17 | 4.54 / +4.54 | 3.91 / **+3.91** | 2.12 / +2.08 |
| 2–5 m | 9.7% | 3.48 | 1.26 | 1.78 | 2.20 |
| 5–10 m | 15.5% | 7.30 | 2.59 | 2.34 | 3.10 |
| 10–20 m | 8.6% | 13.88 | 9.17 | 8.73 / **−8.73** | 6.98 |
| > 20 m | 4.2% | 27.93 | 23.22 | 22.67 / **−22.67** | 11.72 |

### By city (RMSE / MAE / r)
| city | B0-zero | B0-mean | B1 | B2 oracle R1022 |
|---|---|---|---|---|
| DC | 11.35 / 6.87 / – | 9.29 / 6.65 / – | 9.10 / 6.31 / 0.23 | 6.87 / 5.14 / 0.65 |
| PHL | **3.80** / 1.89 / – | 4.33 / 4.02 / – | 4.01 / 3.51 / 0.40 | 2.58 / 1.67 / 0.62 |

## 4. Interpretation

1. **Pretrained relative depth + global calibration barely beats a constant.** B1 improves
   RMSE over "predict the train mean" by only 0.22 m, and it **loses to "predict 0 m" on MAE**
   (4.66 vs 3.94) and on PHL RMSE (4.01 vs 3.80). 62% of valid pixels are within 2 m of the ground,
   so predicting ground everywhere is a strong MAE baseline.
2. **B1 compresses everything into a 2–10 m band** (`figures/scatter_val.png`). The ground is
   overestimated (+3.9 m) and anything above 20 m is underestimated by about 23 m. The fitted
   slope is shallow (6.4 m per unit of d~) because a single global line is a compromise between
   tiles whose depth scales disagree. This is regression toward the mean.
3. **The model does see structure, but its scale changes from tile to tile.**
   - With the per-image oracle scale, RMSE falls to 4.8 m. The pooled r of 0.70 is inflated by the
     oracle's per-tile offset; the **median per-tile r is only 0.37–0.42**.
   - The oracle slope varies about 25× between tiles (10th–90th percentile 0.24–6.3), and
     **55 of 859 tiles (6%) have a negative slope**, i.e. the depth ordering is inverted there.
   - No per-image normalisation computed from d alone can recover this. That is the core
     limitation of zero-shot relative depth on nadir imagery.
4. **A large low-frequency "tilt" dominates the raw output** (visible in every panel). The model
   treats the tile as a slanted scene seen at an angle (its training prior from ground-level
   photos). Buildings and trees are small bumps on top of this gradient, which a monotone map
   of d cannot separate.
5. **Resolution and normalisation hardly matter.** R1022 vs R518 is within noise, and the oracle
   is slightly *better* at R518. Feeding the model near-native resolution doesn't fix a domain
   problem. Linear and isotonic maps are the same (the relationship isn't non-linear; it just
   isn't consistent across tiles).
6. **The best case is large flat roofs** (PHL_6183, RMSE 2.0 m): smooth, well-separated surfaces
   where the relative ordering is right. The worst cases are tall DC buildings and dense trees.
7. The selection optimism from picking 1 of 8 settings on val is at most about 0.04 m (the
   whole B1 spread), which is negligible for any conclusion here.

**Conclusion:** frozen DA-V2-S features are not a usable nDSM estimator on their own. The
problem is not the relative geometry inside a tile; it's the missing, inconsistent metric scale.
That is exactly what supervised RGB → nDSM training (Phase 2) is meant to learn.

## 5. Phase 2 targets (fixed now, before any training)

Phase 2 is judged on the same 859 val tiles with the same metrics:
- **Must beat B1:** val RMSE < **6.60 m**, with the bootstrap CI of the difference excluding 0,
  and building RMSE < **7.10 m**.
- **Must beat B0-zero on MAE** (< **3.94 m**). B1 fails this.
- **Should beat the oracle per-image calibration baseline**: RMSE < **4.77 m**, MAE < **3.03 m**.
  A trained model gets no ground truth at inference, so beating the oracle shows it learned
  metric scale, not just relative shape.
- Report the > 20 m band and per-city results (DC vs PHL) as well. These are where B1 fails.

## 6. Limitations
- Val has no NYC tiles. NYC is covered only through training data and, later, the single test evaluation.
- B0-mean uses the mean of the 4M sampled train pixels (4.71 m), not every train pixel.
- The oracle pooled r is not comparable with the B1 r (see 4.3); the per-tile r is the fair
  measure of the raw geometry.
- These runs recorded `git dirty = true`: `configs/subsets/phase1.yaml` was created after the
  last commit. The exact tile list is stored in each run's `split_manifest.txt`.

## 7. Reproducibility check
The full pipeline was run twice with the same config. All **3,858** reported values (every metric, breakdown and fitted calibration parameter, including the isotonic curves) are **bit-identical** between `runs/20260925-045450_phase1_baseline` and `runs/20260925-052422_phase1_baseline_repro`. The only change is B0-mean's Pearson r, which the first run reported as floating-point noise (±1e-9) and the second as undefined, after a metrics fix between the runs (a constant prediction has no correlation). `runs/results.csv` has rows for both runs; **the official rows are `20260925-052422_phase1_baseline_repro`**.

## Figures (`runs/20260925-052422_phase1_baseline_repro/figures/`)
`scatter_val.png` · `mae_per_height_bin.png` · `rmse_per_class.png` · `rmse_all_b1.png` ·
`calib_curve_r{518,1022}_{pct,medmad}.png` · `panel_*.png` (best, median, worst, tallest,
per-city median, and two others)
