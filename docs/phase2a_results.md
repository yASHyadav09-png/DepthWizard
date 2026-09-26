# Phase 2a results: DA-V2-S, frozen encoder, trained DPT decoder → metric nDSM

Status: **complete, validation only.** The test split was not downloaded, inspected or evaluated.
2b has **not** been started.

| | |
|---|---|
| Run | `runs/20260925-163139_phase2a_frozen/` |
| Code commit at launch | `52eccb3` (tree clean at launch; see §6.2 about the `dirty` flag in `meta.json`) |
| Loss code used in training | **the original loss, WITHOUT the NaN-target sanitisation.** The NaN issue was discovered after this run finished and was fixed afterwards (§6.1). The run and its artifacts are left exactly as produced. |
| Phase 1 commit | `ef540d07840ab8f011f182b6ea6e5187b08ea480` |
| Config | `configs/phase2a_frozen.yaml` (as approved, unchanged) |
| Trainable / frozen parameters | 2,728,513 / 22,056,576 |
| Initial scale | s0 = 3.659840 (fixed buffer, train tiles only) |
| Batch | 8 × 1 (no fallback needed); peak 2.6 GB allocated / 4.2 GB reserved |
| Wall time | 16:31 → 21:12 = **4 h 41 min** (30 epochs + 10 val passes) |
| Selected checkpoint | `best.pt` = epoch 27 of 30 (index 26, step 16,875), lowest val RMSE |

Reproduce the analysis:
`python scripts/eval_ndsm.py --run runs/20260925-163139_phase2a_frozen --phase1 runs/20260925-052422_phase1_baseline_repro --name phase2a`

## 1. Headline (val, 859 tiles, pooled over 891.3M valid pixels)

| method | deployable | RMSE | MAE | Pearson r | bias | RMSE bldg | RMSE tree |
|---|---|---|---|---|---|---|---|
| B0-zero | ✔ | 7.845 | 3.942 | n/a | −3.942 | 10.58 | 13.26 |
| B1 (Phase 1) | ✔ | 6.602 | 4.663 | 0.246 | +0.586 | 7.10 | 10.23 |
| Oracle per-image calibration baseline (R518) | ✘ | 4.766 | 3.026 | 0.712 | +0.020 | 4.94 | 7.05 |
| **Phase 2a** | ✔ | **3.548** | **1.729** | **0.861** | −0.623 | **4.61** | **5.51** |

## 2. Success criteria (pre-registered in `docs/phase2_plan.md`)

Δ = metric(2a) − metric(reference). Paired tile-level bootstrap, 859 tiles, 2,000 resamples, seed 0
(`paired_bootstrap.json`).

| criterion | absolute target | result | statistical claim | Δ [95% CI] | verdict |
|---|---|---|---|---|---|
| vs B1, RMSE (must) | < 6.60 | 3.548 ✅ | upper bound < 0 | −3.054 [−3.303, −2.794] | ✅ significantly better |
| vs B1, building RMSE (must) | < 7.10 | 4.613 ✅ | (no per-class test defined) | – | absolute target met |
| vs B1, MAE | < 4.66 | 1.729 ✅ | upper bound < 0 | −2.933 [−3.046, −2.815] | ✅ significantly better |
| vs B0-zero, MAE (must) | < 3.94 | 1.729 ✅ | upper bound < 0 | −2.213 [−2.394, −2.036] | ✅ significantly better |
| vs oracle R518, RMSE (aim) | < 4.77 | 3.548 ✅ | upper bound < 0 | −1.218 [−1.344, −1.091] | ✅ significantly better |
| vs oracle R518, MAE (aim) | < 3.03 | 1.729 ✅ | upper bound < 0 | −1.297 [−1.374, −1.219] | ✅ significantly better |

Also significantly better than B0-zero on RMSE (−4.30), B0-mean (RMSE −3.28, MAE −3.37) and the
oracle at R1022 (RMSE −1.28, MAE −1.37). All CI upper bounds are < 0.

**Every "must" and every "aim" target is met, and each statistical claim holds.** On the reported
validation metrics, Phase 2a outperformed the Phase 1 oracle per-image calibration baseline. That
baseline is a diagnostic reference (relative depth plus a per-tile affine fit to ground truth),
**not a theoretical upper bound**, so outperforming it is not a claim about any limit.
The building-class RMSE (4.61) is also below the oracle baseline's (4.94) as a point estimate. No per-class
significance test was defined, so no statistical claim is made for it.

Process checks: `best.pt` was re-evaluated twice, **bit-identical**, and **identical to the
in-training evaluation** (`eval_reproducibility.json`). Test split untouched.

## 3. Where the errors are

### By true height band
| band | pixel share | 2a MAE / bias | B1 MAE | oracle MAE |
|---|---|---|---|---|
| 0–2 m | 62% | **0.54** / +0.40 | 3.91 | 2.05 |
| 2–5 m | 9.7% | 2.15 / +0.13 | 1.78 | 2.20 |
| 5–10 m | 15.5% | 2.38 / −1.02 | 2.34 | 2.96 |
| 10–20 m | 8.6% | 4.97 / **−4.01** | 8.73 | 6.93 |
| > 20 m | 4.2% | 9.31 / **−9.04** | 22.67 | 11.58 |

### By class (RMSE / MAE / bias)
| class | 2a | B1 RMSE | oracle RMSE |
|---|---|---|---|
| ground | 1.39 / 0.38 / +0.12 | 4.06 | 3.20 |
| low vegetation | 1.54 / 0.50 / +0.26 | 4.39 | 2.79 |
| road | 2.80 / 1.13 / −0.21 | 4.80 | 4.80 |
| building | 4.61 / 2.74 / −0.84 | 7.10 | 4.94 |
| tree | **5.51 / 3.99 / −2.40** | 10.23 | 7.05 |
| water | 0.75 / 0.08 / −0.03 | 4.88 | 1.05 |

### By city
| city | 2a RMSE / MAE | B1 RMSE / MAE |
|---|---|---|
| DC | 4.96 / 2.85 | 9.10 / 6.31 |
| PHL | 2.05 / 0.94 | 4.01 / 3.51 |

Per tile (supplementary; the headline is the pooled paired-bootstrap result in §2): 2a has lower
RMSE than B1 on 98.8% of tiles and lower than B0-zero on 98.6%. The median per-tile RMSE is 2.42 m (IQR 1.59–3.74). **The 5 worst tiles are all DC**, dense tall
leaf-off forest (e.g. DC_18_24: 12.6 m RMSE, vs 27.7 m for B1).

## 4. Interpretation

1. **The core Phase 1 problem is solved.** The frozen DA-V2-S features *do* contain metric
   height information. A trained DPT decoder reads it out: RMSE halves versus B1, MAE falls by
   63%, and r rises from 0.25 to 0.86. The 2–10 m squeeze is gone (`figures/scatter_val.png`).
2. **On the reported validation metrics, 2a outperformed the Phase 1 oracle per-image
   calibration baseline, while using no ground truth at inference.** This suggests supervision
   gave it more than a better per-image scale: it predicts height structure that an affine
   rescaling of the relative output can't express (e.g. ground MAE 0.54 m vs the oracle
   baseline's 2.05 m). The oracle baseline is a diagnostic reference, not a theoretical upper
   bound.
3. **The remaining error is systematic underestimation of tall objects:** −4 m bias at 10–20 m,
   −9 m above 20 m, and trees at −2.4 m. This is regression toward the mean. It's stronger for
   tall leaf-off DC forest, where bare branches give little texture cue about canopy height.
4. **2a has saturated.** Val RMSE was flat from epoch 18 (3.573) to epoch 30 (3.562), and the
   best (3.548) is within that band.
5. *(Superseded by the 2b result: see `docs/phase2b_results.md` §5.3. L_grad also stayed flat
   in 2b with 4 encoder blocks trainable, so the "frozen encoder limits fine structure"
   explanation below isn't supported.)* **The gradient-matching loss barely moved** (L_grad ≈ 1.0 m at the start and 0.95 m at the
   end), while L1 fell from 4.8 to 1.8 m. The decoder learned the metric level but not finer
   relative structure (edges, crown shapes). That's consistent with the frozen encoder
   limiting spatial detail. `0.5·L_grad / L1` rose from 0.10 to at most 0.30 only because L1
   fell; the gradient term never dominated.
6. Selection optimism: `best.pt` is the best of 10 val evaluations, but 8 of them lie between
   3.55 and 3.61 m, so the optimism is at most about 0.02 m and doesn't affect any conclusion.

## 5. Limitations
- **Val has no NYC tiles.** NYC performance is unknown until the (approved) test evaluation,
  where 1,000 of 2,861 test tiles are NYC.
- Tall-object underestimation (§4.3) matters for building-height analysis in the viewer.
- GPU training isn't bit-reproducible (non-deterministic CUDA backward kernels). A retrained
  2a would give close but not identical numbers. The evaluation of `best.pt` is bit-reproducible.

## 6. Issues found during Phase 2a (both fixed, neither affects the results)

### 6.1 NaN values in GAMUS nDSM (loss logging only)
- A full scan found **14 PHL tiles** (7 train, 7 val) with **24,235 NaN nDSM pixels**. The
  Phase 0 survey sampled only 50 tiles and missed them.
- The validity mask already excluded them (`isfinite`), so **no metric in Phase 1 or Phase 2
  was affected** (metrics use `gt[valid]`).
- In the loss, masking is by multiplication and NaN × 0 = NaN, so the loss **value** was NaN for
  batches that contained such pixels. 63 of 375 logged intervals are NaN (gaps in
  `figures/training_curves.png`).
- **The gradient was unaffected:** at a masked pixel it is exactly 0.
  `tests/test_losses.py::test_nan_target_at_invalid_pixel_gives_finite_loss_and_same_gradient`
  proves the gradients are bit-identical with and without the NaNs. A genuinely NaN gradient
  would have made every weight NaN after one AdamW step; `last.pt` weights and optimiser state
  are all finite.
- Fix for future runs: `losses.sanitize` zeroes targets at invalid pixels before the loss.

### 6.2 `meta.json` reports `git.dirty = true`
The launch check (`train_ndsm.py`) found the tree clean at `52eccb3`; the script refuses to
start otherwise. `utils/runs.new_run` read git status *after* creating the new, untracked run
folder, so it always saw "dirty". This also affected earlier runs. Fixed: git status is now
read before the folder is created.

## 7. Figures (`runs/20260925-163139_phase2a_frozen/figures/`)
`training_curves.png` · `scatter_val.png` (B1 vs 2a) · `rmse_per_class.png` ·
`mae_per_height_bin.png` · `panel_*.png` (the same 8 tiles as Phase 1: RGB | GT | raw DA |
B1 | 2a | error)
