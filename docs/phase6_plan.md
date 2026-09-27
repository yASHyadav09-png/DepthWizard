# Phase 6 plan: out-of-domain validation + scale-augmentation fine-tune

Written 2026-09-27 BEFORE any Phase 6 model output was produced (no AOI had been evaluated with any
model except Pittsburgh with Phase 2b in Phase 5). Anything added after results are seen is marked
as an amendment with its date and reason.

Standing decisions: ground-filter default stays **150 m** (user, 2026-09-27); GAMUS test split is not
touched; the 2b checkpoint is not modified.

## 6a: out-of-domain evaluation of the production model (Phase 2b)

- Areas: `configs/phase6_aois.yaml`: 6 areas of 1.3 x 1.3 km with NAIP 0.6 m and USGS 3DEP LiDAR 2 m:
  hilly urban (Pittsburgh), dense urban (Indianapolis), suburban (Carmel IN), farmland (Champaign IL),
  arid town (Del Rio TX), forest (Grayling MI). Selection rules, the two dropped areas (Tucson: no LiDAR
  coverage; Mendocino: no DTM) and their replacements (data availability only, before any model run)
  are in the config.
- Script: `scripts/phase6_eval_aois.py` (reference nDSM = LiDAR DSM - DTM, comparison on the 2 m grid,
  LiDAR NAVD88 -> EGM2008). Reported per area and pooled: nDSM (overall, height bands, objects > 2 m),
  DTM/DSM vs raw GLO-30 baselines, alignment; DIAGNOSTIC only: filter windows 0/90/150/300 m.
- Known confounders, reported not corrected: 0-3 year gap between NAIP and LiDAR (new or demolished
  buildings, tree growth), NAIP registration (Phase 5: ~4 m), leaf-on vs leaf-off, relief displacement.

## 6b: scale-augmentation fine-tune

Why: Phase 4 showed direct inference degrades with coarser imagery (val RMSE 3.15 m at 0.33 m, 3.87 at
0.66 m, 5.19 at 1 m), and most available aerial imagery (NAIP, many Indian sources) is 0.5-1 m.

- Config `configs/phase6_scaleaug.yaml`: continue from the 2b best checkpoint (same trainable blocks,
  same s0), 10 epochs, LR decoder 2.5e-5 / encoder 2.5e-6 (half of 2b), warm-up 300, cosine.
- Augmentation: with p = 0.5 a training crop shows the scene at 0.33 x s m/px, s ~ U(1.0, 1.95), i.e.
  0.33-0.64 m/px (a round(518 s) px window coarsened to 518 px with the Phase 4 simulation:
  RGB area average, nDSM valid-weighted area average with >= 50% valid, classes nearest;
  `depthwizard/data/coarsen.py`, tested equal to the Phase 4 functions). Flips, rotations, colour
  jitter as in 2b.
- Checkpoint selection (GAMUS val only): lowest mean of val RMSE at 0.33 m and at simulated 0.66 m,
  evaluated every 2 epochs.
- Speed gate (done, `runs/20260927-125854_phase6_scaleaug_speedtest`): 8.7 samples/s, 5.6 GB VRAM,
  data wait < 1%, estimated 1.9 h + the 0.66 m val passes.

### Decision rule (pre-registered)
Compared with 2b; paired bootstrap, 2,000 resamples, seed 0; Delta = 6b - 2b.

| | Data | Unit resampled | Criterion |
|---|---|---|---|
| R1 | GAMUS val, simulated 0.66 m (Phase 4 protocol, direct inference) | tile | Delta RMSE upper 95% bound < 0 |
| R2 | GAMUS val, native 0.33 m | tile | Delta RMSE upper 95% bound < +0.10 m (non-inferiority) |
| R3 | 6 NAIP areas, pooled nDSM vs LiDAR (2 m grid) | 128 m block | Delta RMSE upper 95% bound < 0 |

6b is **recommended** as the new production model only if R1, R2 and R3 all hold. The switch itself
needs the user's approval. If any fails, 2b stays; the result is reported either way.
Also reported (not part of the rule): per area, per height band, objects > 2 m, MAE, bias, and val at
0.5 and 1.0 m. Each model is evaluated on the NAIP areas exactly once for this decision; the areas are
never used for training, selection or thresholds.

Not decided by this rule: whether the "valid" resolution band (0.33 m +-25%) should widen. That would be a
separate proposal to the user, backed by the numbers.

## Amendments
- 2026-09-27, after the 6a evaluation of 2b, before any 6b model existed: Indianapolis' LiDAR DTM contains
  buildings (docs/phase6_results.md), so it is excluded from pooled nDSM/DTM metrics and from the R3
  blocks. R3 therefore uses the 5 remaining areas. R3 counts only complete 128 m blocks (partial edge
  blocks dropped), so its pooled RMSE (2b: 2.14 m) differs slightly from the pixel-pooled 2.17 m.
