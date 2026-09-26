# Phase 2b results: DA-V2-S, last 4 encoder blocks + decoder fine-tuned

Status: **complete, validation only.** The test split was not downloaded, inspected or evaluated.
**Decision (pre-registered rule): keep 2b** (§2).

| | |
|---|---|
| Run | `runs/20260926-003630_phase2b_partial/` |
| Code commit at launch | `69e09e0` (**clean**: `meta.json → git.dirty = false`) |
| Phase 1 commit / 2a commit | `ef540d0` / `e0e18bd` |
| Config | `configs/phase2b_partial.yaml`, the approved design, unchanged |
| Trainable / frozen parameters | 9,829,441 (decoder 2,728,513 + encoder blocks 8–11 7,100,928) / 14,955,648 |
| Initialisation | 2a `best.pt` (epoch 27, sha256 `c0baee4003fc…`), with its fixed s0 = 3.659840 |
| Optimiser | AdamW wd 0.01; decoder LR 5e-5, encoder LR 5e-6; 500 warm-up + cosine; bf16; clip 1.0 |
| Loss | L1 + 0.5·L_grad, **NaN-safe version** (`losses.sanitize`); 0 NaN rows in `train_log.csv` |
| Batch | 8 × 1 (no fallback); peak 3.42 GB allocated / 4.91 GB reserved |
| Wall time | 00:36 → 04:05 = **3 h 28 min** (20 epochs + 7 val passes) |
| Selected checkpoint | `best.pt` = **epoch 20 of 20** (index 19, step 12,500), lowest val RMSE |

Reproduce the analysis:
`python scripts/eval_ndsm.py --run runs/20260926-003630_phase2b_partial --phase1 runs/20260925-052422_phase1_baseline_repro --name phase2b --compare-run runs/20260925-163139_phase2a_frozen --compare-name phase2a`

## 1. Headline (val, the same 859 tiles and 891.3M valid pixels as 2a)

| method | deployable | RMSE | MAE | Pearson r | bias | RMSE bldg | RMSE tree |
|---|---|---|---|---|---|---|---|
| B0-zero | ✔ | 7.845 | 3.942 | n/a | −3.942 | 10.58 | 13.26 |
| B1 (Phase 1) | ✔ | 6.602 | 4.663 | 0.246 | +0.586 | 7.10 | 10.23 |
| Oracle per-image calibration baseline (R518) | ✘ | 4.766 | 3.026 | 0.712 | +0.020 | 4.94 | 7.05 |
| Phase 2a | ✔ | 3.548 | 1.729 | 0.861 | −0.623 | 4.61 | 5.51 |
| **Phase 2b** | ✔ | **3.148** | **1.556** | **0.889** | **−0.417** | **3.84** | **4.88** |

## 2. 2a vs 2b decision (rule fixed before training)

ΔRMSE = RMSE_2b − RMSE_2a. Paired tile-level bootstrap on the same 859 tiles, 2,000 resamples, seed 0.

| | Δ | 95% CI | upper bound < 0? |
|---|---|---|---|
| **ΔRMSE (decision metric)** | **−0.400 m** | **[−0.480, −0.326]** | **yes → keep 2b** |
| ΔMAE (reported) | −0.174 m | [−0.203, −0.148] | yes |

2b was not tuned after seeing validation results. The config is exactly the approved one, and
the checkpoint was chosen by the same rule as 2a (lowest val RMSE).

## 3. Statistical comparisons against the Phase 1 references (for context)

Δ = metric(2b) − metric(reference); a claim is made only when the upper 95% bound is < 0 (all are).

| reference | ΔRMSE [95% CI] | ΔMAE [95% CI] |
|---|---|---|
| B1 | −3.455 [−3.761, −3.152] | −3.107 [−3.241, −2.975] |
| B0-zero | −4.698 [−5.098, −4.307] | −2.386 [−2.588, −2.193] |
| B0-mean | −3.679 [−3.984, −3.386] | −3.547 [−3.676, −3.423] |
| Oracle per-image calibration baseline R518 | −1.618 [−1.734, −1.507] | −1.471 [−1.555, −1.388] |
| Oracle per-image calibration baseline R1022 | −1.683 [−1.795, −1.566] | −1.545 [−1.628, −1.462] |

On the reported validation metrics, 2b outperformed the Phase 1 oracle per-image calibration
baseline. That baseline is a diagnostic reference, not a theoretical upper bound.

Absolute targets: RMSE 3.148 < 6.60 ✔, building RMSE 3.84 < 7.10 ✔, MAE 1.556 < 3.94 ✔, and the aim
targets RMSE < 4.77 ✔ and MAE < 3.03 ✔.

Reproducibility: `best.pt` was re-evaluated twice, **bit-identical**, and **identical to the
in-training evaluation** (`eval_reproducibility.json`).

## 4. Breakdowns (2b vs 2a)

### By class: RMSE / MAE / bias (m)
| class | pixels | 2b | 2a | B1 RMSE | oracle RMSE |
|---|---|---|---|---|---|
| ground | 167.4M | 1.38 / 0.36 / +0.11 | 1.39 / 0.38 / +0.12 | 4.06 | 3.20 |
| low vegetation | 213.4M | 1.63 / 0.52 / +0.29 | **1.54** / 0.50 / +0.26 | 4.39 | 2.79 |
| road | 145.0M | **2.57** / 1.00 / −0.19 | 2.80 / 1.13 / −0.21 | 4.80 | 4.80 |
| **building** | 153.8M | **3.84 / 2.37 / −0.68** | 4.61 / 2.74 / −0.84 | 7.10 | 4.94 |
| **tree** | 196.4M | **4.88 / 3.57 / −1.62** | 5.51 / 3.99 / −2.40 | 10.23 | 7.05 |
| water | 13.9M | 0.74 / 0.08 / −0.02 | 0.75 / 0.08 / −0.03 | 4.88 | 1.05 |
| others | 1.5M | 4.80 / 1.76 / +0.01 | **3.76** / 1.60 / −0.15 | 4.48 | 3.28 |

2b is worse than 2a on **low vegetation** (+0.09 m RMSE) and **others** (+1.04 m, but only 1.5M
pixels). No per-class significance test was defined, so these are point estimates.

### By true height band: MAE / bias (m)
| band | 2b | 2a | B1 MAE | oracle MAE |
|---|---|---|---|---|
| 0–2 m | 0.53 / +0.39 | 0.54 / +0.40 | 3.91 | 2.05 |
| 2–5 m | 2.04 / +0.21 | 2.15 / +0.13 | 1.78 | 2.20 |
| 5–10 m | 2.24 / −0.91 | 2.38 / −1.02 | 2.34 | 2.96 |
| 10–20 m | 4.57 / **−3.19** | 4.97 / −4.01 | 8.73 | 6.93 |
| > 20 m | 6.89 / **−6.33** | 9.31 / −9.04 | 22.67 | 11.58 |

### By city
| city | 2b RMSE / MAE / r | 2a RMSE / MAE / r |
|---|---|---|
| DC | 4.30 / 2.48 / 0.885 | 4.96 / 2.85 / 0.852 |
| PHL | 1.98 / 0.91 / 0.802 | 2.05 / 0.94 / 0.786 |
| NYC | **not evaluated**: the GAMUS val split has no NYC tiles | – |

### Per tile (supplementary; `per_tile_val_final.csv`)
2b has lower RMSE than 2a on 89.9% of tiles (median ΔRMSE −0.10 m). The median tile RMSE is 2.29 m
(IQR 1.50–3.48). The worst tiles are still all DC: DC_50_31 13.05 m, DC_40_29 10.08, DC_42_29 9.77,
DC_18_24 8.96, DC_15_23 8.06. The largest regression against 2a is **DC_50_31 (+2.10 m)**; I haven't
inspected it yet.

## 5. Interpretation

1. **Fine-tuning the last 4 encoder blocks helped, and the gain is significant:** −0.40 m RMSE and
   −0.17 m MAE against 2a. The gain is concentrated in the tall classes: building RMSE −0.77 m,
   tree −0.63 m, and above 20 m, MAE drops from 9.3 to 6.9 m. Ground-level accuracy is unchanged.
2. **Tall objects are still underestimated** (−3.2 m bias at 10–20 m, −6.3 m above 20 m). The
   regression toward the mean is reduced, not removed.
3. **Correction to the 2a interpretation.** In `phase2a_results.md` §4.5 I suggested the flat
   gradient loss meant the frozen encoder limited fine spatial structure. **2b doesn't support
   that:** with 4 encoder blocks trainable, L_grad stayed flat too (0.97 → 0.99 m) while L1 fell
   (1.75 → 1.64 m). The gain came from better height *levels* of tall objects, not sharper edges.
   A plausible but **untested** explanation is that L_grad is dominated by detail finer than the
   model's 14-pixel (about 4.6 m) patch grid, plus LiDAR and relief-displacement noise that the
   RGB can't predict. The panels agree: 2b's canopy is at a better level, but individual crowns
   are smoothed.
4. **2b is close to converged.** Val RMSE changed by only 0.01 m over the last 4 evaluations
   (3.158 → 3.148). The best checkpoint is the final epoch, so a longer run might gain a little
   more. It was not run, because the approved design was 20 epochs.
5. `0.5·L_grad / L1` peaked at 0.31, so the gradient term never dominated.

## 6. Limitations
- **NYC is untested** (not in val); DC forest remains the weakest area.
- Selection optimism: `best.pt` is the best of 7 val evaluations, and the last four lie within
  0.03 m of each other, so the optimism is small. 2b also started from 2a's val-selected
  checkpoint. Both stages used val for checkpoint selection only.
- Training isn't bit-reproducible (memory-efficient attention backward is non-deterministic;
  PyTorch warned about this). Checkpoint evaluation is bit-reproducible.
- The training-code difference between the stages (2b has the NaN-safe loss, 2a doesn't) doesn't
  change gradients (proved by test), so it doesn't confound the 2a-vs-2b comparison.

## 7. Figures (`runs/20260926-003630_phase2b_partial/figures/`)
`training_curves.png` · `scatter_val.png` (B1 | 2a | 2b) · `rmse_per_class.png` ·
`mae_per_height_bin.png` · `panel_*.png` (the same 8 tiles as Phases 1/2a: RGB | GT | raw DA | B1 | 2a |
2b | 2b error)
