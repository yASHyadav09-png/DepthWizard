# Phase 2: supervised RGB → nDSM (approved plan + required modifications)

Approved 2026-09-25 with modifications 1–8 below. Phase 1 commit (source of all targets):
`ef540d07840ab8f011f182b6ea6e5187b08ea480`, recorded in every Phase 2 run (`meta.json → phase1_commit`).

## Scope and splits
- Train: all 5,004 tiles (`configs/subsets/full.yaml`). Val: all 859 tiles.
- **Test: not downloaded, inspected or evaluated in Phase 2.** Code refuses a subset containing `test`.
  Test evaluation happens only after explicit approval.
- 2a (frozen encoder) runs first. **2b is not started automatically.** Results are shown after 2a.
- The val set is used for checkpoint selection and for the single 2a-vs-2b decision, not for
  repeated hyperparameter tuning.

## Model
`depth-anything/Depth-Anything-V2-Small-hf` @ `5426e4f0f36572d16453bbda7a8389317b1bef99`, pretrained weights.

    h = s0 · DA-V2-S(x)        metres, ≥ 0 (the DPT head ends in ReLU)

- 2a trains the DPT neck (2,700,768) + head (27,745) = **2,728,513** parameters. The DINOv2-S
  encoder (**22,056,576**) is frozen and kept in eval mode.
- 2b (if approved later) additionally trains the last 4 of 12 encoder blocks plus the final
  backbone LayerNorm.

### Initial scale s0 (fixed, not trained)
- Fitted **once, before training, on training tiles only**. It is a registered buffer, never an
  optimiser parameter, and no calibration step follows training.
- Equation: `s0 = Σ d·h / Σ d²` over valid pixels (least squares through the origin), where
  `d` = raw DA-V2-S output. Data: one 518×518 crop (`train=False`, centre window) from each of
  50 train tiles drawn with seed 0.
- The value, equation, seed and tile ids are stored in `meta.json → init_scale`.

## Losses (all over valid pixels only; `V` = validity mask from `valid_mask`)

**L1:**  `L_1 = (1/|V|) Σ_{i∈V} |p_i − g_i|`, pooled over the batch.

**Gradient matching** on the residual `R = p − g`, K = 4 scales. At scale k the stride is
`s = 2^k`, on the grid of pixels whose row and column are multiples of s:

    horizontal pairs (i, j)–(i, j+s), vertical pairs (i, j)–(i+s, j)
    a pair counts only if BOTH pixels are valid:  m = v_a · v_b
    G_k = Σ m·|R_a − R_b| / Σ m                  (mean absolute residual difference, metres)
    L_grad = mean of G_k over scales with at least one valid pair

- A constant offset has zero gradient loss; only wrong relative structure (edges, slopes) is
  penalised.
- NoData boundaries create no gradient error, because pairs touching an invalid pixel are
  excluded (tested in `tests/test_losses.py`).
- Each scale is normalised by its own count of valid pairs, so `L_grad` is in metres, like `L_1`.

**Total:** `L = L_1 + λ·L_grad`, λ = 0.5.
L1, L_grad, λ·L_grad and the total are logged separately every 50 optimiser steps
(`train_log.csv`), along with the ratio `λ·L_grad / L_1`. At initialisation the ratio was 0.09,
so the gradient term does not dominate. If the ratio rose above 1, that would be flagged in the report.

## Data and optimisation
- Train input: random 518×518 **window read** at native GSD, random 90° rotation + flip, colour
  jitter (brightness/contrast/saturation ±10%, hue ±0.02 via a YIQ chroma rotation).
  No scale augmentation (that's Phase 4).
- Val input: the whole 1024 tile, reflect-padded to 1036 (a multiple of 14), cropped back. No resampling.
- AdamW, lr 1e-4, weight decay 0.01; 500 warm-up steps, then cosine to 0; gradient clip 1.0;
  bf16 autocast; 30 epochs (one random crop per train tile per epoch).
- Batch: 8. If 8 doesn't fit: 4 × accumulation 2, then 2 × accumulation 4. The architecture is
  never changed to fit memory.
- Validation every 3 epochs on all 859 tiles. `best.pt` = lowest pooled val RMSE.
- Seed 0, cuDNN deterministic, TF32 off. GPU training is not bit-reproducible (some CUDA
  backward kernels are non-deterministic). Evaluation of a saved checkpoint must be.

## Statistical comparisons
- `ΔRMSE = RMSE_model − RMSE_reference`, `ΔMAE = MAE_model − MAE_reference`, both pooled over
  valid val pixels.
- 95% CI from a **paired tile-level bootstrap**: resample the 859 val tiles with replacement
  (2,000 resamples, seed 0), recompute both pooled metrics on the same resample, and take the
  2.5 / 97.5 percentiles of the difference.
- **"Significantly better than the reference"** ⇔ the upper bound of the 95% CI of Δ is < 0.
- **2a vs 2b:** `ΔRMSE = RMSE_2b − RMSE_2a`. Keep 2b only if the upper 95% bound is < 0.

## Success criteria
Absolute targets and statistical claims are reported separately.

| | absolute target (point estimate) | statistical claim (paired bootstrap) |
|---|---|---|
| vs B1 (must) | RMSE < 6.60 m; building RMSE < 7.10 m | ΔRMSE(model − B1) upper 95% < 0 |
| vs B1 MAE | MAE < 4.66 m | ΔMAE(model − B1) upper 95% < 0 |
| vs B0-zero (must) | MAE < 3.94 m | ΔMAE(model − B0-zero) upper 95% < 0 |
| vs oracle per-image calibration baseline (aim) | RMSE < 4.77 m; MAE < 3.03 m | ΔRMSE and ΔMAE (model − oracle R518) upper 95% < 0 |

"Beats the oracle target" is claimed only when the statistical condition holds, not from the
point estimate alone.

Process checks: `best.pt` reproduces its val metrics exactly on re-evaluation; the test split
is untouched.
