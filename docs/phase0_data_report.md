# Phase 0 — GAMUS data report

Status: **PASSED** (2026-09-25). Everything below was checked against the downloaded
files. Nothing here is taken from the paper without verification.

Reproduce:
```bash
python -m depthwizard.data.download --name phase0 --train 40 --val 10 --seed 0
python scripts/inspect_gamus.py --subset phase0
python scripts/check_dataloader.py --subset phase0
python -m pytest -q
```
Outputs: `runs/phase0_inspection/` (`survey.json`, `tiles.csv`, `figures/`).

## 1. What is actually on Hugging Face (`earthflow/GAMUS`)

The current release differs from the arXiv paper (11,507 tiles, 5 cities). OMA and
JAX were removed.

| split | tiles | DC | NYC | PHL |
|---|---|---|---|---|
| train | 5,004 | 1,439 | 1,167 | 2,398 |
| val | 859 | 359 | **0** | 500 |
| test | 2,861 | 361 | 1,000 | 1,500 |

All 8,724 tiles have complete RGB + height + class triplets.

**Limitation:** val has no NYC tiles, so checkpoint selection never sees NYC.
The NYC test numbers therefore also measure cross-city generalisation.

## 2. File format

One HDF5 file per tile and modality. Each holds a single dataset `image` with **no
attributes**.

| modality | path | dtype | shape | notes |
|---|---|---|---|---|
| RGB | `images/<split>/<id>_RGB.h5` (DC, PHL) / `<id>_IMG.h5` (**NYC**) | uint8 | 1024×1024×3 | suffix differs by city |
| nDSM | `heights/<split>/<id>_AGL.h5` | float32 | 1024×1024 | metres above ground |
| classes | `classes/<split>/<id>_CLS.h5` | float32 (DC) / uint8 (NYC, PHL) | 1024×1024 | all integer-valued, 0–6 |

**The tiles carry no georeferencing** (no CRS, no transform, no GSD in the files).
GAMUS can't be used to test the GeoTIFF / absolute-DSM path. That needs other data
(Phase 6). The documented GSD of 0.33 m is used only for GAMUS tiles.

## 3. Units: nDSM is metres (verified)

DC_27_38 contains the Washington National Cathedral. Its central tower is about 91 m tall,
and the tile maximum is **93.7 m**. Per-class medians are also physically sensible:
buildings 7.3 m, trees 8.5 m, ground/road/low-vegetation ≈ 0 m.

## 4. NoData and noise → the validity rule

Implemented once in `depthwizard/data/gamus.py::valid_mask`:

| observation | where | handling |
|---|---|---|
| Values of exactly **-5.0** plus a few in (-5, -4.9) | DC only, mostly water/"others" (up to 55% of a tile) | **invalid** (`ndsm < -2 m`) |
| Small negatives, -0.2 to -1.6 m | NYC, scattered | valid (normal DSM-DTM noise) |
| Isolated spikes up to 203 m on flat ground | PHL_6743, PHL_1364, PHL_3300 (speckle strip at tile edge) | **invalid** if > 20 m above the 5×5 median |
| Real 60–94 m buildings | PHL_2725, DC_27_38 | kept (the spike filter drops only ~30 corner pixels) |
| Small dropout pits on roofs | e.g. PHL_5497 | not handled yet (minor) |
| No NaN / inf | all | — |

No global height cap is applied, because it would delete real skyscrapers.

## 5. Alignment

Checked visually in `figures/*_zoom.png` for each city: building outlines, nDSM > 3 m
edges and the building mask agree to within about 1 px (NYC is best). After the
DataLoader's random crop, rotation and flip, alignment is kept. This is proved exactly
by `tests/test_phase0.py::test_augmentation_keeps_alignment` and shown in
`figures/dataloader_batch.png`.

## 6. Properties that will affect modelling

1. **Heavy class imbalance in height.** About 15% of pixels are exactly 0 m, and most of the
   rest are near 0. Global MAE is dominated by ground, so every report also gives
   building and tree metrics.
2. **Leaf-off imagery (DC, PHL).** Winter trees appear as brown branches but have 10–25 m
   of height. There's a domain gap with leaf-on imagery, including Indian scenes.
3. **Relief displacement.** The RGB is not a true orthophoto, so tall buildings lean in the
   image while the LiDAR nDSM is orthographic. Expect unavoidable errors at tall-building
   edges (visible at the cathedral).
4. **The DC nDSM is blocky** (about 1 m steps), probably upsampled from a coarser LiDAR grid.
5. **Cities differ.** Mean nDSM: DC 6.1 m, NYC 5.0 m, PHL 2.4 m.

## 7. Performance note

The DataLoader gives 11.5 samples/s (4 workers, 518 px crops). The cost is the full-tile
h5 read plus the median filter (~0.27 s per tile). Before full-dataset training
(Phase 2), cache `valid_mask` once per tile.

## 8. Subset used

`configs/subsets/phase0.yaml`: seed 0, city-stratified. Train 40 (DC 12, NYC 9, PHL 19),
val 10 (DC 4, PHL 6). The test split is not downloaded.
