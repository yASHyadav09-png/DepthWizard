# DepthWizard (SIH 2026, PS 26175)

Single RGB remote-sensing image → nDSM (height above ground, metres) → calibrated DSM → 3D flythrough.
Solo developer, Windows 11, RTX 5050 Laptop (8 GB VRAM, sm_120).

## Read first
- `docs/progress.md`: current phase, what is done, next steps
- `docs/phase0_data_report.md`: verified GAMUS format, units, NoData rule

## Environment
- ML venv: `.venv` (Python 3.11, torch 2.11 + cu128). Run with `.venv/Scripts/python.exe`.
- `backend/.venv` belongs to the older Stage-1 prototype (`backend/`, `frontend/`). Don't touch it until Phase 3.
- Tests: `.venv/Scripts/python.exe -m pytest -q`

## Layout
- `depthwizard/` python package: `data/` (gamus.py, download.py), `eval/metrics.py`, `utils/runs.py`
- `scripts/`: runnable entry points
- `configs/subsets/*.yaml`: dataset versions (tile ids per split), tracked in git
- `data/`: downloaded data (gitignored). `runs/<stamp>_<name>/`: experiments, `runs/results.csv`: all results

## Rules
- Supervised target is GAMUS nDSM in metres. Relative depth is never physical height.
- Never use the test split for fitting, calibration, hyperparameters or checkpoint selection. Evaluate on test once per final model.
- Every experiment goes through `depthwizard.utils.runs` (config, seed, git hash, manifest, metrics, results.csv row).
- Never assume GSD/CRS/datum/units without reading them from data. JPG/PNG → GSD unknown.
- Validity mask: only `depthwizard.data.gamus.valid_mask`. Report building/tree metrics as well as the global ones.
- Heights: continuous resampling. Class masks: nearest-neighbour.
