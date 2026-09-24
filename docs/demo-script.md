# 5-minute demo script — SIH judges

A tight walkthrough that shows the pipeline is *real*, not a mockup.

## Before the judges arrive

1. Backend running, model already loaded (the first request after a cold start
   pays the download/warm-up cost — trigger one throwaway run beforehand):
   ```
   cd depthwizard/backend && .venv\Scripts\activate
   uvicorn app.main:app --port 8000
   ```
2. Frontend running: `cd depthwizard/frontend && npm run dev`
3. Two or three images ready in `sample_data/` — ideally one aerial/drone shot of
   buildings and one landscape.
4. Browser at <http://localhost:5173>, window maximised.

## The walkthrough

**0:00 — Frame the problem.**
"PS 26175 asks for height from a *single* view. Conventional DSMs need stereo,
LiDAR or InSAR. We estimate height from one ordinary photo."

**0:30 — Show the header.** Point at the live badges: `Stage 1 · Relative DSM`,
`Depth Anything V2 Small`, `CUDA (…)`, `Backend online`. These are read from
`/api/health` — the device badge proves inference is really on the GPU.

**1:00 — Upload and run.** Drag an image in, press **Generate Terrain**. Narrate
the status stepper as it advances. When it completes, point at the **per-stage
timings** — real inference milliseconds, not an animation.

**1:45 — The raster products.** Original RGB → depth map → relative DSM. Click
the DSM to enlarge; the shaded relief makes the structure obvious.

**2:15 — The 3D terrain.** Rotate, zoom, pan. Say out loud: *"this is the height
field, meshed — the original photo is projected on as texture."*

**2:45 — Prove the geometry is real.**
- Drag **Height exaggeration** from 0 to 1. The surface flattens and rises — the
  relief is data, not a texture trick.
- Toggle **Wireframe**: the 256-grid is visible over the surface.
- Toggle **Elevation view**: RGB drops away, the height ramp remains.
- Switch to **Flythrough** and fly across the terrain with `W A S D`.

**3:45 — Honesty slide (this wins marks).** Point at the amber banner:
**"Relative Height — Not Metric."** Say it plainly:
"Monocular depth is scale- and shift-ambiguous. The *ordering* of heights is
meaningful; the absolute values are not. We refuse to print metres we cannot
justify."

**4:15 — The roadmap.** Open `docs/pipeline.md` §Extension seams, or the README
stage table. Stage 2 GAMUS semantic refinement → Stage 3 SRTM/GCP calibration →
absolute metric DSM → Stage 4 GeoTIFF → Stage 5 RMSE/MAE validation. The
orchestrator already carries the `# [stage-N]` insertion points.

**4:45 — The artefacts.** Download the `.npy` height array and open
`metadata.json` from the statistics panel: `georeferenced: false`,
`height_units: "relative"`, `type: "relative monocular depth"`. Every output is
self-describing.

## Likely questions

**"Is this really AI, or a filter?"** Open `/docs` (FastAPI Swagger), run
`POST /api/process` there, show the raw JSON. Or show the backend log line with
the model name and CUDA device. Upload a photo the judges pick themselves.

**"Why not metres?"** Relative depth from a single image has no absolute scale —
a model plane at 1 m and a real one at 100 m project identically. Metres require
an external reference: GCPs, a DEM, or known camera geometry. That is Stage 3.

**"How accurate is it?"** Not quantified yet, and deliberately so — RMSE/MAE
needs reference LiDAR or an existing DSM, which is Stage 5. What Stage 1
guarantees is a correct, reproducible *relative* surface.

**"Why Depth Anything V2 Small?"** Strong zero-shot generalisation across
domains, and small enough to run interactively on a laptop GPU — which matters
for a live demo. The estimator module is a single swap-in point for a larger
variant or a GAMUS-fine-tuned model.

**"What about overhangs / building facades?"** A height field is 2.5D by
construction — one height per ground cell. True 3D would need multi-view or
volumetric reconstruction, which is outside PS 26175's single-view framing.
