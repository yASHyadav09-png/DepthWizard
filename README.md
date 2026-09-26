# DepthWizard — Single-View Height Estimation & 3D Flythrough

> ### Current status (Phase 3)
> The app now runs the **trained model** (Depth Anything V2 Small fine-tuned on GAMUS, Phase 2b)
> and predicts **height above ground in metres** (nDSM). The Stage-1 description below is kept for
> history; for how to run the current app see **`docs/phase3_demo.md`**, and for project status
> see **`docs/progress.md`**.


**Smart India Hackathon 2026 · Problem Statement 26175 · ISRO**

Turn a single ordinary photograph into an interactive 3D terrain surface driven by
AI-estimated **relative height**.

> ### ⚠ Relative Height — Not Metric
> Stage 1 produces a **unitless, non-georeferenced relative DSM (rDSM)** in the range
> `0–1`. It does **not** output metres, absolute elevation, or any coordinate
> reference system. Metric calibration arrives in a later stage.

---

## 1. The problem

Conventional Digital Surface Models need stereo pairs, LiDAR or InSAR. PS 26175 asks
for height information from a **single view**, plus a way to *see* the result as 3D
terrain you can fly through.

The full DepthWizard vision:

```
RGB / GeoTIFF → monocular depth → relative depth → semantic & geometry-aware refinement
             → DEM / GCP scale calibration → absolute DSM → 3D terrain → flythrough
```

## 2. Stage 1 objective (what this repository does today)

A complete, working, end-to-end prototype of the *relative* half of that pipeline,
for **non-georeferenced JPG/PNG images only**:

```
RGB image
  → Depth Anything V2 Small            (real inference, GPU or CPU)
  → relative depth map
  → relative height field / rDSM       (robust percentile normalisation → 0–1)
  → terrain grid (256 px longest side)
  → RGB texture projection
  → interactive Three.js viewer
```

**Deliberately out of scope for Stage 1:** GAMUS training/fine-tuning, SRTM, GCP
calibration, absolute/metric DSM, GeoTIFF IO, LiDAR validation, semantic fusion. The
architecture leaves explicit seams for all of them (see §8).

## 3. Architecture

```
depthwizard/
├── backend/
│   ├── app/
│   │   ├── main.py                     FastAPI app, CORS, error handlers, /static mount
│   │   ├── config.py                   all tunables + env overrides
│   │   ├── api/
│   │   │   ├── routes.py               /api/health, /api/process, /api/results/{id}
│   │   │   └── schemas.py              pydantic response models
│   │   ├── services/
│   │   │   ├── depth_estimator.py      Depth Anything V2 — loaded once, reused
│   │   │   ├── depth_processor.py      normalisation, rDSM, stats, colour maps
│   │   │   ├── terrain_generator.py    height field → downsampled mesh grid
│   │   │   ├── pipeline.py             stage orchestration (extension points here)
│   │   │   └── storage.py              per-job PNG / .npy / metadata artefacts
│   │   └── utils/                      image IO, validation, typed errors
│   └── tests/                          unit + API + optional real-model tests
│
├── frontend/src/
│   ├── components/                     Header, Upload, Status, Stats, Raster, 3D viewer
│   ├── services/                       API client, terrain buffer decoding
│   └── types/                          shared TypeScript contracts
│
├── sample_data/   docs/   README.md
```

**Data flow.** `POST /api/process` runs the whole pipeline synchronously (off the event
loop, in a worker thread) and returns one JSON payload: statistics, asset URLs and the
terrain grid. Heights travel as a **base64 little-endian float32 buffer** — a 256×256
grid is ~350 KB, versus several MB as JSON numbers. The browser decodes it into a
`Float32Array` and builds a `BufferGeometry` directly.

**Why the mesh is built by hand** (`TerrainMesh.tsx`) rather than by displacing a
`PlaneGeometry`: it keeps UV orientation, triangle winding and the height lookup in one
readable place, and lets the height exaggeration slider re-displace vertices and
recompute normals (scaling the mesh instead would leave the shading wrong).

## 4. Installation

Prerequisites: **Python 3.10–3.12**, **Node.js 18+**, and internet access on first run
(the model weights, ~100 MB, are downloaded from Hugging Face and cached).

### Backend

```bash
cd depthwizard/backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 1) Install the PyTorch build that matches your machine FIRST.
#    NVIDIA GPU (CUDA 12.8 — RTX 40/50 series):
pip install torch --index-url https://download.pytorch.org/whl/cu128
#    CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else.
pip install -r requirements.txt
```

### Frontend

```bash
cd depthwizard/frontend
npm install
```

## 5. Running

**Terminal 1 — backend** (from `depthwizard/backend`, venv active):

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/api/health>

The first start downloads the model and logs the device it will use, e.g.
`Inference device: CUDA (NVIDIA GeForce RTX 5050 Laptop GPU)`.

**Terminal 2 — frontend** (from `depthwizard/frontend`):

```bash
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` and `/static` to port 8000, so the
browser only ever talks to one origin. (`npm run build && npm run preview` serves the
production bundle on port 4173 with the same proxy.)

### Using it

1. Drop a JPG or PNG onto the upload panel.
2. Press **Generate Terrain**.
3. Watch the status stepper; per-stage timings appear when it finishes.
4. Inspect the depth map and relative DSM (click to enlarge).
5. Explore the 3D terrain: **drag** to rotate, **scroll** to zoom, **right-drag** to pan.
6. Toggle RGB texture / elevation view / wireframe, adjust height exaggeration, reset
   the camera, or switch to **Flythrough** (`W A S D` to move, `R`/`F` for up/down,
   drag to look).

A finished run is linkable: the URL becomes `?job=<job_id>`, and reopening it reloads
that result from `/api/results/{job_id}` without re-uploading.

### API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Status, model, device, CUDA availability |
| `POST` | `/api/process` | multipart `image` → full pipeline payload |
| `GET` | `/api/results/{job_id}` | Replay a completed job's payload |
| `GET` | `/api/results/{job_id}/height-array` | Download the raw `.npy` height field |
| `GET` | `/api/jobs` | Recent job ids |
| `GET` | `/static/{job_id}/…` | Generated PNG / NPY / JSON artefacts |

```bash
curl -F "image=@../sample_data/sample_street_scene.jpg" http://127.0.0.1:8000/api/process
```

`terrain` in the response is what the viewer meshes:

```jsonc
"terrain": {
  "width": 256, "height": 171,        // grid columns / rows (aspect preserved)
  "heights_b64": "…",                 // little-endian float32, row-major, 0-1
  "encoding": "float32-le-base64",
  "plane_width": 1.0, "plane_depth": 0.667,
  "height_units": "relative (0-1, unitless)"
}
```

### Outputs per job (`backend/outputs/{job_id}/`)

| File | Contents |
| --- | --- |
| `original.png` | The uploaded image, EXIF-rotated, as RGB |
| `texture.jpg` | Downscaled RGB used as the 3D terrain texture |
| `depth_visualization.png` | Relative depth, `inferno` colour map |
| `relative_dsm.png` | rDSM as shaded relief (`gist_earth` + hillshade) |
| `relative_height.npy` | Numeric float32 height field (at inference resolution — exact shape in `metadata.height_array_width/height`) |
| `metadata.json` | Full job payload — see below |

```jsonc
"metadata": {
  "model": "Depth Anything V2 Small",
  "checkpoint": "depth-anything/Depth-Anything-V2-Small-hf",
  "type": "relative monocular depth",
  "georeferenced": false,
  "height_units": "relative",
  "crs": null,
  "stage": 1
}
```

## 6. Model

| | |
| --- | --- |
| Model | **Depth Anything V2 Small** |
| Checkpoint | `depth-anything/Depth-Anything-V2-Small-hf` (Transformers port) |
| Output | Relative **inverse** depth (disparity-like) |
| Device | CUDA when available, otherwise CPU — chosen automatically |
| Precision | fp16 autocast on CUDA, fp32 on CPU |
| Loading | Once per process, reused across every request |

**Why inverse depth maps to height.** Depth Anything V2 regresses inverse depth: a
large value means *close to the camera*. For downward-looking or oblique scenes,
closer to the sensor means higher off the ground — so the normalised prediction is
already a relative height field. `depth_processor.to_relative_height()` takes an
`is_inverse_depth` flag so a future true-metric-depth model flips instead.

## 7. Testing

```bash
cd depthwizard/backend
pytest                                  # unit + API tests, model stubbed, no download
DW_RUN_MODEL_TESTS=1 pytest             # also runs real Depth Anything V2 inference
```

```bash
cd depthwizard/frontend
npm run typecheck
npm run build
```

## 8. Configuration

Every setting lives in `backend/app/config.py` and is env-overridable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DW_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `DW_MODEL_NAME` | `depth-anything/Depth-Anything-V2-Small-hf` | HF checkpoint |
| `DW_PRELOAD_MODEL` | `1` | Load weights at startup instead of first request |
| `DW_TERRAIN_RESOLUTION` | `256` | Longest side of the terrain grid |
| `DW_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `DW_MAX_INFERENCE_SIDE` | `1536` | Longest side fed to the network |
| `DW_OUTPUT_DIR` | `backend/outputs` | Where artefacts are written |
| `DW_CORS_ORIGINS` | Vite dev origins | Comma-separated allow-list |

## 9. Current limitations

- **Relative, not metric.** Values are unitless `0–1`. No metres, no absolute elevation.
- **Not georeferenced.** No CRS, no geotransform, no GeoTIFF in or out.
- **JPG/PNG only.**
- Monocular depth is **scale- and shift-ambiguous** — relative ordering is meaningful,
  absolute differences are not.
- Accuracy degrades on textureless surfaces, reflections, heavy shadow and thin
  structures; boundaries are softened slightly to remove spiky artefacts.
- The terrain is a **2.5D height field** — no overhangs, facades or true volumes.
- Processing is synchronous and single-job; there is no queue or persistence layer.
- No quantitative validation yet (RMSE/MAE needs reference LiDAR or DSM data).

## 10. Future stages

The pipeline orchestrator (`backend/app/services/pipeline.py`) marks each seam with a
`# [stage-N]` comment; new stages slot in as additional service modules without
touching the API layer.

| Stage | Adds | Where it plugs in |
| --- | --- | --- |
| **2** | GAMUS semantic segmentation, RGB + geometry feature fusion | `services/semantic_refiner.py`, after `to_relative_height` |
| **3** | SRTM / GCP scale calibration → **absolute metric DSM** | `services/scale_calibrator.py`, consumes the rDSM |
| **4** | GeoTIFF read/write, CRS + geotransform propagation | `utils/geo_io.py`, alongside `image_io` |
| **5** | RMSE / MAE validation against LiDAR reference | `services/validator.py` |

GAMUS reference (**not** downloaded or trained in Stage 1):
<https://github.com/EarthNets/RSI-MMSegmentation>

## 11. Credits

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth
- [Three.js](https://github.com/mrdoob/three.js) · [React Three Fiber](https://github.com/pmndrs/react-three-fiber) · [Drei](https://github.com/pmndrs/drei) — 3D rendering
- [FastAPI](https://fastapi.tiangolo.com/) · [PyTorch](https://pytorch.org/) · [Transformers](https://huggingface.co/docs/transformers)
- SIH 2026 · PS 26175 · ISRO — [official repo](https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026)

**Model weights and datasets are never committed to this repository.**
