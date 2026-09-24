# Stage 1 pipeline — what actually happens to your pixels

Reference for the transformations between an uploaded JPG and the 3D surface on
screen. Everything here is **relative and unitless**; nothing produces metres.

---

## 1. Decode and validate — `utils/image_io.py`

- Size, extension and MIME type are checked *before* Pillow touches the bytes.
- `Image.MAX_IMAGE_PIXELS` caps decompression-bomb uploads.
- EXIF orientation is applied (`ImageOps.exif_transpose`) so the terrain matches
  what the user sees, then the image is converted to `RGB`.
- Both sides must be ≥ 32 px.
- For inference only, the longest side is capped at `DW_MAX_INFERENCE_SIDE`
  (1536). Depth Anything internally resizes to 518 px, so larger inputs cost
  memory without adding detail. The **original** resolution is still used for
  the exported artefacts and the texture.

## 2. Depth inference — `services/depth_estimator.py`

`AutoImageProcessor` + `AutoModelForDepthEstimation` on
`depth-anything/Depth-Anything-V2-Small-hf`.

- The model is a **process-wide singleton**, loaded behind a lock, reused by
  every request. A second lock serialises inference so one CUDA context is
  never entered concurrently.
- `torch.inference_mode()`; fp16 autocast on CUDA, fp32 on CPU.
- The network's low-resolution prediction is bicubically upsampled back to the
  input resolution.
- `torch.cuda.OutOfMemoryError` is caught and surfaced as a clean HTTP 503 with
  an actionable message instead of a stack trace.

Output: `float32 (H, W)` **inverse depth** — larger means *closer to the camera*.

## 3. Relative depth → relative height — `services/depth_processor.py`

```
normalised = clip((d - p1) / (p99 - p1), 0, 1)
```

A 1st/99th-percentile stretch rather than raw min/max: a handful of outlier
pixels (sky, specular highlights, sensor noise) would otherwise compress the
entire scene into a narrow band. If the robust range degenerates, it falls back
to true min/max, and a genuinely flat field returns zeros instead of dividing by
zero.

**Why no inversion.** Depth Anything V2 predicts inverse depth, so a large value
already means "near the sensor". For downward-looking or oblique imagery, near
the sensor means high off the ground — the normalised prediction *is* a relative
height field. `to_relative_height(..., is_inverse_depth=False)` flips instead,
which is the path a future true-metric-depth model would take.

**Statistics** (min / max / mean / median / std / p05 / p95) are computed on the
full-resolution field and always carry `height_units = "relative (0-1, unitless)"`.

**Visualisations.** The depth map uses `inferno`; the rDSM uses matplotlib's
`gist_earth` colour map composited with a 315°/45° soft hillshade
(`LightSource`) — the same sun azimuth as the 3D viewer's key light, so the 2D
and 3D products read consistently. `terrain` was tried first and rejected: its
blue low end reads as water, which is meaningless for a relative field.

## 4. Terrain grid — `services/terrain_generator.py`

A 12 MP photo is 12 million potential vertices. Instead:

1. **Resample** the height field so the longest side is `DW_TERRAIN_RESOLUTION`
   (256), preserving aspect — a 1920×1080 image becomes a 256×144 grid. Float
   `BOX` (area-average) resampling downward, bilinear upward; never through an
   8-bit intermediate.
2. **Smooth** with a small separable Gaussian (σ ≈ 0.6 px), hand-rolled in
   numpy — Pillow's `GaussianBlur` refuses mode `"F"` images and going through
   8-bit would quantise the heights. Monocular depth is noisy at object
   boundaries and the raw grid renders as spikes.
3. **Normalise the footprint** so the longest side is 1.0 world unit — every
   image then frames identically in the viewer regardless of aspect ratio.
4. **Encode** as little-endian `float32`, base64. A 256×256 grid is ~350 KB on
   the wire; the same numbers as JSON would be several MB.

## 5. Mesh construction — `frontend/src/components/TerrainMesh.tsx`

The surface is assembled vertex by vertex rather than by displacing a
`PlaneGeometry`, so orientation, winding and height lookup live in one place.

For grid cell `(i, j)` — where `j = 0` is the **top** row of the source image:

| | |
| --- | --- |
| position | `x = (u − 0.5)·planeWidth`, `y = h[j·W+i]`, `z = (v − 0.5)·planeDepth` |
| uv | `(u, 1 − v)` — the `1 −` compensates for Three.js `flipY` on textures |
| indices | `(a, c, b)` and `(b, c, d)` — wound so face normals point **+Y** |
| colour | elevation ramp, sampled per vertex for the elevation view mode |

Index buffer widens to `Uint32Array` above 65 535 vertices (a 256×256 grid is
65 536 — exactly one over the `Uint16` ceiling).

**Height exaggeration** re-writes the `y` component of the position attribute and
calls `computeVertexNormals()`. Scaling the mesh on `y` would be cheaper but
would leave the normals — and therefore the shading — wrong at high values.

## 6. Scene — `frontend/src/components/TerrainViewer.tsx`

- Mesh is built at unit scale and the group is scaled once (`×4`) so camera
  distances and fly speeds are identical for every image.
- Key directional light at 315° azimuth, matching the 2D hillshade, plus a cool
  fill and a hemisphere light.
- `OrbitControls` (rotate / zoom / pan, damped) or `FlyControls` for the
  first-person flythrough. "Reset camera" moves the camera itself (`CameraRig`)
  and then re-targets the controls — re-mounting `OrbitControls` is not enough,
  because it derives its spherical coordinates *from* whatever pose the camera
  already has.
- An `ErrorBoundary` keeps a bad terrain buffer or a WebGL failure from blanking
  the whole dashboard.

---

## Extension seams

`services/pipeline.py` marks each future insertion point with `# [stage-N]`:

```python
relative_height = depth_processor.to_relative_height(...)
# [stage-2] relative_height = semantic_refiner.refine(relative_height, original)
# [stage-3] absolute_dsm    = scale_calibrator.calibrate(relative_height, dem_or_gcps)
```

Each stage lands as a new module under `services/` with the same shape
(array in → array out), so the API layer and the frontend contract do not change
until absolute heights genuinely exist — at which point `height_units`,
`is_metric` and `georeferenced` in the payload flip, and the UI's
"Relative Height — Not Metric" banner is driven by those same fields.
