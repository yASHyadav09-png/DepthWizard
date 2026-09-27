# Phase 7: 3D Explorer (7a, 7c, 7b, 7d, 7e)

Status: **implemented and checked in the browser** (2026-09-26). Built without Phase 5 functionality;
the Phase 5 integration (elevation, map coordinates, GeoTIFF export/validation) is described in
`docs/phase5_geospatial.md` (Viewer section).

## Flow
Dashboard (upload → processing → preview) → **"Enter 3D Explorer ⛶"** → full-screen Explorer →
**◀ Exit** or **Esc** (the first Esc releases the mouse pointer in Fly/Walk, the second exits).

## Coordinate system (real metres, no scene scaling)
```
X = pixel_x × GSD   (0 at the left image edge)
Z = pixel_y × GSD   (0 at the top image edge)
Y = predicted nDSM in metres
```
Grid values sit at cell centres (the backend area-averages the full-resolution nDSM onto the mesh grid).
`heightAt(x, z)` interpolates bilinearly between cell centres and is 0 m (bare ground) outside the
imaged extent (`frontend/src/explorer/terrainModel.ts`). The Explorer always shows true scale
(no exaggeration).

## 7a: navigation, HUD, minimap
| mode | controls | constraint |
|---|---|---|
| Orbit (key 1) | drag / scroll / right-drag | the camera is kept ≥ 2 m above the surface every frame |
| Fly (key 2) | click to capture the mouse · **W A S D** · **Space / C** up/down · **Shift** ×4 · wheel = speed (2–150 m/s) · Esc | camera **≥ 2 m** above the surface |
| Walk (key 3) | same, no vertical keys · Shift ×3 · 0.8–4 m/s | eye **exactly 1.7 m** above the surface; follows the terrain |

**Terrain constraint** (`physics.ts`): movement is integrated in ≤ 0.25 m sub-steps, and every
sub-step is checked **before** it is applied. The camera is never inside the surface and pushed out
afterwards.
- Fly: a step that would go below the clearance keeps as much horizontal motion as possible (sliding
  along walls) as long as the surface ahead rises ≤ 0.5 m above the camera's current height; a
  steeper rise is a wall. A wall doesn't pop the camera onto the roof: you fly up to clear it.
- Walk: step-ups > 0.5 m per sub-step are refused (walls); drops are allowed (walking off a roof).

**HUD:** mode; height, slope and (after validation) error under the cursor (orbit) or at the
screen-centre target (fly/walk); the camera's height above the surface; position (m); speed (m/s).
**Top bar:** file, footprint, GSD (assumed or given), metric validity, model run / epoch / val RMSE /
device. **Minimap:** RGB top-down view = terrain extent, camera position (red when outside the
extent), heading and field-of-view wedge. Click-to-teleport is postponed.

## 7c: slope map
Layer "Slope (degrees)": `slope = atan(√((∂h/∂x)² + (∂h/∂z)²))`, central differences over the real cell
spacing, 0–90° legend. On an nDSM this is the slope of the imaged surface: roofs and canopy, and
near-vertical values at walls and tree edges.

## 7b: profile and measurement
📈 **Profile**: click A and B (drags still orbit). The line is draped 0.4 m above the surface,
sampled every half grid cell, with a chart of height vs distance plus length, min/max and the
steepest local gradient. 📏 **Measure**: ground distance, heights at A and B, height difference B−A,
3D distance.

## 7d: validation panel
Upload a reference height array (`.npy`, metres, the **same pixel grid** as the prediction,
NaN = no data; GeoTIFF with reprojection arrives in Phase 5) → backend `POST /api/results/{job}/validate`.
It uses the **same metric code as the model evaluation**: RMSE, MAE, bias, r, and MAE/bias per true
height band; plus a full-resolution error map (PNG) and the error on the mesh grid. The Explorer
adds an **"Error vs reference"** layer (blue = too low, red = too high, ±10 m) and the error under
the cursor.

## 7e: screenshot and export
📷 saves the current 3D view as PNG (the WebGL drawing buffer is preserved). The Export section links
to the nDSM `.npy` (float32 m), height map, shaded relief, metadata/provenance JSON, and, after
validation, the validation report JSON and error map. GeoTIFF export: Phase 5.

## Verification
| check | result |
|---|---|
| unit tests (vitest) | 26 pass: physics (incl. a 2,000-move random stress test that never ends below the 2 m clearance), slope (30° plane exact at 0.66 m cells, anisotropic cells), profile/measure |
| fly speed live | 22.2 m in 1.5 s at the 15 m/s setting (14.8 m/s); descent 59.8 m in 4 s |
| fly clearance live | a 60 m/s dive stops at exactly 2.0 m |
| walk live | the eye snaps to 1.7 m; 1.4 m/s |
| cursor readout | 14.0 m on a building, 0.0 m on grass; 24.2 m / 54° / −14.8 m error on the tall building |
| profile ↔ measure | length 222.8 m = ground distance 222.80 m; B height 27.84 m = profile max 27.8 m; 3D distance = √(222.8² + 27.84²) |
| validation ↔ evaluation | the DC_48_31 LiDAR reference via the web UI: RMSE 5.137524 / MAE 2.500832 = the evaluated tile metrics |
| backend tests | 29 pass (incl. 3 validation endpoint tests) + 1 opt-in real-model test |
| screenshot | a 1012×735 PNG, 690 KB (real content) |

## Bugs found and fixed during Phase 7
1. **Fly: W did nothing at the 2 m floor while looking slightly down.** A step with a downward
   component was rejected as a whole instead of keeping its horizontal part. Found live and fixed;
   regression test added.
2. A 3D canvas crash during hot-reload (props mismatch) left the scene silently frozen. The Explorer
   canvas is now wrapped in an ErrorBoundary.

## Known limitations
- Mouse-look (pointer lock) can't be exercised by the automated browser; it needs a manual check.
- The Explorer uses the 512-cell mesh grid (≈ 0.66 m cells for a 1024 tile). Readouts and slope are
  at that resolution; the full-resolution nDSM is in the `.npy` export.
- Minimap click-to-teleport is postponed.
