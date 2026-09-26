/**
 * Terrain in REAL METRES for the 3D Explorer. No scene scaling anywhere.
 *
 *   X = pixel_x * gsd   (0 at the left image edge, increasing to the right)
 *   Z = pixel_y * gsd   (0 at the top image edge, increasing downwards)
 *   Y = predicted nDSM in metres (height above ground)
 *
 * The backend area-averages the full-resolution nDSM onto a w x h grid; grid cell
 * (i, j) covers source pixels [i*sw/w, (i+1)*sw/w) etc., so its value belongs to the
 * cell CENTRE: x = (i + 0.5) * dx, z = (j + 0.5) * dz, with dx = plane_width / w.
 */
import * as THREE from 'three'
import type { TerrainGrid } from '../types'
import { decodeHeights } from '../services/terrain'

export interface TerrainModel {
  w: number
  h: number
  dx: number // metres per grid cell along X
  dz: number // metres per grid cell along Z
  width: number // extent along X (m)
  depth: number // extent along Z (m)
  heights: Float32Array // row-major, row 0 = top of the image (Z small)
  maxHeight: number
}

export function makeTerrainModel(grid: TerrainGrid, heights?: Float32Array): TerrainModel {
  const hs = heights ?? decodeHeights(grid)
  let maxHeight = 0
  for (let k = 0; k < hs.length; k += 1) if (hs[k] > maxHeight) maxHeight = hs[k]
  return {
    w: grid.width,
    h: grid.height,
    dx: grid.plane_width / grid.width,
    dz: grid.plane_depth / grid.height,
    width: grid.plane_width,
    depth: grid.plane_depth,
    heights: hs,
    maxHeight,
  }
}

export function insideExtent(t: TerrainModel, x: number, z: number): boolean {
  return x >= 0 && x <= t.width && z >= 0 && z <= t.depth
}

/**
 * Height above ground (m) at a real-world point: bilinear interpolation between
 * cell centres, clamped to the edge cells inside the extent, and 0 m (bare ground)
 * outside the imaged extent.
 */
export function heightAt(t: TerrainModel, x: number, z: number): number {
  if (!insideExtent(t, x, z)) return 0
  const fx = Math.min(Math.max(x / t.dx - 0.5, 0), t.w - 1)
  const fz = Math.min(Math.max(z / t.dz - 0.5, 0), t.h - 1)
  const i0 = Math.floor(fx)
  const j0 = Math.floor(fz)
  const i1 = Math.min(i0 + 1, t.w - 1)
  const j1 = Math.min(j0 + 1, t.h - 1)
  const u = fx - i0
  const v = fz - j0
  const H = t.heights
  const a = H[j0 * t.w + i0]
  const b = H[j0 * t.w + i1]
  const c = H[j1 * t.w + i0]
  const d = H[j1 * t.w + i1]
  return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v
}

/** Mesh with vertices at cell centres, in metres, UVs mapping the full image. */
export function buildTerrainGeometry(t: TerrainModel): THREE.BufferGeometry {
  const { w, h, dx, dz, width, depth, heights } = t
  const count = w * h
  const positions = new Float32Array(count * 3)
  const uvs = new Float32Array(count * 2)
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      const k = j * w + i
      const x = (i + 0.5) * dx
      const z = (j + 0.5) * dz
      positions[k * 3] = x
      positions[k * 3 + 1] = heights[k]
      positions[k * 3 + 2] = z
      uvs[k * 2] = x / width
      uvs[k * 2 + 1] = 1 - z / depth // texture loaded with flipY: top of image at v = 1
    }
  }
  const quads = (w - 1) * (h - 1)
  const indices = count > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6)
  let n = 0
  for (let j = 0; j < h - 1; j += 1) {
    for (let i = 0; i < w - 1; i += 1) {
      const a = j * w + i
      const b = a + 1
      const c = a + w
      const d = c + 1
      // winding with +Y normals for X right / Z down
      indices[n++] = a
      indices[n++] = c
      indices[n++] = b
      indices[n++] = b
      indices[n++] = c
      indices[n++] = d
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  g.setIndex(new THREE.BufferAttribute(indices, 1))
  g.computeVertexNormals()
  g.computeBoundingSphere()
  return g
}

/** Per-vertex colours for a scalar field (e.g. height or slope) using a colour ramp. */
export function colourAttribute(
  values: Float32Array,
  vmin: number,
  vmax: number,
  ramp: (t: number, out: [number, number, number]) => void,
): THREE.BufferAttribute {
  const colours = new Float32Array(values.length * 3)
  const rgb: [number, number, number] = [0, 0, 0]
  const range = Math.max(vmax - vmin, 1e-6)
  for (let k = 0; k < values.length; k += 1) {
    if (!Number.isFinite(values[k])) {
      // no data (e.g. reference gaps in the error layer): neutral grey
      colours[k * 3] = colours[k * 3 + 1] = colours[k * 3 + 2] = 0.35
      continue
    }
    ramp((values[k] - vmin) / range, rgb)
    colours[k * 3] = rgb[0]
    colours[k * 3 + 1] = rgb[1]
    colours[k * 3 + 2] = rgb[2]
  }
  return new THREE.BufferAttribute(colours, 3)
}

/**
 * Surface slope in DEGREES at every grid cell centre, from central differences of the
 * height grid over the real cell spacing (one-sided at the edges):
 *   slope = atan( sqrt( (dh/dx)^2 + (dh/dz)^2 ) )
 * On an nDSM this is the slope of the imaged surface (roofs, canopy, and near-vertical
 * values at building walls and tree edges), at the grid's resolution (dx x dz metres).
 */
export function slopeField(t: TerrainModel): Float32Array {
  const { w, h, dx, dz, heights: H } = t
  const out = new Float32Array(w * h)
  for (let j = 0; j < h; j += 1) {
    const j0 = Math.max(j - 1, 0)
    const j1 = Math.min(j + 1, h - 1)
    for (let i = 0; i < w; i += 1) {
      const i0 = Math.max(i - 1, 0)
      const i1 = Math.min(i + 1, w - 1)
      const gx = (H[j * w + i1] - H[j * w + i0]) / (Math.max(i1 - i0, 1) * dx)
      const gz = (H[j1 * w + i] - H[j0 * w + i]) / (Math.max(j1 - j0, 1) * dz)
      out[j * w + i] = (Math.atan(Math.hypot(gx, gz)) * 180) / Math.PI
    }
  }
  return out
}

/** Value of a per-cell field (e.g. slope) at the cell containing (x, z); null outside the extent. */
export function cellValueAt(t: TerrainModel, field: Float32Array, x: number, z: number): number | null {
  if (!insideExtent(t, x, z)) return null
  const i = Math.min(Math.floor(x / t.dx), t.w - 1)
  const j = Math.min(Math.floor(z / t.dz), t.h - 1)
  return field[j * t.w + i]
}
