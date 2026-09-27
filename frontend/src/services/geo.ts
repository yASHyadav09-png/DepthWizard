/**
 * Phase 5 helpers for georeferenced results.
 *
 * Scene offset: for a DSM job the terrain grid holds ELEVATIONS (m, EGM2008, e.g.
 * 218-371 m). The 3D scenes place the surface at Y = elevation - base, with
 * base = floor(min elevation). This is a rigid vertical TRANSLATION (1 m is still 1 m;
 * nothing is scaled), which keeps the ground plane, physics and camera near Y = 0.
 * Displayed elevations add `base` back. For nDSM jobs base = 0.
 *
 * Map coordinates: scene X/Z are metres from the image's top-left corner
 * (X = col * gsd, Z = row * gsd). The GeoTIFF transform maps pixels to the CRS:
 *   easting = c + col * a,  northing = f + row * e   (north-up, b = d = 0).
 * Longitude/latitude are interpolated bilinearly between the four image corners,
 * which the backend projects exactly; over a few km the error is centimetres,
 * far below the display precision (1e-5 deg ~ 1 m).
 */
import type { ProcessResult, TerrainGrid } from '../types'

export function isDsm(grid: TerrainGrid): boolean {
  return grid.surface_kind === 'dsm'
}

/** Metres to add to a scene Y to get the product's value (elevation for DSM jobs). */
export function sceneBase(grid: TerrainGrid): number {
  return isDsm(grid) ? Math.floor(grid.min_height) : 0
}

export interface MapCoords {
  easting: number
  northing: number
  lon: number | null
  lat: number | null
}

/** CRS coordinates (and lon/lat when available) of a scene point (x, z) in metres. */
export function mapCoords(result: ProcessResult, x: number, z: number): MapCoords | null {
  const hp = result.height_product
  const t = hp.transform
  const gsd = result.terrain.gsd_m
  if (!t || t.length < 6 || !(gsd > 0)) return null
  const col = x / gsd
  const row = z / gsd
  const out: MapCoords = { easting: t[2] + col * t[0] + row * t[1], northing: t[5] + col * t[3] + row * t[4], lon: null, lat: null }
  const c = hp.corners_lonlat
  if (c && c.length === 4) {
    const u = x / result.terrain.plane_width
    const v = z / result.terrain.plane_depth
    const [tl, tr, br, bl] = c
    out.lon = (tl[0] * (1 - u) + tr[0] * u) * (1 - v) + (bl[0] * (1 - u) + br[0] * u) * v
    out.lat = (tl[1] * (1 - u) + tr[1] * u) * (1 - v) + (bl[1] * (1 - u) + br[1] * u) * v
  }
  return out
}

export function formatLonLat(lon: number, lat: number): string {
  return `${Math.abs(lat).toFixed(5)}°${lat >= 0 ? 'N' : 'S'} ${Math.abs(lon).toFixed(5)}°${lon >= 0 ? 'E' : 'W'}`
}

/** Short label for the product's vertical reference, e.g. "EGM2008". */
export function datumLabel(result: ProcessResult): string {
  const vd = result.height_product.vertical_datum
  if (!vd) return 'above ground'
  return vd.includes('EGM2008') ? 'EGM2008' : vd
}

/** Ground resolution for display: 6 significant digits (0.599999999999993 -> "0.6"). */
export function formatGsd(gsdM: number): string {
  return String(Number(gsdM.toPrecision(6)))
}
