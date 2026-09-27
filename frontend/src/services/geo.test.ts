import { describe, expect, it } from 'vitest'
import type { ProcessResult, TerrainGrid } from '../types'
import { formatLonLat, mapCoords, sceneBase } from './geo'
import { aboveGroundAt, heightAt, makeTerrainModel } from '../explorer/terrainModel'
import { constrainMove, EYE_HEIGHT_M } from '../explorer/physics'

const b64 = (a: Float32Array) => {
  const bytes = new Uint8Array(a.buffer)
  let s = ''
  for (let i = 0; i < bytes.length; i += 1) s += String.fromCharCode(bytes[i])
  return btoa(s)
}

/** Float32 at ~250 m resolves ~1.5e-5 m, hence 1e-4 tolerances. 4 x 4 grid, 2 m cells. DSM = 250.4 m ground with a 12 m block in cell (2, 1). */
function dsmGrid(): TerrainGrid {
  const dsm = new Float32Array(16).fill(250.4)
  const ndsm = new Float32Array(16)
  dsm[1 * 4 + 2] = 262.4
  ndsm[1 * 4 + 2] = 12
  return {
    width: 4, height: 4, heights_b64: b64(dsm), encoding: 'float32-le-base64',
    min_height: 250.4, max_height: 262.4, mean_height: 251, aspect_ratio: 1,
    plane_width: 8, plane_depth: 8, source_width: 16, source_height: 16, gsd_m: 0.5,
    display_min: 250, display_max: 263, height_units: 'm', surface_kind: 'dsm', ndsm_b64: b64(ndsm),
  }
}

describe('DSM scene offset', () => {
  it('shifts elevations by floor(min) without scaling', () => {
    const grid = dsmGrid()
    expect(sceneBase(grid)).toBe(250)
    const t = makeTerrainModel(grid)
    expect(t.base).toBe(250)
    expect(heightAt(t, 1, 1)).toBeCloseTo(0.4, 4)
    expect(heightAt(t, 5, 3)).toBeCloseTo(12.4, 4) // block cell centre (x = 2.5 * 2, z = 1.5 * 2)
    // real elevation = Y + base, and 1 m in Y is 1 m
    expect(heightAt(t, 5, 3) + t.base!).toBeCloseTo(262.4, 4)
    expect(heightAt(t, 5, 3) - heightAt(t, 1, 1)).toBeCloseTo(12, 4)
  })
  it('reads height above ground from the nDSM grid, null outside', () => {
    const t = makeTerrainModel(dsmGrid())
    expect(aboveGroundAt(t, 5, 3)).toBeCloseTo(12, 5)
    expect(aboveGroundAt(t, 1, 7)).toBeCloseTo(0, 5)
    expect(aboveGroundAt(t, -1, 3)).toBeNull()
  })
  it('keeps walk physics at eye height above the shifted surface', () => {
    const t = makeTerrainModel(dsmGrid())
    const p = constrainMove(t, { x: 1, y: 0.4 + EYE_HEIGHT_M, z: 7 }, { x: 0.5, y: 0, z: 0 }, 'walk')
    expect(p.y).toBeCloseTo(0.4 + EYE_HEIGHT_M, 4)
  })
  it('leaves nDSM grids unshifted', () => {
    const g = { ...dsmGrid(), surface_kind: undefined, ndsm_b64: undefined }
    expect(sceneBase(g)).toBe(0)
    const t = makeTerrainModel(g)
    expect(t.base).toBe(0)
    expect(aboveGroundAt(t, 1, 1)).toBeCloseTo(250.4, 4) // falls back to the height field itself
  })
})

describe('map coordinates', () => {
  const result = {
    terrain: { ...dsmGrid() },
    height_product: {
      transform: [0.5, 0, 585000, 0, -0.5, 4477000],
      corners_lonlat: [
        [-80.0, 40.5],
        [-79.9, 40.5],
        [-79.9, 40.4],
        [-80.0, 40.4],
      ],
    },
  } as unknown as ProcessResult

  it('maps scene metres to easting/northing through the GeoTIFF transform', () => {
    const c = mapCoords(result, 2, 4)!
    expect(c.easting).toBeCloseTo(585002, 6) // col 4 * 0.5 m
    expect(c.northing).toBeCloseTo(4476996, 6) // row 8 * -0.5 m
  })
  it('interpolates lon/lat between the corners', () => {
    const c = mapCoords(result, 4, 4)! // image centre
    expect(c.lon).toBeCloseTo(-79.95, 9)
    expect(c.lat).toBeCloseTo(40.45, 9)
    expect(formatLonLat(c.lon!, c.lat!)).toBe('40.45000°N 79.95000°W')
  })
  it('returns null without a transform', () => {
    const r = { ...result, height_product: { transform: null } } as unknown as ProcessResult
    expect(mapCoords(r, 1, 1)).toBeNull()
  })
})
