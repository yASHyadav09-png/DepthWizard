import { describe, expect, it } from 'vitest'
import type { TerrainModel } from './terrainModel'
import { measure, sampleProfile } from './measure'

/** 100 x 100 m, 1 m cells, a 20 m building on x in [40, 60), z in [40, 60). */
function scene(): TerrainModel {
  const heights = new Float32Array(100 * 100)
  for (let j = 40; j < 60; j += 1) for (let i = 40; i < 60; i += 1) heights[j * 100 + i] = 20
  return { w: 100, h: 100, dx: 1, dz: 1, width: 100, depth: 100, heights, maxHeight: 20 }
}

describe('profile', () => {
  it('crosses the building: length, min/max and a near-vertical wall', () => {
    const p = sampleProfile(scene(), { x: 10.5, z: 50.5 }, { x: 90.5, z: 50.5 })
    expect(p.length).toBeCloseTo(80)
    expect(p.minH).toBe(0)
    expect(p.maxH).toBe(20)
    expect(p.maxRise).toBeGreaterThan(80) // the wall
    expect(p.samples[0].d).toBe(0)
    expect(p.samples[p.samples.length - 1].d).toBeCloseTo(80)
    // step is half a cell -> at least 160 intervals
    expect(p.samples.length).toBeGreaterThanOrEqual(161)
  })
  it('handles a zero-length line', () => {
    const p = sampleProfile(scene(), { x: 50.5, z: 50.5 }, { x: 50.5, z: 50.5 })
    expect(p.length).toBe(0)
    expect(p.maxH).toBe(20)
  })
})

describe('measure', () => {
  it('ground to roof: horizontal, height difference and 3D distance in metres', () => {
    const m = measure(scene(), { x: 20.5, z: 50.5 }, { x: 50.5, z: 50.5 })
    expect(m.horizontal).toBeCloseTo(30)
    expect(m.hA).toBe(0)
    expect(m.hB).toBe(20)
    expect(m.dh).toBe(20)
    expect(m.slopeDistance).toBeCloseTo(Math.hypot(30, 20))
  })
  it('is directional in dh', () => {
    expect(measure(scene(), { x: 50.5, z: 50.5 }, { x: 20.5, z: 50.5 }).dh).toBe(-20)
  })
})
