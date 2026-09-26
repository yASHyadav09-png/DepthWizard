import { describe, expect, it } from 'vitest'
import type { TerrainModel } from './terrainModel'
import { cellValueAt, slopeField } from './terrainModel'

function plane(w: number, h: number, dx: number, dz: number, f: (x: number, z: number) => number): TerrainModel {
  const heights = new Float32Array(w * h)
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) heights[j * w + i] = f((i + 0.5) * dx, (j + 0.5) * dz)
  return { w, h, dx, dz, width: w * dx, depth: h * dz, heights, maxHeight: Math.max(...heights) }
}

describe('slope field', () => {
  it('is 0 degrees on flat ground', () => {
    const s = slopeField(plane(20, 20, 0.66, 0.66, () => 3))
    expect(Math.max(...s)).toBeCloseTo(0)
  })
  it('recovers a 30 degree plane along X in real metres (non-unit cell size)', () => {
    const tan30 = Math.tan(Math.PI / 6)
    const s = slopeField(plane(20, 20, 0.66, 0.66, (x) => x * tan30))
    for (const v of s) expect(v).toBeCloseTo(30, 4)
  })
  it('combines both axes (45 degrees along the diagonal gradient)', () => {
    const g = Math.SQRT1_2 // |grad| = 1 -> 45 deg
    const s = slopeField(plane(20, 20, 1, 1, (x, z) => g * x + g * z))
    expect(s[10 * 20 + 10]).toBeCloseTo(45, 4)
  })
  it('respects anisotropic cell spacing', () => {
    const s = slopeField(plane(20, 20, 2, 0.5, (_x, z) => z)) // 45 deg along Z
    expect(s[10 * 20 + 10]).toBeCloseTo(45, 4)
  })
  it('looks up the containing cell and is null outside the extent', () => {
    const t = plane(4, 4, 1, 1, (x) => x)
    const s = slopeField(t)
    expect(cellValueAt(t, s, 2.2, 1.7)).toBeCloseTo(45)
    expect(cellValueAt(t, s, -1, 1)).toBeNull()
  })
})
