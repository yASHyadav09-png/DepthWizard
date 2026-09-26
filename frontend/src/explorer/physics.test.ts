import { describe, expect, it } from 'vitest'
import type { TerrainModel } from './terrainModel'
import { heightAt } from './terrainModel'
import { EYE_HEIGHT_M, FLY_CLEARANCE_M, constrainMove, settle } from './physics'

/** 100 x 100 m flat ground (1 m cells) with a 20 m tall "building" on x in [40, 60), z in [40, 60). */
function scene(): TerrainModel {
  const w = 100
  const h = 100
  const heights = new Float32Array(w * h)
  for (let j = 40; j < 60; j += 1) for (let i = 40; i < 60; i += 1) heights[j * w + i] = 20
  return { w, h, dx: 1, dz: 1, width: 100, depth: 100, heights, maxHeight: 20 }
}

describe('terrain model', () => {
  it('samples cell centres exactly and uses real metres', () => {
    const t = scene()
    expect(heightAt(t, 50.5, 50.5)).toBe(20) // cell (50, 50) centre
    expect(heightAt(t, 10.5, 10.5)).toBe(0)
    expect(heightAt(t, 39.5, 50.5)).toBe(0) // just outside the building
  })
  it('interpolates between cell centres', () => {
    const t = scene()
    expect(heightAt(t, 40, 50.5)).toBeCloseTo(10) // halfway between 39.5 (0 m) and 40.5 (20 m)
  })
  it('is bare ground (0 m) outside the imaged extent', () => {
    const t = scene()
    expect(heightAt(t, -5, 50)).toBe(0)
    expect(heightAt(t, 50, 150)).toBe(0)
  })
})

describe('fly mode', () => {
  it('never lets the camera below 2 m above the surface while descending', () => {
    const t = scene()
    const p = constrainMove(t, { x: 50.5, y: 60, z: 50.5 }, { x: 0, y: -100, z: 0 }, 'fly')
    expect(p.y).toBeCloseTo(20 + FLY_CLEARANCE_M)
    const g = constrainMove(t, { x: 10.5, y: 30, z: 10.5 }, { x: 0, y: -100, z: 0 }, 'fly')
    expect(g.y).toBeCloseTo(FLY_CLEARANCE_M)
  })
  it('refuses to fly into a wall below roof level instead of passing through', () => {
    const t = scene()
    const p = constrainMove(t, { x: 30, y: 5, z: 50.5 }, { x: 25, y: 0, z: 0 }, 'fly')
    expect(p.x).toBeLessThan(40) // stopped in front of the wall
    expect(p.y).toBe(5) // not popped onto the roof
  })
  it('cannot tunnel through a building even with a huge single step', () => {
    const t = scene()
    const p = constrainMove(t, { x: 30, y: 5, z: 50.5 }, { x: 500, y: 0, z: 0 }, 'fly')
    expect(p.x).toBeLessThan(40)
  })
  it('slides along a wall on the free axis', () => {
    const t = scene()
    const p = constrainMove(t, { x: 38, y: 5, z: 50.5 }, { x: 5, y: 0, z: 5 }, 'fly')
    expect(p.x).toBeLessThan(40)
    expect(p.z).toBeCloseTo(55.5)
  })
  it('flies freely above the roof', () => {
    const t = scene()
    const p = constrainMove(t, { x: 30, y: 25, z: 50.5 }, { x: 25, y: 0, z: 0 }, 'fly')
    expect(p.x).toBeCloseTo(55)
    expect(p.y).toBe(25)
  })
  it('glides forward along the surface when pitched down at the minimum clearance', () => {
    // regression: found live in the browser, W did nothing while looking slightly down at 2 m
    const t = scene()
    const p = constrainMove(t, { x: 10.5, y: 2, z: 10.5 }, { x: 5, y: -2, z: 0 }, 'fly')
    expect(p.x).toBeCloseTo(15.5)
    expect(p.y).toBeCloseTo(FLY_CLEARANCE_M)
  })
  it('follows a gentle rise when flying low, but not a wall', () => {
    const t = scene()
    for (let j = 0; j < 100; j += 1) for (let i = 0; i < 10; i += 1) t.heights[j * 100 + i] = i * 0.5 // 27 deg ramp
    const p = constrainMove(t, { x: 0.5, y: 2, z: 5.5 }, { x: 8, y: 0, z: 0 }, 'fly')
    expect(p.x).toBeCloseTo(8.5)
    expect(p.y).toBeCloseTo(heightAt(t, 8.5, 5.5) + FLY_CLEARANCE_M)
    const w = constrainMove(t, { x: 30, y: 2, z: 50.5 }, { x: 25, y: 0, z: 0 }, 'fly')
    expect(w.x).toBeLessThan(40)
    for (let k = 0; k < 5; k += 1) expect(w.y).toBeGreaterThanOrEqual(heightAt(t, w.x, w.z) + FLY_CLEARANCE_M - 1e-9)
  })
  it('never ends a move below the clearance anywhere (random stress)', () => {
    const t = scene()
    let s = 1
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1
    let p = { x: 20, y: 10, z: 20 }
    for (let k = 0; k < 2000; k += 1) {
      p = constrainMove(t, p, { x: rnd() * 8, y: rnd() * 8, z: rnd() * 8 }, 'fly')
      expect(p.y).toBeGreaterThanOrEqual(heightAt(t, p.x, p.z) + FLY_CLEARANCE_M - 1e-9)
    }
  })
  it('settle lifts an illegal camera to the clearance', () => {
    expect(settle(scene(), { x: 50.5, y: 3, z: 50.5 }, 'fly').y).toBeCloseTo(22)
  })
})

describe('walk mode', () => {
  it('keeps the eye exactly 1.7 m above the ground', () => {
    const t = scene()
    const p = constrainMove(t, { x: 10.5, y: 50, z: 10.5 }, { x: 3, y: 0, z: 2 }, 'walk')
    expect(p.y).toBeCloseTo(EYE_HEIGHT_M)
    expect(p.x).toBeCloseTo(13.5)
    expect(p.z).toBeCloseTo(12.5)
  })
  it('ignores vertical input', () => {
    const p = constrainMove(scene(), { x: 10.5, y: 1.7, z: 10.5 }, { x: 0, y: 30, z: 0 }, 'walk')
    expect(p.y).toBeCloseTo(EYE_HEIGHT_M)
  })
  it('cannot walk into a building wall', () => {
    const p = constrainMove(scene(), { x: 30, y: 1.7, z: 50.5 }, { x: 25, y: 0, z: 0 }, 'walk')
    expect(p.x).toBeLessThan(40)
    expect(p.y).toBeCloseTo(EYE_HEIGHT_M)
  })
  it('walks off a roof edge and follows the ground down', () => {
    const p = constrainMove(scene(), { x: 55, y: 21.7, z: 50.5 }, { x: 10, y: 0, z: 0 }, 'walk')
    expect(p.x).toBeCloseTo(65)
    expect(p.y).toBeCloseTo(EYE_HEIGHT_M)
  })
  it('climbs gentle slopes', () => {
    const t = scene()
    // a 10 m ramp rising 5 m (27 degrees) on x in [0, 10)
    for (let j = 0; j < 100; j += 1) for (let i = 0; i < 10; i += 1) t.heights[j * 100 + i] = i * 0.5
    const p = constrainMove(t, { x: 0.5, y: 1.7, z: 5.5 }, { x: 8, y: 0, z: 0 }, 'walk')
    expect(p.x).toBeCloseTo(8.5)
    expect(p.y).toBeCloseTo(heightAt(t, 8.5, 5.5) + EYE_HEIGHT_M)
  })
})
