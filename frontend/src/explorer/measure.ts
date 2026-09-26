/**
 * Profile and measurement maths on the real-metre terrain (Phase 7b).
 * Points are ground positions (x, z) in metres; heights come from the terrain.
 */
import type { TerrainModel } from './terrainModel'
import { heightAt } from './terrainModel'

export interface GroundPoint {
  x: number
  z: number
}

export interface ProfileSample {
  d: number // distance along the line from A (m)
  x: number
  z: number
  h: number // surface height above ground at this point (m)
}

export interface Profile {
  samples: ProfileSample[]
  length: number // horizontal length A -> B (m)
  minH: number
  maxH: number
  maxRise: number // steepest local gradient along the line, in degrees (between consecutive samples)
}

/** Heights along A -> B, sampled every `step` metres (default: half a grid cell, never coarser). */
export function sampleProfile(t: TerrainModel, a: GroundPoint, b: GroundPoint, step?: number): Profile {
  const length = Math.hypot(b.x - a.x, b.z - a.z)
  const ds = step ?? Math.min(t.dx, t.dz) / 2
  const n = Math.max(1, Math.ceil(length / ds))
  const samples: ProfileSample[] = []
  for (let k = 0; k <= n; k += 1) {
    const f = k / n
    const x = a.x + (b.x - a.x) * f
    const z = a.z + (b.z - a.z) * f
    samples.push({ d: length * f, x, z, h: heightAt(t, x, z) })
  }
  let minH = Infinity
  let maxH = -Infinity
  let maxRise = 0
  samples.forEach((s, k) => {
    minH = Math.min(minH, s.h)
    maxH = Math.max(maxH, s.h)
    if (k > 0) {
      const dd = s.d - samples[k - 1].d
      if (dd > 0) maxRise = Math.max(maxRise, (Math.atan(Math.abs(s.h - samples[k - 1].h) / dd) * 180) / Math.PI)
    }
  })
  return { samples, length, minH, maxH, maxRise }
}

export interface Measurement {
  horizontal: number // ground distance A -> B (m)
  dh: number // surface height at B minus surface height at A (m)
  slopeDistance: number // straight 3D distance between the two surface points (m)
  hA: number
  hB: number
}

export function measure(t: TerrainModel, a: GroundPoint, b: GroundPoint): Measurement {
  const hA = heightAt(t, a.x, a.z)
  const hB = heightAt(t, b.x, b.z)
  const horizontal = Math.hypot(b.x - a.x, b.z - a.z)
  return { horizontal, dh: hB - hA, slopeDistance: Math.hypot(horizontal, hB - hA), hA, hB }
}
