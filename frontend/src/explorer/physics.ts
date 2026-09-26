/**
 * Camera movement constrained by the terrain (real metres).
 *
 * Movement is integrated in small sub-steps (<= SUBSTEP_M) so a fast camera can
 * never tunnel through a thin wall between two frames. Each sub-step is checked
 * against the terrain BEFORE it is applied; the camera is never allowed inside the
 * surface and pushed out afterwards.
 *
 *   Fly  : free 3D motion; the camera must stay >= FLY_CLEARANCE_M above the surface.
 *          A horizontal step into a structure taller than the camera is refused
 *          (the camera slides along the wall on the free axis), so you fly up to
 *          get over a building rather than popping onto its roof.
 *   Walk : horizontal motion; the eye is kept exactly EYE_HEIGHT_M above the surface.
 *          Steps up of more than MAX_STEP_UP_M per sub-step are refused (walls);
 *          drops are allowed (you walk off a roof edge).
 */
import type { TerrainModel } from './terrainModel'
import { heightAt } from './terrainModel'

export const FLY_CLEARANCE_M = 2.0
export const EYE_HEIGHT_M = 1.7
export const SUBSTEP_M = 0.25
export const MAX_STEP_UP_M = 0.5

export type MoveMode = 'fly' | 'walk'

export interface Vec3 {
  x: number
  y: number
  z: number
}

export function minFlyY(t: TerrainModel, x: number, z: number): number {
  return heightAt(t, x, z) + FLY_CLEARANCE_M
}

function flySubstep(t: TerrainModel, p: Vec3, d: Vec3): Vec3 {
  const full = { x: p.x + d.x, y: p.y + d.y, z: p.z + d.z }
  if (full.y >= minFlyY(t, full.x, full.z)) return full
  // The step would go below the clearance. Keep as much of the horizontal motion as
  // possible (full, then each axis alone = sliding along a wall) as long as the surface
  // there does not rise more than MAX_STEP_UP_M above the camera's CURRENT height; the
  // camera then rides at the clearance. A steeper rise is a wall and blocks that axis.
  const candidates: [number, number][] = [
    [p.x + d.x, p.z + d.z],
    [p.x + d.x, p.z],
    [p.x, p.z + d.z],
  ]
  for (const [x, z] of candidates) {
    const floor = minFlyY(t, x, z)
    if (floor <= p.y + MAX_STEP_UP_M) return { x, y: Math.max(full.y, floor), z }
  }
  return { x: p.x, y: Math.max(full.y, minFlyY(t, p.x, p.z)), z: p.z }
}

function walkSubstep(t: TerrainModel, p: Vec3, d: Vec3): Vec3 {
  const ground = heightAt(t, p.x, p.z)
  const candidates: [number, number][] = [
    [p.x + d.x, p.z + d.z],
    [p.x + d.x, p.z],
    [p.x, p.z + d.z],
  ]
  for (const [x, z] of candidates) {
    const g = heightAt(t, x, z)
    if (g - ground <= MAX_STEP_UP_M) return { x, y: g + EYE_HEIGHT_M, z }
  }
  return { x: p.x, y: ground + EYE_HEIGHT_M, z: p.z }
}

/** Move `p` by `delta` (m) under the terrain constraint of `mode`. Returns the new position. */
export function constrainMove(t: TerrainModel, p: Vec3, delta: Vec3, mode: MoveMode): Vec3 {
  const d = mode === 'walk' ? { x: delta.x, y: 0, z: delta.z } : delta
  const len = Math.hypot(d.x, d.y, d.z)
  const steps = Math.max(1, Math.ceil(len / SUBSTEP_M))
  const s = { x: d.x / steps, y: d.y / steps, z: d.z / steps }
  let q = mode === 'walk' ? settle(t, p, 'walk') : settle(t, p, 'fly')
  for (let k = 0; k < steps; k += 1) {
    q = mode === 'walk' ? walkSubstep(t, q, s) : flySubstep(t, q, s)
  }
  return q
}

/** Put a camera into a legal state for `mode` without moving it horizontally. */
export function settle(t: TerrainModel, p: Vec3, mode: MoveMode): Vec3 {
  const g = heightAt(t, p.x, p.z)
  if (mode === 'walk') return { x: p.x, y: g + EYE_HEIGHT_M, z: p.z }
  return { x: p.x, y: Math.max(p.y, g + FLY_CLEARANCE_M), z: p.z }
}
