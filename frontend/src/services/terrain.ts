import type { TerrainGrid } from '../types'

/** Decode the base64 little-endian float32 height buffer sent by the backend. */
export function decodeHeights(grid: TerrainGrid): Float32Array {
  const binary = atob(grid.heights_b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)

  const expected = grid.width * grid.height
  const heights = new Float32Array(bytes.buffer, 0, Math.min(expected, bytes.length >> 2))
  if (heights.length !== expected) {
    throw new Error(
      `Terrain buffer has ${heights.length} samples but the grid declares ${expected}.`,
    )
  }
  return heights
}
