import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useTexture } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import type { TerrainGrid, ViewMode } from '../types'
import { decodeHeights } from '../services/terrain'
import { sceneBase } from '../services/geo'

/** Elevation ramp used for the "elevation" view mode and its legend (low -> high). */
export const RAMP: [number, [number, number, number]][] = [
  [0.0, [0.10, 0.16, 0.36]],
  [0.2, [0.11, 0.38, 0.52]],
  [0.4, [0.13, 0.55, 0.42]],
  [0.6, [0.62, 0.73, 0.31]],
  [0.8, [0.86, 0.62, 0.32]],
  [1.0, [0.98, 0.98, 0.98]],
]

function rampColor(t: number, out: [number, number, number]) {
  const value = Math.min(1, Math.max(0, t))
  let i = 1
  while (i < RAMP.length - 1 && value > RAMP[i][0]) i += 1
  const [t0, c0] = RAMP[i - 1]
  const [t1, c1] = RAMP[i]
  const f = t1 === t0 ? 0 : (value - t0) / (t1 - t0)
  out[0] = c0[0] + (c1[0] - c0[0]) * f
  out[1] = c0[1] + (c1[1] - c0[1]) * f
  out[2] = c0[2] + (c1[2] - c0[2]) * f
}

/** What the cursor is over: the product value (height above ground, or elevation for a
 *  DSM job) in m, and the ground position (m from top-left). */
export interface HoverInfo {
  heightM: number
  xM: number
  yM: number
}

interface BuiltGeometry {
  geometry: THREE.BufferGeometry
  baseHeights: Float32Array // scene Y before exaggeration
  base: number
}

/**
 * Build the terrain surface in METRES: x/z span the ground footprint
 * (pixels x ground resolution) and y is the predicted height above ground, so
 * the scene has true proportions before any exaggeration. For DSM jobs (Phase 5)
 * y = elevation - base (a vertical shift; exaggeration then scales relief above base).
 *
 * Grid row 0 is the TOP of the source image. It is placed at -Z and given
 * v = 1 so the RGB texture (loaded with flipY) lands the right way up.
 */
function buildGeometry(grid: TerrainGrid): BuiltGeometry {
  const { width: w, height: h, plane_width: pw, plane_depth: pd } = grid
  const heights = decodeHeights(grid)
  const base = sceneBase(grid)
  const shifted = base === 0 ? heights : heights.map((v) => v - base)
  const range = Math.max(grid.display_max - grid.display_min, 1e-6)

  const count = w * h
  const positions = new Float32Array(count * 3)
  const uvs = new Float32Array(count * 2)
  const colors = new Float32Array(count * 3)
  const rgb: [number, number, number] = [0, 0, 0]

  for (let j = 0; j < h; j += 1) {
    const v = h === 1 ? 0 : j / (h - 1)
    for (let i = 0; i < w; i += 1) {
      const u = w === 1 ? 0 : i / (w - 1)
      const index = j * w + i

      positions[index * 3] = (u - 0.5) * pw
      positions[index * 3 + 1] = shifted[index]
      positions[index * 3 + 2] = (v - 0.5) * pd

      uvs[index * 2] = u
      uvs[index * 2 + 1] = 1 - v

      rampColor((heights[index] - grid.display_min) / range, rgb)
      colors[index * 3] = rgb[0]
      colors[index * 3 + 1] = rgb[1]
      colors[index * 3 + 2] = rgb[2]
    }
  }

  // Two triangles per cell, wound so the surface normal points up (+Y).
  const quads = (w - 1) * (h - 1)
  const indices = count > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6)
  let k = 0
  for (let j = 0; j < h - 1; j += 1) {
    for (let i = 0; i < w - 1; i += 1) {
      const a = j * w + i
      const b = a + 1
      const c = a + w
      const d = c + 1
      indices[k++] = a
      indices[k++] = c
      indices[k++] = b
      indices[k++] = b
      indices[k++] = c
      indices[k++] = d
    }
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.setIndex(new THREE.BufferAttribute(indices, 1))
  geometry.computeVertexNormals()

  return { geometry, baseHeights: shifted, base }
}

export function TerrainMesh({
  grid,
  textureUrl,
  viewMode,
  wireframe,
  exaggeration,
  onHover,
}: {
  grid: TerrainGrid
  textureUrl: string
  viewMode: ViewMode
  wireframe: boolean
  exaggeration: number
  onHover?: (info: HoverInfo | null) => void
}) {
  const texture = useTexture(textureUrl)
  const meshRef = useRef<THREE.Mesh>(null)

  const { geometry, baseHeights, base } = useMemo(() => buildGeometry(grid), [grid])

  // Dispose the previous surface when a new job is loaded.
  useEffect(() => () => geometry.dispose(), [geometry])

  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace
    texture.anisotropy = 8
    texture.needsUpdate = true
  }, [texture])

  // Re-displace instead of scaling the mesh: scaling would leave the normals
  // (and therefore the shading) wrong at high exaggeration.
  useEffect(() => {
    const position = geometry.getAttribute('position') as THREE.BufferAttribute
    const array = position.array as Float32Array
    for (let i = 0; i < baseHeights.length; i += 1) {
      array[i * 3 + 1] = baseHeights[i] * exaggeration
    }
    position.needsUpdate = true
    geometry.computeVertexNormals()
    geometry.computeBoundingSphere()
  }, [geometry, baseHeights, exaggeration])

  // Cursor readout: map the hit point back to the grid and report the TRUE
  // (un-exaggerated) predicted height of the nearest grid cell.
  const handleMove = (event: ThreeEvent<PointerEvent>) => {
    if (!onHover || !meshRef.current) return
    const local = meshRef.current.worldToLocal(event.point.clone())
    const u = local.x / grid.plane_width + 0.5
    const v = local.z / grid.plane_depth + 0.5
    if (u < 0 || u > 1 || v < 0 || v > 1) return onHover(null)
    const i = Math.round(u * (grid.width - 1))
    const j = Math.round(v * (grid.height - 1))
    onHover({
      heightM: baseHeights[j * grid.width + i] + base,
      xM: u * grid.plane_width,
      yM: v * grid.plane_depth,
    })
  }

  const textured = viewMode === 'textured'

  return (
    <group>
      <mesh
        ref={meshRef}
        geometry={geometry}
        castShadow
        receiveShadow
        onPointerMove={handleMove}
        onPointerOut={() => onHover?.(null)}
      >
        {/* keyed by mode: three.js does not recompile the shader when
            `vertexColors`/`map` change on an existing material */}
        <meshStandardMaterial
          key={viewMode}
          map={textured ? texture : null}
          vertexColors={!textured}
          roughness={textured ? 0.82 : 0.7}
          metalness={0.02}
          side={THREE.DoubleSide}
          flatShading={false}
        />
      </mesh>

      {wireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial color="#38bdf8" wireframe transparent opacity={0.12} depthTest />
        </mesh>
      )}
    </group>
  )
}
