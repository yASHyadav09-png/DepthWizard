/**
 * Fast pointer picking on large terrain meshes.
 *
 * Every pointer move over the 3D view raycasts the terrain (hover readouts, tool clicks). Plain
 * three.js tests every triangle: ~2 million for a 1024 x 1024 grid, which made hovering and orbiting
 * visibly laggy. A bounding-volume hierarchy (three-mesh-bvh) makes each ray test logarithmic.
 * Results are identical; only the search is faster.
 */
import type * as THREE from 'three'
import { acceleratedRaycast, MeshBVH } from 'three-mesh-bvh'

export { acceleratedRaycast }

type WithBvh = THREE.BufferGeometry & { boundsTree?: MeshBVH }

/** Build (or rebuild after vertices moved) the BVH used by `acceleratedRaycast`. */
export function buildBvh(geometry: THREE.BufferGeometry): void {
  ;(geometry as WithBvh).boundsTree = new MeshBVH(geometry)
}

/** Update the BVH bounds in place after vertex heights changed (same triangles). */
export function refitBvh(geometry: THREE.BufferGeometry): void {
  const g = geometry as WithBvh
  if (g.boundsTree) g.boundsTree.refit()
  else buildBvh(geometry)
}
