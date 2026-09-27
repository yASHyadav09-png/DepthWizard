/**
 * Wall shading for image-draped terrain.
 *
 * The texture is a top-down photo: it has no pixels for vertical surfaces, so near-vertical
 * triangles (building walls, tree edges) stretch the few roof-edge pixels into long streaks.
 * Here steep faces get the LOCAL AVERAGE colour of the image instead (a coarse mipmap level of the
 * same texture), slightly darkened, so walls read as solid, shaded walls and roofs stay sharp.
 * Only the colour changes; geometry and heights are untouched.
 *
 *   wall weight w = 0 for surface slopes <= ~40 deg, 1 for >= ~66 deg (from the vertex normal's y).
 */
import * as THREE from 'three'

const NY_FLAT = 0.75 // cos(41 deg): below this steepness nothing changes
const NY_WALL = 0.4 // cos(66 deg): fully treated as wall
const WALL_LOD = 4.0 // mip level ~16 x 16 texels: local average colour
const WALL_DARKEN = 0.72

/** (Re)compute the per-vertex `wall` weight from the geometry's current normals. */
export function updateWallAttribute(geometry: THREE.BufferGeometry): void {
  const normal = geometry.getAttribute('normal') as THREE.BufferAttribute | undefined
  if (!normal) return
  const n = normal.count
  let attr = geometry.getAttribute('wall') as THREE.BufferAttribute | undefined
  if (!attr || attr.count !== n) {
    attr = new THREE.BufferAttribute(new Float32Array(n), 1)
    geometry.setAttribute('wall', attr)
  }
  const w = attr.array as Float32Array
  for (let i = 0; i < n; i += 1) {
    const ny = Math.abs(normal.getY(i))
    w[i] = Math.min(1, Math.max(0, (NY_FLAT - ny) / (NY_FLAT - NY_WALL)))
  }
  attr.needsUpdate = true
}

/** Patch a textured MeshStandardMaterial so steep faces use the local average colour. */
export function applyWallShading(material: THREE.MeshStandardMaterial): void {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float wall;\nvarying float vWall;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWall = wall;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vWall;')
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
#ifdef USE_MAP
  vec3 wallColour = textureLod( map, vMapUv, ${WALL_LOD.toFixed(1)} ).rgb * ${WALL_DARKEN.toFixed(2)};
  diffuseColor.rgb = mix( diffuseColor.rgb, wallColour, vWall );
#endif`,
      )
  }
  material.customProgramCacheKey = () => 'dw-wall-shading'
  material.needsUpdate = true
}
