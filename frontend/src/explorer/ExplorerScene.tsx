/**
 * Everything inside the Explorer's <Canvas>. The scene is in real metres
 * (see terrainModel.ts); nothing is scaled.
 */
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import { Line, OrbitControls, PointerLockControls, useTexture } from '@react-three/drei'
import type { TerrainModel } from './terrainModel'
import { aboveGroundAt, buildTerrainGeometry, cellValueAt, colourAttribute, heightAt } from './terrainModel'
import { constrainMove, settle } from './physics'
import type { MoveMode } from './physics'
import { errorColor, rampColor, slopeColor } from './ramp'
import { applyWallShading, updateWallAttribute } from '../components/wallShading'
import { acceleratedRaycast, buildBvh } from '../components/fastRaycast'
import type { GroundPoint } from './measure'
import { sampleProfile } from './measure'

export type NavMode = 'orbit' | MoveMode
export type Layer = 'rgb' | 'height' | 'elevation' | 'slope' | 'error'
export type Tool = 'none' | 'profile' | 'measure'

/** Mutable telemetry written by the scene every frame and read by the DOM HUD/minimap. */
export interface Telemetry {
  camera: { x: number; y: number; z: number }
  dir: { x: number; y: number; z: number }
  groundBelow: number
  /** height = scene Y of the surface (elevation - base for DSM jobs); aboveGround = nDSM there. */
  cursor: {
    x: number
    z: number
    height: number
    aboveGround: number | null
    slope: number | null
    error: number | null
    distance: number
  } | null
  pointerLocked: boolean
}

export function newTelemetry(): Telemetry {
  return {
    camera: { x: 0, y: 0, z: 0 },
    dir: { x: 0, y: 0, z: -1 },
    groundBelow: 0,
    cursor: null,
    pointerLocked: false,
  }
}

const finiteOrNull = (v: number | null) => (v === null || !Number.isFinite(v) ? null : v)

const KEY_BINDINGS: Record<string, string> = {
  KeyW: 'forward', ArrowUp: 'forward', KeyS: 'back', ArrowDown: 'back',
  KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right',
  Space: 'up', KeyC: 'down', ShiftLeft: 'fast', ShiftRight: 'fast',
}

function useMovementKeys(enabled: boolean) {
  const keys = useRef<Record<string, boolean>>({})
  useEffect(() => {
    if (!enabled) return
    const set = (down: boolean) => (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
      const action = KEY_BINDINGS[e.code]
      if (!action) return
      keys.current[action] = down
      if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault()
    }
    const down = set(true)
    const up = set(false)
    const blur = () => (keys.current = {})
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
      keys.current = {}
    }
  }, [enabled])
  return keys
}

function Terrain({
  terrain,
  slope,
  textureUrl,
  layer,
  heightMax,
  elevRange,
  meshRef,
  telemetry,
  onPick,
  errorField,
  errorLimit,
}: {
  terrain: TerrainModel
  slope: Float32Array
  textureUrl: string
  layer: Layer
  heightMax: number
  elevRange: [number, number]
  meshRef: React.RefObject<THREE.Mesh | null>
  telemetry: Telemetry
  onPick?: (p: GroundPoint) => void
  errorField: Float32Array | null
  errorLimit: number
}) {
  const texture = useTexture(textureUrl)
  const maxAnisotropy = useThree((s) => s.gl.capabilities.getMaxAnisotropy())
  const geometry = useMemo(() => {
    const g = buildTerrainGeometry(terrain)
    updateWallAttribute(g)
    buildBvh(g) // fast picking for hover readouts, tool clicks and the fly/walk target ray
    return g
  }, [terrain])
  useEffect(() => {
    geometry.setAttribute(
      'color',
      layer === 'slope'
        ? colourAttribute(slope, 0, 90, slopeColor)
        : layer === 'error' && errorField
          ? colourAttribute(errorField, -errorLimit, errorLimit, errorColor)
          : layer === 'elevation'
            ? colourAttribute(terrain.heights, elevRange[0], elevRange[1], rampColor)
            : colourAttribute(terrain.aboveGround ?? terrain.heights, 0, heightMax, rampColor),
    )
  }, [geometry, terrain, slope, layer, heightMax, elevRange, errorField, errorLimit])
  useEffect(() => () => geometry.dispose(), [geometry])
  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace
    texture.anisotropy = maxAnisotropy
    texture.needsUpdate = true
  }, [texture, maxAnisotropy])

  const textured = layer === 'rgb'
  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      raycast={acceleratedRaycast}
      receiveShadow
      onPointerMove={(e) => {
        if (telemetry.pointerLocked) return // fly/walk use the screen-centre ray instead
        telemetry.cursor = {
          x: e.point.x,
          z: e.point.z,
          height: heightAt(terrain, e.point.x, e.point.z),
          aboveGround: aboveGroundAt(terrain, e.point.x, e.point.z),
          slope: cellValueAt(terrain, slope, e.point.x, e.point.z),
          error: errorField ? finiteOrNull(cellValueAt(terrain, errorField, e.point.x, e.point.z)) : null,
          distance: e.distance,
        }
      }}
      onPointerOut={() => {
        if (!telemetry.pointerLocked) telemetry.cursor = null
      }}
      onClick={(e) => {
        // a click, not the end of an orbit drag (R3F: e.delta = pixels moved since pointerdown)
        if (!onPick || e.delta > 4) return
        e.stopPropagation()
        onPick({ x: e.point.x, z: e.point.z })
      }}
    >
      {/* keyed: three.js does not recompile a material when map/vertexColors change */}
      <meshStandardMaterial
        key={layer}
        ref={(m: THREE.MeshStandardMaterial | null) => {
          if (m && textured) applyWallShading(m)
        }}
        map={textured ? texture : null}
        vertexColors={!textured}
        roughness={0.85}
        metalness={0.02}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

/** Fly / walk movement, integrated per frame with the terrain constraint. */
function FirstPerson({
  terrain,
  mode,
  speed,
}: {
  terrain: TerrainModel
  mode: MoveMode
  speed: number
}) {
  const camera = useThree((s) => s.camera)
  const keys = useMovementKeys(true)
  const forward = useMemo(() => new THREE.Vector3(), [])
  const right = useMemo(() => new THREE.Vector3(), [])

  // entering the mode: make the current pose legal (e.g. walk -> snap to eye height)
  useEffect(() => {
    const p = settle(terrain, camera.position, mode)
    camera.position.set(p.x, p.y, p.z)
  }, [mode, terrain, camera])

  useFrame((_, rawDelta) => {
    const dt = Math.min(rawDelta, 0.1)
    const k = keys.current
    camera.getWorldDirection(forward)
    if (mode === 'walk') forward.y = 0
    forward.normalize()
    right.crossVectors(forward, camera.up).normalize()
    const v = new THREE.Vector3()
    if (k.forward) v.add(forward)
    if (k.back) v.sub(forward)
    if (k.right) v.add(right)
    if (k.left) v.sub(right)
    if (mode === 'fly') {
      if (k.up) v.y += 1
      if (k.down) v.y -= 1
    }
    if (v.lengthSq() === 0) return
    v.normalize().multiplyScalar(speed * (k.fast ? (mode === 'walk' ? 3 : 4) : 1) * dt) // metres this frame
    const p = constrainMove(terrain, camera.position, { x: v.x, y: v.y, z: v.z }, mode)
    camera.position.set(p.x, p.y, p.z)
  })
  return null
}

/** Keeps the orbit camera legal and writes telemetry; raycasts the screen centre when locked. */
function Telemetrist({
  terrain,
  slope,
  errorField,
  mode,
  telemetry,
  meshRef,
}: {
  terrain: TerrainModel
  slope: Float32Array
  errorField: Float32Array | null
  mode: NavMode
  telemetry: Telemetry
  meshRef: React.RefObject<THREE.Mesh | null>
}) {
  const camera = useThree((s) => s.camera)
  const raycaster = useMemo(() => new THREE.Raycaster(), [])
  const dir = useMemo(() => new THREE.Vector3(), [])
  const frame = useRef(0)
  useFrame(() => {
    if (mode === 'orbit') {
      // orbiting may never take the camera into the surface
      const p = settle(terrain, camera.position, 'fly')
      if (p.y !== camera.position.y) camera.position.y = p.y
    }
    camera.getWorldDirection(dir)
    telemetry.camera = { x: camera.position.x, y: camera.position.y, z: camera.position.z }
    telemetry.dir = { x: dir.x, y: dir.y, z: dir.z }
    telemetry.groundBelow = heightAt(terrain, camera.position.x, camera.position.z)
    frame.current += 1
    if (telemetry.pointerLocked && meshRef.current && frame.current % 6 === 0) {
      raycaster.setFromCamera(new THREE.Vector2(0, 0), camera)
      const hit = raycaster.intersectObject(meshRef.current, false)[0]
      telemetry.cursor = hit
        ? {
            x: hit.point.x,
            z: hit.point.z,
            height: heightAt(terrain, hit.point.x, hit.point.z),
            aboveGround: aboveGroundAt(terrain, hit.point.x, hit.point.z),
            slope: cellValueAt(terrain, slope, hit.point.x, hit.point.z),
            error: errorField ? finiteOrNull(cellValueAt(terrain, errorField, hit.point.x, hit.point.z)) : null,
            distance: hit.distance,
          }
        : null
    }
  })
  return null
}

/** Picked points (A, B) and the A-B line draped 0.4 m above the surface. */
function Picks({ terrain, picks }: { terrain: TerrainModel; picks: GroundPoint[] }) {
  const line = useMemo(() => {
    if (picks.length < 2) return null
    return sampleProfile(terrain, picks[0], picks[1]).samples.map(
      (s) => [s.x, s.h + 0.4, s.z] as [number, number, number],
    )
  }, [terrain, picks])
  const size = Math.max(terrain.width, terrain.depth)
  const r = Math.max(0.6, size / 400)
  return (
    <group>
      {picks.map((p, k) => (
        <mesh key={k} position={[p.x, heightAt(terrain, p.x, p.z) + r, p.z]}>
          <sphereGeometry args={[r, 16, 12]} />
          <meshBasicMaterial color={k === 0 ? '#fbbf24' : '#f472b6'} depthTest={false} />
        </mesh>
      ))}
      {line && <Line points={line} color="#fbbf24" lineWidth={2.5} depthTest={false} />}
    </group>
  )
}

export function ExplorerScene({
  terrain,
  slope,
  textureUrl,
  layer,
  heightMax,
  elevRange,
  mode,
  speed,
  telemetry,
  resetSignal,
  tool,
  picks,
  onPick,
  errorField,
  errorLimit,
}: {
  tool: Tool
  picks: GroundPoint[]
  onPick: (p: GroundPoint) => void
  errorField: Float32Array | null
  errorLimit: number
  terrain: TerrainModel
  slope: Float32Array
  textureUrl: string
  layer: Layer
  heightMax: number
  elevRange: [number, number]
  mode: NavMode
  speed: number
  telemetry: Telemetry
  resetSignal: number
}) {
  const camera = useThree((s) => s.camera)
  const meshRef = useRef<THREE.Mesh>(null)
  const size = Math.max(terrain.width, terrain.depth)
  const centre = useMemo(() => new THREE.Vector3(terrain.width / 2, 0, terrain.depth / 2), [terrain])

  // home view: south-west of the scene, looking at its centre
  useEffect(() => {
    camera.position.set(centre.x - 0.45 * size, Math.max(0.55 * size, terrain.maxHeight + 30), centre.z + 0.85 * size)
    camera.up.set(0, 1, 0)
    camera.lookAt(centre)
    ;(camera as THREE.PerspectiveCamera).near = 0.1
    ;(camera as THREE.PerspectiveCamera).far = size * 20
    camera.updateProjectionMatrix()
  }, [resetSignal, camera, centre, size, terrain.maxHeight])

  return (
    <>
      <color attach="background" args={['#0b1220']} />
      <fog attach="fog" args={['#0b1220', size * 1.2, size * 6]} />
      <hemisphereLight args={['#dbeafe', '#1e293b', 0.9]} />
      <directionalLight position={[-size, size * 1.2, -size]} intensity={1.6} />
      <directionalLight position={[size, size * 0.6, size]} intensity={0.45} color="#93c5fd" />
      <ambientLight intensity={0.35} />

      <Terrain
        terrain={terrain}
        slope={slope}
        textureUrl={textureUrl}
        layer={layer}
        heightMax={heightMax}
        elevRange={elevRange}
        meshRef={meshRef}
        telemetry={telemetry}
        onPick={tool !== 'none' ? onPick : undefined}
        errorField={errorField}
        errorLimit={errorLimit}
      />
      <Picks terrain={terrain} picks={picks} />
      {/* ground around the imaged extent at Y = 0, as heightAt assumes (0 m nDSM, or the base elevation of a DSM) */}
      <mesh rotation-x={-Math.PI / 2} position={[centre.x, -0.05, centre.z]}>
        <planeGeometry args={[size * 12, size * 12]} />
        <meshStandardMaterial color="#1e293b" roughness={1} />
      </mesh>

      {mode === 'orbit' ? (
        <OrbitControls
          makeDefault
          target={centre}
          enableDamping
          dampingFactor={0.08}
          minDistance={5}
          maxDistance={size * 4}
          maxPolarAngle={Math.PI / 2.05}
        />
      ) : (
        <>
          <PointerLockControls
            makeDefault
            selector="#dw-explorer-canvas"
            onLock={() => (telemetry.pointerLocked = true)}
            onUnlock={() => {
              telemetry.pointerLocked = false
              telemetry.cursor = null
            }}
          />
          <FirstPerson terrain={terrain} mode={mode} speed={speed} />
        </>
      )}
      <Telemetrist terrain={terrain} slope={slope} errorField={errorField} mode={mode} telemetry={telemetry} meshRef={meshRef} />
    </>
  )
}
