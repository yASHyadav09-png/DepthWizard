import { Suspense, useEffect, useRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { FlyControls, Grid, Html, OrbitControls } from '@react-three/drei'
import type { ProcessResult, ViewMode } from '../types'
import { assetUrl } from '../services/api'
import { TerrainMesh } from './TerrainMesh'
import { Badge } from './ui'

/** The mesh is built at unit scale; the scene scales it once so camera
 *  distances and fly speeds are the same for every image. */
const WORLD_SCALE = 4
const CAMERA_HOME: [number, number, number] = [1.7, 2.5, 2.9]
const CAMERA_TARGET: [number, number, number] = [0, 0.35, 0]

export type NavMode = 'orbit' | 'fly'

/**
 * Actually moves the camera home when `resetSignal` changes.
 *
 * Re-mounting the controls is not enough: OrbitControls derives its spherical
 * coordinates *from* the camera on mount, so it happily adopts whatever pose
 * the user left behind. FlyControls has no target at all. So the reset is done
 * on the camera itself, and the controls are re-targeted afterwards.
 */
function CameraRig({ resetSignal }: { resetSignal: number }) {
  const camera = useThree((state) => state.camera)
  const controls = useThree((state) => state.controls) as
    | { target?: { set: (x: number, y: number, z: number) => void }; update?: () => void }
    | null

  // Held in a ref so that swapping Orbit <-> Flythrough (which swaps the
  // default controls) does not itself count as a reset — switching mode should
  // keep whatever the user is currently looking at.
  const controlsRef = useRef(controls)
  controlsRef.current = controls

  useEffect(() => {
    camera.position.set(...CAMERA_HOME)
    camera.up.set(0, 1, 0)
    camera.lookAt(...CAMERA_TARGET)
    camera.updateProjectionMatrix()
    controlsRef.current?.target?.set(...CAMERA_TARGET)
    controlsRef.current?.update?.()
  }, [resetSignal, camera])

  return null
}

function Loader({ label }: { label: string }) {
  return (
    <Html center>
      <div className="flex items-center gap-2 rounded-lg border border-slate-400/20 bg-abyss-900/90 px-3 py-2">
        <span className="size-3 animate-spin rounded-full border-2 border-signal-400 border-t-transparent" />
        <span className="font-mono text-[11px] text-slate-300">{label}</span>
      </div>
    </Html>
  )
}

function EmptyState() {
  return (
    <div className="grid size-full place-items-center px-6 text-center">
      <div>
        <svg
          viewBox="0 0 64 44"
          className="mx-auto w-32 fill-none stroke-slate-700"
          strokeWidth={1.1}
        >
          <path d="M2 30 32 12l30 18-30 12z" />
          <path d="M2 30 32 42M32 12v30M62 30 32 42" opacity={0.6} />
          <path d="M12 24 32 36 52 24" opacity={0.4} />
          <path d="M22 18 32 24 42 18" opacity={0.3} />
        </svg>
        <p className="mt-4 text-sm font-medium text-slate-400">
          No terrain generated yet
        </p>
        <p className="mt-1 max-w-sm text-xs text-slate-600">
          Upload a JPG or PNG and run the pipeline. The relative DSM is meshed into an
          interactive 3D surface with the original image projected as texture.
        </p>
      </div>
    </div>
  )
}

export function TerrainViewer({
  result,
  viewMode,
  wireframe,
  exaggeration,
  navMode,
  resetSignal,
}: {
  result: ProcessResult | null
  viewMode: ViewMode
  wireframe: boolean
  exaggeration: number
  navMode: NavMode
  resetSignal: number
}) {
  if (!result) return <EmptyState />

  return (
    <div className="relative size-full">
      <Canvas
        key={result.job_id}
        shadows
        // `flat` = no ACES tone mapping: the projected RGB texture should read
        // as the photograph, not as a filmic grade of it.
        flat
        dpr={[1, 2]}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ position: CAMERA_HOME, fov: 50, near: 0.01, far: 200 }}
      >
        <color attach="background" args={['#070b12']} />
        <fog attach="fog" args={['#070b12', 9, 26]} />

        {/* Key light matches the 315° azimuth used by the 2D hillshade. */}
        <directionalLight
          position={[-6, 8, -6]}
          intensity={1.7}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <directionalLight position={[6, 5, 6]} intensity={0.55} color="#7dd3fc" />
        <hemisphereLight args={['#bfdbfe', '#0b1220', 0.7]} />
        <ambientLight intensity={0.55} />

        <Suspense fallback={<Loader label="Building terrain mesh…" />}>
          <group scale={WORLD_SCALE}>
            <TerrainMesh
              grid={result.terrain}
              textureUrl={assetUrl(result.assets.texture)}
              viewMode={viewMode}
              wireframe={wireframe}
              exaggeration={exaggeration}
            />
          </group>
        </Suspense>

        <CameraRig resetSignal={resetSignal} />

        <Grid
          position={[0, -0.02, 0]}
          args={[40, 40]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#16202f"
          sectionSize={2.5}
          sectionThickness={0.8}
          sectionColor="#14415a"
          fadeDistance={17}
          fadeStrength={1.8}
          infiniteGrid
        />

        {navMode === 'orbit' ? (
          <OrbitControls
            makeDefault
            enablePan
            enableZoom
            enableRotate
            enableDamping
            dampingFactor={0.08}
            minDistance={0.6}
            maxDistance={22}
            maxPolarAngle={Math.PI / 2.02}
            target={CAMERA_TARGET}
          />
        ) : (
          <FlyControls
            makeDefault
            movementSpeed={2.2}
            rollSpeed={0.5}
            dragToLook
          />
        )}
      </Canvas>

      <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap gap-2">
        <Badge tone="warn">Relative Height — Not Metric</Badge>
        <Badge tone="neutral">
          {result.terrain.width}×{result.terrain.height} grid
        </Badge>
      </div>

      <div className="pointer-events-none absolute right-3 bottom-3 rounded-lg border border-slate-400/12 bg-abyss-950/80 px-2.5 py-1.5 font-mono text-[10px] leading-relaxed text-slate-500 backdrop-blur">
        {navMode === 'orbit' ? (
          <>
            drag <span className="text-slate-300">rotate</span> · scroll{' '}
            <span className="text-slate-300">zoom</span> · right-drag{' '}
            <span className="text-slate-300">pan</span>
          </>
        ) : (
          <>
            <span className="text-slate-300">W A S D</span> move ·{' '}
            <span className="text-slate-300">R / F</span> up-down · drag{' '}
            <span className="text-slate-300">look</span>
          </>
        )}
      </div>
    </div>
  )
}
