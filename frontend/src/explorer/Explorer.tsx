/**
 * Full-screen 3D Explorer (Phase 7a). Entered from the dashboard once a result exists.
 * Real-metre scene (terrainModel.ts), Orbit / Fly / Walk navigation constrained by the
 * terrain (physics.ts), minimap, status HUD and model/GSD status.
 * Phase 5: georeferenced DSM jobs show elevation (EGM2008), height above ground,
 * easting/northing and lon/lat; GeoTIFF exports; validation against GeoTIFF references.
 */
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { useProgress } from '@react-three/drei'
import type { ProcessResult, ValidationResult } from '../types'
import { ApiError, assetUrl, decodeFloat32, validateJob } from '../services/api'
import type { ValidationTarget } from '../services/api'
import { datumLabel, formatGsd, formatLonLat, isDsm, mapCoords } from '../services/geo'
import { Badge } from '../components/ui'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { ExplorerScene, newTelemetry } from './ExplorerScene'
import type { Layer, NavMode, Tool } from './ExplorerScene'
import type { GroundPoint } from './measure'
import { measure, sampleProfile } from './measure'
import { MeasureCard, ProfileChart } from './ToolPanels'
import { Minimap } from './Minimap'
import { makeTerrainModel, slopeField } from './terrainModel'
import { EYE_HEIGHT_M, FLY_CLEARANCE_M } from './physics'
import { errorRampCss, rampCss, slopeRampCss } from './ramp'

const LAYERS_NDSM: Layer[] = ['rgb', 'height', 'slope', 'error']
const LAYERS_DSM: Layer[] = ['rgb', 'elevation', 'height', 'slope', 'error']

const FLY_SPEEDS = [2, 5, 10, 15, 25, 40, 60, 100, 150] // m/s
const WALK_SPEEDS = [0.8, 1.4, 2.5, 4] // m/s

const MODES: { key: NavMode; label: string; icon: string; hint: string }[] = [
  { key: 'orbit', label: 'Orbit', icon: '⟳', hint: 'Overview: drag to rotate, scroll to zoom, right-drag to pan' },
  { key: 'fly', label: 'Fly', icon: '✈', hint: 'First-person flight, stays ≥ 2 m above the surface' },
  { key: 'walk', label: 'Walk', icon: '🚶', hint: 'Ground level, eye height 1.7 m, follows the terrain' },
]

function Loader() {
  const { active, progress } = useProgress()
  if (!active) return null
  return (
    <div className="pointer-events-none absolute inset-0 grid place-items-center">
      <div className="rounded-lg border border-slate-400/20 bg-abyss-900/90 px-3 py-2 font-mono text-[11px] text-slate-300">
        Loading terrain… {Math.round(progress)}%
      </div>
    </div>
  )
}

export function Explorer({ result, onExit }: { result: ProcessResult; onExit: () => void }) {
  const grid = result.terrain
  const hp = result.height_product
  const terrain = useMemo(() => makeTerrainModel(grid), [grid])
  const slope = useMemo(() => slopeField(terrain), [terrain])
  const dsm = isDsm(grid)
  const base = terrain.base ?? 0
  const geo = Boolean(hp.transform)
  // colour ranges in scene units: height above ground from the nDSM statistics,
  // elevation from the DSM display range shifted by the scene base
  const heightMax = dsm ? result.statistics.display_max : grid.display_max
  const elevRange = useMemo<[number, number]>(
    () => [grid.display_min - base, grid.display_max - base],
    [grid.display_min, grid.display_max, base],
  )
  const [validationTarget, setValidationTarget] = useState<ValidationTarget>('auto')
  const telemetry = useMemo(() => newTelemetry(), [])
  const [mode, setMode] = useState<NavMode>('orbit')
  const [layer, setLayer] = useState<Layer>('rgb')
  const [flyIdx, setFlyIdx] = useState(3)
  const [walkIdx, setWalkIdx] = useState(1)
  const [resetSignal, setResetSignal] = useState(0)
  const [panelOpen, setPanelOpen] = useState(true)
  const [tool, setTool] = useState<Tool>('none')
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [validating, setValidating] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)
  const errorField = useMemo(
    () =>
      validation
        ? decodeFloat32(validation.error_grid.values_b64, validation.error_grid.width * validation.error_grid.height)
        : null,
    [validation],
  )
  const runValidation = async (file: File) => {
    setValidating(true)
    setValidationError(null)
    try {
      const v = await validateJob(result.job_id, file, validationTarget)
      setValidation(v)
      setLayer('error')
    } catch (err) {
      setValidationError(err instanceof ApiError ? err.message : 'Validation failed.')
    } finally {
      setValidating(false)
    }
  }
  const [picks, setPicks] = useState<GroundPoint[]>([])
  const profile = useMemo(
    () => (tool === 'profile' && picks.length === 2 ? sampleProfile(terrain, picks[0], picks[1]) : null),
    [tool, picks, terrain],
  )
  const measurement = useMemo(
    () => (tool === 'measure' && picks.length === 2 ? measure(terrain, picks[0], picks[1]) : null),
    [tool, picks, terrain],
  )
  const pick = (p: GroundPoint) => setPicks((prev) => (prev.length >= 2 ? [p] : [...prev, p]))
  const chooseTool = (t: Tool) => {
    setPicks([])
    setTool((cur) => (cur === t ? 'none' : t))
    setMode('orbit') // points are picked in the orbit view
  }
  const clearTool = () => {
    setTool('none')
    setPicks([])
  }
  const [hud, setHud] = useState(() => ({ ...telemetry }))
  const rootRef = useRef<HTMLDivElement>(null)
  const speed = mode === 'walk' ? WALK_SPEEDS[walkIdx] : FLY_SPEEDS[flyIdx]
  const valid = hp.metric_validity === 'valid'
  const textureUrl = assetUrl(result.assets.texture)

  // HUD refresh at 10 Hz from the scene's telemetry (no per-frame React renders)
  useEffect(() => {
    const id = window.setInterval(
      () => setHud({ ...telemetry, camera: { ...telemetry.camera }, cursor: telemetry.cursor && { ...telemetry.cursor } }),
      100,
    )
    return () => window.clearInterval(id)
  }, [telemetry])

  // Esc: the browser first releases pointer lock; a second Esc (unlocked) exits the Explorer.
  // Mouse wheel adjusts the fly/walk speed.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Escape' && !document.pointerLockElement) onExit()
      if (e.code === 'Digit1') setMode('orbit')
      if (e.code === 'Digit2') setMode('fly')
      if (e.code === 'Digit3') setMode('walk')
    }
    const onWheel = (e: WheelEvent) => {
      if (mode === 'orbit') return
      const up = e.deltaY < 0
      if (mode === 'fly') setFlyIdx((i) => Math.min(FLY_SPEEDS.length - 1, Math.max(0, i + (up ? 1 : -1))))
      else setWalkIdx((i) => Math.min(WALK_SPEEDS.length - 1, Math.max(0, i + (up ? 1 : -1))))
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('wheel', onWheel, { passive: true })
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('wheel', onWheel)
    }
  }, [mode, onExit])

  // lock page scroll while the Explorer is open
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
      if (document.fullscreenElement) void document.exitFullscreen()
    }
  }, [])

  // 7e: save the current 3D view (the WebGL canvas keeps its drawing buffer for this)
  const screenshot = () => {
    const canvas = document.querySelector<HTMLCanvasElement>('#dw-explorer-canvas canvas')
    if (!canvas) return
    const a = document.createElement('a')
    const stem = result.source.filename.replace(/\.[^.]+$/, '')
    a.download = `depthwizard_${stem}_${mode}_${layer}_${new Date().toISOString().replace(/[:.]/g, '-')}.png`
    a.href = canvas.toDataURL('image/png')
    a.click()
  }

  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void rootRef.current?.requestFullscreen()
  }

  const altitude = hud.camera.y - hud.groundBelow
  const cursorMap = hud.cursor && geo ? mapCoords(result, hud.cursor.x, hud.cursor.z) : null
  const cameraMap = geo ? mapCoords(result, hud.camera.x, hud.camera.z) : null
  const elev = dsm && result.statistics.elevation && 'dsm_min' in result.statistics.elevation
    ? result.statistics.elevation
    : null
  const btn = 'rounded-md border border-slate-400/20 px-2.5 py-1 text-[11px] text-slate-300 transition hover:border-signal-400/50 hover:text-signal-300'

  return (
    <div ref={rootRef} className="fixed inset-0 z-50 flex flex-col bg-abyss-950 text-slate-200">
      {/* ---- top bar: scene + model / GSD status ---- */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-400/12 px-3 py-2">
        <button type="button" onClick={onExit} className={btn} title="Exit (Esc)">
          ◀ Exit
        </button>
        <span className="font-mono text-xs text-slate-100">{result.source.filename}</span>
        <span className="font-mono text-[11px] text-slate-500">
          {Math.round(grid.plane_width)} × {Math.round(grid.plane_depth)} m ·{' '}
          {formatGsd(grid.gsd_m)} m/px{hp.gsd_source === 'assumed_training_gsd' ? ' (assumed)' : ''}
        </span>
        <Badge tone={valid ? 'relief' : 'warn'}>
          {dsm ? `Surface elevation · m ${datumLabel(result)}` : valid ? 'Height above ground · m' : 'Estimated m'}
          {valid ? '' : ' · resolution uncertain'}
        </Badge>
        {geo && <Badge tone="neutral">{hp.crs}</Badge>}
        <span className="ml-auto hidden font-mono text-[10px] text-slate-500 md:inline">
          model {result.model.run} · epoch {result.model.epoch} · val RMSE {result.model.val_rmse_m.toFixed(2)} m ·{' '}
          {result.model.device_label}
        </span>
        <button type="button" onClick={toggleFullscreen} className={btn} title="Browser full screen">
          ⛶
        </button>
      </div>

      <div className="relative flex min-h-0 flex-1">
        {/* ---- left toolbar: navigation modes ---- */}
        <div className="z-10 flex flex-col gap-1.5 border-r border-slate-400/12 bg-abyss-950/90 p-1.5">
          {MODES.map((m, i) => (
            <button
              key={m.key}
              type="button"
              title={`${m.hint} (key ${i + 1})`}
              onClick={() => {
                setMode(m.key)
                if (m.key !== 'orbit') clearTool()
              }}
              className={`flex w-14 flex-col items-center rounded-md px-1 py-1.5 text-[10px] transition ${
                mode === m.key ? 'bg-signal-500/20 text-signal-300' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span className="text-base leading-none">{m.icon}</span>
              {m.label}
            </button>
          ))}
          <div className="my-1 border-t border-slate-400/12" />
          <button
            type="button"
            title="Reset view"
            onClick={() => {
              setMode('orbit')
              setResetSignal((n) => n + 1)
            }}
            className="w-14 rounded-md px-1 py-1.5 text-[10px] text-slate-400 hover:text-slate-200"
          >
            <span className="block text-base leading-none">⌂</span>
            Home
          </button>
          <div className="my-1 border-t border-slate-400/12" />
          {([
            ['profile', '📈', 'Profile', 'Height profile: click point A, then point B on the terrain'],
            ['measure', '📏', 'Measure', 'Distance and height difference between two clicked points'],
          ] as const).map(([key, icon, label, hint]) => (
            <button
              key={key}
              type="button"
              title={hint}
              onClick={() => chooseTool(key)}
              className={`flex w-14 flex-col items-center rounded-md px-1 py-1.5 text-[10px] transition ${
                tool === key ? 'bg-amber-400/20 text-amber-300' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span className="text-base leading-none">{icon}</span>
              {label}
            </button>
          ))}
          <div className="my-1 border-t border-slate-400/12" />
          <button
            type="button"
            title="Save the current 3D view as PNG"
            onClick={screenshot}
            className="flex w-14 flex-col items-center rounded-md px-1 py-1.5 text-[10px] text-slate-400 hover:text-slate-200"
          >
            <span className="text-base leading-none">📷</span>
            Shot
          </button>
        </div>

        {/* ---- the scene ---- */}
        <div className="relative min-w-0 flex-1">
          <ErrorBoundary resetKey={result.job_id} fallbackTitle="The 3D scene failed to render">
          <Canvas
            id="dw-explorer-canvas"
            dpr={[1, 2]}
            gl={{ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true }}
            camera={{ fov: 60, near: 0.1, far: 10000 }}
          >
            <Suspense fallback={null}>
              <ExplorerScene
                terrain={terrain}
                slope={slope}
                textureUrl={textureUrl}
                layer={layer}
                heightMax={heightMax}
                elevRange={elevRange}
                mode={mode}
                speed={speed}
                telemetry={telemetry}
                resetSignal={resetSignal}
                tool={tool}
                picks={picks}
                onPick={pick}
                errorField={errorField}
                errorLimit={validation?.error_limit_m ?? 10}
              />
            </Suspense>
          </Canvas>
          </ErrorBoundary>
          <Loader />

          {mode !== 'orbit' && !hud.pointerLocked && (
            <div className="pointer-events-none absolute inset-x-0 top-4 flex justify-center">
              <div className="rounded-lg border border-signal-400/30 bg-abyss-950/85 px-3 py-2 text-center text-xs text-slate-300">
                <b className="text-signal-300">Click the scene</b> to look around with the mouse ·{' '}
                <b>W A S D</b> move{mode === 'fly' ? ' · Space / C up-down' : ''} · <b>Shift</b> faster · wheel = speed ·{' '}
                <b>Esc</b> release
              </div>
            </div>
          )}
          {hud.pointerLocked && (
            <div className="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-lg text-white/80">
              +
            </div>
          )}

          {tool !== 'none' && picks.length < 2 && (
            <div className="pointer-events-none absolute inset-x-0 top-4 flex justify-center">
              <div className="rounded-lg border border-amber-400/40 bg-abyss-950/85 px-3 py-2 text-xs text-slate-300">
                {tool === 'profile' ? 'Profile' : 'Measure'}: click point <b className="text-amber-300">{picks.length === 0 ? 'A' : 'B'}</b> on
                the terrain (drag still orbits)
              </div>
            </div>
          )}
          <div className="absolute bottom-3 left-3 z-10">
            {profile && <ProfileChart profile={profile} onClose={clearTool} base={base} elevation={dsm} />}
            {measurement && <MeasureCard m={measurement} onClose={clearTool} base={base} elevation={dsm} />}
          </div>

          <div className="absolute right-3 bottom-3">
            <Minimap textureUrl={textureUrl} width={grid.plane_width} depth={grid.plane_depth} telemetry={telemetry} />
          </div>
        </div>

        {/* ---- right panel: layers, legend, stats ---- */}
        {panelOpen ? (
          <div className="z-10 w-60 space-y-3 overflow-y-auto border-l border-slate-400/12 bg-abyss-950/90 p-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] tracking-widest text-slate-500 uppercase">Layers</span>
              <button type="button" className="text-slate-500 hover:text-slate-300" onClick={() => setPanelOpen(false)}>
                ✕
              </button>
            </div>
            {(dsm ? LAYERS_DSM : LAYERS_NDSM).map((l) => (
              <label
                key={l}
                className={`flex items-center gap-2 ${l === 'error' && !validation ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'}`}
              >
                <input
                  type="radio"
                  name="layer"
                  checked={layer === l}
                  disabled={l === 'error' && !validation}
                  onChange={() => setLayer(l)}
                />
                {l === 'rgb'
                  ? 'RGB image'
                  : l === 'height'
                    ? 'Height above ground'
                    : l === 'elevation'
                      ? `Elevation (${datumLabel(result)})`
                    : l === 'slope'
                      ? 'Slope (degrees)'
                      : 'Error vs reference'}
              </label>
            ))}
            <p className="text-[10px] text-slate-600">Tools: 📈 profile and 📏 measure in the left toolbar.</p>
            {layer === 'slope' && (
              <div>
                <div className="h-2 rounded-sm" style={{ background: `linear-gradient(to right, ${slopeRampCss()})` }} />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-300">
                  <span>0°</span>
                  <span>15°</span>
                  <span>30°</span>
                  <span>45°</span>
                  <span>90°</span>
                </div>
                <p className="mt-1 text-[10px] leading-snug text-slate-500">
                  Slope of the {dsm ? 'surface (DSM)' : 'imaged surface'} on the {terrain.dx.toFixed(2)} m grid: roofs,
                  canopy{dsm ? ', hillsides' : ''}, and near-vertical values at building walls and tree edges.
                </p>
              </div>
            )}
            {layer === 'error' && validation && (
              <div>
                <div className="h-2 rounded-sm" style={{ background: `linear-gradient(to right, ${errorRampCss()})` }} />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-300">
                  <span>−{validation.error_limit_m} m</span>
                  <span>0</span>
                  <span>+{validation.error_limit_m} m</span>
                </div>
                <p className="mt-1 text-[10px] leading-snug text-slate-500">
                  predicted − reference · blue = too low, red = too high, grey = no reference data
                </p>
              </div>
            )}
            {layer === 'height' && (
              <div>
                <div className="h-2 rounded-sm" style={{ background: `linear-gradient(to right, ${rampCss()})` }} />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-300">
                  <span>0 m</span>
                  <span>{(heightMax / 2).toFixed(0)} m</span>
                  <span>≥{heightMax.toFixed(0)} m</span>
                </div>
              </div>
            )}
            {layer === 'elevation' && (
              <div>
                <div className="h-2 rounded-sm" style={{ background: `linear-gradient(to right, ${rampCss()})` }} />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-300">
                  <span>≤{grid.display_min.toFixed(0)} m</span>
                  <span>{((grid.display_min + grid.display_max) / 2).toFixed(0)} m</span>
                  <span>≥{grid.display_max.toFixed(0)} m</span>
                </div>
                <p className="mt-1 text-[10px] leading-snug text-slate-500">
                  Surface elevation (DSM = DTM + nDSM), metres above the EGM2008 geoid.
                </p>
              </div>
            )}
            <div className="space-y-1 border-t border-slate-400/12 pt-2 font-mono text-[10px] text-slate-400">
              {elev && (
                <>
                  <div>
                    elevation {elev.dsm_min.toFixed(1)} – {elev.dsm_max.toFixed(1)} m
                  </div>
                  <div>ground relief {elev.relief_m.toFixed(1)} m (DTM)</div>
                  <div className="text-slate-500">height above ground:</div>
                </>
              )}
              <div>median {result.statistics.median.toFixed(1)} m · p95 {result.statistics.p95.toFixed(1)} m</div>
              <div>max {result.statistics.max.toFixed(1)} m · &gt;2 m: {(result.statistics.frac_above_2m * 100).toFixed(0)}%</div>
              <div>scale: true (1 m = 1 m, no exaggeration)</div>
              {base !== 0 && <div>scene Y 0 = {base} m elevation (shift only)</div>}
            </div>
            <div className="space-y-1.5 border-t border-slate-400/12 pt-2">
              <span className="font-mono text-[10px] tracking-widest text-slate-500 uppercase">Validation</span>
              <p className="text-[10px] leading-snug text-slate-500">
                Reference heights: <code>.npy</code> in metres on the same {result.height_product.width}×
                {result.height_product.height} px grid (NaN = no data)
                {geo
                  ? ', or a GeoTIFF in any CRS (reprojected onto this grid; NAVD88 converted to EGM2008).'
                  : '. GeoTIFF references need a GeoTIFF input.'}
              </p>
              {dsm && (
                <label className="flex items-center gap-2 text-[10px] text-slate-400">
                  compare
                  <select
                    value={validationTarget}
                    onChange={(e) => setValidationTarget(e.target.value as ValidationTarget)}
                    className="rounded border border-slate-400/20 bg-abyss-900 px-1 py-0.5 text-slate-200"
                  >
                    <option value="auto">DSM (elevation)</option>
                    <option value="ndsm">nDSM (height above ground)</option>
                  </select>
                </label>
              )}
              <label className="block cursor-pointer rounded-md border border-dashed border-slate-400/30 px-2 py-1.5 text-center text-[11px] text-slate-300 hover:border-signal-400/50">
                {validating ? 'Validating…' : validation ? 'Replace reference…' : `Upload reference ${geo ? '.tif / ' : ''}.npy…`}
                <input
                  type="file"
                  accept={geo ? '.npy,.tif,.tiff' : '.npy'}
                  className="hidden"
                  disabled={validating}
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    e.target.value = ''
                    if (f) void runValidation(f)
                  }}
                />
              </label>
              {validationError && <p className="text-[10px] text-red-400">{validationError}</p>}
              {validation && (
                <div className="space-y-1 font-mono text-[10px] text-slate-300">
                  <div className="truncate text-slate-500" title={validation.reference.filename}>
                    {validation.reference.filename} · {(validation.reference.valid_fraction * 100).toFixed(1)}% valid
                  </div>
                  <div className="text-slate-500">
                    target <b className="text-slate-300">{validation.target.toUpperCase()}</b>
                    {validation.reference.datum_conversion && (
                      <span className="block leading-snug" title={validation.reference.datum_conversion}>
                        datum: {validation.reference.datum_conversion}
                      </span>
                    )}
                  </div>
                  <div>
                    RMSE <b className="text-slate-100">{validation.overall.rmse?.toFixed(2)} m</b> · MAE{' '}
                    <b className="text-slate-100">{validation.overall.mae?.toFixed(2)} m</b>
                  </div>
                  <div>
                    bias {validation.overall.bias !== null && validation.overall.bias >= 0 ? '+' : ''}
                    {validation.overall.bias?.toFixed(2)} m · r {validation.overall.pearson_r?.toFixed(3)}
                  </div>
                  <table className="mt-1 w-full">
                    <tbody>
                      {Object.entries(validation.per_height_band).map(([band, v]) => (
                        <tr key={band} className="text-slate-400">
                          <td>{band}</td>
                          <td className="text-right">MAE {v.mae?.toFixed(2)}</td>
                          <td className="text-right">bias {v.bias?.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <a href={assetUrl(validation.error_map)} target="_blank" rel="noreferrer" className="text-signal-300">
                    ↗ full-resolution error map
                  </a>
                </div>
              )}
            </div>
            <div className="space-y-1 border-t border-slate-400/12 pt-2">
              <span className="font-mono text-[10px] tracking-widest text-slate-500 uppercase">
                {mode === 'walk' ? 'Walk speed' : 'Fly speed'}
              </span>
              <input
                type="range"
                className="w-full"
                min={0}
                max={(mode === 'walk' ? WALK_SPEEDS : FLY_SPEEDS).length - 1}
                value={mode === 'walk' ? walkIdx : flyIdx}
                disabled={mode === 'orbit'}
                onChange={(e) => (mode === 'walk' ? setWalkIdx : setFlyIdx)(Number(e.target.value))}
              />
              <div className="font-mono text-[11px] text-slate-200">{mode === 'orbit' ? '–' : `${speed} m/s`}</div>
            </div>
            <div className="space-y-1 border-t border-slate-400/12 pt-2 text-[11px]">
              <span className="font-mono text-[10px] tracking-widest text-slate-500 uppercase">Export</span>
              {(
                [
                  ...Object.entries(hp.geotiffs ?? {}).map(
                    ([k, url]) =>
                      [
                        `${k.toUpperCase()} GeoTIFF (float32 m, ${k === 'ndsm' ? 'above ground' : datumLabel(result)})`,
                        url,
                      ] as [string, string],
                  ),
                  ...(result.assets.dsm_array
                    ? ([['DSM elevations (.npy, float32 m)', result.assets.dsm_array]] as [string, string][])
                    : []),
                  ['nDSM heights (.npy, float32 m)', hp.ndsm_array ?? result.assets.height_array],
                  ['Height map (PNG)', result.assets.height_map],
                  ['Shaded relief (PNG)', result.assets.hillshade],
                  ['Metadata + provenance (JSON)', `/static/${result.job_id}/metadata.json`],
                  ...(validation
                    ? ([
                        ['Validation report (JSON)', `/static/${result.job_id}/validation.json`],
                        ['Error map (PNG)', validation.error_map],
                      ] as [string, string][])
                    : []),
                ] as [string, string][]
              ).map(([label, url]) => (
                <a key={url} href={assetUrl(url)} download className="block text-signal-300 hover:text-signal-200">
                  ↓ {label}
                </a>
              ))}
              {!geo && (
                <p className="text-[10px] text-slate-600">GeoTIFF export needs a georeferenced (GeoTIFF) input.</p>
              )}
            </div>
          </div>
        ) : (
          <button
            type="button"
            className="absolute top-3 right-3 z-10 rounded-md border border-slate-400/20 bg-abyss-950/90 px-2 py-1 text-[11px]"
            onClick={() => setPanelOpen(true)}
          >
            Layers ▸
          </button>
        )}
      </div>

      {/* ---- status HUD ---- */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-slate-400/12 px-3 py-1.5 font-mono text-[11px] text-slate-400">
        <span>
          mode <b className="text-slate-100">{MODES.find((m) => m.key === mode)?.label}</b>
        </span>
        <span>
          {hud.cursor ? (
            <>
              {hud.pointerLocked ? 'target' : 'cursor'}{' '}
              {dsm ? (
                <>
                  <b className="text-slate-100">{(hud.cursor.height + base).toFixed(1)} m</b> {datumLabel(result)}
                  {hud.cursor.aboveGround !== null && (
                    <>
                      {' '}· <b className="text-slate-100">{hud.cursor.aboveGround.toFixed(1)} m</b> above ground
                    </>
                  )}
                </>
              ) : (
                <>
                  <b className="text-slate-100">{hud.cursor.height.toFixed(1)} m</b> above ground
                </>
              )}
              {hud.cursor.slope !== null && (
                <>
                  {' '}· slope <b className="text-slate-100">{hud.cursor.slope.toFixed(0)}°</b>
                </>
              )}
              {hud.cursor.error !== null && (
                <>
                  {' '}· error{' '}
                  <b className="text-slate-100">
                    {hud.cursor.error >= 0 ? '+' : ''}
                    {hud.cursor.error.toFixed(1)} m
                  </b>
                </>
              )}{' '}
              ·{' '}
              {hud.cursor.distance.toFixed(0)} m away
            </>
          ) : (
            <span className="text-slate-600">{mode === 'orbit' ? 'hover the terrain' : 'aim at the terrain'}</span>
          )}
        </span>
        <span>
          you <b className="text-slate-100">{altitude.toFixed(1)} m</b> above surface
          {mode === 'fly' ? ` (min ${FLY_CLEARANCE_M} m)` : mode === 'walk' ? ` (eye ${EYE_HEIGHT_M} m)` : ''}
        </span>
        {cursorMap && (
          <span title={`${hp.crs} easting / northing; lon/lat WGS 84`}>
            E {cursorMap.easting.toFixed(1)} · N {cursorMap.northing.toFixed(1)}
            {cursorMap.lon !== null && cursorMap.lat !== null && <> · {formatLonLat(cursorMap.lon, cursorMap.lat)}</>}
          </span>
        )}
        <span>
          {cameraMap
            ? `camera E ${cameraMap.easting.toFixed(0)} · N ${cameraMap.northing.toFixed(0)}`
            : `position (${hud.camera.x.toFixed(0)}, ${hud.camera.z.toFixed(0)}) m`}{' '}
          · {dsm ? `elev ${(hud.camera.y + base).toFixed(1)} m` : `Y ${hud.camera.y.toFixed(1)} m`}
        </span>
        <span>speed {mode === 'orbit' ? '–' : `${speed} m/s`}</span>
      </div>
    </div>
  )
}
