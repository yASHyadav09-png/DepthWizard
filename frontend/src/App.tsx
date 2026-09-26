import { useCallback, useEffect, useRef, useState } from 'react'
import { Header } from './components/Header'
import { UploadPanel } from './components/UploadPanel'
import { StatusPanel } from './components/StatusPanel'
import { StatsPanel } from './components/StatsPanel'
import { RasterPanel } from './components/RasterPanel'
import { TerrainViewer } from './components/TerrainViewer'
import type { NavMode } from './components/TerrainViewer'
import { ViewerControls } from './components/ViewerControls'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Badge, Panel } from './components/ui'
import { ApiError, getHealth, getResult, processImage } from './services/api'
import type { HealthResponse, PipelineStage, ProcessResult, ViewMode } from './types'

const DEFAULT_EXAGGERATION = 1 // true vertical scale

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [file, setFile] = useState<File | null>(null)
  const [gsd, setGsd] = useState('')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [stage, setStage] = useState<PipelineStage>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ProcessResult | null>(null)

  const [viewMode, setViewMode] = useState<ViewMode>('textured')
  const [wireframe, setWireframe] = useState(false)
  const [exaggeration, setExaggeration] = useState(DEFAULT_EXAGGERATION)
  const [navMode, setNavMode] = useState<NavMode>('orbit')
  const [resetSignal, setResetSignal] = useState(0)

  const abortRef = useRef<AbortController | null>(null)
  const previewRef = useRef<string | null>(null)

  // --- backend health ----------------------------------------------------
  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then((body) => {
        setHealth(body)
        setHealthError(null)
      })
      .catch((err: unknown) => {
        if ((err as Error)?.name === 'AbortError') return
        setHealthError(
          err instanceof ApiError ? err.message : 'Backend is not reachable.',
        )
      })
    return () => controller.abort()
  }, [])

  // Deep link: ?job=<id> reloads a completed run straight from the backend,
  // so a result can be shared, bookmarked or replayed without re-uploading.
  useEffect(() => {
    const jobId = new URLSearchParams(window.location.search).get('job')
    if (!jobId) return
    setStage('inferring')
    getResult(jobId)
      .then((payload) => {
        setResult(payload)
        setResetSignal((n) => n + 1)
        setStage('done')
      })
      .catch((err: unknown) => {
        setStage('error')
        setError(
          err instanceof ApiError ? err.message : `Could not load job '${jobId}'.`,
        )
      })
  }, [])

  // Revoke the last object URL whenever a new preview replaces it.
  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    }
  }, [])

  const setPreview = useCallback((next: File | null) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = next ? URL.createObjectURL(next) : null
    previewRef.current = url
    setPreviewUrl(url)
  }, [])

  const handleSelect = useCallback(
    (selected: File, localError: string | null) => {
      abortRef.current?.abort()
      setResult(null)
      setStage('idle')
      if (localError) {
        setFile(null)
        setPreview(null)
        setError(localError)
        return
      }
      setError(null)
      setFile(selected)
      setPreview(selected)
    },
    [setPreview],
  )

  const handleReset = useCallback(() => {
    abortRef.current?.abort()
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setStage('idle')
    setExaggeration(DEFAULT_EXAGGERATION)
    setWireframe(false)
    setViewMode('textured')
    setNavMode('orbit')
  }, [setPreview])

  const handleGenerate = useCallback(async () => {
    if (!file) return
    const gsdTrim = gsd.trim()
    const gsdM = gsdTrim === '' ? null : Number(gsdTrim)
    if (gsdM !== null && !(Number.isFinite(gsdM) && gsdM >= 0.01 && gsdM <= 100)) {
      setStage('error')
      setError('Ground resolution must be a number between 0.01 and 100 m per pixel (or empty).')
      return
    }
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setError(null)
    setResult(null)
    setStage('uploading')

    // The backend runs the whole pipeline in one request, so the intermediate
    // stages are timed indications rather than server-pushed progress.
    const toInferring = window.setTimeout(() => setStage('inferring'), 350)

    try {
      const payload = await processImage(file, gsdM, controller.signal)
      window.clearTimeout(toInferring)
      setStage('building')
      setResult(payload)
      setNavMode('orbit')
      setResetSignal((n) => n + 1)
      setStage('done')
      // Make the finished run linkable / reloadable.
      window.history.replaceState(null, '', `?job=${payload.job_id}`)
    } catch (err) {
      window.clearTimeout(toInferring)
      if ((err as Error)?.name === 'AbortError') return
      setStage('error')
      setError(
        err instanceof ApiError
          ? err.message
          : 'Processing failed for an unknown reason.',
      )
    }
  }, [file, gsd])

  const hasTerrain = Boolean(result)

  return (
    <div className="relative z-10 flex min-h-full flex-col">
      <Header health={health} healthError={healthError} />

      <main className="mx-auto w-full max-w-[1800px] flex-1 px-5 py-5">
        {/* 1 column on phones, 2 on laptops (viewer full-width on top),
            3-column dashboard from 1280px up. */}
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-[21rem_minmax(0,1fr)_21rem]">
          {/* ---- left: input + status ---- */}
          <div className="order-2 flex flex-col gap-4 xl:order-none">
            <UploadPanel
              file={file}
              previewUrl={previewUrl}
              stage={stage}
              maxUploadMb={health?.max_upload_mb ?? null}
              gsd={gsd}
              onGsd={setGsd}
              onSelect={handleSelect}
              onGenerate={handleGenerate}
              onReset={handleReset}
            />
            <StatusPanel stage={stage} error={error} result={result} />
          </div>

          {/* ---- centre: 3D terrain + rasters ---- */}
          <div className="order-1 flex min-w-0 flex-col gap-4 lg:col-span-2 xl:order-none xl:col-span-1">
            <Panel
              title="3 · 3D Terrain (height above ground)"
              aside={
                result ? (
                  <Badge tone="relief">
                    {navMode === 'orbit' ? 'Orbit' : 'Flythrough'}
                  </Badge>
                ) : null
              }
              bodyClassName="p-0"
              className="min-h-[30rem] xl:min-h-[34rem]"
            >
              <div className="size-full overflow-hidden rounded-b-[0.8rem]">
                <ErrorBoundary
                  resetKey={result?.job_id ?? 'none'}
                  fallbackTitle="Could not render the terrain"
                >
                  <TerrainViewer
                    result={result}
                    viewMode={viewMode}
                    wireframe={wireframe}
                    exaggeration={exaggeration}
                    navMode={navMode}
                    resetSignal={resetSignal}
                  />
                </ErrorBoundary>
              </div>
            </Panel>

            <RasterPanel result={result} previewUrl={previewUrl} />
          </div>

          {/* ---- right: stats + viewer controls ---- */}
          <div className="order-3 flex flex-col gap-4 xl:order-none">
            <StatsPanel result={result} />
            <ViewerControls
              enabled={hasTerrain}
              viewMode={viewMode}
              onViewMode={setViewMode}
              wireframe={wireframe}
              onWireframe={setWireframe}
              exaggeration={exaggeration}
              onExaggeration={setExaggeration}
              navMode={navMode}
              onNavMode={setNavMode}
              onResetCamera={() => setResetSignal((n) => n + 1)}
            />
          </div>
        </div>
      </main>

      <footer className="mx-auto w-full max-w-[1800px] px-5 pt-1 pb-5">
        <p className="text-[11px] text-slate-600">
          DepthWizard Phase 3 · height above ground (nDSM, metres) from a single
          top-down view with Depth Anything V2 Small fine-tuned on GAMUS. Resolution
          handling (Phase 4) and GeoTIFF / absolute DSM (Phase 5) arrive next.
        </p>
      </footer>
    </div>
  )
}
