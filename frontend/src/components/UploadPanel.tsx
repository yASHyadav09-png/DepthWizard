import { useCallback, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import type { PipelineStage } from '../types'
import { Panel } from './ui'

const ACCEPT = 'image/jpeg,image/png,image/tiff,.tif,.tiff'
const GCP_ACCEPT = '.csv,text/csv'
const MAX_MB_FALLBACK = 25

export function UploadPanel({
  file,
  previewUrl,
  stage,
  maxUploadMb,
  gsd,
  onGsd,
  gcpFile,
  onGcpFile,
  gcpModel,
  onGcpModel,
  onSelect,
  onGenerate,
  onReset,
}: {
  file: File | null
  previewUrl: string | null
  stage: PipelineStage
  maxUploadMb: number | null
  gsd: string
  onGsd: (value: string) => void
  gcpFile: File | null
  onGcpFile: (file: File | null) => void
  gcpModel: 'offset' | 'plane'
  onGcpModel: (model: 'offset' | 'plane') => void
  onSelect: (file: File, localError: string | null) => void
  onGenerate: () => void
  onReset: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const gcpRef = useRef<HTMLInputElement>(null)
  const geoTiff = Boolean(file && /\.tiff?$/i.test(file.name))
  const [dragging, setDragging] = useState(false)
  const busy = stage === 'uploading' || stage === 'inferring' || stage === 'building'
  const limitMb = maxUploadMb ?? MAX_MB_FALLBACK

  const accept = useCallback(
    (candidate: File | undefined | null) => {
      if (!candidate) return
      const okType = /^image\/(jpeg|png|tiff?)$/i.test(candidate.type)
      const okExt = /\.(jpe?g|png|tiff?)$/i.test(candidate.name)
      if (!okType && !okExt) {
        onSelect(candidate, 'JPG, PNG or GeoTIFF (.tif) images only.')
        return
      }
      if (candidate.size > limitMb * 1024 * 1024) {
        onSelect(candidate, `Image is larger than the ${limitMb} MB upload limit.`)
        return
      }
      onSelect(candidate, null)
    },
    [limitMb, onSelect],
  )

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (busy) return
    accept(event.dataTransfer.files?.[0])
  }

  return (
    <Panel title="1 · Input Image" bodyClassName="space-y-3 p-4">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          if (!busy) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if ((event.key === 'Enter' || event.key === ' ') && !busy) {
            inputRef.current?.click()
          }
        }}
        className={`relative grid cursor-pointer place-items-center overflow-hidden rounded-xl border border-dashed transition ${
          dragging
            ? 'border-signal-400 bg-signal-400/10'
            : 'border-slate-400/25 bg-slate-400/4 hover:border-signal-400/50 hover:bg-signal-400/5'
        } ${busy ? 'pointer-events-none opacity-60' : ''} ${
          previewUrl ? 'aspect-4/3' : 'py-9'
        }`}
      >
        {geoTiff && !previewUrl ? (
          <div className="px-4 text-center">
            <p className="font-mono text-xs text-signal-300">GeoTIFF</p>
            <p className="mt-1 truncate font-mono text-[11px] text-slate-300">{file?.name}</p>
            <p className="mt-1 font-mono text-[10px] text-slate-500">
              {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : ''} · CRS and resolution are read from the file ·
              click to replace
            </p>
          </div>
        ) : previewUrl ? (
          <>
            <img
              src={previewUrl}
              alt="Selected source"
              className="absolute inset-0 size-full object-contain"
            />
            <div className="absolute inset-x-0 bottom-0 bg-linear-to-t from-abyss-950/95 to-transparent px-3 pt-8 pb-2">
              <p className="truncate font-mono text-[11px] text-slate-300">
                {file?.name}
              </p>
              <p className="font-mono text-[10px] text-slate-500">
                {file ? `${(file.size / 1024).toFixed(0)} KB` : ''} · click to replace
              </p>
            </div>
          </>
        ) : (
          <div className="px-4 text-center">
            <svg
              viewBox="0 0 24 24"
              className="mx-auto size-8 fill-none stroke-slate-500"
              strokeWidth={1.4}
            >
              <path
                d="M12 16V4m0 0L8 8m4-4 4 4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
            </svg>
            <p className="mt-2.5 text-sm font-medium text-slate-300">
              Drop a JPG, PNG or GeoTIFF, or click to browse
            </p>
            <p className="mt-1 font-mono text-[11px] text-slate-500">
              Top-down aerial / satellite image · max {limitMb} MB · GeoTIFF → absolute DSM
            </p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(event) => {
            accept(event.target.files?.[0])
            event.target.value = ''
          }}
        />
      </div>

      <label className="block">
        <span className="text-[11px] text-slate-400">
          Ground resolution <span className="text-slate-600">(m / pixel, optional)</span>
        </span>
        <input
          type="number"
          inputMode="decimal"
          min={0.01}
          max={100}
          step={0.01}
          value={geoTiff ? '' : gsd}
          disabled={busy || geoTiff}
          onChange={(event) => onGsd(event.target.value)}
          placeholder={geoTiff ? 'read from the GeoTIFF' : 'unknown: assume 0.33 (training resolution)'}
          className="mt-1 w-full rounded-lg border border-slate-400/20 bg-slate-400/5 px-3 py-2 font-mono text-xs text-slate-200 placeholder:text-slate-600 focus:border-signal-400/50 focus:outline-none disabled:opacity-50"
        />
        <span className="mt-1 block text-[10px] leading-snug text-slate-600">
          The model was trained on ~0.33 m/px aerial imagery. Other resolutions give less
          reliable heights (Phase 4: RMSE 3.2 m at 0.33 m/px, 5.2 m at 1 m/px).
        </span>
      </label>

      {geoTiff && (
        <div className="space-y-1.5 rounded-lg border border-slate-400/12 bg-slate-400/4 p-2.5">
          <span className="text-[11px] text-slate-400">
            Ground control points <span className="text-slate-600">(optional CSV)</span>
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => gcpRef.current?.click()}
              className="min-w-0 flex-1 truncate rounded-md border border-dashed border-slate-400/30 px-2 py-1 text-left font-mono text-[11px] text-slate-300 hover:border-signal-400/50 disabled:opacity-50"
            >
              {gcpFile ? gcpFile.name : 'choose .csv…'}
            </button>
            {gcpFile && (
              <button
                type="button"
                disabled={busy}
                onClick={() => onGcpFile(null)}
                className="text-[11px] text-slate-500 hover:text-slate-200"
              >
                ✕
              </button>
            )}
            <select
              value={gcpModel}
              disabled={busy || !gcpFile}
              onChange={(event) => onGcpModel(event.target.value as 'offset' | 'plane')}
              className="rounded border border-slate-400/20 bg-abyss-900 px-1 py-1 text-[11px] text-slate-200 disabled:opacity-50"
            >
              <option value="offset">offset (≥1 point)</option>
              <option value="plane">plane (≥3 points)</option>
            </select>
          </div>
          <input
            ref={gcpRef}
            type="file"
            accept={GCP_ACCEPT}
            className="hidden"
            onChange={(event) => {
              onGcpFile(event.target.files?.[0] ?? null)
              event.target.value = ''
            }}
          />
          <span className="block text-[10px] leading-snug text-slate-600">
            Columns <code>x,y,z</code> (image CRS) or <code>lon,lat,z</code>; z = ground elevation in metres
            above EGM2008. Corrects the DEM ground before adding the model's heights.
          </span>
        </div>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={onGenerate}
          disabled={!file || busy}
          className="flex-1 rounded-lg bg-linear-to-r from-signal-500 to-relief-500 px-4 py-2.5 text-sm font-semibold text-abyss-950 shadow-lg shadow-signal-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:shadow-none"
        >
          {busy ? 'Processing…' : 'Generate Terrain'}
        </button>
        <button
          type="button"
          onClick={onReset}
          disabled={busy || (!file && stage === 'idle')}
          className="rounded-lg border border-slate-400/20 px-3.5 py-2.5 text-sm font-medium text-slate-400 transition hover:border-slate-400/40 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </Panel>
  )
}
