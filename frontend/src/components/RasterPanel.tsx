import { useState } from 'react'
import type { ProcessResult } from '../types'
import { Panel } from './ui'

type LayerKey = 'original' | 'depth' | 'dsm'

const LAYERS: {
  key: LayerKey
  label: string
  caption: string
  asset: (r: ProcessResult) => string
}[] = [
  {
    key: 'original',
    label: 'Original RGB',
    caption: 'Uploaded single view',
    asset: (r) => r.assets.original,
  },
  {
    key: 'depth',
    label: 'Depth Map',
    caption: 'Relative inverse depth · Depth Anything V2',
    asset: (r) => r.assets.depth_map,
  },
  {
    key: 'dsm',
    label: 'Relative DSM',
    caption: 'Normalised rDSM · shaded relief',
    asset: (r) => r.assets.relative_dsm,
  },
]

function Tile({
  label,
  caption,
  src,
  active,
  onZoom,
}: {
  label: string
  caption: string
  src: string | null
  active: boolean
  onZoom: () => void
}) {
  return (
    <figure className="min-w-0">
      <button
        type="button"
        onClick={onZoom}
        disabled={!src}
        className={`group relative block aspect-4/3 w-full overflow-hidden rounded-lg border bg-abyss-900 transition ${
          active ? 'border-signal-400/50' : 'border-slate-400/12 hover:border-slate-400/30'
        } disabled:cursor-default`}
      >
        {src ? (
          <img
            src={src}
            alt={label}
            loading="lazy"
            className="size-full object-cover transition duration-300 group-hover:scale-[1.03]"
          />
        ) : (
          <span className="grid size-full place-items-center font-mono text-[10px] tracking-widest text-slate-600 uppercase">
            awaiting run
          </span>
        )}
      </button>
      <figcaption className="mt-1.5">
        <p className="text-xs font-medium text-slate-200">{label}</p>
        <p className="truncate text-[10px] text-slate-500">{caption}</p>
      </figcaption>
    </figure>
  )
}

export function RasterPanel({
  result,
  previewUrl,
}: {
  result: ProcessResult | null
  previewUrl: string | null
}) {
  const [zoom, setZoom] = useState<LayerKey | null>(null)

  const srcFor = (layer: (typeof LAYERS)[number]): string | null => {
    if (result) return layer.asset(result)
    return layer.key === 'original' ? previewUrl : null
  }

  const zoomed = LAYERS.find((l) => l.key === zoom)
  const zoomSrc = zoomed ? srcFor(zoomed) : null

  return (
    <>
      <Panel
        title="2 · Raster Products"
        aside={
          result ? (
            <span className="font-mono text-[10px] text-slate-500 normal-case">
              click to enlarge
            </span>
          ) : null
        }
        bodyClassName="grid grid-cols-3 gap-3 p-4"
      >
        {LAYERS.map((layer) => (
          <Tile
            key={layer.key}
            label={layer.label}
            caption={layer.caption}
            src={srcFor(layer)}
            active={zoom === layer.key}
            onZoom={() => setZoom(layer.key)}
          />
        ))}
      </Panel>

      {zoomed && zoomSrc && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={zoomed.label}
          onClick={() => setZoom(null)}
          className="fixed inset-0 z-50 grid place-items-center bg-abyss-950/92 p-6 backdrop-blur-sm"
        >
          <figure className="max-h-full max-w-6xl" onClick={(e) => e.stopPropagation()}>
            <img
              src={zoomSrc}
              alt={zoomed.label}
              className="max-h-[80vh] rounded-xl border border-slate-400/20 object-contain"
            />
            <figcaption className="mt-3 flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-slate-100">{zoomed.label}</p>
                <p className="text-xs text-slate-500">{zoomed.caption}</p>
              </div>
              <button
                type="button"
                onClick={() => setZoom(null)}
                className="rounded-lg border border-slate-400/25 px-3 py-1.5 text-xs text-slate-300 transition hover:border-slate-400/50"
              >
                Close
              </button>
            </figcaption>
          </figure>
        </div>
      )}
    </>
  )
}
