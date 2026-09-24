import type { ProcessResult } from '../types'
import { assetUrl } from '../services/api'
import { Panel, RelativeHeightNotice } from './ui'

function Row({
  label,
  value,
  accent = false,
}: {
  label: string
  value: string
  accent?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-slate-400/8 py-1.5 last:border-0">
      <span className="text-[11px] text-slate-500">{label}</span>
      <span
        className={`truncate text-right font-mono text-[11px] ${
          accent ? 'text-signal-300' : 'text-slate-200'
        }`}
        title={value}
      >
        {value}
      </span>
    </div>
  )
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-400/12 bg-slate-400/5 px-2.5 py-2">
      <p className="font-mono text-[9px] tracking-widest text-slate-500 uppercase">
        {label}
      </p>
      <p className="mt-0.5 font-mono text-sm text-slate-100">{value}</p>
    </div>
  )
}

export function StatsPanel({ result }: { result: ProcessResult | null }) {
  if (!result) {
    return (
      <Panel title="Statistics" bodyClassName="p-4">
        <p className="text-xs text-slate-500">
          Statistics appear once an image has been processed.
        </p>
      </Panel>
    )
  }

  const s = result.statistics
  const fmt = (value: number) => value.toFixed(4)

  return (
    <Panel title="Statistics" bodyClassName="space-y-3 p-4">
      <RelativeHeightNotice />

      <div className="grid grid-cols-3 gap-2">
        <Tile label="Min" value={fmt(s.min_relative_height)} />
        <Tile label="Mean" value={fmt(s.mean_relative_height)} />
        <Tile label="Max" value={fmt(s.max_relative_height)} />
      </div>

      <div>
        <Row
          label="Image dimensions"
          value={`${s.image_width} × ${s.image_height} px`}
        />
        <Row label="Median relative height" value={fmt(s.median_relative_height)} />
        <Row label="Std. deviation" value={fmt(s.std_relative_height)} />
        <Row
          label="P05 – P95 range"
          value={`${fmt(s.p05_relative_height)} – ${fmt(s.p95_relative_height)}`}
        />
        <Row label="Height units" value={s.height_units} />
        <Row label="Model" value={s.model_name} accent />
        <Row label="Processing device" value={s.processing_device} accent />
        <Row label="Terrain grid" value={`${result.terrain.width} × ${result.terrain.height}`} />
        <Row label="Georeferenced" value={s.georeferenced ? 'yes' : 'no'} />
        <Row label="Metric" value={s.is_metric ? 'yes' : 'no'} />
        <Row label="Job ID" value={result.job_id} />
      </div>

      <div className="flex flex-wrap gap-2 pt-0.5">
        <a
          href={assetUrl(result.assets.height_array)}
          download
          className="rounded-md border border-slate-400/20 px-2.5 py-1.5 font-mono text-[10px] tracking-wide text-slate-400 uppercase transition hover:border-signal-400/40 hover:text-signal-300"
        >
          ↓ height .npy
        </a>
        <a
          href={assetUrl(`/static/${result.job_id}/metadata.json`)}
          target="_blank"
          rel="noreferrer"
          className="rounded-md border border-slate-400/20 px-2.5 py-1.5 font-mono text-[10px] tracking-wide text-slate-400 uppercase transition hover:border-signal-400/40 hover:text-signal-300"
        >
          ↗ metadata.json
        </a>
      </div>
    </Panel>
  )
}
