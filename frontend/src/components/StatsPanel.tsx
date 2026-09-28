import type { ProcessResult } from '../types'
import { assetUrl } from '../services/api'
import { MetricNotice, Panel } from './ui'
import { formatGsd } from '../services/geo'

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
  const hp = result.height_product
  const m = (value: number) => `${value.toFixed(2)} m`
  const elev = s.elevation && 'dsm_min' in s.elevation ? s.elevation : null
  const btn =
    'rounded-md border border-slate-400/20 px-2.5 py-1.5 font-mono text-[10px] tracking-wide text-slate-400 uppercase transition hover:border-signal-400/40 hover:text-signal-300'

  return (
    <Panel title="Statistics" bodyClassName="space-y-3 p-4">
      <MetricNotice validity={hp.metric_validity} note={hp.validity_note} />

      <div className="grid grid-cols-3 gap-2">
        <Tile label="Median" value={m(s.median)} />
        <Tile label="P95" value={m(s.p95)} />
        <Tile label="Max" value={m(s.max)} />
      </div>
      {hp.kind === 'dsm' && (
        <p className="-mt-1 text-[10px] text-slate-600">tiles: height above ground (nDSM)</p>
      )}

      <div>
        <Row label="Image" value={`${result.source.width} × ${result.source.height} px`} />
        <Row
          label="Ground resolution"
          value={`${formatGsd(hp.gsd_m)} m/px${hp.gsd_source === 'assumed_training_gsd' ? ' (assumed)' : ''}`}
        />
        <Row
          label="Footprint"
          value={`${result.terrain.plane_width.toFixed(0)} × ${result.terrain.plane_depth.toFixed(0)} m`}
        />
        {elev && (
          <>
            <Row label="Elevation (DSM)" value={`${elev.dsm_min.toFixed(1)} – ${elev.dsm_max.toFixed(1)} m`} accent />
            <Row label="Ground (DTM)" value={`${elev.dtm_min.toFixed(1)} – ${elev.dtm_max.toFixed(1)} m`} />
            <Row label="Vertical datum" value={hp.vertical_datum ?? '–'} />
            <Row
              label="DEM"
              value={`${hp.dem?.source ?? '–'} · ${hp.dem?.how ?? ''}${
                hp.dem?.ground_filter ? ` · filter ${hp.dem.ground_filter.window_m} m` : ''
              }`}
            />
            <Row
              label="GCP correction"
              value={
                hp.gcp
                  ? `${hp.gcp.model} · offset ${hp.gcp.params.a >= 0 ? '+' : ''}${hp.gcp.params.a.toFixed(2)} m`
                  : 'none'
              }
            />
          </>
        )}
        {hp.kind === 'ndsm' && hp.crs && hp.dem && <Row label="DEM" value={hp.dem.how} />}
        <Row label="Mean height" value={m(s.mean)} />
        <Row label="P05 – P99" value={`${s.p05.toFixed(2)} – ${s.p99.toFixed(2)} m`} />
        <Row label="Area above 2 m" value={`${(s.frac_above_2m * 100).toFixed(1)} %`} />
        <Row label="Product" value={`${hp.kind.toUpperCase()} · ${hp.description}`} />
        <Row label="Model" value={result.model.name} accent />
        <Row label="Model run / epoch" value={`${result.model.run} / ${result.model.epoch}`} />
        <Row label="Checkpoint sha256" value={result.model.checkpoint_sha256.slice(0, 16) + '…'} />
        <Row label="Val RMSE (GAMUS)" value={m(result.model.val_rmse_m)} />
        <Row label="Processing device" value={result.model.device_label} accent />
        <Row label="Georeferenced" value={s.georeferenced ? `yes · ${hp.crs}` : 'no (JPG/PNG input)'} />
        <Row label="Job ID" value={result.job_id} />
      </div>

      {hp.notes && hp.notes.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-4 text-[10px] leading-snug text-amber-300/80">
          {hp.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap gap-2 pt-0.5">
        {Object.entries(hp.geotiffs ?? {}).map(([k, url]) => (
          <a key={k} href={assetUrl(url)} download className={btn}>
            ↓ {k} .tif
          </a>
        ))}
        <a href={assetUrl(hp.ndsm_array ?? result.assets.height_array)} download className={btn}>
          ↓ nDSM .npy (m)
        </a>
        <a
          // metadata.json sits next to the other per-job assets; derive its path from one
          // of those (which already carries the correct, possibly non-"/static", prefix --
          // see RasterPanel.tsx) instead of hardcoding "/static".
          href={assetUrl(result.assets.original.replace(/[^/]+$/, 'metadata.json'))}
          target="_blank"
          rel="noreferrer"
          className={btn}
        >
          ↗ metadata.json
        </a>
      </div>
    </Panel>
  )
}
