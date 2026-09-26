/** DOM panels for the 7b tools: height profile chart and two-point measurement. */
import type { Measurement, Profile } from './measure'

const W = 640
const H = 150
const PAD = { l: 44, r: 12, t: 12, b: 26 }

function ticks(min: number, max: number, n: number): number[] {
  const span = max - min
  if (span <= 0) return [min]
  const raw = span / n
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(6)))
  return out
}

export function ProfileChart({ profile, onClose }: { profile: Profile; onClose: () => void }) {
  const { samples, length, minH, maxH, maxRise } = profile
  const y0 = Math.min(0, minH)
  const y1 = Math.max(maxH + 2, y0 + 5)
  const L = Math.max(length, 1e-6)
  const sx = (d: number) => PAD.l + (d / L) * (W - PAD.l - PAD.r)
  const sy = (h: number) => PAD.t + (1 - (h - y0) / (y1 - y0)) * (H - PAD.t - PAD.b)
  const line = samples.map((s) => `${sx(s.d).toFixed(1)},${sy(s.h).toFixed(1)}`).join(' ')
  const area = `${sx(0)},${sy(y0)} ${line} ${sx(L)},${sy(y0)}`
  return (
    <div className="rounded-lg border border-slate-400/20 bg-abyss-950/92 p-2 backdrop-blur">
      <div className="mb-1 flex items-center gap-3 font-mono text-[10px] text-slate-400">
        <span className="tracking-widest text-slate-500 uppercase">Height profile</span>
        <span>
          length <b className="text-slate-100">{length.toFixed(1)} m</b>
        </span>
        <span>
          min <b className="text-slate-100">{minH.toFixed(1)} m</b> · max <b className="text-slate-100">{maxH.toFixed(1)} m</b>
        </span>
        <span>
          steepest <b className="text-slate-100">{maxRise.toFixed(0)}°</b>
        </span>
        <button type="button" onClick={onClose} className="ml-auto text-slate-500 hover:text-slate-200">
          ✕
        </button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-[640px] max-w-full">
        {ticks(y0, y1, 4).map((v) => (
          <g key={`y${v}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={sy(v)} y2={sy(v)} stroke="#334155" strokeWidth={0.6} />
            <text x={PAD.l - 5} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="#94a3b8">
              {v} m
            </text>
          </g>
        ))}
        {ticks(0, L, 6).map((v) => (
          <text key={`x${v}`} x={sx(v)} y={H - 8} textAnchor="middle" fontSize={9} fill="#94a3b8">
            {v} m
          </text>
        ))}
        <polygon points={area} fill="rgba(56,189,248,0.18)" />
        <polyline points={line} fill="none" stroke="#38bdf8" strokeWidth={1.4} />
        <text x={sx(0)} y={PAD.t + 8} fontSize={9} fill="#fbbf24">
          A
        </text>
        <text x={sx(L) - 8} y={PAD.t + 8} fontSize={9} fill="#fbbf24">
          B
        </text>
      </svg>
      <p className="font-mono text-[9px] text-slate-600">
        height above ground along A→B, sampled every {(samples.length > 1 ? samples[1].d : 0).toFixed(2)} m · vertical
        axis is scaled to fit
      </p>
    </div>
  )
}

export function MeasureCard({ m, onClose }: { m: Measurement; onClose: () => void }) {
  const row = (k: string, v: string) => (
    <div className="flex justify-between gap-4">
      <span className="text-slate-500">{k}</span>
      <b className="text-slate-100">{v}</b>
    </div>
  )
  return (
    <div className="w-64 space-y-0.5 rounded-lg border border-slate-400/20 bg-abyss-950/92 p-2.5 font-mono text-[11px] backdrop-blur">
      <div className="mb-1 flex items-center">
        <span className="text-[10px] tracking-widest text-slate-500 uppercase">Measurement</span>
        <button type="button" onClick={onClose} className="ml-auto text-slate-500 hover:text-slate-200">
          ✕
        </button>
      </div>
      {row('ground distance', `${m.horizontal.toFixed(2)} m`)}
      {row('height at A', `${m.hA.toFixed(2)} m`)}
      {row('height at B', `${m.hB.toFixed(2)} m`)}
      {row('height difference B−A', `${m.dh >= 0 ? '+' : ''}${m.dh.toFixed(2)} m`)}
      {row('3D distance', `${m.slopeDistance.toFixed(2)} m`)}
    </div>
  )
}
