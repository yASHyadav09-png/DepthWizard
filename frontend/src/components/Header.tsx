import type { HealthResponse } from '../types'
import { Badge, Dot } from './ui'

export function Header({
  health,
  healthError,
}: {
  health: HealthResponse | null
  healthError: string | null
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-400/12 bg-abyss-950/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1800px] flex-wrap items-center gap-x-5 gap-y-3 px-5 py-3">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl border border-signal-400/30 bg-linear-to-br from-signal-500/25 to-relief-500/20">
            <svg
              viewBox="0 0 24 24"
              className="size-5.5 fill-none stroke-signal-300"
              strokeWidth={1.6}
            >
              <path
                d="M2 17.5 8 11l4 3.2L16.2 9 22 15.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path d="M2 21h20" strokeLinecap="round" opacity={0.5} />
              <circle cx="17.5" cy="4.8" r="2.3" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg leading-tight font-semibold tracking-tight text-slate-100">
              Depth<span className="text-signal-400">Wizard</span>
            </h1>
            <p className="text-[11px] leading-tight text-slate-500">
              Single-View Height Estimation &amp; 3D Flythrough
            </p>
          </div>
        </div>

        <div className="hidden h-8 w-px bg-slate-400/12 lg:block" />

        <div className="hidden flex-col lg:flex">
          <span className="font-mono text-[10px] tracking-widest text-slate-500 uppercase">
            SIH 2026 · ISRO
          </span>
          <span className="font-mono text-xs text-slate-300">PS 26175</span>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Badge tone="signal">Stage 1 · Relative DSM</Badge>
          {health && (
            <>
              <Badge tone="neutral">{health.model_name}</Badge>
              <Badge tone={health.cuda_available ? 'relief' : 'neutral'}>
                {health.device}
              </Badge>
            </>
          )}
          <Badge tone={healthError ? 'danger' : health ? 'relief' : 'warn'}>
            <Dot tone={healthError ? 'danger' : health ? 'relief' : 'warn'} />
            {healthError ? 'Backend offline' : health ? 'Backend online' : 'Connecting'}
          </Badge>
        </div>
      </div>
    </header>
  )
}
