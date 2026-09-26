import type { ReactNode } from 'react'

export function Panel({
  title,
  aside,
  children,
  className = '',
  bodyClassName = 'p-4',
}: {
  title?: string
  aside?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={`panel flex flex-col overflow-hidden ${className}`}>
      {title && (
        <header className="panel-heading">
          <span>{title}</span>
          {aside}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  )
}

const TONES = {
  neutral: 'border-slate-400/20 bg-slate-400/10 text-slate-300',
  signal: 'border-signal-400/30 bg-signal-400/10 text-signal-300',
  relief: 'border-relief-400/30 bg-relief-400/10 text-relief-400',
  warn: 'border-warn-400/35 bg-warn-400/10 text-warn-400',
  danger: 'border-rose-400/35 bg-rose-400/10 text-rose-300',
} as const

export function Badge({
  children,
  tone = 'neutral',
  className = '',
}: {
  children: ReactNode
  tone?: keyof typeof TONES
  className?: string
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono text-[10px] font-semibold tracking-wider uppercase ${TONES[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function Dot({ tone = 'relief' }: { tone?: 'relief' | 'warn' | 'danger' }) {
  const color =
    tone === 'relief'
      ? 'bg-relief-400'
      : tone === 'warn'
        ? 'bg-warn-400'
        : 'bg-rose-400'
  return (
    <span className="relative flex h-1.5 w-1.5">
      <span
        className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${color}`}
      />
      <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${color}`} />
    </span>
  )
}

/** The disclaimer that must never be missing from a Stage 1 screen. */
/** States how far the metres on screen can be trusted (height_product.metric_validity). */
export function MetricNotice({
  validity,
  note,
  className = '',
}: {
  validity: 'valid' | 'uncertain'
  note: string
  className?: string
}) {
  const ok = validity === 'valid'
  return (
    <div
      className={`flex items-start gap-2.5 rounded-lg border px-3 py-2 ${
        ok ? 'border-relief-400/30 bg-relief-400/8' : 'border-warn-400/30 bg-warn-400/8'
      } ${className}`}
    >
      <svg viewBox="0 0 20 20" className={`mt-0.5 size-4 shrink-0 ${ok ? 'fill-relief-400' : 'fill-warn-400'}`}>
        <path d="M10 1.8 1 17.5h18L10 1.8Zm0 4.6a.9.9 0 0 1 .9.9v4.4a.9.9 0 1 1-1.8 0V7.3a.9.9 0 0 1 .9-.9Zm0 8.9a1.05 1.05 0 1 1 0-2.1 1.05 1.05 0 0 1 0 2.1Z" />
      </svg>
      <p className={`font-mono text-[11px] leading-snug font-semibold tracking-wide ${ok ? 'text-relief-400' : 'text-warn-400'}`}>
        {ok ? 'HEIGHT ABOVE GROUND · METRES' : 'ESTIMATED METRES · RESOLUTION UNCERTAIN'}
        <span className="mt-0.5 block font-sans font-normal tracking-normal text-slate-400">{note}</span>
      </p>
    </div>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
  hint,
  disabled = false,
}: {
  label: string
  checked: boolean
  onChange: (next: boolean) => void
  hint?: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={hint ?? label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`group flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-2 text-left transition disabled:cursor-not-allowed disabled:opacity-40 ${
        checked
          ? 'border-signal-400/40 bg-signal-400/10'
          : 'border-slate-400/12 bg-slate-400/5 hover:border-slate-400/25'
      }`}
    >
      <span
        className={`text-xs font-medium ${checked ? 'text-signal-300' : 'text-slate-400'}`}
      >
        {label}
      </span>
      <span
        className={`relative h-4 w-8 shrink-0 rounded-full transition ${
          checked ? 'bg-signal-500' : 'bg-slate-600'
        }`}
      >
        <span
          className={`absolute top-0.5 size-3 rounded-full bg-white transition-all ${
            checked ? 'left-4.5' : 'left-0.5'
          }`}
        />
      </span>
    </button>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v: number) => v.toFixed(2),
  disabled = false,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (next: number) => void
  format?: (value: number) => string
  disabled?: boolean
}) {
  return (
    <label className="block space-y-1.5">
      <span className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-400">{label}</span>
        <span className="font-mono text-[11px] text-signal-300">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-signal-400 disabled:cursor-not-allowed disabled:opacity-40"
      />
    </label>
  )
}
