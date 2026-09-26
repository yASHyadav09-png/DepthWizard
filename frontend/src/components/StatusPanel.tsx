import type { PipelineStage, ProcessResult } from '../types'
import { Panel } from './ui'

const STEPS = [
  { key: 'upload', label: 'Upload & validate', timing: 'load_image_ms' },
  { key: 'depth', label: 'nDSM inference (trained DA-V2-S, tiled)', timing: 'ndsm_inference_ms' },
  { key: 'height', label: 'Height statistics (m)', timing: 'statistics_ms' },
  { key: 'terrain', label: 'Terrain mesh grid', timing: 'terrain_generation_ms' },
  { key: 'export', label: 'Artefact export', timing: 'artefact_export_ms' },
] as const

/** How far the visual stepper has progressed for a given pipeline stage. */
function activeIndex(stage: PipelineStage): number {
  switch (stage) {
    case 'uploading':
      return 0
    case 'inferring':
      return 1
    case 'building':
      return 3
    case 'done':
      return STEPS.length
    default:
      return -1
  }
}

function formatMs(ms: number | undefined): string {
  if (ms === undefined) return '—'
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`
}

export function StatusPanel({
  stage,
  error,
  result,
}: {
  stage: PipelineStage
  error: string | null
  result: ProcessResult | null
}) {
  const current = activeIndex(stage)
  const failed = stage === 'error'

  return (
    <Panel
      title="Processing Status"
      aside={
        result && stage === 'done' ? (
          <span className="font-mono text-[10px] text-relief-400 normal-case">
            {formatMs(result.timings_ms.total_ms)} total
          </span>
        ) : null
      }
      bodyClassName="space-y-2.5 p-4"
    >
      {STEPS.map((step, index) => {
        const complete = stage === 'done' || index < current
        const running = !failed && index === current && stage !== 'done'
        const errored = failed && index === Math.max(current, 0)

        return (
          <div key={step.key} className="flex items-center gap-3">
            <span
              className={`grid size-5 shrink-0 place-items-center rounded-full border text-[10px] font-bold ${
                errored
                  ? 'border-rose-400/50 bg-rose-400/15 text-rose-300'
                  : complete
                    ? 'border-relief-400/50 bg-relief-400/15 text-relief-400'
                    : running
                      ? 'border-signal-400/60 bg-signal-400/15 text-signal-300'
                      : 'border-slate-500/30 text-slate-600'
              }`}
            >
              {errored ? '!' : complete ? '✓' : index + 1}
            </span>

            <div className="min-w-0 flex-1">
              <p
                className={`truncate text-xs ${
                  complete || running ? 'text-slate-300' : 'text-slate-600'
                }`}
              >
                {step.label}
              </p>
              {running && (
                <div className="animate-sweep relative mt-1 h-0.5 overflow-hidden rounded-full bg-slate-700" />
              )}
            </div>

            <span className="shrink-0 font-mono text-[10px] text-slate-500">
              {result ? formatMs(result.timings_ms[step.timing]) : running ? '···' : ''}
            </span>
          </div>
        )
      })}

      {error && (
        <p className="rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-xs leading-relaxed text-rose-200">
          {error}
        </p>
      )}

      {stage === 'idle' && !error && (
        <p className="pt-1 text-xs text-slate-500">
          Upload an image and press <span className="text-slate-300">Generate Terrain</span> to
          run the pipeline.
        </p>
      )}
    </Panel>
  )
}
