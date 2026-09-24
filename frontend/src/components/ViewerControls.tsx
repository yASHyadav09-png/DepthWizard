import type { ViewMode } from '../types'
import type { NavMode } from './TerrainViewer'
import { Panel, Slider, Toggle } from './ui'

export function ViewerControls({
  enabled,
  viewMode,
  onViewMode,
  wireframe,
  onWireframe,
  exaggeration,
  onExaggeration,
  navMode,
  onNavMode,
  onResetCamera,
}: {
  enabled: boolean
  viewMode: ViewMode
  onViewMode: (mode: ViewMode) => void
  wireframe: boolean
  onWireframe: (next: boolean) => void
  exaggeration: number
  onExaggeration: (value: number) => void
  navMode: NavMode
  onNavMode: (mode: NavMode) => void
  onResetCamera: () => void
}) {
  return (
    <Panel title="Viewer Controls" bodyClassName="space-y-3 p-4">
      <div className="grid grid-cols-2 gap-1 rounded-lg border border-slate-400/12 bg-slate-400/5 p-1">
        {(['orbit', 'fly'] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            disabled={!enabled}
            onClick={() => onNavMode(mode)}
            className={`rounded-md px-2 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${
              navMode === mode
                ? 'bg-signal-500/20 text-signal-300'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {mode === 'orbit' ? 'Orbit' : 'Flythrough'}
          </button>
        ))}
      </div>

      <Toggle
        label="RGB texture"
        hint="Project the original image onto the terrain"
        checked={viewMode === 'textured'}
        disabled={!enabled}
        onChange={(next) => onViewMode(next ? 'textured' : 'elevation')}
      />
      <Toggle
        label="Elevation view"
        hint="Colour the surface by relative height instead of RGB"
        checked={viewMode === 'elevation'}
        disabled={!enabled}
        onChange={(next) => onViewMode(next ? 'elevation' : 'textured')}
      />
      <Toggle
        label="Wireframe"
        hint="Overlay the terrain mesh wireframe"
        checked={wireframe}
        disabled={!enabled}
        onChange={onWireframe}
      />

      <Slider
        label="Height exaggeration"
        value={exaggeration}
        min={0}
        max={1.2}
        step={0.01}
        disabled={!enabled}
        onChange={onExaggeration}
        format={(v) => `${v.toFixed(2)}×`}
      />

      <button
        type="button"
        onClick={onResetCamera}
        disabled={!enabled}
        className="w-full rounded-lg border border-slate-400/20 px-3 py-2 text-xs font-medium text-slate-300 transition hover:border-signal-400/40 hover:text-signal-300 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Reset camera
      </button>

      <p className="text-[10px] leading-relaxed text-slate-600">
        Exaggeration is a <span className="text-slate-500">visual</span> scale on a
        unitless 0–1 field. It does not represent metres and does not change the
        underlying rDSM.
      </p>
    </Panel>
  )
}
