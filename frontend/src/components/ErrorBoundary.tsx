import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Changing this value clears a previous error (e.g. a new job id). */
  resetKey?: string | number
  fallbackTitle?: string
}

interface State {
  message: string | null
  resetKey?: string | number
}

/** Keeps a bad terrain buffer or a WebGL failure from blanking the dashboard. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { message: null, resetKey: this.props.resetKey }

  static getDerivedStateFromProps(props: Props, state: State): State | null {
    if (props.resetKey !== state.resetKey) {
      return { message: null, resetKey: props.resetKey }
    }
    return null
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { message: error.message || 'Unknown rendering error.' }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('DepthWizard render error:', error, info.componentStack)
  }

  render() {
    if (this.state.message) {
      return (
        <div className="grid size-full place-items-center p-6 text-center">
          <div>
            <p className="text-sm font-medium text-rose-300">
              {this.props.fallbackTitle ?? 'Rendering failed'}
            </p>
            <p className="mt-1.5 max-w-sm font-mono text-[11px] leading-relaxed text-slate-500">
              {this.state.message}
            </p>
            <p className="mt-3 text-xs text-slate-600">
              Try another image, or check that your browser has WebGL enabled.
            </p>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
