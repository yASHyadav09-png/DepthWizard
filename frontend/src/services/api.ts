import type { HealthResponse, ProcessResult } from '../types'

/** Dev uses the Vite proxy (same origin); override with VITE_API_BASE if the
 *  backend runs elsewhere. */
const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string = 'request_failed',
    readonly status: number = 0,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let detail = `Request failed with status ${response.status}.`
  let code = 'request_failed'
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') detail = body.detail
    if (typeof body?.error === 'string') code = body.error
  } catch {
    /* non-JSON error body — keep the generic message */
  }
  return new ApiError(detail, code, response.status)
}

export function assetUrl(path: string): string {
  return `${API_BASE}${path}`
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`, { signal })
  if (!response.ok) throw await parseError(response)
  return response.json()
}

export async function processImage(
  file: File,
  gsdM: number | null,
  signal?: AbortSignal,
): Promise<ProcessResult> {
  const form = new FormData()
  form.append('image', file)
  if (gsdM !== null) form.append('gsd_m', String(gsdM))

  let response: Response
  try {
    response = await fetch(`${API_BASE}/api/process`, {
      method: 'POST',
      body: form,
      signal,
    })
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err
    throw new ApiError(
      'Could not reach the DepthWizard backend. Is it running on port 8000?',
      'network_error',
    )
  }
  if (!response.ok) throw await parseError(response)
  return response.json()
}

export async function getResult(jobId: string): Promise<ProcessResult> {
  const response = await fetch(`${API_BASE}/api/results/${jobId}`)
  if (!response.ok) throw await parseError(response)
  return response.json()
}
