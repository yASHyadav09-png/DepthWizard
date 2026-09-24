/** Shapes returned by the DepthWizard Stage 1 backend. */

export interface TerrainGrid {
  width: number
  height: number
  /** Row-major float32 little-endian heights in [0,1], base64 encoded. */
  heights_b64: string
  encoding: string
  min_height: number
  max_height: number
  mean_height: number
  aspect_ratio: number
  plane_width: number
  plane_depth: number
  source_width: number
  source_height: number
  height_units: string
}

export interface Statistics {
  image_width: number
  image_height: number
  min_relative_height: number
  max_relative_height: number
  mean_relative_height: number
  median_relative_height: number
  std_relative_height: number
  p05_relative_height: number
  p95_relative_height: number
  height_units: string
  model_name: string
  processing_device: string
  is_metric: boolean
  georeferenced: boolean
}

export interface Assets {
  original: string
  texture: string
  depth_map: string
  relative_dsm: string
  height_array: string
}

export interface ProcessResult {
  job_id: string
  status: string
  created_at: string
  stage: number
  stage_label: string
  source: {
    filename: string
    bytes: number
    width: number
    height: number
    inference_width: number
    inference_height: number
  }
  model: {
    name: string
    checkpoint: string
    type: string
    precision: string
    device: string
    device_label: string
  }
  statistics: Statistics
  assets: Assets
  terrain: TerrainGrid
  timings_ms: Record<string, number>
  disclaimer: string
  metadata: Record<string, unknown>
}

export interface HealthResponse {
  status: string
  app: string
  version: string
  stage: string
  model_name: string
  model_checkpoint: string
  model_loaded: boolean
  device: string
  cuda_available: boolean
  torch_version: string
  terrain_resolution: number
  max_upload_mb: number
}

export type PipelineStage =
  | 'idle'
  | 'uploading'
  | 'inferring'
  | 'building'
  | 'done'
  | 'error'

export type ViewMode = 'textured' | 'elevation'
