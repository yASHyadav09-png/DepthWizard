/** Shapes returned by the DepthWizard backend (Phase 3: metric nDSM; Phase 5: georeferenced DSM). */

export interface TerrainGrid {
  width: number
  height: number
  /** Row-major float32 little-endian heights in METRES, base64 encoded. */
  heights_b64: string
  encoding: string
  min_height: number
  max_height: number
  mean_height: number
  aspect_ratio: number
  /** Footprint in metres (source pixels x gsd_m). */
  plane_width: number
  plane_depth: number
  source_width: number
  source_height: number
  gsd_m: number
  /** Colour-scale range in metres, shared by elevation mode, 2D map and legend. */
  display_min: number
  display_max: number
  height_units: string
  /** Phase 5: "dsm" = heights_b64 holds surface ELEVATION (m, EGM2008); "ndsm" = height above ground. */
  surface_kind?: 'ndsm' | 'dsm'
  /** Phase 5 (georeferenced jobs): height above ground on the same grid, float32-le-base64. */
  ndsm_b64?: string | null
}

export type MetricValidity = 'valid' | 'uncertain'

/** Hand-off contract between the model side and the viewer (docs/phase3_demo.md). */
export interface HeightProduct {
  kind: 'ndsm' | 'dsm'
  description: string
  height_units: 'm'
  width: number
  height: number
  height_array: string
  gsd_m: number
  gsd_source: 'user' | 'assumed_training_gsd' | 'geotiff'
  metric_validity: MetricValidity
  validity_note: string
  nodata: unknown
  crs: string | null
  /** GDAL order [a, b, c, d, e, f]: x = c + col*a, y = f + row*e (north-up). */
  transform: number[] | null
  vertical_datum: string | null
  /** Phase 5 (GeoTIFF input) */
  ndsm_array?: string
  crs_epsg?: number | null
  crs_wkt?: string
  bounds?: number[]
  /** lon/lat (WGS 84) of the image corners: top-left, top-right, bottom-right, bottom-left. */
  corners_lonlat?: [number, number][]
  dem?: {
    how: string
    source?: string
    vertical_datum?: string
    ground_filter?: { method: string; window_m: number; sigma_px: number }
    coverage_fraction?: number
  }
  gcp?: {
    model: string
    params: { a: number; b: number; c: number }
    n_used?: number
    n_total?: number
    [key: string]: unknown
  } | null
  geotiffs?: Record<string, string>
  notes?: string[]
  model: {
    run: string
    checkpoint_sha256: string
    git_commit: string | null
    val_rmse_m: number
  }
}

export interface Statistics {
  min: number
  max: number
  mean: number
  median: number
  std: number
  p05: number
  p95: number
  p99: number
  frac_above_2m: number
  units: string
  display_min: number
  display_max: number
  is_metric: boolean
  metric_validity: MetricValidity
  georeferenced: boolean
  /** Phase 5 DSM jobs: elevation ranges in m (EGM2008). */
  elevation?: { dsm_min: number; dsm_max: number; dtm_min: number; dtm_max: number; relief_m: number } | Record<string, never>
}

export interface Assets {
  original: string
  texture: string
  height_map: string
  hillshade: string
  height_array: string
  dsm_array?: string
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
  }
  model: {
    name: string
    run: string
    checkpoint: string
    checkpoint_sha256: string
    epoch: number
    val_rmse_m: number
    git_commit: string | null
    type: string
    precision: string
    device: string
    device_label: string
  }
  height_product: HeightProduct
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

export interface MetricBlock {
  n: number
  mae: number | null
  rmse: number | null
  pearson_r: number | null
  bias: number | null
}

/** Response of POST /api/results/{job}/validate (Phase 7d). error = predicted - reference. */
export interface ValidationResult {
  job_id: string
  reference: {
    filename: string
    valid_pixels: number
    valid_fraction: number
    units: string
    crs?: string
    vertical_crs?: string | null
    datum_conversion?: string
    resampling?: string
  }
  /** What was compared: the job's DSM (elevation) or nDSM (height above ground). */
  target: 'dsm' | 'ndsm'
  definition: string
  overall: MetricBlock
  per_height_band: Record<string, MetricBlock>
  error_map: string
  error_limit_m: number
  error_grid: { width: number; height: number; encoding: string; values_b64: string }
}
