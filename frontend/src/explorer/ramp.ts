/** Height colour ramp shared by the Explorer mesh and its legend (low -> high). */
export const RAMP: [number, [number, number, number]][] = [
  [0.0, [0.10, 0.16, 0.36]],
  [0.2, [0.11, 0.38, 0.52]],
  [0.4, [0.13, 0.55, 0.42]],
  [0.6, [0.62, 0.73, 0.31]],
  [0.8, [0.86, 0.62, 0.32]],
  [1.0, [0.98, 0.98, 0.98]],
]

export function rampColor(t: number, out: [number, number, number]) {
  const value = Math.min(1, Math.max(0, t))
  let i = 1
  while (i < RAMP.length - 1 && value > RAMP[i][0]) i += 1
  const [t0, c0] = RAMP[i - 1]
  const [t1, c1] = RAMP[i]
  const f = t1 === t0 ? 0 : (value - t0) / (t1 - t0)
  out[0] = c0[0] + (c1[0] - c0[0]) * f
  out[1] = c0[1] + (c1[1] - c0[1]) * f
  out[2] = c0[2] + (c1[2] - c0[2]) * f
}

export function rampCss(): string {
  return RAMP.map(
    ([t, [r, g, b]]) => `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${t * 100}%`,
  ).join(', ')
}

/** Slope ramp in degrees: flat (green) -> 15 (yellow) -> 30 (orange) -> 45+ (red) -> 90 (dark red). */
export const SLOPE_RAMP: [number, [number, number, number]][] = [
  [0 / 90, [0.13, 0.55, 0.33]],
  [15 / 90, [0.93, 0.85, 0.25]],
  [30 / 90, [0.96, 0.55, 0.16]],
  [45 / 90, [0.86, 0.2, 0.15]],
  [1, [0.45, 0.05, 0.08]],
]

function rampFrom(stops: [number, [number, number, number]][]) {
  return (t: number, out: [number, number, number]) => {
    const value = Math.min(1, Math.max(0, t))
    let i = 1
    while (i < stops.length - 1 && value > stops[i][0]) i += 1
    const [t0, c0] = stops[i - 1]
    const [t1, c1] = stops[i]
    const f = t1 === t0 ? 0 : (value - t0) / (t1 - t0)
    out[0] = c0[0] + (c1[0] - c0[0]) * f
    out[1] = c0[1] + (c1[1] - c0[1]) * f
    out[2] = c0[2] + (c1[2] - c0[2]) * f
  }
}

export const slopeColor = rampFrom(SLOPE_RAMP)

export function slopeRampCss(): string {
  return SLOPE_RAMP.map(
    ([t, [r, g, b]]) => `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${t * 100}%`,
  ).join(', ')
}

/** Diverging error ramp: blue = prediction below reference, white = 0, red = above. */
export const ERROR_RAMP: [number, [number, number, number]][] = [
  [0, [0.13, 0.4, 0.75]],
  [0.5, [0.97, 0.97, 0.97]],
  [1, [0.8, 0.15, 0.15]],
]
export const errorColor = rampFrom(ERROR_RAMP)
export function errorRampCss(): string {
  return ERROR_RAMP.map(
    ([t, [r, g, b]]) => `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${t * 100}%`,
  ).join(', ')
}
