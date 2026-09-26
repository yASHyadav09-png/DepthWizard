/**
 * Top-down minimap: the RGB image (= the terrain extent), the camera position and its
 * viewing direction with a field-of-view wedge. Reads the scene's telemetry on its own
 * animation frame, so it never re-renders React.
 */
import { useEffect, useRef } from 'react'
import type { Telemetry } from './ExplorerScene'

const SIZE = 190 // longest side in CSS px

export function Minimap({
  textureUrl,
  width,
  depth,
  telemetry,
  fovDeg = 50,
}: {
  textureUrl: string
  width: number // terrain extent in metres along X
  depth: number // along Z
  telemetry: Telemetry
  fovDeg?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scale = SIZE / Math.max(width, depth) // px per metre
  const cw = Math.round(width * scale)
  const ch = Math.round(depth * scale)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = cw * dpr
    canvas.height = ch * dpr
    ctx.scale(dpr, dpr)
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = textureUrl
    let raf = 0
    const half = ((fovDeg / 2) * Math.PI) / 180

    const draw = () => {
      ctx.clearRect(0, 0, cw, ch)
      if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, cw, ch)
      else {
        ctx.fillStyle = '#1e293b'
        ctx.fillRect(0, 0, cw, ch)
      }
      ctx.strokeStyle = 'rgba(56,189,248,0.9)'
      ctx.lineWidth = 1.5
      ctx.strokeRect(0.75, 0.75, cw - 1.5, ch - 1.5) // terrain extent

      const { x, z } = telemetry.camera
      const inside = x >= 0 && x <= width && z >= 0 && z <= depth
      const px = Math.min(Math.max(x * scale, 4), cw - 4)
      const pz = Math.min(Math.max(z * scale, 4), ch - 4)
      const heading = Math.atan2(telemetry.dir.z, telemetry.dir.x) // X right, Z down on the map
      const reach = 34
      ctx.fillStyle = 'rgba(251,191,36,0.28)'
      ctx.beginPath()
      ctx.moveTo(px, pz)
      ctx.lineTo(px + Math.cos(heading - half) * reach, pz + Math.sin(heading - half) * reach)
      ctx.lineTo(px + Math.cos(heading + half) * reach, pz + Math.sin(heading + half) * reach)
      ctx.closePath()
      ctx.fill()
      ctx.fillStyle = inside ? '#fbbf24' : '#f87171'
      ctx.strokeStyle = '#0b1220'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(px, pz, 4.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [textureUrl, cw, ch, scale, width, depth, telemetry, fovDeg])

  return (
    <div className="rounded-lg border border-slate-400/20 bg-abyss-950/85 p-1.5 backdrop-blur">
      <canvas ref={canvasRef} style={{ width: cw, height: ch }} className="block rounded" />
      <p className="mt-1 text-center font-mono text-[9px] text-slate-500">
        {Math.round(width)} × {Math.round(depth)} m · N ↑ (image top)
      </p>
    </div>
  )
}
