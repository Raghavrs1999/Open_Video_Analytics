/**
 * DetectionCanvas
 *
 * A <canvas> element that is absolutely positioned over the video.
 * It reads live detections from the Zustand store and draws:
 *   - Bounding boxes (colour-coded per class)
 *   - Track ID + class label + confidence badge
 *   - Thin centroid dot
 *
 * The canvas dimensions are kept in sync with the displayed video via
 * a ResizeObserver on the parent wrapper element.
 */

import { useEffect, useRef, useCallback } from 'react'
import { useStore, getClassColor } from '../store'

// Hex → rgba() helper
function hexRgba(hex, alpha = 1) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

export default function DetectionCanvas({ wrapperRef }) {
  const canvasRef   = useRef(null)
  const detections  = useStore((s) => s.detections)
  const showLabels  = useStore((s) => s.showLabels)
  const frameWidth  = useStore((s) => s.videoWidth)
  const frameHeight = useStore((s) => s.videoHeight)

  // Sync canvas size to video wrapper
  useEffect(() => {
    if (!wrapperRef?.current) return
    const obs = new ResizeObserver(() => {
      const canvas = canvasRef.current
      const wr     = wrapperRef.current
      if (!canvas || !wr) return
      canvas.width  = wr.clientWidth
      canvas.height = wr.clientHeight
    })
    obs.observe(wrapperRef.current)
    return () => obs.disconnect()
  }, [wrapperRef])

  // Draw on every detections change
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!detections.length) return

    // Compute letterbox scaling: the raw frame is scaled to fit inside canvas
    const canvasW = canvas.width
    const canvasH = canvas.height
    const srcW    = frameWidth  || canvasW
    const srcH    = frameHeight || canvasH

    const scale  = Math.min(canvasW / srcW, canvasH / srcH)
    const offsetX = (canvasW - srcW * scale) / 2
    const offsetY = (canvasH - srcH * scale) / 2

    const toCanvasX = (x) => x * scale + offsetX
    const toCanvasY = (y) => y * scale + offsetY

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox
      const cx1 = toCanvasX(x1)
      const cy1 = toCanvasY(y1)
      const cx2 = toCanvasX(x2)
      const cy2 = toCanvasY(y2)
      const bw  = cx2 - cx1
      const bh  = cy2 - cy1

      const hex   = getClassColor(det.class_id)
      const solid = hexRgba(hex, 1.0)
      const fill  = hexRgba(hex, 0.08)
      const glow  = hexRgba(hex, 0.4)

      // Box fill
      ctx.fillStyle = fill
      ctx.fillRect(cx1, cy1, bw, bh)

      // Box border with glow
      ctx.shadowColor  = glow
      ctx.shadowBlur   = 8
      ctx.strokeStyle  = solid
      ctx.lineWidth    = 1.5
      ctx.strokeRect(cx1, cy1, bw, bh)
      ctx.shadowBlur   = 0

      // Corner accents
      const cs = Math.min(bw, bh, 12)
      ctx.lineWidth = 2
      ctx.strokeStyle = solid
      ;[
        [cx1, cy1,  1,  1],
        [cx2, cy1, -1,  1],
        [cx1, cy2,  1, -1],
        [cx2, cy2, -1, -1],
      ].forEach(([sx, sy, dx, dy]) => {
        ctx.beginPath()
        ctx.moveTo(sx + dx * cs, sy)
        ctx.lineTo(sx, sy)
        ctx.lineTo(sx, sy + dy * cs)
        ctx.stroke()
      })

      // Centroid dot
      const [centX, centY] = det.centroid
      const ccx = toCanvasX(centX)
      const ccy = toCanvasY(centY)
      ctx.beginPath()
      ctx.arc(ccx, ccy, 3, 0, Math.PI * 2)
      ctx.fillStyle = solid
      ctx.fill()

      // Label badge
      if (showLabels) {
        const label = `${det.class_name} #${det.track_id} ${Math.round(det.confidence * 100)}%`
        const fontSize = Math.max(10, Math.min(13, bw / 5))
        ctx.font = `500 ${fontSize}px 'JetBrains Mono', monospace`
        const tm   = ctx.measureText(label)
        const padH = 4, padV = 2
        const bx = cx1
        const by = cy1 - fontSize - padV * 2 - 2

        // Badge background
        ctx.fillStyle = hexRgba(hex, 0.85)
        ctx.beginPath()
        ctx.roundRect(bx, by, tm.width + padH * 2, fontSize + padV * 2, 3)
        ctx.fill()

        ctx.fillStyle = '#090d17'
        ctx.fillText(label, bx + padH, by + fontSize + padV - 1)
      }
    })
  }, [detections, showLabels, frameWidth, frameHeight])

  useEffect(() => { draw() }, [draw])

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
    />
  )
}
