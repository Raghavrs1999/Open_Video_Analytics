/**
 * VideoPlayer — Phase 2 upgraded
 *
 * Renders the live video stream in one of two modes:
 *   1. HLS mode  — uses hls.js to play the MediaMTX HLS playlist.
 *                  Sub-second H.264 stream with burned-in YOLO boxes.
 *                  Activated when `hlsUrl` is set in the store.
 *   2. MJPEG mode — falls back to the Redis-relayed MJPEG <img> stream.
 *                   Works without MediaMTX.
 *
 * A transparent <canvas> overlays the <video>/<img> in both modes
 * to draw WebSocket-sourced bounding boxes in real time (canvas overlay
 * is always accurate to the latest WS frame regardless of stream mode).
 */

import { useRef, useEffect, useState } from 'react'
import Hls from 'hls.js'
import { useStore } from '../store'
import DetectionCanvas from './DetectionCanvas'

// ── HLS Player ────────────────────────────────────────────────────────────

function HlsPlayer({ hlsUrl, videoRef }) {
  const [hlsError, setHlsError] = useState(null)
  const hlsRef = useRef(null)

  useEffect(() => {
    if (!hlsUrl || !videoRef.current) return

    setHlsError(null)

    if (Hls.isSupported()) {
      const hls = new Hls({
        lowLatencyMode:     true,
        backBufferLength:   4,
        maxBufferLength:    8,
        liveSyncDuration:   1.5,
        liveMaxLatencyDuration: 5,
      })
      hls.loadSource(hlsUrl)
      hls.attachMedia(videoRef.current)
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        videoRef.current?.play().catch(() => {})
      })
      hls.on(Hls.Events.ERROR, (_, data) => {
        if (data.fatal) {
          setHlsError(`HLS error: ${data.type} — ${data.details}`)
        }
      })
      hlsRef.current = hls
    } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari native HLS
      videoRef.current.src = hlsUrl
      videoRef.current.play().catch(() => {})
    }

    return () => {
      hlsRef.current?.destroy()
      hlsRef.current = null
    }
  }, [hlsUrl, videoRef])

  if (hlsError) {
    return (
      <div className="absolute inset-0 flex items-center justify-center">
        <p className="text-xs text-cyber-red font-mono text-center px-4">{hlsError}</p>
      </div>
    )
  }

  return (
    <video
      ref={videoRef}
      className="w-full h-full object-contain"
      autoPlay
      muted
      playsInline
    />
  )
}

// ── MJPEG Player (fallback) ───────────────────────────────────────────────

function MjpegPlayer({ streamUrl, videoRef }) {
  return (
    <img
      ref={videoRef}
      src={streamUrl}
      alt="Live video stream"
      className="w-full h-full object-contain"
      onError={(e) => { e.target.style.opacity = '0.3' }}
      onLoad={(e)  => { e.target.style.opacity = '1' }}
    />
  )
}

// ── VideoPlayer (main) ────────────────────────────────────────────────────

export default function VideoPlayer() {
  const sessionId  = useStore((s) => s.sessionId)
  const wsStatus   = useStore((s) => s.wsStatus)
  const detections = useStore((s) => s.detections)
  const frameId    = useStore((s) => s.frameId)
  const fps        = useStore((s) => s.fps)
  const hlsUrl     = useStore((s) => s.hlsUrl)
  const gatewayUrl = useStore((s) => s.gatewayUrl)

  const videoRef   = useRef(null)
  const [dims, setDims] = useState({ w: 0, h: 0 })

  const streamUrl  = sessionId
    ? `${gatewayUrl}/video/${sessionId}/stream`
    : null

  const useHls    = !!hlsUrl
  const isLive    = wsStatus === 'connected'

  // Track rendered video dimensions for canvas alignment
  useEffect(() => {
    const el = videoRef.current
    if (!el) return
    const handler = () => setDims({ w: el.offsetWidth, h: el.offsetHeight })
    const ro = new ResizeObserver(handler)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const isActive = sessionId && streamUrl

  return (
    <div className="panel flex flex-col h-full overflow-hidden">
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-cyber-border shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-cyber-accent tracking-wide uppercase">
            Video Feed
          </h2>
          {useHls && (
            <span className="badge text-[10px]"
              style={{ color: '#00ff88', borderColor: 'rgba(0,255,136,0.3)', border: '1px solid' }}>
              HLS
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {fps > 0 && (
            <span className="text-xs font-mono text-cyber-muted">
              {fps} <span className="text-cyber-accent">fps</span>
            </span>
          )}
          {isActive && (
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyber-red animate-pulse-slow" />
              <span className="text-xs font-mono text-cyber-red tracking-wider">LIVE</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Video area ── */}
      <div className="relative flex-1 bg-black overflow-hidden">
        {isActive ? (
          <>
            {useHls ? (
              <HlsPlayer hlsUrl={hlsUrl} videoRef={videoRef} />
            ) : (
              <MjpegPlayer streamUrl={streamUrl} videoRef={videoRef} />
            )}

            {/* Canvas overlay (always on top) */}
            <DetectionCanvas
              detections={detections}
              containerRef={videoRef}
            />

            {/* Stats bar */}
            {detections.length > 0 && (
              <div className="absolute bottom-2 left-2 flex items-center gap-2">
                <span className="badge text-xs">
                  {detections.length} object{detections.length !== 1 ? 's' : ''}
                </span>
                <span className="badge text-xs text-cyber-muted">
                  #{frameId}
                </span>
              </div>
            )}
          </>
        ) : (
          /* ── No-stream placeholder ── */
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
            <div className="w-16 h-16 rounded-2xl border border-cyber-border flex items-center justify-center opacity-30">
              <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 text-cyber-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M15 10l4.553-2.069A1 1 0 0121 9v6a1 1 0 01-1.447.894L15 14M5 8h10a2 2 0 012 2v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4a2 2 0 012-2z" />
              </svg>
            </div>
            <p className="text-xs text-cyber-muted font-mono text-center max-w-[200px] leading-relaxed">
              Configure a video source and start a session to begin streaming
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
