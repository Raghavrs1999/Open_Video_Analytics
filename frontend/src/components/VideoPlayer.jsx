/**
 * VideoPlayer
 *
 * Renders the video stream and layers the DetectionCanvas on top.
 * Supports two stream modes:
 *   1. MJPEG — <img> element refreshed via the gateway /video/{id}/stream endpoint
 *   2. Placeholder — shown when no session is active
 */

import { useRef } from 'react'
import { useStore, GATEWAY_URL } from '../store'
import DetectionCanvas from './DetectionCanvas'

export default function VideoPlayer() {
  const wrapperRef = useRef(null)
  const sessionId  = useStore((s) => s.sessionId)
  const wsStatus   = useStore((s) => s.wsStatus)
  const frameId    = useStore((s) => s.frameId)
  const fps        = useStore((s) => s.fps)
  const detections = useStore((s) => s.detections)

  const streamUrl = sessionId
    ? `${GATEWAY_URL}/video/${sessionId}/stream`
    : null

  const isLive = wsStatus === 'connected'

  return (
    <div className="flex flex-col h-full gap-2">
      {/* ── Video frame ── */}
      <div ref={wrapperRef} className="video-wrapper flex-1">
        {streamUrl ? (
          <img
            className="mjpeg"
            src={streamUrl}
            alt="Live video stream"
            onError={(e) => { e.target.style.display = 'none' }}
          />
        ) : (
          /* Placeholder grid */
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-cyber-muted">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-16 h-16 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M4 8h7a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1V9a1 1 0 011-1z" />
            </svg>
            <p className="text-sm">No active session — configure a source and click&nbsp;<strong className="text-cyber-accent">Start Session</strong></p>
          </div>
        )}

        {/* Detection canvas overlay */}
        {isLive && <DetectionCanvas wrapperRef={wrapperRef} />}

        {/* LIVE badge */}
        {isLive && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2 py-1 rounded-md bg-cyber-bg/80 border border-cyber-red text-cyber-red text-xs font-mono font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-cyber-red animate-pulse-slow" />
            LIVE
          </div>
        )}
      </div>

      {/* ── Stats bar ── */}
      <div className="flex items-center gap-4 px-3 py-1.5 panel text-xs font-mono text-cyber-muted shrink-0">
        <span>
          <span className="text-cyber-text font-semibold">{fps}</span> fps
        </span>
        <span className="text-cyber-border">|</span>
        <span>
          Frame <span className="text-cyber-text font-semibold">#{frameId.toLocaleString()}</span>
        </span>
        <span className="text-cyber-border">|</span>
        <span>
          <span className="text-cyber-text font-semibold">{detections.length}</span> detections
        </span>
        <span className="ml-auto text-cyber-muted">
          {sessionId ? `session: ${sessionId}` : '—'}
        </span>
      </div>
    </div>
  )
}
