/**
 * CameraGrid
 *
 * Renders N simultaneous session feeds — each with its own MJPEG
 * <img> stream and a compact WebSocket-driven detection count badge.
 *
 * Currently displays all sessions returned by the gateway /sessions
 * endpoint, polling every 5 seconds.
 *
 * Usage: drop <CameraGrid /> anywhere in the layout.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { GATEWAY_URL, getClassColor } from '../store'

// A single camera tile
function CameraTile({ session }) {
  const { session_id, process_status, meta = {} } = session
  const streamUrl = `${GATEWAY_URL}/video/${session_id}/stream`
  const isRunning = process_status === 'running'
  const [detCount, setDetCount] = useState(0)
  const wsRef = useRef(null)

  // Lightweight WS just for detection count
  useEffect(() => {
    if (!isRunning) return
    const ws = new WebSocket(`${GATEWAY_URL.replace(/^http/, 'ws')}/ws/${session_id}`)
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (!msg.type && Array.isArray(msg.detections)) {
          setDetCount(msg.detections.length)
        }
      } catch (_) {}
    }
    wsRef.current = ws
    return () => ws.close()
  }, [session_id, isRunning])

  return (
    <div className="panel overflow-hidden flex flex-col">
      {/* Tile header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-cyber-border shrink-0">
        <span className="text-xs font-mono text-cyber-text truncate max-w-[100px]">
          {session_id}
        </span>
        <div className="flex items-center gap-2">
          {isRunning && (
            <span className="text-xs font-mono text-cyber-accent">
              {detCount} obj
            </span>
          )}
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isRunning ? 'bg-cyber-accent animate-pulse-slow' : 'bg-cyber-muted'
            }`}
          />
        </div>
      </div>

      {/* Video */}
      <div className="relative flex-1 bg-black min-h-[140px]">
        {isRunning ? (
          <img
            src={streamUrl}
            alt={`session ${session_id}`}
            className="w-full h-full object-contain"
            style={{ maxHeight: '180px' }}
            onError={(e) => { e.target.style.display = 'none' }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-cyber-muted text-xs font-mono">
            {process_status}
          </div>
        )}

        {/* Model badge */}
        {meta.model_name && (
          <div className="absolute bottom-1 right-1 badge" style={{
            backgroundColor: 'rgba(9,13,23,0.8)',
            border: '1px solid rgba(0,212,255,0.3)',
            color: '#00d4ff',
          }}>
            {meta.model_name}
          </div>
        )}
      </div>
    </div>
  )
}

export default function CameraGrid() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading]   = useState(true)

  const fetchSessions = useCallback(async () => {
    try {
      const res  = await fetch(`${GATEWAY_URL}/sessions`)
      const data = await res.json()
      setSessions(data.sessions || [])
    } catch (_) {}
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    fetchSessions()
    const interval = setInterval(fetchSessions, 5000)
    return () => clearInterval(interval)
  }, [fetchSessions])

  if (loading) {
    return (
      <div className="panel p-4 text-xs text-cyber-muted font-mono animate-pulse">
        Loading sessions…
      </div>
    )
  }

  if (!sessions.length) {
    return (
      <div className="panel p-4 text-xs text-cyber-muted font-mono text-center">
        No active sessions — start a session to see it here.
      </div>
    )
  }

  const cols = sessions.length === 1 ? 1 : sessions.length <= 4 ? 2 : 3

  return (
    <div
      className="grid gap-2"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {sessions.map((s) => (
        <CameraTile key={s.session_id} session={s} />
      ))}
    </div>
  )
}
