/**
 * AlertFeed
 *
 * Scrolling live list of detection events received over WebSocket.
 * Shows class name, track ID, confidence, and relative timestamp.
 * Auto-scrolls to the latest event.
 */

import { useEffect, useRef } from 'react'
import { useStore, getClassColor } from '../store'

function timeAgo(ts) {
  const secs = Math.floor((Date.now() - ts) / 1000)
  if (secs < 2)  return 'now'
  if (secs < 60) return `${secs}s ago`
  return `${Math.floor(secs / 60)}m ago`
}

export default function AlertFeed() {
  const alerts     = useStore((s) => s.alerts)
  const wsStatus   = useStore((s) => s.wsStatus)
  const bottomRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [alerts.length])

  return (
    <div className="panel flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-cyber-border shrink-0">
        <h2 className="text-sm font-semibold text-cyber-accent tracking-wide uppercase">
          Detection Feed
        </h2>
        <span className="text-xs font-mono text-cyber-muted">
          {alerts.length} events
        </span>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto py-2 px-2 flex flex-col-reverse gap-0.5">
        <div ref={bottomRef} />
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-cyber-muted">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p className="text-xs text-center">
              {wsStatus === 'connected'
                ? 'Waiting for detections…'
                : 'Start a session to see live detections'}
            </p>
          </div>
        ) : (
          [...alerts].reverse().map((alert) => {
            const hex = getClassColor(0) // use accent for simplicity; real: pass class_id
            return (
              <div
                key={alert.id}
                className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-white/5 transition-colors animate-slide-up"
              >
                {/* Class colour dot */}
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: getClassColor(alert.trackId % 10) }}
                />

                {/* Class name */}
                <span className="text-xs font-mono text-cyber-text flex-1 truncate">
                  {alert.className}
                </span>

                {/* Track ID */}
                <span className="text-xs font-mono text-cyber-muted">
                  #{alert.trackId}
                </span>

                {/* Confidence */}
                <span
                  className="badge"
                  style={{
                    color: getClassColor(alert.trackId % 10),
                    borderColor: `${getClassColor(alert.trackId % 10)}44`,
                    border: '1px solid',
                  }}
                >
                  {Math.round(alert.confidence * 100)}%
                </span>

                {/* Time */}
                <span className="text-xs text-cyber-muted w-12 text-right shrink-0">
                  {timeAgo(alert.timestamp)}
                </span>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
