/**
 * AlertFeed — upgraded
 *
 * Shows both:
 *   - Detection events (INFO, dim)
 *   - Spatial risk alerts (WARNING flash yellow, CRITICAL flash red)
 *
 * Spatial alerts are promoted to the top and visually distinct.
 */

import { useEffect, useRef } from 'react'
import { useStore, getClassColor, getSeverityColor } from '../store'

function timeAgo(ts) {
  const secs = Math.floor((Date.now() - ts) / 1000)
  if (secs < 2)  return 'now'
  if (secs < 60) return `${secs}s ago`
  return `${Math.floor(secs / 60)}m ago`
}

const SEVERITY_BG = {
  CRITICAL: 'rgba(255,68,102,0.12)',
  WARNING:  'rgba(255,215,0,0.08)',
  INFO:     'transparent',
}
const SEVERITY_BORDER = {
  CRITICAL: 'rgba(255,68,102,0.4)',
  WARNING:  'rgba(255,215,0,0.3)',
  INFO:     'transparent',
}

function AlertRow({ alert }) {
  const color   = getSeverityColor(alert.severity)
  const isSpatial = alert.kind === 'spatial'

  return (
    <div
      className="flex items-start gap-2 px-2 py-1.5 rounded-lg transition-colors"
      style={{
        background:   SEVERITY_BG[alert.severity]   || 'transparent',
        border:       `1px solid ${SEVERITY_BORDER[alert.severity] || 'transparent'}`,
        marginBottom: '2px',
      }}
    >
      {/* Severity dot */}
      <span
        className={`w-1.5 h-1.5 rounded-full shrink-0 mt-1 ${
          alert.severity === 'CRITICAL' ? 'animate-pulse-slow' : ''
        }`}
        style={{ backgroundColor: color }}
      />

      <div className="flex-1 min-w-0">
        {/* Message */}
        <p className="text-xs font-mono leading-snug break-words" style={{ color }}>
          {isSpatial && (
            <span className="mr-1 text-[10px] font-bold tracking-wider opacity-70">
              [{alert.severity}]
            </span>
          )}
          {alert.message}
        </p>

        {/* Zone label for spatial alerts */}
        {alert.zoneName && (
          <span className="text-[10px] font-mono opacity-60" style={{ color }}>
            ⬡ {alert.zoneName}
          </span>
        )}
      </div>

      {/* Time */}
      <span className="text-[10px] font-mono text-cyber-muted shrink-0 mt-0.5">
        {timeAgo(alert.timestamp)}
      </span>
    </div>
  )
}

export default function AlertFeed() {
  const alerts   = useStore((s) => s.alerts)
  const wsStatus = useStore((s) => s.wsStatus)
  const bottomRef = useRef(null)

  const spatialCount = alerts.filter((a) => a.kind === 'spatial').length
  const criticalCount = alerts.filter((a) => a.severity === 'CRITICAL').length

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
        <div className="flex items-center gap-2">
          {criticalCount > 0 && (
            <span className="badge animate-pulse-slow"
              style={{ color: '#ff4466', borderColor: 'rgba(255,68,102,0.4)', border: '1px solid' }}>
              ⚠ {criticalCount}
            </span>
          )}
          <span className="text-xs font-mono text-cyber-muted">{alerts.length} events</span>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto py-2 px-2 flex flex-col-reverse gap-0">
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
          [...alerts].reverse().map((alert) => (
            <AlertRow key={alert.id} alert={alert} />
          ))
        )}
      </div>

      {/* Spatial alert summary footer */}
      {spatialCount > 0 && (
        <div className="px-3 py-2 border-t border-cyber-border flex items-center gap-2 shrink-0">
          <span className="w-1.5 h-1.5 rounded-full bg-cyber-red animate-pulse-slow" />
          <span className="text-xs font-mono text-cyber-muted">
            {spatialCount} spatial risk event{spatialCount > 1 ? 's' : ''}
          </span>
        </div>
      )}
    </div>
  )
}
