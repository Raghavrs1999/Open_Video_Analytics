/**
 * SystemTelemetryStrip
 *
 * A compact horizontal bar at the bottom or top of the UI showing
 * live CPU, RAM, and GPU utilisation sourced from the gateway's
 * telemetry WebSocket (/ws/telemetry).
 *
 * Connects automatically on mount and reconnects if disconnected.
 */

import { useEffect } from 'react'
import { useStore } from '../store'

function Bar({ label, value, max = 100, color }) {
  const pct = Math.min(100, Math.round((value / max) * 100))
  const warn = pct > 80
  const barColor = warn ? '#ff4466' : color
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <span className="text-xs font-mono text-cyber-muted w-8 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-cyber-border overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
      <span
        className="text-xs font-mono w-8 text-right shrink-0"
        style={{ color: barColor }}
      >
        {pct}%
      </span>
    </div>
  )
}

function MemBar({ label, used, total, color }) {
  if (!total) return null
  const pct = Math.min(100, Math.round((used / total) * 100))
  const warn = pct > 85
  const barColor = warn ? '#ff4466' : color
  return (
    <div className="flex items-center gap-2 min-w-[160px]">
      <span className="text-xs font-mono text-cyber-muted w-8 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-cyber-border overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
      <span className="text-xs font-mono text-cyber-muted shrink-0">
        {used}/<span style={{ color: barColor }}>{total}</span>GB
      </span>
    </div>
  )
}

export default function SystemTelemetryStrip() {
  const connectTelemetry = useStore((s) => s.connectTelemetry)
  const telemetry        = useStore((s) => s.telemetry)

  useEffect(() => {
    connectTelemetry()
  }, [connectTelemetry])

  if (!telemetry) {
    return (
      <div className="flex items-center gap-3 px-4 py-2 border-t border-cyber-border text-xs text-cyber-muted font-mono">
        <span className="animate-pulse">▸ System telemetry connecting…</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-6 px-4 py-2 border-t border-cyber-border overflow-x-auto shrink-0"
      style={{ background: 'rgba(9,13,23,0.9)' }}>
      <span className="text-xs font-mono text-cyber-muted shrink-0 tracking-widest uppercase">
        System
      </span>

      {telemetry.cpu_percent != null && (
        <Bar label="CPU" value={telemetry.cpu_percent} color="#00d4ff" />
      )}

      {telemetry.ram_percent != null && (
        <Bar label="RAM" value={telemetry.ram_percent} color="#a855f7" />
      )}

      {telemetry.ram_used_gb != null && (
        <MemBar
          label="MEM"
          used={telemetry.ram_used_gb}
          total={telemetry.ram_total_gb}
          color="#a855f7"
        />
      )}

      {telemetry.gpu_percent != null && (
        <Bar label="GPU" value={telemetry.gpu_percent} color="#00ff88" />
      )}

      {telemetry.gpu_mem_used_gb != null && (
        <MemBar
          label="VRAM"
          used={telemetry.gpu_mem_used_gb}
          total={telemetry.gpu_mem_total_gb}
          color="#00ff88"
        />
      )}

      {telemetry.gpu_name && (
        <span className="text-xs font-mono text-cyber-muted ml-auto shrink-0 truncate max-w-[200px]">
          {telemetry.gpu_name}
        </span>
      )}
    </div>
  )
}
