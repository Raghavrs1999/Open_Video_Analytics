/**
 * Zustand global store — single source of truth for:
 *   - WebSocket + connection state
 *   - Live detections from inference worker
 *   - Spatial risk alerts (WARNING / CRITICAL) from the Risk Engine
 *   - System telemetry (CPU / RAM / GPU) from the gateway
 *   - Session metadata (model name, class labels, video dimensions)
 *   - UI toggles (show labels, class filter)
 */

import { create } from 'zustand'

export const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000'
const WS_BASE            = GATEWAY_URL.replace(/^http/, 'ws')

// ── Class colour palette ──────────────────────────────────────────────────
const PALETTE = [
  '#00d4ff','#00ff88','#ffd700','#ff4466','#a855f7',
  '#f97316','#14b8a6','#ec4899','#3b82f6','#84cc16',
]
export const getClassColor = (classId) => PALETTE[classId % PALETTE.length]

const SEVERITY_COLOR = {
  CRITICAL: '#ff4466',
  WARNING:  '#ffd700',
  INFO:     '#00d4ff',
}
export const getSeverityColor = (severity) => SEVERITY_COLOR[severity] || '#94a3b8'

// ── Store ────────────────────────────────────────────────────────────────
export const useStore = create((set, get) => ({
  // ── Connection ──────────────────────────────────────────────────────────
  wsStatus:   'disconnected',   // 'connecting' | 'connected' | 'disconnected' | 'error'
  sessionId:  '',
  gatewayUrl: GATEWAY_URL,

  // ── Session metadata ────────────────────────────────────────────────────
  modelName:   '',
  videoWidth:  0,
  videoHeight: 0,

  // ── Live detections (current frame) ─────────────────────────────────────
  detections: [],
  frameId:    0,
  fps:        0,

  // ── Alert feed (detection events + spatial risk alerts) ──────────────────
  alerts:    [],
  maxAlerts: 300,

  // ── System telemetry ─────────────────────────────────────────────────────
  telemetry:  null,
  _telemetryWs: null,

  // ── UI state ─────────────────────────────────────────────────────────────
  showLabels: true,

  // ── Internal ─────────────────────────────────────────────────────────────
  _ws:          null,
  _fpsCounter:  { frames: 0, lastTs: Date.now() },

  // ═══════════════════════════════════════════════════════════════════════
  // Actions
  // ═══════════════════════════════════════════════════════════════════════

  toggleLabels: () => set((s) => ({ showLabels: !s.showLabels })),

  // ── Connect detection WebSocket ───────────────────────────────────────────
  connect: (sessionId) => {
    const { _ws } = get()
    if (_ws) _ws.close()

    set({ wsStatus: 'connecting', sessionId })
    const ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`)

    ws.onopen    = () => set({ wsStatus: 'connected', _ws: ws })
    ws.onclose   = () => set({ wsStatus: 'disconnected', _ws: null })
    ws.onerror   = () => set({ wsStatus: 'error', _ws: null })
    ws.onmessage = (evt) => {
      try { get()._handleMessage(JSON.parse(evt.data)) }
      catch (_) {}
    }

    set({ _ws: ws })
  },

  // ── Disconnect ───────────────────────────────────────────────────────────
  disconnect: () => {
    const { _ws } = get()
    if (_ws) _ws.close()
    set({ wsStatus: 'disconnected', _ws: null, detections: [], frameId: 0 })
  },

  // ── Connect telemetry WebSocket ───────────────────────────────────────────
  connectTelemetry: () => {
    const { _telemetryWs } = get()
    if (_telemetryWs) return

    const ws = new WebSocket(`${WS_BASE}/ws/telemetry`)
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'telemetry') set({ telemetry: msg.data })
      } catch (_) {}
    }
    ws.onclose = () => set({ _telemetryWs: null })
    set({ _telemetryWs: ws })
  },

  // ── Handle incoming WS message ────────────────────────────────────────────
  _handleMessage: (msg) => {
    // --- Spatial risk alert from the gateway alert channel ---
    if (msg.type === 'alert') {
      const alert = msg.data
      const entry = {
        id:        `alert-${alert.track_id}-${alert.frame_id}-${Date.now()}`,
        kind:      'spatial',
        severity:  alert.severity,
        message:   alert.message,
        zoneName:  alert.zone_name,
        className: alert.class_name,
        trackId:   alert.track_id,
        frameId:   alert.frame_id,
        timestamp: Date.now(),
      }
      set((s) => ({ alerts: [entry, ...s.alerts].slice(0, s.maxAlerts) }))
      return
    }

    // --- Meta / heartbeat ---
    if (msg.type === 'meta') {
      set({ modelName: msg.data?.model_name || '' })
      return
    }
    if (['heartbeat','pong','connected'].includes(msg.type)) return

    // --- Detection frame payload ---
    const { detections = [], frame_id, frame_width, frame_height, alerts: spatialAlerts = [] } = msg
    const { _fpsCounter, alerts, maxAlerts } = get()

    // Update FPS
    const now    = Date.now()
    const delta  = (now - _fpsCounter.lastTs) / 1000
    const frames = _fpsCounter.frames + 1
    if (delta >= 0.5) {
      set({ _fpsCounter: { frames: 0, lastTs: now }, fps: Math.round(frames / delta) })
    } else {
      set({ _fpsCounter: { frames, lastTs: _fpsCounter.lastTs } })
    }

    // Build detection-level alert entries (top 5 to avoid flooding)
    const detAlerts = detections.slice(0, 5).map((d) => ({
      id:        `det-${frame_id}-${d.track_id}`,
      kind:      'detection',
      severity:  'INFO',
      message:   `${d.class_name} #${d.track_id} (${Math.round(d.confidence * 100)}%)`,
      className: d.class_name,
      trackId:   d.track_id,
      frameId:   frame_id,
      timestamp: Date.now(),
    }))

    // Spatial alerts embedded in payload
    const riskAlerts = spatialAlerts.map((a) => ({
      id:        `spatial-${a.track_id}-${a.frame_id}`,
      kind:      'spatial',
      severity:  a.severity,
      message:   a.message,
      zoneName:  a.zone_name,
      className: a.class_name,
      trackId:   a.track_id,
      frameId:   a.frame_id,
      timestamp: Date.now(),
    }))

    set({
      detections,
      frameId:     frame_id,
      videoWidth:  frame_width  || get().videoWidth,
      videoHeight: frame_height || get().videoHeight,
      alerts: [...riskAlerts, ...detAlerts, ...alerts].slice(0, maxAlerts),
    })
  },
}))
