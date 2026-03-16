/**
 * Zustand global store — single source of truth for:
 *   - WebSocket + connection state
 *   - Live detections from inference worker
 *   - Session metadata (model name, class labels, video dimensions)
 *   - Alert history
 *   - UI toggles (show labels, class filter)
 */

import { create } from 'zustand'

const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000'
const WS_URL      = GATEWAY_URL.replace(/^http/, 'ws')

// ── Class colour palette ──────────────────────────────────────────────────
const PALETTE = [
  '#00d4ff','#00ff88','#ffd700','#ff4466','#a855f7',
  '#f97316','#14b8a6','#ec4899','#3b82f6','#84cc16',
]
export const getClassColor = (classId) => PALETTE[classId % PALETTE.length]

// ── Store ────────────────────────────────────────────────────────────────
export const useStore = create((set, get) => ({
  // ── Connection ──────────────────────────────────────────────────────────
  wsStatus:   'disconnected',   // 'connecting' | 'connected' | 'disconnected' | 'error'
  sessionId:  '',
  gatewayUrl: GATEWAY_URL,

  // ── Session metadata ────────────────────────────────────────────────────
  modelName:   '',
  modelNames:  {},   // {0: 'person', 1: 'bicycle', ...}
  videoWidth:  0,
  videoHeight: 0,

  // ── Live detections (current frame) ─────────────────────────────────────
  detections: [],      // [{track_id, class_id, class_name, confidence, bbox, centroid}]
  frameId:    0,
  fps:        0,

  // ── Alert feed ───────────────────────────────────────────────────────────
  alerts:    [],      // last N detection events for display
  maxAlerts: 200,

  // ── UI state ─────────────────────────────────────────────────────────────
  showLabels:    true,
  showHeatmap:   false,
  classFiler:    null,  // null = all, Set of class_ids = filtered

  // ── Internal ─────────────────────────────────────────────────────────────
  _ws:            null,
  _fpsCounter:    { frames: 0, lastTs: Date.now() },

  // ═══════════════════════════════════════════════════════════════════════
  // Actions
  // ═══════════════════════════════════════════════════════════════════════

  setGatewayUrl: (url) => set({ gatewayUrl: url }),
  setSessionId:  (id)  => set({ sessionId: id }),
  toggleLabels:  ()    => set((s) => ({ showLabels: !s.showLabels })),

  // ── Connect WebSocket ────────────────────────────────────────────────────
  connect: (sessionId) => {
    const { _ws } = get()
    if (_ws) _ws.close()

    const wsUrl = `${WS_URL.replace(/^http/, 'ws')}/ws/${sessionId}`
    set({ wsStatus: 'connecting', sessionId })

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      set({ wsStatus: 'connected', _ws: ws })
    }

    ws.onclose = () => {
      set({ wsStatus: 'disconnected', _ws: null })
    }

    ws.onerror = () => {
      set({ wsStatus: 'error', _ws: null })
    }

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        get()._handleMessage(msg)
      } catch (_) { /* ignore parse errors */ }
    }

    set({ _ws: ws })
  },

  // ── Disconnect ───────────────────────────────────────────────────────────
  disconnect: () => {
    const { _ws } = get()
    if (_ws) _ws.close()
    set({ wsStatus: 'disconnected', _ws: null, detections: [], frameId: 0 })
  },

  // ── Handle incoming WS message ────────────────────────────────────────────
  _handleMessage: (msg) => {
    if (msg.type === 'meta') {
      set({
        modelName:  msg.data?.model_name   || '',
        modelNames: msg.data?.model_names  || {},
      })
      return
    }
    if (msg.type === 'heartbeat' || msg.type === 'pong' || msg.type === 'connected') {
      return
    }

    // Detection frame payload
    const { detections = [], frame_id, frame_width, frame_height } = msg
    const { _fpsCounter, alerts, maxAlerts } = get()

    // Update FPS counter
    const now   = Date.now()
    const delta = (now - _fpsCounter.lastTs) / 1000
    const frames = _fpsCounter.frames + 1
    let fps = get().fps
    if (delta >= 0.5) {
      fps = Math.round(frames / delta)
      set({ _fpsCounter: { frames: 0, lastTs: now }, fps })
    } else {
      set({ _fpsCounter: { frames, lastTs: _fpsCounter.lastTs } })
    }

    // Append new detections to alert feed
    const newAlerts = detections.map((d) => ({
      id:         `${frame_id}-${d.track_id}`,
      frameId:    frame_id,
      className:  d.class_name,
      trackId:    d.track_id,
      confidence: d.confidence,
      timestamp:  Date.now(),
    }))

    set({
      detections,
      frameId:    frame_id,
      videoWidth:  frame_width  || get().videoWidth,
      videoHeight: frame_height || get().videoHeight,
      alerts: [...newAlerts, ...alerts].slice(0, maxAlerts),
    })
  },
}))

export { GATEWAY_URL }
