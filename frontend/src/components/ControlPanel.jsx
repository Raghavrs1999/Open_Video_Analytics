/**
 * ControlPanel
 *
 * Sidebar panel for:
 *   - Gateway URL (env-configured by default)
 *   - Session ID
 *   - Video source (RTSP URL | file path | 0 for webcam)
 *   - Model picker (fetched from /models endpoint)
 *   - Confidence threshold
 *   - Target FPS
 *   - Toggle: publish frames for MJPEG relay
 *   - Start / Stop session buttons
 *   - Connection status indicator
 */

import { useState, useEffect } from 'react'
import { useStore, GATEWAY_URL } from '../store'

const DEFAULT_GATEWAY = GATEWAY_URL

export default function ControlPanel() {
  const wsStatus    = useStore((s) => s.wsStatus)
  const storeConnId = useStore((s) => s.sessionId)
  const connect     = useStore((s) => s.connect)
  const disconnect  = useStore((s) => s.disconnect)
  const modelName   = useStore((s) => s.modelName)

  const [gateway,       setGateway]       = useState(DEFAULT_GATEWAY)
  const [sessionId,     setSessionId]     = useState('default')
  const [videoSource,   setVideoSource]   = useState('0')
  const [model,         setModel]         = useState('yolov8n')
  const [models,        setModels]        = useState([])
  const [confidence,    setConfidence]    = useState(0.25)
  const [targetFps,     setTargetFps]     = useState(30)
  const [publishFrames, setPublishFrames] = useState(true)
  const [loading,       setLoading]       = useState(false)
  const [error,         setError]         = useState('')

  // Fetch available models from gateway
  useEffect(() => {
    const url = `${gateway}/models`
    fetch(url)
      .then((r) => r.json())
      .then((d) => setModels(d.models || []))
      .catch(() => setModels([]))
  }, [gateway])

  const statusMeta = {
    connected:    { cls: 'status-connected',    label: 'Connected'    },
    connecting:   { cls: 'status-connecting',   label: 'Connecting…'  },
    disconnected: { cls: 'status-disconnected', label: 'Disconnected' },
    error:        { cls: 'status-error',        label: 'Error'        },
  }[wsStatus] || { cls: 'status-disconnected', label: wsStatus }

  const handleStart = async () => {
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`${gateway}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:     sessionId,
          video_source:   videoSource,
          model_name:     model,
          confidence:     parseFloat(confidence),
          target_fps:     parseInt(targetFps),
          publish_frames: publishFrames,
        }),
      })
      if (!res.ok) {
        const d = await res.json()
        throw new Error(d.detail || `HTTP ${res.status}`)
      }
      connect(sessionId)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    disconnect()
    try {
      await fetch(`${gateway}/session/stop/${storeConnId}`, { method: 'POST' })
    } catch (_) {}
  }

  const isRunning = wsStatus === 'connected' || wsStatus === 'connecting'

  return (
    <div className="flex flex-col gap-4">
      {/* ── Status ── */}
      <div className="flex items-center justify-between panel px-3 py-2">
        <span className="text-xs text-cyber-muted font-medium">Gateway Status</span>
        <div className="flex items-center gap-1.5">
          <span className={`status-dot ${statusMeta.cls}`} />
          <span className="text-xs font-mono text-cyber-text">{statusMeta.label}</span>
        </div>
      </div>

      {/* ── Configuration ── */}
      <div className="panel p-4 flex flex-col gap-4">
        <h2 className="text-sm font-semibold text-cyber-accent tracking-wide uppercase">
          Session Configuration
        </h2>

        {/* Gateway URL */}
        <div>
          <label className="label">Gateway URL</label>
          <input className="input" value={gateway}
            onChange={(e) => setGateway(e.target.value)}
            placeholder="http://localhost:8000" />
        </div>

        {/* Session ID */}
        <div>
          <label className="label">Session ID</label>
          <input className="input font-mono" value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            placeholder="default" />
        </div>

        {/* Video Source */}
        <div>
          <label className="label">Video Source</label>
          <input className="input font-mono text-xs" value={videoSource}
            onChange={(e) => setVideoSource(e.target.value)}
            placeholder="rtsp://... or /path/to/video.mp4 or 0" />
          <p className="mt-1 text-xs text-cyber-muted">Use 0 for default webcam</p>
        </div>

        {/* Model */}
        <div>
          <label className="label">Model</label>
          {models.length > 0 ? (
            <select className="input" value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          ) : (
            <input className="input font-mono text-xs" value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="yolov8n" />
          )}
        </div>

        {/* Confidence + FPS */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label">Confidence</label>
            <div className="flex items-center gap-2">
              <input type="range" min="0.05" max="0.95" step="0.05"
                value={confidence} onChange={(e) => setConfidence(e.target.value)}
                className="flex-1 accent-cyber-accent" />
              <span className="text-xs font-mono text-cyber-text w-8 text-right">
                {parseFloat(confidence).toFixed(2)}
              </span>
            </div>
          </div>
          <div>
            <label className="label">Target FPS</label>
            <input className="input font-mono" type="number" min="1" max="60"
              value={targetFps} onChange={(e) => setTargetFps(e.target.value)} />
          </div>
        </div>

        {/* Publish frames toggle */}
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <div
            onClick={() => setPublishFrames(!publishFrames)}
            className={`relative w-9 h-5 rounded-full transition-colors duration-200 cursor-pointer
              ${publishFrames ? 'bg-cyber-accent' : 'bg-cyber-muted'}`}
          >
            <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white
              transition-transform duration-200 ${publishFrames ? 'translate-x-4' : ''}`} />
          </div>
          <span className="text-xs text-cyber-text">Stream video via gateway (MJPEG)</span>
        </label>

        {/* Error */}
        {error && (
          <div className="text-xs text-cyber-red bg-cyber-red/10 border border-cyber-red/30 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        {/* Actions */}
        {!isRunning ? (
          <button
            className="btn-primary w-full"
            onClick={handleStart}
            disabled={loading}
          >
            {loading ? 'Starting…' : '▶ Start Session'}
          </button>
        ) : (
          <button className="btn-danger w-full" onClick={handleStop}>
            ⏹ Stop Session
          </button>
        )}
      </div>
    </div>
  )
}
