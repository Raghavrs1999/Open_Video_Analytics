/**
 * App — Root layout
 *
 * Three-column command-center layout:
 *   [Control Panel] | [Video + Stats] | [Alert Feed]
 *
 * Responsive: on narrow screens the sidebar collapses.
 */

import './index.css'
import ControlPanel from './components/ControlPanel'
import VideoPlayer  from './components/VideoPlayer'
import AlertFeed    from './components/AlertFeed'
import { useStore } from './store'

function Header() {
  const wsStatus   = useStore((s) => s.wsStatus)
  const modelName  = useStore((s) => s.modelName)
  const fps        = useStore((s) => s.fps)
  const showLabels = useStore((s) => s.showLabels)
  const toggle     = useStore((s) => s.toggleLabels)

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-cyber-border shrink-0"
      style={{ background: 'linear-gradient(90deg, #090d17 0%, #111827 100%)' }}>
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-cyber-accent/10 border border-cyber-accent/30">
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-cyber-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        </div>
        <div>
          <h1 className="text-sm font-bold text-white tracking-tight">Open Video Analytics</h1>
          <p className="text-xs text-cyber-muted">Real-time inference pipeline</p>
        </div>
      </div>

      {/* Centre — connection indicator */}
      <div className="hidden md:flex items-center gap-4 text-xs font-mono text-cyber-muted">
        {modelName && (
          <span className="px-2 py-1 rounded bg-cyber-bg border border-cyber-border text-cyber-text">
            model: {modelName}
          </span>
        )}
      </div>

      {/* Right — tools */}
      <div className="flex items-center gap-3">
        <button
          onClick={toggle}
          className={`btn-ghost text-xs py-1.5 ${showLabels ? 'border-cyber-accent text-cyber-accent' : ''}`}
        >
          Labels {showLabels ? 'ON' : 'OFF'}
        </button>
      </div>
    </header>
  )
}

export default function App() {
  return (
    <div className="dark flex flex-col h-screen bg-cyber-bg overflow-hidden">
      <Header />

      <div className="flex flex-1 overflow-hidden gap-0">
        {/* ── Left sidebar: Control Panel ── */}
        <aside className="w-72 shrink-0 flex flex-col overflow-y-auto p-3 gap-3 border-r border-cyber-border">
          <ControlPanel />
        </aside>

        {/* ── Centre: Video ── */}
        <main className="flex-1 p-3 overflow-hidden">
          <VideoPlayer />
        </main>

        {/* ── Right sidebar: Alert Feed ── */}
        <aside className="w-64 shrink-0 flex flex-col p-3 border-l border-cyber-border overflow-hidden">
          <AlertFeed />
        </aside>
      </div>
    </div>
  )
}
