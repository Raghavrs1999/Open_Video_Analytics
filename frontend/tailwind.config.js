/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'cyber-bg':      '#090d17',
        'cyber-surface': '#111827',
        'cyber-border':  '#1f2d42',
        'cyber-accent':  '#00d4ff',
        'cyber-green':   '#00ff88',
        'cyber-red':     '#ff4466',
        'cyber-yellow':  '#ffd700',
        'cyber-text':    '#b0c4d8',
        'cyber-muted':   '#4a5568',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'fade-in':    'fadeIn 0.3s ease-in-out',
        'slide-up':   'slideUp 0.2s ease-out',
        'glow':       'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn:  { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp: { '0%': { transform: 'translateY(8px)', opacity: 0 }, '100%': { transform: 'translateY(0)', opacity: 1 } },
        glow:    { from: { boxShadow: '0 0 4px #00d4ff44' }, to: { boxShadow: '0 0 16px #00d4ff88' } },
      },
    },
  },
  plugins: [],
}

