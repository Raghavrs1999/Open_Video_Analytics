import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // hls.js is ~700 kB — raise the warning threshold to avoid false build failures
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        // Split hls.js into its own chunk for better caching
        manualChunks: {
          'hls': ['hls.js'],
        },
      },
    },
  },
})
