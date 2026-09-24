import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Backend API + generated artefacts are proxied so the browser only ever talks
// to one origin, in both `npm run dev` and `npm run preview`.
const proxy = {
  '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/static': { target: 'http://127.0.0.1:8000', changeOrigin: true },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
})
