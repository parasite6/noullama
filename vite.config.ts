import { defineConfig } from 'vite';

export default defineConfig({
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  envPrefix: ['VITE_', 'TAURI_ENV_*'],
  build: {
    target: 'es2022',
    rollupOptions: { output: { manualChunks(id) {
      if (id.includes('/@codemirror/') || id.includes('/@lezer/')) return 'json-editor';
    } } },
  },
});
