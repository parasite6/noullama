import { readFile } from 'node:fs/promises';
import { defineConfig } from 'vite';

export default defineConfig({
  clearScreen: false,
  plugins: [{
    name: 'frontend-license-notices',
    apply: 'build',
    async generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'licenses/THIRD_PARTY_NOTICES.md',
        source: await readFile(new URL('./THIRD_PARTY_NOTICES.md', import.meta.url)),
      });
    },
  }],
  server: { port: 1420, strictPort: true },
  envPrefix: ['VITE_', 'TAURI_ENV_*'],
  build: {
    target: 'es2022',
    rollupOptions: { output: { manualChunks(id) {
      if (id.includes('/@codemirror/') || id.includes('/@lezer/')) return 'json-editor';
    } } },
  },
});
