import { defineConfig, type Plugin } from 'vite';
import solid from 'vite-plugin-solid';
import unocss from 'unocss/vite';
import { fileURLToPath } from 'url';
import { copyFileSync, createReadStream, existsSync, statSync } from 'fs';
import { join, normalize, resolve } from 'path';

// The dashboard's data lives next to the page: `index.json` plus per-run files under `runs/`
// (static mode), or the obs-server API under `api/` (server mode).
//   npm run dev                       proxies /api to OBS_SERVER (default http://127.0.0.1:8790)
//                                     and serves ../index.json and ../runs for static mode
//   npm run build                     -> dist/ (index.html, dashboard.html, assets incl. wasm + worker)
const OBS_DIR = fileURLToPath(new URL('..', import.meta.url));
const SERVER = process.env.OBS_SERVER || 'http://127.0.0.1:8790';

/** dev only: serve the static export (index.json, runs/) from the obs/ directory */
function staticData(): Plugin {
  return {
    name: 'obs-static-data',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url || '').split('?')[0]);
        if (url !== '/index.json' && !url.startsWith('/runs/')) return next();
        const full = normalize(join(OBS_DIR, url));
        if (!full.startsWith(resolve(OBS_DIR)) || !existsSync(full) || !statSync(full).isFile()) return next();
        res.setHeader('Content-Type', 'application/json');
        createReadStream(full).pipe(res);
      });
    },
  };
}

/** obs-server serves dashboard.html at `/`; ship the app under both names */
function dashboardAlias(): Plugin {
  let outDir = 'dist';
  return {
    name: 'obs-dashboard-alias',
    apply: 'build',
    configResolved(c) { outDir = c.build.outDir; },
    closeBundle() {
      const src = resolve(outDir, 'index.html');
      if (existsSync(src)) copyFileSync(src, resolve(outDir, 'dashboard.html'));
    },
  };
}

export default defineConfig({
  base: './',
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@pkg': fileURLToPath(new URL('../pkg', import.meta.url)),
    },
  },
  plugins: [unocss(), solid(), staticData(), dashboardAlias()],
  worker: { format: 'es' },
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022', assetsInlineLimit: 0 },
  server: {
    port: 5178,
    fs: { allow: [OBS_DIR] },
    proxy: { '/api': { target: SERVER, changeOrigin: true } },
  },
  preview: { port: 5178 },
});
