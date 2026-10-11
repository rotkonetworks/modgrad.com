# modgrad obs dashboard (obs/web)

Vite + SolidJS + TypeScript + UnoCSS, the same stack and conventions as modgrad.com.
All graphs, fits and forecasts run in the browser on the Rust analysis module (wasm);
dropped log files are parsed in a Web Worker by the Rust ingest module (wasm).

## Build

```sh
JOBS=3 scripts/build_obs_wasm.sh      # once, and after Rust changes: obs/pkg/*.{js,d.ts,wasm}
cd obs/web && npm ci && npm run build # tsc --noEmit, then vite build -> obs/web/dist
```

`dist/` is self-contained: `index.html` (also copied to `dashboard.html`, which is what
obs-server opens at `/`), the JS/CSS bundle, the drop worker and both wasm modules.

## Serve

- **Server mode** (live SSE, `#run=`, `#project=`, `#new`, `#board=`):
  `obs-server --root <obs dir> --static obs/web/dist --listen 127.0.0.1:8790 [--dev]`
- **Static mode** (no server): put `index.json` and the run folders it points to
  (`runs/<source>/<run>/…`, written by `scripts/obs_collect.sh`) next to the files of
  `dist/`, and serve that directory with any static server:
  ```sh
  mkdir site && cp -r obs/web/dist/* site/ && cp -r obs/index.json obs/runs site/
  python3 -m http.server --directory site 8781
  ```
  Saved boards go in `site/boards/<name>.json` for `#board=<name>`. The page needs HTTP
  (not `file://`): it fetches JSON and instantiates wasm.
- **From the obs/ folder** (old setups: `python3 -m http.server --directory obs`, or obs-server
  with its default `--static <root>`): `obs/dashboard.html` redirects to
  `web/dist/index.html?data=../../`; `?data=` (only `../` sequences) makes the app fetch
  `index.json`, `runs/`, `boards/` and `api/` from that level, keeping the URL hash.
- **Dev**: `npm run dev` (port 5178) proxies `/api` to `OBS_SERVER` (default
  `http://127.0.0.1:8790`) and serves `../index.json`, `../runs/`, `../boards/` for static mode.

## Addresses (URL hash)

`#run=<id>`, `#project=<p>`, `#new` (anonymous project), `#board=<name>` (a board file from
the MCP `make_board` tool: shown unsaved, with "save as my board"), `#view=cross|arrays|runs`,
`#panel=<id>` (focus view; the old `#panel-<id>` still works).

## Layout

| path | DOM? | contents |
|---|---|---|
| `src/logic/` | no | board model and query text ops, data/forecast/ETA, palette commands, live client, drop pipeline + worker |
| `src/stores/` | no | Solid stores and signals: runs (+ per-run data versions, series cache), board (+ undo, presets, focus), live, drop, ui |
| `src/charts/` | yes | the imperative RRD drawing: `chart2d`, `Scene`/surfaces, heatmap, array views |
| `src/board/` | yes | panel drawing (`drawPanel`: cells, forecasts, overlays, legend controls), pointer drag and drop, stream loader |
| `src/components/`, `src/views/` | yes | Solid UI: header, rail (series browser, run picker), panels, inspector, palette, focus view, tabs |

Reactivity: a live row appends to that run's cached series and bumps that run's version;
a panel re-evaluates only when its query, view, settings, the theme, or the version of a run
its query plans to read changes.

Theme tokens live in `src/app.css`: paper/ink tokens, data series `--c1..6`, sequential and
diverging scales, and one `--accent` used only for live, selection, alerts and ETA/targets.

Storage (all behind try/catch): `obs-builder-board-v2` (per data target:
`…:run=<id>`, `…:project=<p>`), `obs-builder-settings-v1`, `obs-theme`, `obs-rail`,
IndexedDB `modgrad-obs-local` for dropped runs.
