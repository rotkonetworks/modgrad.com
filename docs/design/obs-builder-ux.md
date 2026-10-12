# modgrad-obs graph builder: UX design

Status: implemented in `obs/web` (Vite + SolidJS + TypeScript; ported from the vanilla `obs/builder.js` in Oct 2026) and
`crates/modgrad-obs-analysis/src/{query,expr,curve}.rs` (Rust, compiled to wasm). See "Module boundaries".

## 1. Three objects, no modes

| object | what it is |
|---|---|
| **stream** | one run × one metric (or `array.stat` for arrays logged over ≥ 2 steps), static or live |
| **panel** | a **query** (text) plus a **view** (hidden/solo series, right-axis series, follow, forecast window/target/band/pool) |
| **board** | ordered panels, saved as JSON: `{version: 2, panels: [{id, title, q, view}]}` |

The board persists per viewer in `localStorage` (`obs-builder-board-v2`, all access in try/catch),
exports/imports as a JSON file, and survives reloads. Version-1 layouts (series lists) are
converted on load.

## 2. Panels are queries

Every gesture writes a query; the query is the only truth, so new runs that match appear by
themselves.

| gesture | query written |
|---|---|
| drag (or `+`) a stream | `loss{run="bkk07/small"}` |
| drag (or `+`) several selected streams | `loss{run=~"a\|b"}` (one entry per metric) |
| drag (or `+`) a metric heading | `loss` (every run) |
| drag (or `+`) a run heading | `{run="bkk07/small"} \| split metric` (small multiples) |
| drop onto a panel | appends the entries before the stages (overlay, same x axis) |
| drop a panel title on another panel's middle | merge: entries appended, letters renamed so formulas keep working |
| drop a panel title on another panel's edge | reorder |
| drag a legend row out of the board, or its `x` | adds a `run!~` exclusion (or hides an aggregate/formula row) |

### Grammar (PromQL-like; parsed and evaluated in Rust, `query.rs`)

```
loss{run=~"bkk07/*", lr>1e-4}; nat_none{run="bkk07/small"}; d = a - b
    | ema .9 | by size | mean±sd over seed | x tokens | forecast to 10k
```

- Entries are separated by `;` and lettered `a, b, c, …` (formula names are skipped).
- Matchers: `=`, `!=`, `=~`/`!~` (anchored glob: `*`, `?`, `|` between alternatives; **not** full
  regex), `< <= > >=` on numeric config fields. Fields: `run`/`id`, `name`, `source`, `project`,
  else a config key. A missing field reads as `""` for string ops and never matches a numeric
  comparison.
- Stages (several may share one `|` segment): `ema w`, `by field`, `mean±sd|mean|median|min|max over
  field`, `split field` (`metric`, `run`, `letter` or config), `x step|time|tokens|<metric>`,
  `forecast [to N] [auto|ridge|powerlaw|last]`, `log`, `norm z|minmax|first`,
  `baseline <run> [diff|ratio]`, `align intersection|union|exact`, `show letters`.
- Numbers accept `k`/`M`/`G` suffixes (`10k`).

### Chips and text round-trip

The header shows the query as chips (`[loss] [runs: bkk07/*] [lr>1e-4] [ema .9] …`). Each chip is a
real `<button>` that opens an inline editor, with an `x` button next to it. Edits go through
`query_replace_chip` in wasm, which re-composes and re-parses, so chips and text cannot diverge.
`text` on the panel (or the inspector's query field) edits the whole query; errors show inline with
the failing span underlined.

### Evaluation order (Rust, `query::eval`)

selectors → formulas per run (a letter matching one stream broadcasts) → `show` → `baseline` (per
run, aligned) → `norm` → aggregation → x mapping (joined on step, never extrapolated) → `ema`
(display only; legend stats stay raw) → block-mean downsampling to 1500 points when x is step/time.

Alignment for every binary operation (formulas, baseline, aggregation, metric x axes):
`intersection` (default: the overlap range, union of steps inside it, linear interpolation),
`union` (NaN outside each series' range), `exact` (shared steps only). Never extrapolates; NaN next
to a gap.

## 3. One-gesture operations

Each is a chip, a `+` menu item on the panel, an inspector control, and a palette command:

- **mean±sd over seed**: runs whose configs agree on every field except the `over` field form a
  group (a field none of them has is ignored, so replicate runs with no seed field group too);
  drawn as a line plus sd band, `n` in the legend.
- **by field**: colour by config value; numeric fields get a gradient (log when the range spans
  > 20×) with a key under the plot, categorical fields the palette.
- **split field**: small multiples; y axes are locked when every cell shows the same metric, x is
  linked within the panel.
- **baseline run [diff|ratio]**.
- **formula** over letters (`c = a - b`), per run.
- **x axis**: step, time (`elapsed_min × 60`), tokens (first metric containing "token"), any metric.

## 4. Forecasts

- Methods: ridge (existing SearchCast-style, `forecast_n` in wasm) and a saturating power law
  `y = a + b·x^-c` (`curve.rs`, grid over `c`, least squares for `a, b`, backtested on the last
  quarter vs last-value). `auto` picks the method with skill and the lower test MAE; with no skill
  the ghost is the last value.
- **Badge** (always visible when forecasting, a button): `forecast · ridge · 2.9× vs last-value (159
  test pts)`; grey `no skill` when it does not beat naive. It opens the forecast popover: on/off,
  horizon, method (auto/ridge/power law/last), band (50/80/95%; ridge's 10-90% residual band
  rescaled by the normal-quantile ratio), pooling (none / same metric / same letter), live refit
  on/off (off freezes the current forecast into the view), target, fit window.
- **Direct manipulation** on the plot (SVG handles; every one also has an input in the popover and
  inspector):
  - horizon tab at the forecast's end: drag right past the plot to extend, left to shorten
    (writes `forecast to N`);
  - fit-window bracket just under the x axis: drag its ends to exclude warmup or the tail;
  - target line: `+ target` places one, drag it vertically; the ETA readout under the plot says
    `reaches 2.450 at ~step 11.1k (11.1k–>12.0k)`, `not reached 2.45 by 8000`, or `reached … at`
    when the data already crossed. ETA comes from `forecast_eta` (mean crossing; band crossings in
    the same direction).
- **Accountability**: when new rows arrive and the forecast refits, the previous forecast stays as a
  faint dotted trace, so forecast vs reality is visible (kept in memory per page).

## 5. Live

- Queries re-evaluate on `obs:runs-changed` (fired by `window.obsAddRuns` and by reload). Rows of
  runs that newly match fade in with a `new` tag for 20 s.
- **Follow** (`> follow` button per panel): the zoom window keeps its width and slides to the newest
  x. Zooming the chart pauses it; the button shows the paused state and resumes on click.
- **Live dot** per legend row: green when the newest row is recent, amber when older than 10 min and
  the run is not finished; no dot for finished runs or rows older than 6 h.
- **Event markers**: the runs' event logs as small markers with tooltips on all step-axis panels.
- **Spike markers**: upward spikes (> 6 robust sd above the trailing 10-point median, floor 1% of
  the median), detected in wasm.

## 6. Clutter control and keyboard (Vimium-C friendly)

- Three levels: drag or `+` buttons (no knowledge needed) → chips and the `+` stage menu → text
  query, inspector, popovers, palette.
- One **inspector**, in the sidebar, for the selected panel only.
- Legend: more than 8 series collapses to current values; hovering a row dims the other series;
  clicking a name (a button) solos it, shift-click hides/shows it; `show all series` resets.
- **No single-letter shortcuts.** Every control is a real `<button>`/`<input>`/`<select>` with an
  aria-label/title, so Vimium-C link hints reach it; nothing is hover-only; every drag has a click
  path (`+ … add to…` menus listing panels plus "new panel", panel `menu` for merge/move/duplicate,
  `x` on chips and legend rows).
- Global keys: **Ctrl/Cmd+K** command palette only (toggle in settings). **Ctrl/Cmd+Z** undo only
  while focus is inside the board or focus view (plus a visible undo button and undo in toasts).
  **Delete** removes the focused panel. **Esc** resets a focused chart's zoom (existing) and closes
  the focus view only when focus is inside it; nothing else.
- Palette commands mirror the chips: `add loss`, `split by seed`, `colour by lr`, `mean±sd over
  seed`, `forecast to 10k`, `x axis: tokens`, `follow newest data`, `new panel <query>`, `expand
  panel`, `back to board`, presets, export/import, undo.
- **Auto boards**: pre-training (loss, lr, grad norm, throughput, tokens), eval (heldout/val
  loss/nll, accuracy), RL (reward, kl, entropy, response length) naming conventions produce the
  default board, with `mean±sd over seed` when replicate runs exist and `x tokens` for losses when a
  tokens metric exists; demo runs are kept out with `{synthetic!=true}` rather than pinning run ids.
  Falls back to the History view (one panel per History metric over the selected runs), which is
  also a preset.

## 7. Focus view

- Every panel has an `expand` button. It opens the same panel object in a full-viewport view with
  the series browser and inspector docked alongside; everything works there (click/drag add, chips,
  text, formulas, forecast handles, zoom, follow). Optional browser fullscreen button (Fullscreen
  API; rejection shows a toast).
- `< back to board` returns to the grid with the scroll position restored. Esc closes it only when
  focus is inside the focus view.
- Deep link `#panel=<id>` (ids are letters/digits/dash; the old `#panel-<id>` still works), combinable with `run=`/`project=`/`board=`; restored on reload; browser Back closes it.
- The grid is the default view; the focus view is temporary.

## 8. Engine

- Rust/wasm (`modgrad-obs-analysis`): `query_parse`, `query_replace_chip`, `query_plan`,
  `query_eval`, `eval_expr`/`expr_refs`, `align2`, `normalize`, `powerlaw_forecast`, `forecast_n`,
  `forecast_eta`, `spikes`, `array_stats`, plus the existing kernels. Pure functions over flat
  arrays and strings (catalog as tab-separated lines, results as JSON metadata + one flat
  `Float64Array`), so they can move into a worker unchanged; today they run on the main thread.
- Series store (`obs/web/src/stores/runs.ts`): one cached series per run and metric; live rows are
  appended in place and bump that run's version, so only panels whose query plan reads the run
  re-evaluate. New metric keys bump the catalog version (every query re-plans).

## Module boundaries (obs/web)

| module | DOM? | contents |
|---|---|---|
| `src/logic/board.ts` | no | board/panel model, validation, (de)serialisation, undo stack, query text ops (`addEntries`, `setStage`, `mergeQueries`, `excludeRun`, `selectorFor`), presets, auto boards |
| `src/logic/data.ts` | no | catalog text, stream list, `planOf`, `evalQuery` (injected loader), forecasts, badge/ETA text, spikes, live age, colour mapping |
| `src/logic/commands.ts` | no | palette: input text → `[{label, action}]`, actions are plain data |
| `src/logic/liveClient.ts`, `src/logic/drop.ts` (+ `dropWorker.ts`) | no | SSE/polling client; file grouping, worker protocol, IndexedDB |
| `src/stores/*.ts` | no | Solid stores: runs (+ series store, per-run versions, selection), board (`commit()`, undo, presets, focus, shared boards), live, drop, ui |
| `src/charts/*.ts`, `src/board/drawPanel.ts` | yes | the imperative RRD drawing (`chart2d`, `Scene`, heatmap, surfaces, array views) and one panel's chart, legend controls and forecast handles |
| `src/components`, `src/views` | yes | Solid UI: header + tabs (Board / Across runs / Arrays / 3D / Runs), rail (series browser, run picker), panels, inspector (on demand), palette, focus view |
| `crates/modgrad-obs-analysis` | no | parser, evaluator, numeric kernels |

The old History section is the `history` preset (selected runs × picked or headline metrics); its
EMA, x axis and forecast toggles are chips, and "link x zoom" is a board setting.

## API for other modules

- `window.obsAddRuns(runs, {select = true})` (= `upsertRuns` in `stores/runs.ts`): upsert runs by
  `id` (defaults filled), select them, drop cached data of an updated run, re-evaluate, and fire
  `document` event `obs:runs-changed` with
  `{added: [ids], updated: [ids]}`. Rows can be inline: `run.data = {metric: {step: [...],
  value: [...]}}` with `run.metrics = {metric: {n}}`; `metricSeries` reads them before fetching.
- The reload button also fires `obs:runs-changed`.
- `window.OBS`: `{S, A, upsertRuns, builder}`; `window.OBS.builder`: `{board(), setBoard(b),
  addPanel(q), openFocus(id), closeFocus(), runAction(a)}`.
