// Shapes shared across the app: index.json / api/index run entries, the board, query results.

export type Scalar = string | number | boolean | null;
export interface MetricInfo { file?: string; n: number }
export interface ArrayInfo { file?: string; kind: string; shape: number[]; n_steps: number }
export interface RunMeta { git?: string | null; hostname?: string | null; start_unix?: number | null; legacy_log?: string; project?: string; [k: string]: unknown }
export interface Rows { step: number[]; value: (number | null)[] }
/** a run parsed in this browser from dropped files (drop pipeline) */
export interface LocalData {
  metrics: Record<string, { step: ArrayLike<number>; value: ArrayLike<number> }>;
  t: { step: ArrayLike<number>; value: ArrayLike<number> } | null;
  arrays: Record<string, ArrayFile>;
  events: RunEvent[];
  files?: unknown;
  added: number;
}
export interface Run {
  id: string; name?: string; source: string; project?: string; path?: string | null;
  config: Record<string, Scalar | object>; summary: Record<string, unknown>; meta: RunMeta;
  metrics: Record<string, MetricInfo>; arrays: Record<string, ArrayInfo>;
  finished?: boolean; last_step?: number | null; last_t?: number | null;
  n_rows?: number; n_events?: number; skipped_lines?: number; unparsed: string[]; has_small?: boolean;
  /** text samples logged (samples.jsonl); 0 or absent = none */
  n_samples?: number;
  /** inline rows (live feed, other modules): read before fetching */
  data?: Record<string, Rows>;
  local?: LocalData;
}
export interface Index { runs: Run[]; generated_unix: number; errors?: unknown[]; server?: boolean; dev?: boolean }
export interface RunEvent { step: number; t?: number; text: string }
export interface ArrayFile {
  kind: string; step: number[]; shape: number[][]; b64: string[]; labels_b64?: (string | null)[]; edges?: number[][];
}
export interface Series { steps: ArrayLike<number>; values: Float64Array }

// ------------------------------------------------------------ board
export interface ForecastView {
  window: [number, number] | null; target: number | null; band: number; pool: 'none' | 'metric' | 'group';
  live: boolean; frozen: Record<string, FitForecast> | null;
}
export interface PanelView { follow: boolean; hidden: string[]; solo: string | null; y2: string | null; fc: ForecastView; samplesK?: number }
export interface Panel { id: string; title: string; q: string; view: PanelView }
/** where a run started, set in the browser (Runs view) for runs that never declared it; the
 *  query engine reads it as the run's config parent_run / parent_step / stage (query.rs "Lineage") */
export interface Lineage { parent: string; step?: number | null; stage?: string | null }
export interface Board { version: number; panels: Panel[]; lineage?: Record<string, Lineage> }

// ------------------------------------------------------------ query engine (query.rs)
export interface Chip { kind: string; text: string; letter?: string | null; entry: number | null }
/** a failed parse: `start..end` (string indices) is the span, `suggest` replacements for it ("did you mean") */
export type Parsed = { ok: true; canonical: string; chips: Chip[] } | { ok: false; error: string; start: number; end: number; suggest: string[] };
/** a "did you mean" fix from evaluation: the whole query with one name replaced */
export interface QueryFix { msg: string; label: string; query: string }
export interface TraceMeta {
  label: string; letter: string; metric: string; runs: string[]; color: string; cell?: string; band: boolean; off: number; n: number; [k: string]: unknown;
}
export interface Trace extends TraceMeta { xs: Float64Array; ys: Float64Array; raw: Float64Array; lo: Float64Array; hi: Float64Array; steps: Float64Array }
/** one layer x step matrix of `heatmap over layer` (query.rs LayerHeatmap) */
export interface HeatmapMeta { run: string; family: string; layers: number[]; metrics: string[]; ncols: number; off: number }
export interface LayerHeatmap extends HeatmapMeta { cols: Float64Array; z: Float64Array }
export interface QueryMeta {
  traces: TraceMeta[]; heatmaps?: HeatmapMeta[]; warnings: string[]; fixes?: QueryFix[]; cells: string[]; x_label: string; ylog: boolean;
  by?: string | null; by_numeric?: boolean; forecast?: { to?: number | null; method?: string } | null; downsampled?: number | null; stitch?: StitchSpan[]; [k: string]: unknown;
}
/** `| stitch`: one plotted run's place on the axis (query.rs StitchSpan) */
export interface StitchSpan { run: string; parent: string | null; stage: string; x0: number; x1: number; step_off: number; x_off: number }
export type QueryResult = { ok: true; meta: QueryMeta; traces: Trace[]; heatmaps: LayerHeatmap[]; streams: number } | { ok: false; error: string; start?: number; end?: number };

export interface FitForecast {
  none?: undefined; method: string; skill: boolean; mae: number; naive: number; nTest: number; cfg: string;
  xs: ArrayLike<number>; ys: ArrayLike<number>; lo: ArrayLike<number>; hi: ArrayLike<number>; lastX: number;
}
export interface NoForecast { none: string }
export type Forecast = FitForecast | NoForecast;
