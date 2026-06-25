import { For, Show, createSignal } from "solid-js";

// ── per-region neuromod / stats the engine can attach to a step ──────────────
// Optional and forward-looking: if the wasm engine starts emitting per-region
// dopamine / pain channels, we surface them; otherwise the panel degrades to
// the activation-RMS bars we can always compute from the tick acts.
export type RegionTel = {
  name: string;
  rms: number;
  peak: number;
  dopamine?: number;
  pain?: number;
};

// Same palette the 3D brain in Play.tsx uses, so a region's dot here matches
// its colour in the particle cloud.
export const REGION_COLORS = [
  "#4488ff",
  "#44ff88",
  "#ff8844",
  "#ff4488",
  "#aa66ff",
  "#66ff44",
  "#ff66aa",
  "#ffdd44",
] as const;

// Short, stable labels for the 8 regions (falls back to the raw name).
const SHORT: Record<string, string> = {
  input: "input",
  attention: "attn",
  output: "output",
  motor: "motor",
  cerebellum: "cereb",
  basal_ganglia: "basal",
  insula: "insula",
  hippocampus: "hippo",
};
const shortName = (n: string): string => SHORT[n] ?? n;

export interface RegionTelemetryProps {
  /** Region structure from brain_solver_reference.json (name + neuron count). */
  regions: { name: string; d_model: number }[];
  /** Per-region activation vectors for the current tick, or null when idle. */
  acts: number[][] | null;
  /** Normalised global-sync magnitude at the current tick (0..1). */
  global: number;
  /** Adaptive-compute exit gate λ for this step, or null if it never fired. */
  exitLambda: number | null;
  ticksUsed: number;
  ticksTotal: number;
  /** Optional richer per-region stats from the engine (neuromods etc.). */
  telemetry?: RegionTel[];
  /** Expand the panel by default (used when it anchors a column). */
  defaultOpen?: boolean;
}

// activation RMS for one region's vector — the same measure Play.tsx bars on.
function rmsOf(a: number[] | undefined): { rms: number; peak: number } {
  if (!a || !a.length) return { rms: 0, peak: 0 };
  let s = 0;
  let peak = 0;
  for (const v of a) {
    s += v * v;
    const av = v < 0 ? -v : v;
    if (av > peak) peak = av;
  }
  return { rms: Math.sqrt(s / a.length), peak };
}

export function RegionTelemetry(props: RegionTelemetryProps) {
  // Build a row per region, preferring engine telemetry when it's present and
  // otherwise computing RMS/peak straight from the tick acts.
  const rows = (): (RegionTel & { d_model: number; color: string })[] => {
    const tel = props.telemetry;
    return props.regions.map((rg, i) => {
      const fromEngine = tel?.find((t) => t.name === rg.name);
      const base = fromEngine ?? { name: rg.name, ...rmsOf(props.acts?.[i]) };
      return {
        ...base,
        d_model: rg.d_model,
        color: REGION_COLORS[i % REGION_COLORS.length],
      };
    });
  };

  // Normalise bar lengths to the loudest region so they're comparable.
  const maxRms = (): number => {
    let m = 1e-6;
    for (const r of rows()) if (r.rms > m) m = r.rms;
    return m;
  };

  const hasNeuromod = (): boolean =>
    !!props.telemetry?.some((t) => t.dopamine != null || t.pain != null);

  const pct = (v: number, scale = 1): string =>
    `${Math.max(0, Math.min(100, (v / scale) * 100)).toFixed(0)}%`;

  const [open, setOpen] = createSignal(props.defaultOpen ?? false);

  return (
    <div class="card">
      <button
        type="button"
        class="flex items-center justify-between w-full text-left"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open()}
      >
        <div class="eyebrow flex items-center gap-1.5">
          <span
            class="inline-block transition-transform"
            style={{ transform: open() ? "rotate(90deg)" : "none" }}
          >
            ▸
          </span>
          Region telemetry
        </div>
        <div class="font-mono text-xs text-mute tabular-nums">
          {props.regions.length} regions · live
        </div>
      </button>

      <Show when={open()}>
      <div class="flex flex-col gap-2 mt-3">
        <For each={rows()}>
          {(r) => {
            const level = () => Math.min(1, r.rms / maxRms());
            return (
              <div>
                <div class="flex items-center gap-2 text-[.72rem] font-mono">
                  <i
                    class="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                    style={{ background: r.color }}
                  />
                  <span class="w-12 shrink-0 text-dim">{shortName(r.name)}</span>
                  <span class="w-12 shrink-0 text-mute tabular-nums">
                    {r.d_model}n
                  </span>
                  <span class="ml-auto text-mute tabular-nums">
                    rms {r.rms.toFixed(3)}
                  </span>
                </div>
                {/* activation RMS bar */}
                <div
                  class="mt-1 h-1.5 rounded-full overflow-hidden"
                  style={{ background: "var(--bg-2)" }}
                >
                  <div
                    class="h-full rounded-full"
                    style={{
                      width: pct(level()),
                      background: r.color,
                      transition: "width .12s linear",
                    }}
                  />
                </div>

                {/* engine neuromod bars — only when the engine supplies them */}
                <Show when={r.dopamine != null || r.pain != null}>
                  <div class="mt-1 flex gap-2">
                    <Show when={r.dopamine != null}>
                      <div class="flex-1 flex items-center gap-1.5">
                        <span class="text-[.6rem] font-mono text-mute w-7 shrink-0">
                          DA
                        </span>
                        <div
                          class="flex-1 h-1 rounded-full overflow-hidden"
                          style={{ background: "var(--bg-2)" }}
                        >
                          <div
                            class="h-full rounded-full"
                            style={{
                              width: pct(r.dopamine ?? 0),
                              background: "#44ff88",
                            }}
                          />
                        </div>
                      </div>
                    </Show>
                    <Show when={r.pain != null}>
                      <div class="flex-1 flex items-center gap-1.5">
                        <span class="text-[.6rem] font-mono text-mute w-7 shrink-0">
                          pain
                        </span>
                        <div
                          class="flex-1 h-1 rounded-full overflow-hidden"
                          style={{ background: "var(--bg-2)" }}
                        >
                          <div
                            class="h-full rounded-full"
                            style={{
                              width: pct(r.pain ?? 0),
                              background: "#e0564e",
                            }}
                          />
                        </div>
                      </div>
                    </Show>
                  </div>
                </Show>
              </div>
            );
          }}
        </For>
      </div>

      {/* footer: global-sync, exit λ, ticks used */}
      <div class="flex flex-wrap items-center gap-x-5 gap-y-1 mt-4 pt-3 border-t border-line text-[.7rem] font-mono text-mute tabular-nums">
        <span class="inline-flex items-center gap-1.5">
          <span class="text-dim">global-sync</span>
          {props.global.toFixed(2)}
        </span>
        <span class="inline-flex items-center gap-1.5">
          <span class="text-dim">exit λ</span>
          {props.exitLambda == null ? "—" : props.exitLambda.toFixed(3)}
        </span>
        <span class="inline-flex items-center gap-1.5">
          <span class="text-dim">ticks</span>
          {props.ticksUsed}/{props.ticksTotal}
        </span>
        <Show when={hasNeuromod()}>
          <span class="text-mute ml-auto">DA = dopamine · neuromod live</span>
        </Show>
      </div>
      </Show>
    </div>
  );
}

export default RegionTelemetry;
