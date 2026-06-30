import { For, Show, createSignal } from "solid-js";
import { A } from "@solidjs/router";
import type { RegionTel } from "./RegionTelemetry";

// ── what the parent passes us each step ──────────────────────────────────────
// Every field is a *real* signal off the running brain. We only light a feature
// when its signal is actually present this step — no decorative "always on".
export interface SdkFeatureState {
  /** Recurrent ticks the CTM actually ran this step. */
  ticksUsed: number;
  /** Adaptive-compute exit gate λ, or null if it never fired. */
  exitLambda: number | null;
  /** True when the visual retina/cortex produced feature maps this step. */
  visionActive: boolean;
  /** Magnitude of any three-factor plastic weight change this step (≥0). */
  plasticDelta: number;
  /** Pain / reward neuromodulator signal this step (e.g. wall-hit, reach). */
  signal: number;
  /** Optional per-region engine stats (drives the memory/region readouts). */
  telemetry?: RegionTel[];
  /** Episodic-memory recall this step: nearest past situation by similarity. */
  episodic?: {
    recalled: boolean;
    sim: number; // cosine similarity to the nearest stored episode
    id: number; // which episode matched
    size: number; // episodes currently held
  } | null;
}

type Feature = {
  key: string;
  name: string;
  desc: string;
  docHref: string;
  isActive: (s: SdkFeatureState) => boolean;
  liveValue?: (s: SdkFeatureState) => string;
  /** 0..1 live intensity that fills the row's meter bar. */
  intensity: (s: SdkFeatureState) => number;
  /** bar colour — constant, or a function of state (e.g. pain vs reward). */
  color: string | ((s: SdkFeatureState) => string);
};
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const resolveColor = (f: Feature, s: SdkFeatureState) =>
  typeof f.color === "function" ? f.color(s) : f.color;

// Data-driven list of the modgrad SDK features this demo exercises. Each row is
// honest about *when* it's active and shows its live value when it is.
const FEATURES: Feature[] = [
  {
    key: "ctm",
    name: "Continuous Thought Machine",
    desc: "Each region thinks over recurrent ticks, not one forward pass.",
    docHref: "/docs/continuous-thought-machine",
    isActive: (s) => s.ticksUsed > 0,
    liveValue: (s) =>
      `${s.ticksUsed} ticks` +
      (s.exitLambda != null ? ` · λ ${s.exitLambda.toFixed(2)}` : ""),
    intensity: (s) => clamp01(s.ticksUsed / 12),
    color: "#62e6ff",
  },
  {
    key: "vision",
    name: "Visual retina / cortex",
    desc: "Each cell is seen through a retina → V1 → V2 → V4 pathway.",
    docHref: "/docs/multimodal",
    isActive: (s) => s.visionActive,
    liveValue: (s) => (s.visionActive ? "retina→V4 lit" : "no vision"),
    intensity: (s) => (s.visionActive ? 1 : 0),
    color: "#62e6ff",
  },
  {
    key: "regions",
    name: "8-region brain",
    desc: "input · attention · output · motor · cerebellum · basal-ganglia · insula · hippocampus.",
    docHref: "/docs/brain-composition",
    // The wired brain is always the substrate; treat it as active whenever the
    // brain is thinking (ticks ran) so the row reflects real computation.
    isActive: (s) => s.ticksUsed > 0,
    liveValue: () => "8 regions wired",
    intensity: (s) => {
      const t = s.telemetry;
      if (!t || t.length === 0) return s.ticksUsed > 0 ? 0.5 : 0;
      const mean = t.reduce((a, r) => a + (r.rms ?? 0), 0) / t.length;
      return clamp01(mean / 1.2);
    },
    color: "#8a7dff",
  },
  {
    key: "vin",
    name: "Value Iteration Network",
    desc: "The planner, distributed across regions: basal ganglia value each cell (dopamine = RPE), the hippocampus replays value across its cognitive map (a Bellman backup), motor reads the move ego-centrically.",
    docHref: "/docs/brain-composition",
    // The VIN drives every step the agent takes — active whenever it's running.
    isActive: (s) => s.ticksUsed > 0,
    liveValue: () => "planning",
    intensity: (s) => (s.ticksUsed > 0 ? 1 : 0),
    color: "#7c6cf0",
  },
  {
    key: "plasticity",
    name: "Three-factor plasticity",
    desc: "A per-cell bias adapts live on the planner's prior — three-factor, no backprop.",
    docHref: "/docs/bio-inspired",
    // Lit on any step the live escape bias actually moved (‖Δ‖ > 0).
    isActive: (s) => s.plasticDelta > 0,
    liveValue: (s) => `‖Δ‖ ${s.plasticDelta.toFixed(3)}`,
    intensity: (s) => clamp01(s.plasticDelta / 0.3),
    color: "#3aa86c",
  },
  {
    key: "neuromod",
    name: "Pain / neuromodulators",
    desc: "Pain from no progress / revisits, dopamine on progress and the goal.",
    docHref: "/docs/bio-inspired",
    // Lit whenever a (signed) neuromodulator signal was emitted this step.
    isActive: (s) => Math.abs(s.signal) > 1e-4,
    liveValue: (s) => (s.signal >= 0 ? `+${s.signal.toFixed(2)}` : s.signal.toFixed(2)),
    intensity: (s) => clamp01(Math.abs(s.signal) / 1.5),
    color: (s) => (s.signal < 0 ? "#e0564e" : "#3aa86c"),
  },
  {
    key: "memory",
    name: "Episodic memory",
    desc: "Recalls the nearest past situation it has seen, by similarity.",
    docHref: "/docs/memory",
    isActive: (s) => !!s.episodic?.recalled,
    liveValue: (s) => {
      const e = s.episodic;
      if (!e || e.size === 0) return "empty";
      if (e.recalled) return `recall #${e.id} · sim ${e.sim.toFixed(2)}`;
      return `${e.size} stored`;
    },
    intensity: (s) => {
      const e = s.episodic;
      if (!e || e.size === 0) return 0;
      return e.recalled ? clamp01(e.sim) : clamp01(e.size / 64) * 0.4;
    },
    color: "#e06ce0",
  },
  {
    key: "adaptive",
    name: "Adaptive compute",
    desc: "An exit gate decides when it has thought enough and stops early.",
    docHref: "/docs/continuous-thought-machine",
    isActive: (s) => s.exitLambda != null,
    liveValue: (s) =>
      s.exitLambda != null ? `gate λ ${s.exitLambda.toFixed(3)}` : "off",
    intensity: (s) => (s.exitLambda != null ? clamp01(s.exitLambda) : 0),
    color: "#e0a23c",
  },
];

export interface SdkFeaturesProps {
  state: SdkFeatureState;
}

export function SdkFeatures(props: SdkFeaturesProps) {
  const activeCount = () =>
    FEATURES.filter((f) => f.isActive(props.state)).length;
  const [open, setOpen] = createSignal(true);

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
          SDK features in use
        </div>
        <span class="tag" classList={{ "tag-live": activeCount() > 0 }}>
          {activeCount()}/{FEATURES.length} live
        </span>
      </button>
      <Show when={open()}>
      <p class="text-mute text-[.72rem] leading-relaxed mb-3 mt-2">
        Each row is a modgrad SDK capability this demo runs — the meter is its
        live intensity this step, off when its real signal is absent.
      </p>

      <div class="flex flex-col gap-3">
        <For each={FEATURES}>
          {(f) => {
            const active = () => f.isActive(props.state);
            const level = () => (active() ? f.intensity(props.state) : 0);
            const col = () => resolveColor(f, props.state);
            return (
              <div class="flex flex-col gap-1">
                {/* label row: name + live value */}
                <div class="flex items-baseline justify-between gap-2">
                  <span
                    class="font-mono text-[.74rem]"
                    classList={{ "text-dim": !active() }}
                    title={f.desc}
                  >
                    {f.name}
                  </span>
                  <span
                    class="font-mono text-[.68rem] tabular-nums shrink-0"
                    classList={{ "text-mute": !active() }}
                    style={active() ? { color: col() } : undefined}
                  >
                    {active() && f.liveValue ? f.liveValue(props.state) : "idle"}
                  </span>
                </div>
                {/* meter bar — fills to live intensity, like the other panels */}
                <div
                  class="rounded-full overflow-hidden"
                  style={{ background: "var(--bg-2)", height: "6px" }}
                >
                  <div
                    class="h-full rounded-full"
                    style={{
                      width: `${Math.round(level() * 100)}%`,
                      background: col(),
                      opacity: active() ? "1" : "0",
                      transition: "width .18s ease, opacity .18s ease",
                      "box-shadow": active()
                        ? `0 0 8px color-mix(in srgb, ${col()} 55%, transparent)`
                        : "none",
                    }}
                  />
                </div>
                <div class="flex items-center justify-between gap-2">
                  <span class="text-mute text-[.68rem] leading-snug">
                    {f.desc}
                  </span>
                  <A
                    href={f.docHref}
                    class="text-accent text-[.66rem] font-mono shrink-0"
                  >
                    docs →
                  </A>
                </div>
              </div>
            );
          }}
        </For>
      </div>
      </Show>
    </div>
  );
}

export default SdkFeatures;
