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
};

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
  },
  {
    key: "vision",
    name: "Visual retina / cortex",
    desc: "Each cell is seen through a retina → V1 → V2 → V4 pathway.",
    docHref: "/docs/multimodal",
    isActive: (s) => s.visionActive,
    liveValue: (s) => (s.visionActive ? "retina→V4 lit" : "no vision"),
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
  },
  {
    key: "vin",
    name: "Value Iteration Network",
    desc: "The learned planner: propagates value across the grid, reads the move ego-centrically.",
    docHref: "/docs/brain-composition",
    // The VIN drives every step the agent takes — active whenever it's running.
    isActive: (s) => s.ticksUsed > 0,
    liveValue: () => "planning",
  },
  {
    key: "plasticity",
    name: "Three-factor plasticity",
    desc: "A per-cell bias adapts live on the planner's prior — three-factor, no backprop.",
    docHref: "/docs/bio-inspired",
    // Lit on any step the live escape bias actually moved (‖Δ‖ > 0).
    isActive: (s) => s.plasticDelta > 0,
    liveValue: (s) => `‖Δ‖ ${s.plasticDelta.toFixed(3)}`,
  },
  {
    key: "neuromod",
    name: "Pain / neuromodulators",
    desc: "Pain from no progress / revisits, dopamine on progress and the goal.",
    docHref: "/docs/bio-inspired",
    // Lit whenever a (signed) neuromodulator signal was emitted this step.
    isActive: (s) => Math.abs(s.signal) > 1e-4,
    liveValue: (s) => (s.signal >= 0 ? `+${s.signal.toFixed(2)}` : s.signal.toFixed(2)),
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
  },
  {
    key: "adaptive",
    name: "Adaptive compute",
    desc: "An exit gate decides when it has thought enough and stops early.",
    docHref: "/docs/continuous-thought-machine",
    isActive: (s) => s.exitLambda != null,
    liveValue: (s) =>
      s.exitLambda != null ? `gate λ ${s.exitLambda.toFixed(3)}` : "off",
  },
];

export interface SdkFeaturesProps {
  state: SdkFeatureState;
}

export function SdkFeatures(props: SdkFeaturesProps) {
  const activeCount = () =>
    FEATURES.filter((f) => f.isActive(props.state)).length;
  const [open, setOpen] = createSignal(false);

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
        Each row is a modgrad SDK capability this demo runs. The dot lights only
        when its real signal is present this step.
      </p>

      <div class="flex flex-col">
        <For each={FEATURES}>
          {(f, i) => {
            const active = () => f.isActive(props.state);
            return (
              <div
                class="flex items-start gap-3 py-2.5"
                classList={{ "border-t border-line": i() > 0 }}
              >
                {/* live activity indicator */}
                <span
                  class="mt-1 w-2.5 h-2.5 rounded-full shrink-0"
                  title={active() ? "active this step" : "idle this step"}
                  style={{
                    background: active()
                      ? "var(--pos, #3CB371)"
                      : "var(--bg-2)",
                    "box-shadow": active()
                      ? "0 0 0 3px color-mix(in srgb, var(--pos, #3CB371) 22%, transparent)"
                      : "none",
                    border: active()
                      ? "none"
                      : "1px solid var(--line)",
                    transition: "background .12s ease, box-shadow .12s ease",
                  }}
                />

                <div class="min-w-0 flex-1">
                  <div class="flex items-baseline justify-between gap-2">
                    <span
                      class="text-sm font-medium"
                      classList={{
                        "text-dim": !active(),
                      }}
                    >
                      {f.name}
                    </span>
                    <Show when={active() && f.liveValue}>
                      <span class="font-mono text-[.68rem] text-accent tabular-nums shrink-0">
                        {f.liveValue!(props.state)}
                      </span>
                    </Show>
                  </div>
                  <div class="text-mute text-[.72rem] leading-relaxed mt-0.5">
                    {f.desc}
                  </div>
                  <A
                    href={f.docHref}
                    class="text-accent text-[.7rem] font-mono inline-block mt-1"
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
