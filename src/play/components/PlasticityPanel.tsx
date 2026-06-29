import { createSignal, createEffect, Show, For } from "solid-js";

// ── PlasticityPanel ───────────────────────────────────────────────────────
// The "learns while it plays" component. Honest, measured-signal visualization
// of the engine's local three-factor plasticity rule (no backprop, no training
// run). The parent feeds it the per-step signals; this panel keeps the rolling
// learning stats (streak / record / wall-hits / wall-hit-rate history) itself.
//
// Self-contained: no Play.tsx imports. Matches Play.tsx styling — `card`,
// `eyebrow`, `tag`, `text-dim`, `text-mute`, `font-mono`, accent var(--accent).

type Verdict = "ok" | "wall" | "astray" | "wait";
// graded neuromodulator tier the agent earned this step (Part A).
type Neuromod = "dopamine" | "reward" | "disappointment" | "pain";

export type PlasticityPanelProps = {
  enabled: boolean; // plasticity available + on (engine exposes apply_plasticity)
  signal: number; // the three-factor signal fed in this step (pain<0 / reward>0)
  neuromod?: Neuromod; // graded tier (dopamine/reward/disappointment/pain); falls back to signal
  plasticDelta: number; // ||ΔW|| applied to the readout this step
  verdict: Verdict | null; // the brain's prediction vs the maze, this step
  reached: boolean; // did the agent reach the goal this step?
  loss?: number; // move-head cross-entropy vs the target move (the curve that drops)
  lr?: number; // three-factor learning rate θ used this step (constant)
  efficiency?: number; // live steps ÷ shortest (→1.0 as it learns; >1 = wandering)
  efficiencyFinal?: number; // per-solve efficiency point (one per solved maze, for the trend)
  onReset: () => void; // revert the readout to the frozen weights
};

// ── tiny utils (in-file, no deps) ─────────────────────────────────────────

// moving average of a 0/1 (or any numeric) series over a trailing window.
// Returns one averaged point per input point, so the line has the same length.
function movingAvg(series: number[], window: number): number[] {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < series.length; i++) {
    sum += series[i];
    if (i >= window) sum -= series[i - window];
    const n = Math.min(i + 1, window);
    out.push(sum / n);
  }
  return out;
}

// inline SVG sparkline — no deps. Renders `values` (assumed ∈ [0,1]) as a
// polyline. Higher y at top is a HIGHER value, so we invert for screen space.
function Sparkline(props: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}) {
  const w = () => props.width ?? 220;
  const h = () => props.height ?? 40;
  const color = () => props.color ?? "var(--accent)";

  const points = () => {
    const vals = props.values;
    const W = w();
    const H = h();
    const pad = 3;
    if (vals.length === 0) return "";
    if (vals.length === 1) {
      const y = pad + (1 - clamp01(vals[0])) * (H - pad * 2);
      return `${pad.toFixed(1)},${y.toFixed(1)} ${(W - pad).toFixed(1)},${y.toFixed(1)}`;
    }
    const span = W - pad * 2;
    return vals
      .map((v, i) => {
        const x = pad + (i / (vals.length - 1)) * span;
        const y = pad + (1 - clamp01(v)) * (H - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  // area fill under the line (closes the polyline down to the baseline)
  const fillPoints = () => {
    const p = points();
    if (!p) return "";
    const W = w();
    const H = h();
    const pad = 3;
    return `${pad},${H - pad} ${p} ${W - pad},${H - pad}`;
  };

  return (
    <svg
      width={w()}
      height={h()}
      viewBox={`0 0 ${w()} ${h()}`}
      preserveAspectRatio="none"
      class="w-full"
      aria-hidden="true"
    >
      <Show when={props.values.length > 0}>
        <polygon
          points={fillPoints()}
          fill={color()}
          opacity="0.1"
        />
        <polyline
          points={points()}
          fill="none"
          stroke={color()}
          stroke-width="1.5"
          stroke-linejoin="round"
          stroke-linecap="round"
          vector-effect="non-scaling-stroke"
        />
      </Show>
    </svg>
  );
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

const HISTORY_WINDOW = 40; // rolling window of recent wall-hit (0/1) samples
const MA_WINDOW = 8; // moving-average window for the wall-hit-rate line

export default function PlasticityPanel(props: PlasticityPanelProps) {
  // ── internal learning state, derived reactively from the verdict stream ──
  const [streak, setStreak] = createSignal(0); // moves since last wall-hit
  const [record, setRecord] = createSignal(0); // best clean streak so far
  const [wallHits, setWallHits] = createSignal(0); // cumulative wall-hits
  const [history, setHistory] = createSignal<number[]>([]); // 0/1 wall-hit samples
  const [lossHist, setLossHist] = createSignal<number[]>([]); // move-head loss samples
  const [effHist, setEffHist] = createSignal<number[]>([]); // per-solve efficiency points

  // Fold each incoming verdict into the running stats. We track the verdict
  // identity so a re-render with the SAME verdict object doesn't double-count;
  // a fresh step always sets a new verdict (or the same value on a new step),
  // so we key on a monotonically-changing step token from the parent instead.
  // Since the parent updates `verdict` (and the other props) every step, we use
  // the prop tuple as the effect's dependency and process exactly one sample.
  let lastProcessed = -1;
  let stepToken = 0;

  createEffect(() => {
    // touch every per-step prop so the effect re-runs each step the parent
    // pushes new values — verdict + delta + signal + reached together identify
    // a step. We bump a token and only process once per token.
    const v = props.verdict;
    void props.plasticDelta;
    void props.signal;
    void props.reached;
    void props.loss;

    // a null verdict means "no step yet" (reset/maze load) — skip counting,
    // but DON'T wipe learned stats (Restart / New maze keep what was learned).
    if (v == null) return;

    const token = ++stepToken;
    if (token === lastProcessed) return;
    lastProcessed = token;

    const isWall = v === "wall";
    setHistory((h) => {
      const next = h.length >= HISTORY_WINDOW ? h.slice(1) : h.slice();
      next.push(isWall ? 1 : 0);
      return next;
    });
    const l = props.loss;
    if (typeof l === "number" && Number.isFinite(l)) {
      setLossHist((h) => {
        const next = h.length >= HISTORY_WINDOW ? h.slice(1) : h.slice();
        next.push(l);
        return next;
      });
    }
    if (isWall) setWallHits((n) => n + 1);
    // CLEAN STREAK = consecutive OPTIMAL steps only. A wrong-direction step
    // (astray / wait) breaks it just like a wall — wandering is not "clean".
    const isClean = v === "ok";
    if (isClean) {
      setStreak((s) => {
        const ns = s + 1;
        setRecord((r) => (ns > r ? ns : r));
        return ns;
      });
    } else {
      setStreak(0);
    }
  });

  // ── efficiency trend (Part A) ──
  // Push one point per solved maze (efficiencyFinal changes on each solve). Guard
  // against double-counting a re-render with the same value via `lastEff`.
  let lastEff = -1;
  createEffect(() => {
    const ef = props.efficiencyFinal;
    if (typeof ef !== "number" || !Number.isFinite(ef)) return;
    if (ef === lastEff) return;
    lastEff = ef;
    setEffHist((h) => {
      const next = h.length >= HISTORY_WINDOW ? h.slice(1) : h.slice();
      next.push(ef);
      return next;
    });
  });

  // wall-hit-rate moving average over the rolling window (lower = learning)
  const rate = () => movingAvg(history(), MA_WINDOW);
  const currentRate = () => {
    const r = rate();
    return r.length ? r[r.length - 1] : 0;
  };

  // training loss (move-head cross-entropy), smoothed. The Sparkline expects
  // [0,1], so we normalize by the window's own max — the curve fills the height
  // and its downward trend (the thing you watch in training) stays obvious.
  const lossMA = () => movingAvg(lossHist(), MA_WINDOW);
  const lossNorm = () => {
    const ma = lossMA();
    const max = Math.max(...ma, 1e-6);
    return ma.map((v) => v / max);
  };
  const currentLoss = () => {
    const ma = lossMA();
    return ma.length ? ma[ma.length - 1] : 0;
  };

  // efficiency trend: normalize each per-solve point to (v-1)/(max-1) so the
  // Sparkline DROPS toward the baseline as the agent approaches the 1.0 optimum.
  const effNorm = (): number[] => {
    const h = effHist();
    const max = Math.max(...h, 1.0);
    const denom = max - 1 || 1;
    return h.map((v) => clamp01((v - 1) / denom));
  };
  // the live efficiency ratio (steps ÷ shortest); falls back to the last solve.
  const liveEff = (): number => {
    const e = props.efficiency;
    if (typeof e === "number" && Number.isFinite(e)) return e;
    const h = effHist();
    return h.length ? h[h.length - 1] : 0;
  };

  // live 4-state pulse chip — prefers the graded `neuromod` tier, falling back
  // to the signed `signal` for back-compat (old callers / non-self-drive modes).
  type PulseCls = "dopamine" | "reward" | "disappointment" | "pain" | "flat";
  const pulse = (): { label: string; cls: PulseCls } => {
    const nm = props.neuromod;
    if (nm === "dopamine") return { label: "✦ dopamine", cls: "dopamine" };
    if (nm === "reward") return { label: "↑ reward", cls: "reward" };
    if (nm === "disappointment") return { label: "↘ disappointment", cls: "disappointment" };
    if (nm === "pain") return { label: "↓ pain", cls: "pain" };
    // fallback: derive a coarse tier from the signed signal
    const s = props.signal;
    if (s > 0) return { label: "↑ reward", cls: "reward" };
    if (s < 0) return { label: "↓ pain", cls: "pain" };
    return { label: "—", cls: "flat" };
  };
  // cls → {bg, fg} lookup (green / brighter-green / amber / red), no nested ternaries.
  const PULSE_STYLE: Record<PulseCls, { bg: string; fg: string }> = {
    dopamine: { bg: "color-mix(in srgb, #3CB371 22%, transparent)", fg: "#2f8c5b" },
    reward: { bg: "color-mix(in srgb, #3CB371 14%, transparent)", fg: "#3aa86c" },
    disappointment: { bg: "color-mix(in srgb, #e0a23c 18%, transparent)", fg: "#b07a1e" },
    pain: { bg: "color-mix(in srgb, #e0564e 16%, transparent)", fg: "#c63d35" },
    flat: { bg: "var(--bg-2)", fg: "var(--text-mute)" },
  };

  // revert the readout to frozen weights AND clear the local learning stats,
  // so the panel and the engine agree on "nothing learned yet".
  const handleReset = () => {
    props.onReset();
    setStreak(0);
    setRecord(0);
    setWallHits(0);
    setHistory([]);
    setLossHist([]);
    setEffHist([]);
    lastProcessed = -1;
    stepToken = 0;
    lastEff = -1;
  };

  const readouts = (): { label: string; value: () => number }[] => [
    { label: "clean streak", value: streak },
    { label: "record", value: record },
    { label: "wall-hits", value: wallHits },
  ];

  return (
    <div class="card">
      <style>{`
        @keyframes pp-pulse {
          0%,100% { opacity: 1; }
          50%     { opacity: .55; }
        }
        .pp-live { animation: pp-pulse 1.6s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) {
          .pp-live { animation: none; }
        }
      `}</style>

      <div class="flex items-center justify-between mb-3">
        <div class="eyebrow">Planner</div>
        <span class="tag tag-live">trained · live</span>
      </div>

      {/* ── one honest line: trained offline, planning live, no solver ── */}
      <div class="text-[.72rem] text-mute font-mono mb-4 leading-relaxed">
        Trained offline; planning live. No solver in the loop — these are its
        live accuracy against the (hidden) optimal route.
      </div>

      {/* ── stat tiles ── */}
      <div class="grid grid-cols-3 gap-3 mb-4">
        <For each={readouts()}>
          {(r) => (
            <div
              class="rounded-lg px-3 py-2.5 text-center"
              style={{ background: "var(--bg-2)" }}
            >
              <div class="font-mono text-[1.35rem] leading-none tabular-nums text-accent">
                {r.value()}
              </div>
              <div class="eyebrow mt-1.5 text-[.62rem]">{r.label}</div>
            </div>
          )}
        </For>
      </div>

      {/* ── metric sparklines: terse label + value, no captions ── */}
      <Show when={lossHist().length > 0}>
        <div class="mb-3">
          <div class="flex items-center justify-between mb-1 font-mono text-[.72rem]">
            <span class="text-dim">gap from optimal</span>
            <span class="text-mute tabular-nums">{currentLoss().toFixed(3)}</span>
          </div>
          <div class="rounded-lg overflow-hidden" style={{ background: "var(--bg-2)", height: "40px" }}>
            <Sparkline values={lossNorm()} height={40} color="#7c6cf0" />
          </div>
        </div>
      </Show>

      <div class="mb-3">
        <div class="flex items-center justify-between mb-1 font-mono text-[.72rem]">
          <span class="text-dim">efficiency</span>
          <span class="text-mute tabular-nums">{liveEff().toFixed(2)}×</span>
        </div>
        <div class="rounded-lg overflow-hidden" style={{ background: "var(--bg-2)", height: "40px" }}>
          <Show
            when={effHist().length > 0}
            fallback={<div class="grid place-items-center h-full text-[.7rem] text-mute font-mono">solving…</div>}
          >
            <Sparkline values={effNorm()} height={40} color="#3aa86c" />
          </Show>
        </div>
      </div>

      <div class="mb-4">
        <div class="flex items-center justify-between mb-1 font-mono text-[.72rem]">
          <span class="text-dim">wall-hit rate</span>
          <span class="text-mute tabular-nums">{(currentRate() * 100).toFixed(0)}%</span>
        </div>
        <div class="rounded-lg overflow-hidden" style={{ background: "var(--bg-2)", height: "40px" }}>
          <Show
            when={history().length > 0}
            fallback={<div class="grid place-items-center h-full text-[.7rem] text-mute font-mono">—</div>}
          >
            <Sparkline values={rate()} height={40} />
          </Show>
        </div>
      </div>

      {/* ── learning rate + reset to initial state ── */}
      <div class="flex items-center justify-between gap-3 pt-1 border-t border-[var(--border)]/60 mt-1">
        <span
          class="font-mono text-[.72rem] text-mute tabular-nums"
          title="three-factor learning rate θ applied per live update (frozen substrate, plastic surround)"
        >
          learning rate θ{" "}
          <span class="text-dim">
            {props.lr != null ? props.lr.toFixed(3) : "—"}
          </span>
        </span>
        <button
          type="button"
          onClick={handleReset}
          class="font-mono text-[.72rem] px-2.5 py-1 rounded-md transition-colors"
          style={{
            background: "var(--bg-2)",
            border: "1px solid var(--border)",
            color: "var(--text-mute)",
          }}
          title="revert the planner to its trained-offline weights, undoing every live update and dream consolidation"
        >
          ↺ reset learning
        </button>
      </div>
    </div>
  );
}
