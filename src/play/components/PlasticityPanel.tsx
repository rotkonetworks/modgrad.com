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

export type PlasticityPanelProps = {
  enabled: boolean; // plasticity available + on (engine exposes apply_plasticity)
  signal: number; // the three-factor signal fed in this step (pain<0 / reward>0)
  plasticDelta: number; // ||ΔW|| applied to the readout this step
  verdict: Verdict | null; // the brain's prediction vs the maze, this step
  reached: boolean; // did the agent reach the goal this step?
  loss?: number; // move-head cross-entropy vs the target move (the curve that drops)
  lr?: number; // three-factor learning rate θ used this step (constant)
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
    if (isWall) {
      setWallHits((n) => n + 1);
      setStreak(0);
    } else {
      setStreak((s) => {
        const ns = s + 1;
        setRecord((r) => (ns > r ? ns : r));
        return ns;
      });
    }
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

  // live pulse chip state from the raw signal
  const pulse = () => {
    const s = props.signal;
    if (s > 0) return { label: "↑ dopamine", cls: "up" as const };
    if (s < 0) return { label: "↓ pain", cls: "down" as const };
    return { label: "—", cls: "flat" as const };
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
    lastProcessed = -1;
    stepToken = 0;
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
        <div class="eyebrow">Plasticity</div>
        <span
          class="tag"
          classList={{ "tag-live": props.enabled }}
        >
          {props.enabled ? "learning" : "frozen"}
        </span>
      </div>

      {/* ── banner: only while plasticity is on ── */}
      <Show when={props.enabled}>
        <div
          class="rounded-lg px-3.5 py-3 mb-4 text-sm leading-relaxed"
          style={{
            background: "color-mix(in srgb, var(--accent) 9%, transparent)",
            border:
              "1px solid color-mix(in srgb, var(--accent) 28%, transparent)",
          }}
        >
          <span class="font-mono text-accent">Learning — right now, in your browser.</span>{" "}
          <span class="text-dim">
            No training run, no optimizer: a local three-factor rule updates the
            readout each step — pain on wall-hits, dopamine on the goal.{" "}
          </span>
          <span class="text-accent">Inference IS learning.</span>
        </div>
      </Show>

      {/* ── live pulse chip + ΔW ── */}
      <div class="flex flex-wrap items-center gap-2 mb-4">
        <span
          class="pp-live inline-flex items-center gap-1.5 font-mono text-xs px-2 py-1 rounded"
          style={{
            background:
              pulse().cls === "up"
                ? "color-mix(in srgb, #3CB371 16%, transparent)"
                : pulse().cls === "down"
                  ? "color-mix(in srgb, #e0564e 16%, transparent)"
                  : "var(--bg-2)",
            color:
              pulse().cls === "up"
                ? "#2f8c5b"
                : pulse().cls === "down"
                  ? "#c63d35"
                  : "var(--text-mute)",
          }}
        >
          {pulse().label}
        </span>
        <span class="font-mono text-xs text-mute tabular-nums">
          ΔW {props.plasticDelta.toFixed(3)}
        </span>
        <Show when={props.lr != null}>
          <span class="font-mono text-xs text-mute tabular-nums" title="three-factor learning rate θ (constant, set in the engine)">
            lr {props.lr!.toFixed(3)}
          </span>
        </Show>
        <Show when={props.reached}>
          <span
            class="font-mono text-xs px-2 py-1 rounded"
            style={{
              background: "color-mix(in srgb, var(--accent) 16%, transparent)",
              color: "var(--accent)",
            }}
          >
            ✦ goal · dopamine burst
          </span>
        </Show>
      </div>

      {/* ── three readouts ── */}
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

      {/* ── training loss curve (move-head cross-entropy → 0 = learned) ── */}
      <Show when={lossHist().length > 0}>
        <div class="mb-4">
          <div class="flex items-center justify-between mb-1.5">
            <div class="text-[.72rem] text-mute font-mono">
              training loss (lower = learning)
            </div>
            <div class="font-mono text-[.72rem] text-dim tabular-nums">
              {currentLoss().toFixed(3)}
            </div>
          </div>
          <div
            class="rounded-lg overflow-hidden"
            style={{ background: "var(--bg-2)", height: "46px" }}
          >
            <Sparkline values={lossNorm()} height={46} color="#7c6cf0" />
          </div>
          <div class="text-[.66rem] text-mute font-mono mt-1.5">
            move-head cross-entropy vs the step it should take — the curve you
            watch drop in training, here it drops at inference
          </div>
        </div>
      </Show>

      {/* ── wall-hit-rate sparkline (trending down = learning) ── */}
      <div class="mb-4">
        <div class="flex items-center justify-between mb-1.5">
          <div class="text-[.72rem] text-mute font-mono">
            wall-hit rate (lower = learning)
          </div>
          <div class="font-mono text-[.72rem] text-dim tabular-nums">
            {(currentRate() * 100).toFixed(0)}%
          </div>
        </div>
        <div
          class="rounded-lg overflow-hidden"
          style={{ background: "var(--bg-2)", height: "46px" }}
        >
          <Show
            when={history().length > 0}
            fallback={
              <div class="grid place-items-center h-full text-[.7rem] text-mute font-mono">
                waiting for the first step…
              </div>
            }
          >
            <Sparkline values={rate()} height={46} />
          </Show>
        </div>
        <div class="text-[.66rem] text-mute font-mono mt-1.5">
          each point is a measured wall-hit over the last {HISTORY_WINDOW} moves
        </div>
      </div>

      {/* ── reset + framing copy ── */}
      <div class="pt-4 border-t border-line">
        <div class="flex flex-wrap items-center gap-3">
          <button
            class="btn"
            onClick={handleReset}
            title="revert the readout to the frozen weights"
          >
            Reset learning
          </button>
          <div class="text-[.72rem] text-mute leading-relaxed flex-1 min-w-[200px]">
            <span class="text-dim">Restart</span> and{" "}
            <span class="text-dim">New maze</span> keep what was learned.{" "}
            <span class="text-dim">Reset</span> reverts the readout to the frozen
            weights.
          </div>
        </div>
        <div class="text-[.66rem] text-mute font-mono mt-3 leading-relaxed">
          Bounded by design: ΔW and the readout weights are clamped each step, so
          the local rule can't blow up or drift away from the trained brain.
        </div>
      </div>
    </div>
  );
}
