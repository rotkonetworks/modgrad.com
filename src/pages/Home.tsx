import { A } from "@solidjs/router";
import { For } from "solid-js";
import { reveal } from "@/lib/reveal";
import { useDocMeta } from "@/lib/meta";

// keep the directive referenced for the compiler
false && reveal;
import { site } from "@/data/site";
import {
  crates,
  differentiators,
  heroCode,
  howItWorks,
  pillars,
  proofStat,
  regions,
  runningToday,
  useCases,
} from "@/data/content";

export default function Home() {
  useDocMeta(() => ({ path: "/" }));

  return (
    <>
      {/* ── hero ─────────────────────────────────────── */}
      <section class="relative overflow-hidden border-b border-line">
        <div class="grid-bg" />
        <div class="hero-glow" />
        <div class="container-page relative pt-16 pb-20 sm:pt-24 sm:pb-28">
          <div class="grid lg:grid-cols-[1.05fr_0.95fr] gap-12 lg:gap-10 items-center">
            <div>
              <div class="chip mb-6">
                <span class="chip-dot" />
                Rust · Continuous Thought Machines · open source
              </div>
              <h1 class="text-[clamp(2.6rem,6.4vw,4.3rem)] leading-[1.04] font-650 tracking-[-0.03em]">
                Build your <span class="grad-text">own brain.</span>
              </h1>
              <p class="mt-6 text-[1.12rem] text-dim max-w-[46ch] leading-relaxed">
                Composable Rust crates for building brains: Continuous Thought
                Machines, multi-region composition, bio-inspired learning, and
                full-residency GPU training. No framework. No YAML. Just functions.
              </p>
              <div class="mt-8 flex flex-wrap gap-3">
                <A href="/docs" class="btn btn-primary">Get started →</A>
                <a href={site.repo} target="_blank" rel="noopener" class="btn btn-ghost">
                  View on GitHub ↗
                </a>
              </div>
              <div class="mt-7 font-mono text-[.82rem] text-mute">
                <span class="text-dim">$</span> cargo add modgrad-ctm
              </div>
            </div>

            {/* code card */}
            <div class="glow-card overflow-hidden">
              <div class="flex items-center gap-2 px-4 h-10 border-b border-line">
                <span class="w-3 h-3 rounded-full" style={{ background: "#ff5f56" }} />
                <span class="w-3 h-3 rounded-full" style={{ background: "#ffbd2e" }} />
                <span class="w-3 h-3 rounded-full" style={{ background: "#27c93f" }} />
                <span class="ml-2 font-mono text-xs text-mute">brain.rs</span>
              </div>
              <pre class="code !border-0 !rounded-none !m-0 text-[.8rem]"><code>{heroCode}</code></pre>
            </div>
          </div>
        </div>
      </section>

      {/* ── the bet ──────────────────────────────────── */}
      <section class="container-page py-20 sm:py-24">
        <div use:reveal class="max-w-[760px]">
          <div class="eyebrow mb-4">The bet</div>
          <h2 class="text-[clamp(1.7rem,4vw,2.5rem)] tracking-[-0.025em] leading-[1.12]">
            Betting on architecture, not scale.
          </h2>
          <div class="mt-7 text-[1.1rem] text-dim leading-relaxed flex flex-col gap-4">
            <p class="dropcap">
              The frontier scaled a single architecture, the transformer, to the limits of
              capital. modgrad takes a different bet: that intelligence comes from specialized
              regions that <strong>think recurrently</strong>, route information between each
              other, remember, and learn from their own surprise. Each is a part you can build
              and test on its own.
            </p>
            <p>
              Those parts ship as plain Rust crates: a Continuous Thought Machine, graph
              composition, neuromodulated learning, multimodal codecs, a transformer stack,
              GPU residency. You assemble the brain you want. The same SDK trains a
              187k-parameter router and mounts a frozen LLM as a cerebellum, and it runs on one
              workstation.
            </p>
          </div>
        </div>
      </section>

      {/* ── proof ────────────────────────────────────── */}
      <section use:reveal class="container-page pb-8">
        <div class="grid lg:grid-cols-[0.9fr_1.1fr] gap-6">
          <div class="card glow-card">
            <div class="eyebrow mb-4">Measured, reproducible</div>
            <div class="text-[1.15rem] font-600 mb-5 leading-snug">{proofStat.headline}</div>
            <div class="flex flex-col gap-3">
              <For each={proofStat.rows}>
                {(r) => (
                  <div class="flex items-center gap-4">
                    <div class="w-[120px] text-sm text-dim">{r.label}</div>
                    <div class="font-mono text-xs text-mute w-12">{r.params}</div>
                    <div class="flex-1 h-2 rounded-full bg-raised overflow-hidden">
                      <div
                        class="h-full rounded-full"
                        style={{
                          width: r.metric,
                          background: r.pos ? "var(--grad)" : "var(--line-2)",
                        }}
                      />
                    </div>
                    <div class="stat-num !text-[1.15rem]" classList={{ "grad-text": r.pos }}>
                      {r.metric}
                    </div>
                  </div>
                )}
              </For>
            </div>
            <p class="mt-5 text-sm text-mute leading-relaxed">{proofStat.foot}</p>
          </div>

          <div class="card">
            <div class="eyebrow mb-4">Running today</div>
            <ul class="flex flex-col">
              <For each={runningToday}>
                {([title, body], i) => (
                  <li
                    class="flex gap-4 py-3.5"
                    classList={{ "border-t border-line": i() > 0 }}
                  >
                    <span class="mt-1 text-pos">▸</span>
                    <div>
                      <div class="font-550">{title}</div>
                      <div class="text-sm text-dim leading-relaxed mt-0.5">{body}</div>
                    </div>
                  </li>
                )}
              </For>
            </ul>
          </div>
        </div>
      </section>

      {/* ── pillars ──────────────────────────────────── */}
      <section use:reveal class="container-page py-20">
        <div class="eyebrow mb-3">Capabilities</div>
        <h2 class="text-[clamp(1.6rem,3.6vw,2.3rem)] tracking-[-0.025em] mb-10">
          Compose what you need.
        </h2>
        <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <For each={pillars}>
            {(p) => (
              <A href={p.href} class="card card-hover group flex flex-col">
                <span class="tag w-fit mb-4">{p.tag}</span>
                <h3 class="text-[1.12rem] mb-2">{p.title}</h3>
                <p class="text-sm text-dim leading-relaxed flex-1">{p.body}</p>
                <span class="mt-4 text-sm text-accent">Learn more →</span>
              </A>
            )}
          </For>
        </div>
      </section>

      {/* ── brain regions ────────────────────────────── */}
      <section use:reveal class="container-page py-12">
        <div class="card glow-card">
          <div class="grid lg:grid-cols-[0.8fr_1.2fr] gap-8 items-center">
            <div>
              <div class="eyebrow mb-3">The 8-region brain</div>
              <h2 class="text-[1.7rem] tracking-[-0.02em] mb-3">
                One graph, eight specialists.
              </h2>
              <p class="text-dim leading-relaxed text-[.98rem]">
                Each region is its own CTM with its own neuron count, memory depth, and tick
                budget. They talk over learned inter-region synapses: a cortical loop plus
                subcortical value, salience, and episodic memory pathways.
              </p>
              <A href="/docs/brain-composition" class="inline-block mt-5 text-sm text-accent">
                Read the architecture →
              </A>
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <For each={regions}>
                {([name, role]) => (
                  <div class="rounded-xl border border-line bg-panel p-3 hover:border-line-2 transition">
                    <div class="font-mono text-[.82rem] text-accent font-600">{name}</div>
                    <div class="text-[.72rem] text-mute leading-snug mt-1">{role}</div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </div>
      </section>

      {/* ── how it works ─────────────────────────────── */}
      <section use:reveal class="container-page py-20">
        <div class="eyebrow mb-3">How it works</div>
        <h2 class="text-[clamp(1.6rem,3.6vw,2.3rem)] tracking-[-0.025em] mb-10">
          Think, compose, learn.
        </h2>
        <div class="border border-line rounded-xl overflow-hidden">
          <For each={howItWorks}>
            {([title, body], i) => (
              <div
                class="bg-card p-6 sm:p-8 grid sm:grid-cols-[3rem_1fr] gap-x-5 items-baseline"
                classList={{ "border-t border-line": i() > 0 }}
              >
                <div class="font-mono text-sm text-mute tabular-nums">
                  {String(i() + 1).padStart(2, "0")}
                </div>
                <div>
                  <h3 class="text-[1.15rem] mb-2">{title}</h3>
                  <p class="text-dim leading-relaxed max-w-[62ch]">{body}</p>
                </div>
              </div>
            )}
          </For>
        </div>
      </section>

      {/* ── differentiators ──────────────────────────── */}
      <section use:reveal class="container-page py-20">
        <div class="eyebrow mb-3">Design</div>
        <h2 class="text-[clamp(1.6rem,3.6vw,2.3rem)] tracking-[-0.025em] mb-10">
          Four deliberate constraints.
        </h2>
        <div class="grid sm:grid-cols-2 gap-4">
          <For each={differentiators}>
            {([title, body], i) => (
              <div class="card flex gap-4">
                <span class="font-mono text-sm text-mute mt-1 tabular-nums">
                  {String(i() + 1).padStart(2, "0")}
                </span>
                <div>
                  <h3 class="text-[1.05rem] mb-1.5">{title}</h3>
                  <p class="text-sm text-dim leading-relaxed">{body}</p>
                </div>
              </div>
            )}
          </For>
        </div>
      </section>

      {/* ── what you can build ───────────────────────── */}
      <section use:reveal class="container-page py-20">
        <div class="eyebrow mb-3">What you can build</div>
        <h2 class="text-[clamp(1.6rem,3.6vw,2.3rem)] tracking-[-0.025em] mb-10">
          A toolkit, not a model.
        </h2>
        <dl class="border-t border-line">
          <For each={useCases}>
            {([term, desc]) => (
              <div class="py-4 border-b border-line grid sm:grid-cols-[12rem_1fr] gap-x-6 gap-y-1 items-baseline">
                <dt class="text-[1.02rem]" style={{ "font-weight": 600 }}>{term}</dt>
                <dd class="text-dim leading-relaxed max-w-[60ch]">{desc}</dd>
              </div>
            )}
          </For>
        </dl>
      </section>

      {/* ── crates ───────────────────────────────────── */}
      <section use:reveal class="container-page py-12">
        <div class="flex items-end justify-between mb-8 flex-wrap gap-3">
          <div>
            <div class="eyebrow mb-3">The SDK</div>
            <h2 class="text-[clamp(1.6rem,3.6vw,2.3rem)] tracking-[-0.025em]">
              Fourteen crates. Use any of them.
            </h2>
          </div>
          <A href="/docs/crates" class="btn btn-ghost h-10">Full reference →</A>
        </div>
        <div class="grid sm:grid-cols-2 gap-x-8 gap-y-0">
          <For each={crates}>
            {([name, desc], i) => (
              <div
                class="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4 py-3 border-t border-line"
                classList={{ "sm:border-t-0": i() < 2 }}
              >
                <code class="font-mono text-[.82rem] text-accent w-[180px] shrink-0">{name}</code>
                <span class="text-sm text-dim">{desc}</span>
              </div>
            )}
          </For>
        </div>
      </section>

      {/* ── CTA band ─────────────────────────────────── */}
      <section use:reveal class="container-page py-20">
        <div class="glow-card p-10 sm:p-14 text-center relative overflow-hidden">
          <div class="hero-glow" />
          <div class="relative">
            <h2 class="text-[clamp(1.8rem,4.4vw,2.8rem)] tracking-[-0.03em] leading-[1.1]">
              Start with <span class="grad-text">twenty lines</span> of Rust.
            </h2>
            <p class="mt-5 text-dim max-w-[54ch] mx-auto leading-relaxed">
              modgrad is open source, MIT-licensed, and built in the open by {site.org}.
              If you're building on it, or want to fund where it's headed, get in touch.
            </p>
            <div class="mt-8 flex flex-wrap gap-3 justify-center">
              <A href="/docs" class="btn btn-primary">Read the docs →</A>
              <A href="/contact" class="btn btn-ghost">Talk to us →</A>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
