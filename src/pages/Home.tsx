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
  researchStatus,
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
                A Rust SDK for <span class="grad-text">composable brains.</span>
              </h1>
              <p class="mt-6 text-[1.12rem] text-dim max-w-[46ch] leading-relaxed">
                Rust crates for Continuous Thought Machines and networks of
                them: training with backpropagation through every tick,
                optional bio-inspired learning rules, CPU and GPU backends. You
                write the training loop.
              </p>
              <div class="mt-8 flex flex-wrap gap-3">
                <A href="/docs" class="btn btn-primary">Get started →</A>
                <A href="/play" class="btn btn-ghost">
                  <span class="inline-block w-1.5 h-1.5 rounded-full bg-accent mr-2 align-middle" />
                  Watch it think →
                </A>
                <a href={site.repo} target="_blank" rel="noopener" class="btn btn-ghost">
                  View on GitHub ↗
                </a>
              </div>
              <div class="mt-7 font-mono text-[.82rem] text-mute break-all">
                <span class="text-dim">$</span> cargo add --git {site.repo} modgrad-ctm
              </div>
              <div class="mt-1.5 text-[.75rem] text-mute">
                Not on crates.io yet; depend on the git repository.
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
          <div class="eyebrow mb-4">Approach</div>
          <h2 class="text-[clamp(1.7rem,4vw,2.5rem)] tracking-[-0.025em] leading-[1.12]">
            Specialization, not scale.
          </h2>
          <div class="mt-7 text-[1.1rem] text-dim leading-relaxed flex flex-col gap-4">
            <p class="dropcap">
              The hypothesis we are testing: useful capability can come from
              specialized regions that <strong>think recurrently</strong>, pass
              information between each other and remember, rather than only from
              scaling one transformer. Each region is a part you can build and
              test on its own.
            </p>
            <p>
              The parts are Rust crates: a Continuous Thought Machine, graph
              composition, optional neuromodulated learning, multimodal codecs, a
              transformer stack, GPU residency. The same SDK trains a
              187k-parameter brain and mounts a frozen LLM as a cerebellum, on
              one workstation.
            </p>
          </div>
        </div>
      </section>

      {/* ── proof ────────────────────────────────────── */}
      <section use:reveal class="container-page pb-8">
        <div class="grid lg:grid-cols-[0.9fr_1.1fr] gap-6">
          <div class="card glow-card">
            <div class="eyebrow mb-4">Measured</div>
            <div class="text-[1.15rem] font-600 mb-1.5 leading-snug">{proofStat.headline}</div>
            <div class="text-sm text-mute mb-5">{proofStat.detail}</div>
            <div class="flex flex-col gap-3">
              <For each={proofStat.rows}>
                {(r) => (
                  <div class="flex items-center gap-4">
                    <div class="w-[140px] shrink-0 text-sm text-dim">{r.label}</div>
                    <div class="flex-1 h-2 rounded-full bg-raised overflow-hidden">
                      <div
                        class="h-full rounded-full"
                        style={{ width: r.width, background: "var(--grad)" }}
                      />
                    </div>
                    <div class="font-mono text-sm tabular-nums w-[84px] text-right">{r.value}</div>
                  </div>
                )}
              </For>
            </div>
            <p class="mt-5 text-sm text-mute leading-relaxed">{proofStat.foot}</p>
            <A href="/play" class="mt-3 inline-block text-sm text-accent">See it on a maze →</A>
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
            <div class="eyebrow mt-6 mb-2">Research status</div>
            <ul class="flex flex-col">
              <For each={researchStatus}>
                {([title, body], i) => (
                  <li class="py-3" classList={{ "border-t border-line": i() > 0 }}>
                    <div class="font-550">{title}</div>
                    <div class="text-sm text-dim leading-relaxed mt-0.5">{body}</div>
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
          Six parts. Use the ones you need.
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
                Eight regions, one graph.
              </h2>
              <p class="text-dim leading-relaxed text-[.98rem]">
                Each region is its own CTM with its own neuron count, memory depth, and tick
                budget. They are connected by learned inter-region synapses: a cortical
                loop plus subcortical value, salience and memory pathways. The names are
                anatomical analogies, not claims of equivalence.
              </p>
              <div class="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-sm">
                <A href="/play" class="text-accent">See it predict a maze route →</A>
                <A href="/docs/brain-composition" class="text-accent">Read the architecture →</A>
              </div>
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
          Four design choices.
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
          A toolkit, not a trained model.
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
              The main crates.
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
              Read the code.
            </h2>
            <p class="mt-5 text-dim max-w-[54ch] mx-auto leading-relaxed">
              Open source (MIT), developed in the open by {site.org}. If you use
              it, or want to fund the work, get in touch.
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
