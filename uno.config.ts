import { defineConfig, presetMini } from "unocss";

export default defineConfig({
  presets: [presetMini()],
  rules: [
    // text colors → theme vars
    ["text-base", { color: "var(--text)" }],
    ["text-dim", { color: "var(--text-dim)" }],
    ["text-mute", { color: "var(--text-mute)" }],
    ["text-accent", { color: "var(--accent-2)" }],
    ["text-pos", { color: "var(--pos)" }],
    ["text-warn", { color: "var(--warn)" }],
    // backgrounds → theme vars
    ["bg-base", { background: "var(--bg)" }],
    ["bg-panel", { background: "var(--bg-2)" }],
    ["bg-raised", { background: "var(--bg-3)" }],
    // borders → theme vars
    ["border-line", { "border-color": "var(--line)" }],
    ["border-line-2", { "border-color": "var(--line-2)" }],
    // fonts
    ["font-mono", { "font-family": "var(--mono)" }],
    // text-transform (not in presetMini)
    ["tabular-nums", { "font-variant-numeric": "tabular-nums" }],
    ["uppercase", { "text-transform": "uppercase" }],
    ["lowercase", { "text-transform": "lowercase" }],
    ["capitalize", { "text-transform": "capitalize" }],
  ],
  shortcuts: {
    "container-page": "w-full max-w-[1120px] mx-auto px-5 sm:px-7",
    "eyebrow": "font-mono text-xs tracking-widest uppercase text-mute",
  },
});
