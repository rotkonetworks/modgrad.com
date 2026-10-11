import { defineConfig, presetMini } from 'unocss';

// Utilities map onto the theme tokens in src/app.css (same pattern as modgrad.com).
export default defineConfig({
  presets: [presetMini()],
  rules: [
    ['text-ink', { color: 'var(--ink)' }],
    ['text-mute', { color: 'var(--muted)' }],
    ['text-accent', { color: 'var(--accent)' }],
    ['bg-base', { background: 'var(--bg)' }],
    ['bg-hl', { background: 'var(--hl)' }],
    ['border-rule', { 'border-color': 'var(--rule)' }],
    ['border-frame', { 'border-color': 'var(--frame)' }],
    ['font-mono', { 'font-family': 'var(--mono)' }],
    ['font-sans', { 'font-family': 'var(--sans)' }],
    ['tabular-nums', { 'font-variant-numeric': 'tabular-nums' }],
  ],
  shortcuts: {
    note: 'text-[11px] text-mute',
  },
});
