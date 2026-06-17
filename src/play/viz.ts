// Small, dependency-free drawing helpers for the /play visualisation.
// Colours are read from CSS variables where possible so the panels track the
// paper / dark theme, with hand-tuned diverging + heat ramps for the data.

export const MOVES = ["Up", "Down", "Left", "Right"] as const;

export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((v) => v / s);
}

/** Read a CSS custom property off :root, with a fallback. */
export function cssVar(name: string, fallback: string): string {
  if (typeof getComputedStyle === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Diverging neuron colour around 0: cool violet (neg) → paper (0) → warm (pos).
 *  `v` is the raw activation, `scale` the value mapped to full saturation. */
export function neuronColor(v: number, scale: number): string {
  const t = Math.max(-1, Math.min(1, v / scale)); // -1..1
  const a = Math.abs(t);
  if (t >= 0) {
    // 0 → faint paper, +1 → accent violet
    return `rgba(98, 67, 217, ${0.06 + a * 0.9})`;
  }
  // negative → teal/blue, reads as "inhibited"
  return `rgba(41, 199, 214, ${0.06 + a * 0.85})`;
}

/** Attention heat: transparent → warm accent overlay. */
export function heatAlpha(t: number): number {
  // gentle gamma so mid values stay legible over the maze
  return Math.pow(Math.max(0, Math.min(1, t)), 1.15) * 0.78;
}
