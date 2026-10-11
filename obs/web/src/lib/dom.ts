// Tiny DOM builders for the imperative chart code (kept from the vanilla dashboard so the
// drawing stays pixel-identical), plus the shared hover tooltip.
const NS = 'http://www.w3.org/2000/svg';
type Attrs = Record<string, string | number | boolean>;

export function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, parent?: Element | null): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  if (parent) parent.appendChild(e);
  return e;
}
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, parent?: Element | null, text?: string | number): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  if (text !== undefined) e.textContent = String(text);
  if (parent) parent.appendChild(e);
  return e;
}
export function btn(parent: Element | null, label: string, title: string, on?: ((ev: MouseEvent) => void) | null, cls = ''): HTMLButtonElement {
  const b = h('button', { type: 'button', title, 'aria-label': title, class: cls }, parent, label);
  if (on) b.addEventListener('click', on);
  return b;
}

let tipEl: HTMLDivElement | null = null;
function tip(): HTMLDivElement {
  if (!tipEl) { tipEl = h('div', { class: 'tip', role: 'tooltip' }, document.body); }
  return tipEl;
}
export function showTip(ev: { clientX: number; clientY: number }, text: string): void {
  const t = tip();
  t.textContent = text; t.style.display = 'block';
  const x = Math.min(ev.clientX + 12, innerWidth - t.offsetWidth - 4);
  t.style.left = x + 'px'; t.style.top = (ev.clientY + 12) + 'px';
}
export function hideTip(): void { if (tipEl) tipEl.style.display = 'none'; }
