import { createEffect, onCleanup } from "solid-js";

const SITE = "https://modgrad.com";
const DEFAULT_TITLE = "modgrad — a Rust SDK for composable brains";
const DEFAULT_DESC =
  "A Rust SDK for Continuous Thought Machines and multi-region brains: graph composition, bio-inspired learning, and GPU-resident training. No framework, no YAML.";

function setMeta(selector: string, attr: string, value: string) {
  let el = document.head.querySelector<HTMLElement>(selector);
  if (!el) {
    el = document.createElement("meta");
    const [, key, val] = selector.match(/\[(.+?)="(.+?)"\]/) ?? [];
    if (key && val) el.setAttribute(key, val);
    document.head.appendChild(el);
  }
  el.setAttribute(attr, value);
}

function setCanonical(href: string) {
  let el = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!el) {
    el = document.createElement("link");
    el.rel = "canonical";
    document.head.appendChild(el);
  }
  el.href = href;
}

export type DocMeta = { title?: string; description?: string; path?: string };

/** Update document title + description + canonical + OG/Twitter tags for a route. */
export function useDocMeta(meta: () => DocMeta) {
  createEffect(() => {
    const m = meta();
    const title = m.title ? `${m.title} — modgrad` : DEFAULT_TITLE;
    const desc = m.description ?? DEFAULT_DESC;
    const url = SITE + (m.path ?? (typeof location !== "undefined" ? location.pathname : "/"));

    document.title = title;
    setMeta('meta[name="description"]', "content", desc);
    setMeta('meta[property="og:title"]', "content", title);
    setMeta('meta[property="og:description"]', "content", desc);
    setMeta('meta[property="og:url"]', "content", url);
    setMeta('meta[name="twitter:title"]', "content", title);
    setMeta('meta[name="twitter:description"]', "content", desc);
    setCanonical(url);
  });

  onCleanup(() => {
    document.title = DEFAULT_TITLE;
  });
}
