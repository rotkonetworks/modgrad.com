import { onCleanup } from "solid-js";

/** `use:reveal` — fade-and-rise an element in once it scrolls into view.
 *  Adds the `reveal` class at create time (before first paint, so no flash),
 *  then `in` when it intersects. Falls back to visible if motion is reduced or
 *  IntersectionObserver is unavailable. Crawlers / no-JS never get the hidden
 *  state, since the class is only applied by this directive. */
export function reveal(el: HTMLElement) {
  const reduced =
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduced || typeof IntersectionObserver === "undefined") {
    el.classList.add("in");
    return;
  }

  el.classList.add("reveal");
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          el.classList.add("in");
          io.unobserve(el);
        }
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.06 },
  );
  io.observe(el);
  onCleanup(() => io.disconnect());
}

declare module "solid-js" {
  namespace JSX {
    interface Directives {
      reveal: boolean;
    }
  }
}
