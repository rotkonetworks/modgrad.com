import { createSignal } from "solid-js";

type Theme = "dark" | "light";

const initial = (): Theme =>
  (document.documentElement.dataset.theme as Theme) || "light";

export const [theme, setTheme] = createSignal<Theme>(initial());

export function toggleTheme() {
  const next: Theme = theme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch (e) {}
  setTheme(next);
}
