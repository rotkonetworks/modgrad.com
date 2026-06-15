import { A, useLocation } from "@solidjs/router";
import { createSignal, For, Show } from "solid-js";
import Logo from "@/components/Logo";
import { nav, site } from "@/data/site";
import { theme, toggleTheme } from "@/lib/theme";

export default function Header() {
  const loc = useLocation();
  const [open, setOpen] = createSignal(false);
  const isActive = (href: string) =>
    !href.startsWith("http") && href !== "/" && loc.pathname.startsWith(href);

  return (
    <header
      class="sticky top-0 z-50 border-b border-line"
      style={{
        background: "color-mix(in srgb, var(--bg) 78%, transparent)",
        "backdrop-filter": "blur(14px)",
        "-webkit-backdrop-filter": "blur(14px)",
      }}
    >
      <div class="container-page flex items-center justify-between h-[58px]">
        <A href="/" class="flex items-center gap-2.5" onClick={() => setOpen(false)}>
          <Logo size={26} id="mg-head" />
          <span class="text-[1.06rem] font-600 tracking-tight">modgrad</span>
        </A>

        <nav class="hidden md:flex items-center gap-1">
          <For each={nav}>
            {(item) =>
              item.ext ? (
                <a
                  href={item.href}
                  target="_blank"
                  rel="noopener"
                  class="px-3 py-1.5 text-[.92rem] text-dim hover:text-base transition-colors"
                >
                  {item.label} ↗
                </a>
              ) : (
                <A
                  href={item.href}
                  class="px-3 py-1.5 text-[.92rem] transition-colors"
                  classList={{ "text-base": isActive(item.href), "text-dim hover:text-base": !isActive(item.href) }}
                >
                  {item.label}
                </A>
              )
            }
          </For>
          <button
            onClick={toggleTheme}
            class="ml-2 w-9 h-9 grid place-items-center rounded-lg border border-line text-dim hover:text-base hover:border-line-2 transition"
            aria-label="toggle theme"
            title="toggle theme"
          >
            {theme() === "dark" ? "☾" : "☀"}
          </button>
          <a href="/docs" class="hidden lg:inline-flex btn btn-primary h-9 ml-2 px-4 text-[.88rem]">
            Get started
          </a>
        </nav>

        <button
          class="md:hidden w-9 h-9 grid place-items-center rounded-lg border border-line text-dim"
          aria-label="menu"
          onClick={() => setOpen(!open())}
        >
          {open() ? "✕" : "☰"}
        </button>
      </div>

      <Show when={open()}>
        <div class="md:hidden border-t border-line bg-panel">
          <div class="container-page py-3 flex flex-col gap-1">
            <For each={nav}>
              {(item) =>
                item.ext ? (
                  <a href={item.href} target="_blank" rel="noopener" class="py-2 text-dim">
                    {item.label} ↗
                  </a>
                ) : (
                  <A href={item.href} class="py-2 text-dim" onClick={() => setOpen(false)}>
                    {item.label}
                  </A>
                )
              }
            </For>
            <div class="flex items-center gap-3 pt-2">
              <a href="/docs" class="btn btn-primary h-10 flex-1 justify-center" onClick={() => setOpen(false)}>
                Get started
              </a>
              <button onClick={toggleTheme} class="btn h-10 px-4" aria-label="toggle theme">
                {theme() === "dark" ? "☾ dark" : "☀ light"}
              </button>
            </div>
          </div>
        </div>
      </Show>
    </header>
  );
}
