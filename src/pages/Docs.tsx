import { A, useParams } from "@solidjs/router";
import { For, Show, createMemo } from "solid-js";
import { useDocMeta } from "@/lib/meta";
import { adjacent, getPage, sections } from "@/lib/docs";

export default function Docs() {
  const params = useParams();
  const page = createMemo(() => getPage(params.slug));

  useDocMeta(() => {
    const p = page();
    return p
      ? { title: p.title, description: p.description, path: `/docs/${p.slug}` }
      : { title: "Docs", path: "/docs" };
  });

  return (
    <div class="container-page grid lg:grid-cols-[210px_minmax(0,1fr)] xl:grid-cols-[210px_minmax(0,1fr)_180px] gap-10 py-10">
      {/* sidebar */}
      <aside class="hidden lg:block">
        <nav class="sticky top-[78px] flex flex-col gap-6 max-h-[calc(100vh-100px)] overflow-y-auto pr-2">
          <For each={sections}>
            {(sec) => (
              <div>
                <div class="eyebrow mb-2.5">{sec.title}</div>
                <ul class="flex flex-col gap-0.5 border-l border-line">
                  <For each={sec.pages}>
                    {(p) => {
                      const active = () => params.slug === p.slug;
                      return (
                        <li>
                          <A
                            href={`/docs/${p.slug}`}
                            class="block pl-3.5 py-1 text-[.875rem] -ml-px border-l transition-colors"
                            classList={{
                              "border-transparent text-dim hover:text-base": !active(),
                              "text-base font-550": active(),
                            }}
                            style={active() ? { "border-image": "var(--grad) 1", "border-left-width": "2px" } : {}}
                          >
                            {p.title}
                          </A>
                        </li>
                      );
                    }}
                  </For>
                </ul>
              </div>
            )}
          </For>
        </nav>
      </aside>

      {/* content */}
      <main class="min-w-0">
        <Show
          when={page()}
          fallback={
            <div class="prose">
              <h1>Page not found</h1>
              <p>
                That doc doesn’t exist. <A href="/docs">Back to the docs index →</A>
              </p>
            </div>
          }
        >
          {(p) => (
            <>
              <div class="eyebrow mb-3">{p().section}</div>
              <article class="prose" innerHTML={p().html} />

              <nav class="grad-rule mt-14 mb-6" />
              <PrevNext slug={p().slug} />
            </>
          )}
        </Show>
      </main>

      {/* on-page toc */}
      <aside class="hidden xl:block">
        <Show when={page()?.toc.length}>
          <div class="sticky top-[78px]">
            <div class="eyebrow mb-3">On this page</div>
            <ul class="flex flex-col gap-1.5 text-[.82rem]">
              <For each={page()!.toc}>
                {(t) => (
                  <li classList={{ "pl-3": t.level === 3 }}>
                    <a href={`#${t.id}`} class="text-mute hover:text-base transition-colors">
                      {t.text}
                    </a>
                  </li>
                )}
              </For>
            </ul>
          </div>
        </Show>
      </aside>
    </div>
  );
}

function PrevNext(props: { slug: string }) {
  const nav = createMemo(() => adjacent(props.slug));
  return (
    <div class="grid sm:grid-cols-2 gap-3">
      <Show when={nav().prev} fallback={<span />}>
        {(prev) => (
          <A href={`/docs/${prev().slug}`} class="card card-hover">
            <div class="text-xs text-mute mb-1">← Previous</div>
            <div class="font-550">{prev().title}</div>
          </A>
        )}
      </Show>
      <Show when={nav().next}>
        {(next) => (
          <A href={`/docs/${next().slug}`} class="card card-hover text-right">
            <div class="text-xs text-mute mb-1">Next →</div>
            <div class="font-550">{next().title}</div>
          </A>
        )}
      </Show>
    </div>
  );
}
