import { A } from "@solidjs/router";
import Logo from "@/components/Logo";
import { site } from "@/data/site";

const cols = [
  {
    title: "Architecture",
    links: [
      ["Continuous Thought Machine", "/docs/continuous-thought-machine"],
      ["Brain composition", "/docs/brain-composition"],
      ["Bio-inspired learning", "/docs/bio-inspired"],
      ["Memory & multiplicity", "/docs/memory"],
    ],
  },
  {
    title: "Systems",
    links: [
      ["Compute & GPU", "/docs/compute-gpu"],
      ["Multimodal token space", "/docs/multimodal"],
      ["Foundation models", "/docs/foundation-models"],
    ],
  },
  {
    title: "Reference",
    links: [
      ["SDK crates", "/docs/crates"],
      ["CLI & runtime", "/docs/runtime-cli"],
      ["FAQ", "/docs/faq"],
    ],
  },
];

export default function Footer() {
  return (
    <footer class="border-t border-line mt-24">
      <div class="container-page py-14">
        <div class="grid gap-10 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div>
            <A href="/" class="flex items-center gap-2.5 mb-3">
              <Logo size={24} id="mg-foot" />
              <span class="text-[1.02rem] font-600">modgrad</span>
            </A>
            <p class="text-dim text-sm max-w-[30ch] leading-relaxed">
              Composable Rust crates for building brains. No framework. No YAML. Just functions.
            </p>
            <div class="flex flex-wrap gap-2 mt-4">
              <a href={site.repo} target="_blank" rel="noopener" class="chip">GitHub ↗</a>
              <a href={site.discussions} target="_blank" rel="noopener" class="chip">Discussions ↗</a>
              <A href="/contact" class="chip">Contact</A>
            </div>
          </div>

          {cols.map((col) => (
            <div>
              <div class="eyebrow mb-3">{col.title}</div>
              <ul class="flex flex-col gap-2">
                {col.links.map(([label, href]) => (
                  <li>
                    <A href={href} class="text-sm text-dim hover:text-base transition-colors">
                      {label}
                    </A>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div class="grad-rule my-9" />
        <div class="flex flex-col sm:flex-row justify-between gap-3 text-xs text-mute font-mono">
          <span>© {site.org} · MIT licensed</span>
          <a
            href="https://arxiv.org/abs/2505.05522"
            target="_blank"
            rel="noopener"
            class="hover:text-base transition-colors"
          >
            Built on the Sakana AI CTM ↗
          </a>
        </div>
      </div>
    </footer>
  );
}
