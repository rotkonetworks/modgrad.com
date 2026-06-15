// Post-build: write a per-route index.html with route-specific title,
// description, canonical, and OG/Twitter tags. The SPA still hydrates the same
// bundle — this is purely so crawlers and link unfurlers see the right metadata.
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { buildDocs } from "./docs-build";

const dist = join(dirname(fileURLToPath(import.meta.url)), "dist");
const template = readFileSync(join(dist, "index.html"), "utf-8");
const SITE = "https://modgrad.com";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;");

function variant(title: string, desc: string, path: string): string {
  const url = SITE + path;
  const t = esc(title);
  const d = esc(desc);
  return template
    .replace(/<title>[\s\S]*?<\/title>/, `<title>${t}</title>`)
    .replace(/(<meta name="description" content=")[^"]*(")/, `$1${d}$2`)
    .replace(/(<link rel="canonical" href=")[^"]*(")/, `$1${url}$2`)
    .replace(/(<meta property="og:title" content=")[^"]*(")/, `$1${t}$2`)
    .replace(/(<meta property="og:description" content=")[^"]*(")/, `$1${d}$2`)
    .replace(/(<meta property="og:url" content=")[^"]*(")/, `$1${url}$2`)
    .replace(/(<meta name="twitter:title" content=")[^"]*(")/, `$1${t}$2`)
    .replace(/(<meta name="twitter:description" content=")[^"]*(")/, `$1${d}$2`);
}

function emit(path: string, html: string) {
  const dir = path === "/" ? dist : join(dist, path);
  if (path !== "/") mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "index.html"), html);
  console.log(`  ${path}`);
}

const DEFAULT_DESC =
  "Composable Rust crates for building brains: Continuous Thought Machines, multi-region composition, bio-inspired learning, and full-residency GPU training. No framework. No YAML. Just functions.";

// home
emit("/", variant("modgrad — build your own brain", DEFAULT_DESC, "/"));

// docs index
emit(
  "/docs",
  variant(
    "Documentation — modgrad",
    "modgrad documentation — architecture, the SDK crates, compute, and the runtime.",
    "/docs",
  ),
);

// community / contact
emit(
  "/contact",
  variant(
    "Community & contact — modgrad",
    "Ask questions in the open on GitHub Discussions, file issues, or get in touch about building on modgrad or funding where it's headed.",
    "/contact",
  ),
);

// one page per doc
const { pages } = buildDocs();
for (const p of pages) {
  emit(`/docs/${p.slug}`, variant(`${p.title} — modgrad`, p.description || DEFAULT_DESC, `/docs/${p.slug}`));
}

console.log(`prerendered ${pages.length + 3} routes`);
