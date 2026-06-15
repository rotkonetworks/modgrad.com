// Build-time docs pipeline. Reads src/docs/*.md, parses frontmatter, renders
// HTML with markdown-it, injects heading ids + a table of contents.
// Shared by the Vite virtual-module plugin (dev/build) and prerender.ts (SEO).
import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import MarkdownIt from "markdown-it";

const DOCS_DIR = join(dirname(fileURLToPath(import.meta.url)), "src/docs");

const md = new MarkdownIt({ html: false, linkify: true, typographer: true });

export type TocItem = { level: number; text: string; id: string };
export type DocPage = {
  slug: string;
  title: string;
  section: string;
  order: number;
  description: string;
  html: string;
  toc: TocItem[];
};
export type DocSection = { title: string; pages: { slug: string; title: string }[] };

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function parseFrontmatter(raw: string): { meta: Record<string, string>; body: string } {
  if (!raw.startsWith("---")) return { meta: {}, body: raw };
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return { meta: {}, body: raw };
  const block = raw.slice(3, end).trim();
  const body = raw.slice(end + 4).replace(/^\s*\n/, "");
  const meta: Record<string, string> = {};
  for (const line of block.split("\n")) {
    const i = line.indexOf(":");
    if (i === -1) continue;
    meta[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, "");
  }
  return { meta, body };
}

function render(body: string): { html: string; toc: TocItem[] } {
  const toc: TocItem[] = [];
  let html = md.render(body);
  html = html.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (_m, lvl, inner) => {
    const text = inner.replace(/<[^>]+>/g, "").trim();
    const id = slugify(text);
    toc.push({ level: Number(lvl), text, id });
    return `<h${lvl} id="${id}">${inner}</h${lvl}>`;
  });
  return { html, toc };
}

export function buildDocs(): { pages: DocPage[]; sections: DocSection[] } {
  const files = readdirSync(DOCS_DIR).filter((f) => f.endsWith(".md"));
  const pages: DocPage[] = files.map((file) => {
    const raw = readFileSync(join(DOCS_DIR, file), "utf-8");
    const { meta, body } = parseFrontmatter(raw);
    const { html, toc } = render(body);
    return {
      slug: file.replace(/\.md$/, ""),
      title: meta.title ?? file,
      section: meta.section ?? "Docs",
      order: Number(meta.order ?? 999),
      description: meta.description ?? "",
      html,
      toc,
    };
  });

  pages.sort((a, b) => a.order - b.order);

  const sections: DocSection[] = [];
  for (const p of pages) {
    let sec = sections.find((s) => s.title === p.section);
    if (!sec) {
      sec = { title: p.section, pages: [] };
      sections.push(sec);
    }
    sec.pages.push({ slug: p.slug, title: p.title });
  }

  return { pages, sections };
}
