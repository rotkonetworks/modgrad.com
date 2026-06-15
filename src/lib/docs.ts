import { pages, sections, type DocPage, type DocSection } from "virtual:docs";

export type { DocPage, DocSection };
export { pages, sections };

export const firstSlug = pages[0]?.slug ?? "introduction";

export function getPage(slug: string | undefined): DocPage | undefined {
  if (!slug) return pages[0];
  return pages.find((p) => p.slug === slug);
}

export function adjacent(slug: string): { prev?: DocPage; next?: DocPage } {
  const i = pages.findIndex((p) => p.slug === slug);
  if (i === -1) return {};
  return { prev: pages[i - 1], next: pages[i + 1] };
}
