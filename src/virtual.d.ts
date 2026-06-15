declare module "virtual:uno.css";

declare module "virtual:docs" {
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
  export const pages: DocPage[];
  export const sections: DocSection[];
}
