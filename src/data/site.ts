export const site = {
  name: "modgrad",
  tagline: "a Rust SDK for composable brains",
  description:
    "A Rust SDK for Continuous Thought Machines and multi-region brains: graph composition, bio-inspired learning, and GPU-resident training. No framework, no YAML.",
  url: "https://modgrad.com",
  repo: "https://github.com/rotkonetworks/modgrad",
  discussions: "https://github.com/rotkonetworks/modgrad/discussions",
  issues: "https://github.com/rotkonetworks/modgrad/issues",
  org: "Rotko Networks",
} as const;

/** Primary header navigation. External links carry `ext: true`; `app: true`
 *  links go to modgrad-server under /app (a full page load, not a route here). */
export const nav: readonly { label: string; href: string; ext?: boolean; app?: boolean }[] = [
  { label: "architecture", href: "/docs/continuous-thought-machine" },
  { label: "sdk", href: "/docs/crates" },
  { label: "docs", href: "/docs" },
  { label: "play", href: "/play" },
  { label: "community", href: "/contact" },
  // the observability app, served by modgrad-server under /app (not part of this SPA)
  { label: "app", href: "/app/", app: true },
  { label: "sign in", href: "/app/login", app: true },
  { label: "github", href: site.repo, ext: true },
];
