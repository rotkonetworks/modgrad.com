export const site = {
  name: "modgrad",
  tagline: "a Rust SDK for composable brains",
  description:
    "Rust crates for Continuous Thought Machines and multi-region networks of them: backpropagation through every tick, optional bio-inspired learning rules, CPU and GPU backends. Open source, early research.",
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
