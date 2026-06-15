export const site = {
  name: "modgrad",
  tagline: "build your own brain",
  description:
    "Composable Rust crates for building brains: Continuous Thought Machines, multi-region composition, bio-inspired learning, and full-residency GPU training.",
  url: "https://modgrad.com",
  repo: "https://github.com/rotkonetworks/modgrad",
  discussions: "https://github.com/rotkonetworks/modgrad/discussions",
  issues: "https://github.com/rotkonetworks/modgrad/issues",
  org: "Rotko Networks",
} as const;

/** Primary header navigation. External links carry `ext: true`. */
export const nav = [
  { label: "architecture", href: "/docs/continuous-thought-machine" },
  { label: "sdk", href: "/docs/crates" },
  { label: "docs", href: "/docs" },
  { label: "community", href: "/contact" },
  { label: "github", href: site.repo, ext: true },
] as const;
