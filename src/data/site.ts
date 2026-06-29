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

/** Primary header navigation. External links carry `ext: true`. */
export const nav = [
  { label: "architecture", href: "/docs/continuous-thought-machine" },
  { label: "sdk", href: "/docs/crates" },
  { label: "docs", href: "/docs" },
  { label: "play", href: "/play" },
  { label: "community", href: "/contact" },
  { label: "github", href: site.repo, ext: true },
] as const;
