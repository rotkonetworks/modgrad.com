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

/** Primary header navigation. External links carry `ext: true`. */
export const nav = [
  { label: "architecture", href: "/docs/continuous-thought-machine" },
  { label: "sdk", href: "/docs/crates" },
  { label: "docs", href: "/docs" },
  { label: "play", href: "/play" },
  { label: "community", href: "/contact" },
  { label: "github", href: site.repo, ext: true },
] as const;
