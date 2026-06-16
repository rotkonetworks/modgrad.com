// Landing-page content. Every number here is grounded in the modgrad
// source tree (maze benchmark, GPU benches, crate manifest). Keep it honest.

export const proofStat = {
  headline: "187k-param brain beats a 450k single model",
  detail: "21×21 maze routing, 5000 steps, 3 seeds, held-out eval.",
  rows: [
    { label: "8-region brain", params: "187k", metric: "79.2%", pos: true },
    { label: "single CTM", params: "450k", metric: "51.3%", pos: false },
  ],
  foot: "first-step accuracy · the brain wins every metric with 2.4× fewer parameters",
};

export const runningToday = [
  ["Qwen2.5-0.5B inference", "Loads safetensors, decodes coherent text on the resident GPU runtime."],
  ["End-to-end LM training", "lm_validate drives cross-entropy 5.72 → 0.74 in 10 steps on real text."],
  ["Full-residency GPU path", "Weights stay on-device across steps, up to 5.6× faster per call (1024×512, RX 7600M XT)."],
  ["Live 3D brain debugger", "Attach over TCP and watch neurons, token streams, and sync in real time."],
];

export const pillars = [
  {
    tag: "core",
    title: "Continuous Thought Machine",
    body: "A neuron pool that thinks over internal ticks instead of stacking layers. NLM memory traces, U-Net synapse, sync readout, learned early-exit, full BPTT through every tick.",
    href: "/docs/continuous-thought-machine",
  },
  {
    tag: "compose",
    title: "Multi-region brains",
    body: "Wire many CTMs into a directed graph of cortical and subcortical regions with learned routing and episodic memory. From 187k params to a billion.",
    href: "/docs/brain-composition",
  },
  {
    tag: "learn",
    title: "Bio-inspired learning",
    body: "Pain as relative surprise, neuromodulators, dream consolidation during sleep, episodic replay, and multiple personalities sharing one set of weights. Opt in per module.",
    href: "/docs/bio-inspired",
  },
  {
    tag: "run",
    title: "Full-residency compute",
    body: "CPU (AVX-512), CUDA, AMD ROCm, KFD, and Vulkan behind one backend trait. Resident kernels keep weights on the GPU; quantized loaders target 7B-class models on 8 GB.",
    href: "/docs/compute-gpu",
  },
  {
    tag: "sense",
    title: "Unified token space",
    body: "Text bytes, image VQ codes, audio frames, timestamps, and actions share one embedding table and one next-token objective. A learnable V1→V4 visual cortex feeds the brain.",
    href: "/docs/multimodal",
  },
  {
    tag: "transformers",
    title: "Transformers & foundation models",
    body: "A full transformer stack (MHA, RoPE, KV cache, resident GPT) runs native Qwen2.5 inference today. Plus a byte-latent transformer (BLT) and mounting a frozen LLM as the brain's cerebellum.",
    href: "/docs/foundation-models",
  },
];

export const regions = [
  ["input", "perception + motor feedback"],
  ["attention", "gating, routing"],
  ["output", "evidence accumulation"],
  ["motor", "action selection"],
  ["cerebellum", "forward model, timing"],
  ["basal ganglia", "value estimation"],
  ["insula", "interoception, salience"],
  ["hippocampus", "episodic binding"],
];

export const crates: [string, string][] = [
  ["modgrad-ctm", "single CTM + graph composition, NeuralComputer, bio modules"],
  ["modgrad-compute", "Linear, tensor ops, batched + resident GPU dispatch"],
  ["modgrad-device", "CPU / CUDA / ROCm / KFD / Vulkan backends, resident kernels"],
  ["modgrad-transformer", "blocks, MHA, RoPE, KV cache, resident GPT, Qwen loader"],
  ["modgrad-blt", "byte-latent transformer: entropy patcher, byteify recipe"],
  ["modgrad-substrate", "Q4_K residency, streaming loaders, 7B on 8 GB target"],
  ["modgrad-codec", "VisualRetina V1→V4, VQ-VAE, audio codec, FSQ"],
  ["modgrad-ffn", "SwiGLU language prior + FrozenCerebellum trait"],
  ["modgrad-data", "type-safe multimodal tokenization + streaming"],
  ["modgrad-training", "AdamW / Adam / SGD, schedulers, dream replay"],
  ["modgrad-memory", "episodic memory with valence, content-addressable recall"],
  ["modgrad-io", "telemetry, safetensors + ONNX + GGUF backends"],
  ["modgrad-persist", "wincode / JSON save-load, f32/f16/i8 quantization"],
  ["modgrad-traits", "Brain, Encoder, LossFn: the compile-time foundation"],
];

export const differentiators = [
  ["Crates, not a framework", "modgrad-compute doesn't know about modgrad-ctm. Import one crate or fourteen. You own the training loop; nothing runs behind your back."],
  ["A compile-time brain/host boundary", "The Brain trait is pure: (Weights, State, Input) → (Output, State). The type system stops a brain from doing I/O. State is explicit and checkpointable, never hidden behind &mut self."],
  ["Depth from thinking, not size", "A CTM gets its depth from deliberation time, not parameter count. Harder inputs get more compute through learned exit gates."],
  ["A faithful port, with tests", "A line-for-line Rust port of the Sakana AI CTM (arXiv 2505.05522), with full BPTT. No architectural shortcuts, and the tests to back it."],
];

export const howItWorks: [string, string][] = [
  [
    "It thinks",
    "A Continuous Thought Machine iterates a single neuron pool over internal ticks, reading its memory, attending to the input, and accumulating evidence, then exits when it's confident. Depth comes from deliberation time, not stacked layers.",
  ],
  [
    "It composes",
    "Wire many CTMs into a directed graph of specialized regions that route information, remember, and feed action back into perception. Eight regions at 187k parameters beat a single 450k model.",
  ],
  [
    "It learns",
    "Full backpropagation through every tick. And, if you want, from its own surprise: pain relative to expectation, neuromodulators, dream replay, and sleep consolidation.",
  ],
];

export const useCases: [string, string][] = [
  ["Language models", "Byte-level CTMs and byte-latent transformers. No tokenizer, raw bytes in."],
  ["Reasoning agents", "The NeuralComputer holds state across turns: generate, observe, act, repeat."],
  ["Multimodal systems", "One embedding table for text, images, audio, timestamps, and actions."],
  ["Bio-inspired research", "Episodic memory, neuromodulated learning, even multiple selves on one brain."],
  ["LLM composition", "Mount a frozen Qwen2.5 (or a byte-latent transformer) as the cerebellum."],
];

export const heroCode = `use modgrad_ctm::graph::{RegionalConfig, RegionalWeights,
    RegionalAdamW, RegionalGradients, regional_train_token,
    NeuralComputer};

// an 8-region brain (187k params)
let cfg = RegionalConfig::eight_region_small(128, 256, 16);
let mut w   = RegionalWeights::new(cfg);
let mut opt = RegionalAdamW::new(&w).with_lr(3e-4);

// train on your own data
let mut g = RegionalGradients::zeros(&w);
for (token, target) in your_data {
    let (loss, _) = regional_train_token(&w, &mut g, token, target);
    opt.step(&mut w, &g);
    g.zero();
}

// run it as a neural computer
let mut nc = NeuralComputer::new(w);
let reply  = nc.chat("hello", 100, 0.8);`;
