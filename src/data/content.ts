// Landing-page content. Every number here comes from the modgrad source tree
// (the SDK README's maze benchmark and GPU numbers, the crate manifests) or
// from a measurement noted next to it. Keep it that way.

// From the SDK README: `cargo run -p mazes --release [-- --brain] --size 21
// --steps 5000 --seed {..}`, three seeds, mean ± std.
export const proofStat = {
  headline: "21×21 maze routing: an 8-region brain vs a single CTM",
  detail: "5000 training steps, 3 seeds, mean ± std.",
  rows: [
    { label: "8-region brain", params: "187k", metric: "79.2%", spread: "± 13.4", pos: true },
    { label: "single CTM", params: "450k", metric: "51.3%", spread: "± 2.0", pos: false },
  ],
  foot:
    "First-step accuracy. Per-step accuracy is 38.1% vs 27.4%, and the correct prefix is 2.1 vs 1.2 of 20 steps: the brain is better with 2.4× fewer parameters, but neither model solves these mazes.",
};

export const runningToday: [string, string][] = [
  ["Qwen2.5-0.5B inference", "Loads safetensors and decodes text on the GPU-resident runtime (qwen_chat)."],
  ["Training-loop check", "lm_validate fits a small batch of real text, cross-entropy 5.72 → 0.74 in 10 steps. A correctness check that gradients flow, not a benchmark."],
  ["GPU residency", "Weights stay on the device across steps: 5.6× faster per call than copying through the host, at 1024×512 on an RX 7600M XT."],
  ["3D brain debugger", "Attach over TCP to a running brain and watch its neurons, token streams and synchronization."],
];

// Short, factual research status. Update when it changes.
export const researchStatus: [string, string][] = [
  ["Models", "The SDK has the CTM and the CMM (Continuous Memory Machine, Sakana 2026)."],
  ["Episodic memory for frozen LLMs", "A memory read that starts from the model's own retrieval heads recalls stored facts with their text absent, on the fact types it was trained on. It does not yet transfer to unseen fact types."],
  ["Mazes", "A supervised CTM trained as in the paper solves 45% of 11×11 mazes and fails on larger boards. The in-browser demo's brain gets the first step right and drifts after."],
];

export const pillars = [
  {
    tag: "core",
    title: "Continuous Thought Machine",
    body: "A neuron pool iterated over internal ticks instead of stacked layers: per-neuron memory traces, a U-Net synapse, a synchronization read-out, a learned early-exit gate, and full backpropagation through every tick.",
    href: "/docs/continuous-thought-machine",
  },
  {
    tag: "compose",
    title: "Multi-region brains",
    body: "Several CTMs wired into a directed graph of regions, connected by learned inter-region synapses and routing. The 8-region preset is 187k parameters.",
    href: "/docs/brain-composition",
  },
  {
    tag: "learn",
    title: "Bio-inspired learning",
    body: "Optional modules: pain as relative surprise, neuromodulators, dream replay and sleep consolidation, episodic replay, several personalities sharing one set of weights. Experimental; most have not yet shown a measured benefit.",
    href: "/docs/bio-inspired",
  },
  {
    tag: "run",
    title: "Compute backends",
    body: "CPU (AVX-512), CUDA, AMD ROCm, KFD and Vulkan behind one backend trait. Resident kernels keep weights on the GPU. Quantized loaders for 7B-class models in 8 GB are in progress.",
    href: "/docs/compute-gpu",
  },
  {
    tag: "sense",
    title: "Unified token space",
    body: "Text bytes, image VQ codes, audio frames, timestamps and actions share one embedding table and one next-token objective. A trainable retina → V1 → V4 cascade turns images into tokens.",
    href: "/docs/multimodal",
  },
  {
    tag: "transformers",
    title: "Transformers & foundation models",
    body: "A transformer stack (MHA, RoPE, KV cache, resident GPT) that runs Qwen2.5 inference, a byte-latent transformer (forward path), and a frozen LLM mounted as a brain's cerebellum.",
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
  ["modgrad-ctm", "single CTM, CMM, graph composition, NeuralComputer, bio modules"],
  ["modgrad-compute", "Linear, tensor ops, batched + resident GPU dispatch"],
  ["modgrad-device", "CPU / CUDA / ROCm / KFD / Vulkan backends, resident kernels"],
  ["modgrad-transformer", "blocks, MHA, RoPE, KV cache, resident GPT, Qwen loader"],
  ["modgrad-blt", "byte-latent transformer: entropy patcher, byteify recipe"],
  ["modgrad-substrate", "Q4_K residency and streaming loaders (7B in 8 GB is the goal)"],
  ["modgrad-codec", "visual retina V1→V4, VQ-VAE, audio codec, FSQ"],
  ["modgrad-ffn", "SwiGLU language prior + FrozenCerebellum trait"],
  ["modgrad-data", "typed multimodal tokenization + streaming"],
  ["modgrad-training", "AdamW / Adam / SGD, schedulers, dream replay"],
  ["modgrad-memory", "episodic memory with valence, content-addressable recall"],
  ["modgrad-lm-memory", "episodic memory for a frozen LLM: store + read paths"],
  ["modgrad-io", "telemetry, safetensors + ONNX + GGUF backends"],
  ["modgrad-persist", "save / load, f32 / f16 / i8 quantization"],
  ["modgrad-traits", "Brain, Encoder, LossFn: the shared interfaces"],
  ["modgrad-wasm", "WebAssembly bindings (the engine behind /play)"],
];

export const differentiators = [
  ["Crates, not a framework", "modgrad-compute doesn't know about modgrad-ctm. Import one crate or several. You write the training loop; nothing runs behind your back."],
  ["A brain/host boundary in the types", "The Brain trait is pure: (Weights, State, Input) → (Output, State). A brain can't do I/O, and its state is explicit and checkpointable rather than hidden behind &mut self."],
  ["Depth from ticks, not layers", "A CTM's depth is the number of internal ticks it runs. A learned exit gate lets easy inputs stop early."],
  ["Follows the paper, checked by tests", "The CTM implements Sakana AI's design (arXiv 2505.05522) with full backpropagation through time; finite-difference tests check the gradients."],
];

export const howItWorks: [string, string][] = [
  [
    "It thinks",
    "A Continuous Thought Machine iterates one neuron pool over internal ticks. Each tick it reads its neurons' recent history, attends to the input and updates a synchronization read-out; a learned gate can stop it early.",
  ],
  [
    "It composes",
    "Several CTMs are wired into a graph of regions that pass activations to each other. On a 21×21 maze benchmark, eight regions at 187k parameters beat a single 450k CTM on first-step accuracy; neither solves the mazes.",
  ],
  [
    "It learns",
    "Backpropagation through every tick. Optionally, experimental bio-inspired signals as well: pain relative to expectation, neuromodulators, dream replay.",
  ],
];

export const useCases: [string, string][] = [
  ["Language models", "Byte-level CTMs and byte-latent transformers: raw bytes in, no tokenizer."],
  ["Agents with state", "The NeuralComputer keeps state across turns: generate, observe, act, repeat."],
  ["Multimodal models", "One embedding table for text, images, audio, timestamps and actions."],
  ["Memory research", "Episodic memory for CTM brains and for frozen LLMs."],
  ["LLM composition", "Mount a frozen Qwen2.5 (or a byte-latent transformer) as a brain's cerebellum."],
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
    opt.step(&mut w, &mut g);
    g.zero();
}

// run it as a neural computer
let mut nc = NeuralComputer::new(w);
let reply  = nc.chat("hello", 100, 0.8);`;
