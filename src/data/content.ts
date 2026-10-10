// Landing-page content. Every number here comes from the modgrad source tree
// (the SDK README's maze benchmark and GPU numbers, the crate manifests) or
// from a measurement noted next to it. Keep it that way.

// Measured headless with the shipped wasm engine and weights (the /play brain,
// cortex 32): 200 random 9×9 mazes, the final-tick route predicted from the
// start image, argmax per step, compared with the unique shortest route.
export const proofStat = {
  headline: "The /play brain on 200 random 9×9 mazes",
  detail: "Its predicted route from the start image, compared with the shortest route.",
  rows: [
    { label: "first step right", value: "199 / 200", width: "99.5%" },
    { label: "steps matching", value: "4.3 / 17.4", width: "24.7%" },
    { label: "route correct to goal", value: "0 / 200", width: "0%" },
  ],
  foot:
    "It gets the first steps right and drifts after; most routes end in a predicted step into a wall. This is where the in-browser model stands today.",
};

export const runningToday: [string, string][] = [
  ["Qwen2.5-0.5B inference", "Loads safetensors and decodes text on the GPU-resident runtime (qwen_chat)."],
  ["End-to-end LM training", "lm_validate trains the transformer stack on real text and reports held-out loss."],
  ["GPU residency", "Weights stay on the device across steps: 5.6× faster per call than copying through the host, at 1024×512 on an RX 7600M XT."],
  ["3D brain debugger", "Attach over TCP to a running brain and watch its neurons, token streams and synchronization."],
];

// Short, factual research status. Update when it changes.
export const researchStatus: [string, string][] = [
  ["Models", "The SDK has the CTM and the CMM (Continuous Memory Machine, Sakana 2026)."],
  ["Episodic memory for frozen LLMs", "A memory read that starts from the model's own retrieval heads recalls stored facts with their text absent, on the fact types it was trained on. It does not yet transfer to unseen fact types."],
  ["Mazes", "A supervised CTM trained as in the paper solves 45% of 11×11 mazes and fails on larger boards."],
  ["Regions vs one block", "A pre-registered, parameter-matched test found a graph of regions no better than a single CTM block on two tasks, so new work uses a single block. An earlier 21×21 result that favoured the 8-region brain predates a synchronization bug fix and has not been re-run."],
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
    body: "CPU (AVX-512) and AMD ROCm are tested; CUDA, KFD and Vulkan sit behind the same backend trait, with CUDA not yet validated on NVIDIA hardware. Resident kernels keep weights on the GPU between steps.",
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
    body: "A transformer stack (MHA, RoPE, KV cache, resident GPT) that runs Qwen2.5 inference, a byte-latent transformer, and a frozen LLM mounted as a brain's cerebellum.",
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
  ["modgrad-ctm", "the CTM and CMM, graph composition, NeuralComputer, bio modules"],
  ["modgrad-compute", "Linear, tensor ops, batched + resident GPU dispatch"],
  ["modgrad-device", "GPU / hardware backends, resident kernels"],
  ["modgrad-transformer", "transformer blocks, MHA, RoPE, KV cache, Qwen loader"],
  ["modgrad-blt", "byte-latent transformer with entropy-based patching"],
  ["modgrad-codec", "modality codecs: visual cortex V1→V4, audio, text"],
  ["modgrad-ffn", "SwiGLU MLP language model"],
  ["modgrad-data", "tokenization and streaming data pipeline"],
  ["modgrad-training", "optimizers, schedulers, checkpointing"],
  ["modgrad-memory", "episodic memory and gated-delta fast weights"],
  ["modgrad-lm-memory", "episodic memory for a frozen LLM: store + read paths"],
  ["modgrad-io", "IO, serialization and inference backends"],
  ["modgrad-persist", "serialization and persistence"],
  ["modgrad-traits", "core traits: Brain, LossFn, Optimizer"],
  ["modgrad-substrate", "host observability: CPU frequency, throttling, thermals"],
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
    "Several CTMs can be wired into a graph of regions that pass activations to each other. Whether that beats one larger block is an open question; our latest parameter-matched test found no difference.",
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
