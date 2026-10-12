# modgrad-obs: LLM views (per-layer heatmaps, samples, evals, anomalies)

Status: implemented on branch `obs-llmviews` (verified in Chrome against `llm_synthetic` runs, desktop and 390 px). This note fixes the **naming conventions and
record shapes** that clients (the Python client, `obs run`, agents) write, and describes the
four views the dashboard derives from them. The server only stores and serves; every view is
computed in the browser (Rust → wasm in `crates/modgrad-obs-analysis`, plus TS in `obs/web`).

## 1. Per-layer metrics

### Convention

```
layers/<i>/<stat>          e.g. layers/0/grad_norm, layers/17/act_rms, layers/3/attn/entropy
```

- `<i>` is a non-negative decimal integer with no padding (`layers/7/…`, not `layers/07/…`;
  padded indices still parse, they are just not canonical).
- `<stat>` is any metric path (may contain `/`): `grad_norm`, `act_rms`, `weight_norm`,
  `update_ratio`, `attn/entropy`, `moe/load`, ….
- A prefix is allowed: `model/layers/<i>/<stat>`, `actor/layers/<i>/<stat>` form their own families.
- One history row may carry every layer at once: `{"step": 100, "layers": {"0": {"grad_norm": …}, "1": {…}}}`
  is flattened by the server into `layers/0/grad_norm`, `layers/1/grad_norm`, ….

Why this form and not `<stat>/layer.<i>`: nested JSON flattens into it naturally (the server joins
nested keys with `/`), a glob over the index (`layers/*/grad_norm`) selects one family, and
`layers/<i>/` groups every statistic of one layer together in a sorted key list.

### Also recognised (read side only, so existing logs work)

| pattern | example | family |
|---|---|---|
| `…/layers/<i>/…` | `layers/3/grad_norm` | `layers/*/grad_norm` |
| `…/layer.<i>` or `…/layer_<i>` or `…/layer<i>` at the end | `grad_norm/layer.3` | `grad_norm/layer.*` |
| `….layers.<i>.…` (PyTorch module names) | `model.layers.3.mlp.grad_norm` | `model.layers.*.mlp.grad_norm` |
| `…/blocks/<i>/…`, `….blocks.<i>.…`, `….h.<i>.…` (GPT-2) | `transformer.h.3.attn` | `transformer.h.*.attn` |

A **family** is the set of metric names that are equal after replacing the layer index with `*`;
it needs at least 2 distinct indices. Detection is `layer_index()` in
`crates/modgrad-obs-analysis/src/query.rs` (tested).

### Heatmap view

Query stage: `heatmap [over layer] [rownorm]`, e.g.

```
layers/*/grad_norm{run="pt/gpt-24l"} | heatmap over layer rownorm | log
```

- One heatmap per (run, family): rows = layers in numeric order, columns = steps (one shared
  grid for every row, block-averaged to at most 400 columns).
- `rownorm` divides each row by its median, so a value of 1 is "typical for this layer" and a
  spike in one layer stands out even when layers differ by orders of magnitude.
- `log` (the existing stage) switches to a log colour scale.
- Clicking a cell draws that layer's line chart under the heatmap with a marker at the step;
  "open as panel" adds it to the board.
- The series browser shows families as one row ("layers/*/grad_norm, 24 layers"); dragging or `+`
  writes the heatmap query. The auto board adds heatmaps for grad-norm and activation-RMS
  families when present.

## 2. Text samples (generations)

### Record

One line of `samples.jsonl` per sample:

```json
{"step": 1200, "t": 812.4, "key": "train", "prompt": "…", "completion": "…",
 "reward": 0.73, "score": 1.0, "meta": {"temperature": 0.7, "id": "gsm8k/1534"}}
```

| field | type | meaning |
|---|---|---|
| `step` | int ≥ 0, optional | the training step; missing = the run's last history step |
| `t` | number, optional | seconds since run start; default = arrival time |
| `key` | string, optional | which stream: `train`, `eval/gsm8k`, `rollout`, …; default `"default"` |
| `prompt` | string, optional | default `""` |
| `completion` | string, **required** | the model output |
| `reward` | number, optional | RL reward |
| `score` | number, optional | any other scalar (judge score, pass@1, exact match) |
| `meta` | object, optional | anything else, shown verbatim |

Compare-across-steps matches samples by (`key`, `prompt`): log the same prompts at every
evaluation step to get diffs.

### Writing

- Rust: `run.log_sample(step, &Sample { key, prompt, completion, reward, score, meta })`.
- HTTP: `POST /api/v1/runs/<id>/samples` with one sample object, an array, NDJSON, or
  `{"samples": [...]}`. Same auth and run auto-create as `…/log`.
- Read: `GET /api/v1/runs/<id>/samples[?from=&to=&key=&limit=]` →
  `{"samples": [...], "steps": [...], "keys": [...], "n": N}` (newest `limit`, default 5000).
- Static export: `samples.json` next to `events.json`; the index entry carries `n_samples`.

Limits: a run's `n_samples` comes with the index, so a run that starts logging samples after the
page loaded shows them after a reload (runs that already had samples refresh live).

Consistency: `samples.jsonl` sits outside the committed-length snapshot the other three files
use, so a read may see samples a little newer than the metrics snapshot it was served with. A torn
last line is never returned. Samples are not pushed over SSE: the Samples view refetches when the
run's rows change (live runs log rows every step) and on reload.

### Samples view (tab "Samples", and a board panel)

Run and key pickers, a step slider, search (prompt and completion), filter by reward range,
sort by reward / score / length, a completion-length histogram for the current step, "copy" per
sample, and **compare**: the same prompt across steps with a word-level diff of the completions
(`diff_words` in wasm, LCS over whitespace tokens). On the board, the query `samples{run="…"}`
is a panel that shows the latest samples of the matched runs.

## 3. Evals

### Convention

```
eval/<benchmark>/<metric>          e.g. eval/gsm8k/acc, eval/mmlu/acc, eval/humaneval/pass@1, eval/wikitext/ppl
```

logged at the checkpoint's step. Lower is better when the metric name contains `loss`, `nll`,
`ppl`, `perplexity`, `error`, `err`, `wer`, `cer` or `bpb`; otherwise higher is better.

### Eval table view (tab "Evals")

Rows = checkpoints (run × step), columns = `<benchmark>/<metric>`, the best value per column
marked with the accent, a sparkline per column header, sortable by any column, and "add to board"
per column (`eval/<b>/<m>`) and per cell (that run's curve).

## 4. Spikes and anomalies

A compact list over the selected runs: loss and grad-norm spikes (the wasm `spikes` kernel:
> 6 robust sd above the trailing 10-point median) and non-finite values. The writer stores
NaN and inf as `null`, so the browser cannot tell NaN from inf from a missing value; only
gaps inside an otherwise dense series are reported, as "non-finite or missing". Per-layer grad
norms are scanned too and summarised per step ("6 of 24 layers (18-23)"). Spikes smaller than a
chosen multiple of the trailing median (default x1.5) are hidden: on smooth curves the 6-sd test
alone flags few-percent bumps. Each entry has "jump": every step-axis panel zooms to a window
around that step, and panels on a metric x axis (`x tokens_seen`) zoom to the same window mapped
through their own step -> x. Following panels pause. Series are scanned as loaded (server mode
strides runs longer than 4000 points).
