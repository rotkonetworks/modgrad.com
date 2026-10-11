# obs-agent-bench

Can a local model answer questions about training runs through `obs mcp`?

The bench has two suites. `base` has 14 one-step tasks; `hard`
(`hard_tasks.py`) has 13 multi-step tasks that separate tool designs. Each
task gets a fresh data directory, filled through
the MCP tools (`ingest_log`, `log`) from tables in `bench.py`, so the expected
answers are computed in the bench and do not depend on the server's own
analysis. A task is a user prompt plus a checker on the final answer. For
write tasks, the checker also reads the stored `metrics.jsonl` or board file.
The tasks cover every tool, a misspelled run name (`exp/meduim`), a metric
alias (`lr`), an honest "no clear trend" on noise, an ETA, and mean ± sd
across seeds.

```
cargo build -p modgrad-obs                          # builds target/debug/obs

# 1. validate the bench: run each task's canonical tool calls, no model
python3 bench/obs_agent_bench/bench.py --scripted --data /some/scratch

# 2. a model behind an OpenAI-compatible endpoint (Ollama shown; the
#    results below used this separate server, models stored on /steam)
OLLAMA_MODELS=/steam/rotko/ollama-models OLLAMA_HOST=127.0.0.1:11435 \
  HSA_OVERRIDE_GFX_VERSION=11.0.2 OLLAMA_KEEP_ALIVE=30m OLLAMA_CONTEXT_LENGTH=16384 \
  ollama serve &
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3:30b-a3b
python3 bench/obs_agent_bench/bench.py --data /some/scratch \
    --endpoint http://127.0.0.1:11435/v1 --model qwen3:30b-a3b
```

Requires `python3` with `requests`.

Run `--scripted --suite all` first. It must pass 27/27; otherwise a model score says more
about the bench than about the model. qwen3 runs with `/no_think` unless you
pass `--think`. Other flags: `--only a,b`, `--max-turns`, `--temperature`,
`--seed`, `--timeout`, `--max-tokens` (default 3000; a reply that hits it
without a tool call is recorded as "no answer"), `--suite base|hard|all`,
`--trial N` (offsets the sampling seed so trials differ), `--remote` (each
task against its own `obs-server --dev` through `obs mcp --url`).

### The hard suite

All 13 tasks share one fixture project of 25 runs (sweeps, seed ablations,
RL runs, a fine-tune with an lr change), generated deterministically in
`hard_tasks.py` and loaded through `ingest_log`. Ground truth comes from the
generators, not from the tools.

| task | what the agent must do |
|---|---|
| diverged | find which of 4 sweep runs diverged and roughly when (blow-up from step 750, NaN from 900) |
| continue_yes / continue_no | "is it worth continuing?" for a run still improving vs one on a noisy plateau |
| eta_unreachable | "when will it reach 2.0?" for a run plateaued at 2.5: no confident step |
| seeds_noise / seeds_real | mean ± sd over 3 seeds per config, and whether the difference is within or beyond seed noise |
| spike_lr | find a loss spike and the lr change (1e-4 to 1e-3) just before it |
| wrong_metric | "validation loss" when the run logs `eval/nll` and `train/loss` |
| rl_board | a board of reward, KL and entropy for the two PPO runs (metric names `rollout/ep_reward_mean`, `train/approx_kl`, `train/entropy_loss`) |
| overfit | which of 4 runs has val loss rising while train loss falls |
| paste_compare | store two pasted logs in different formats, compare them at one step |
| instrument_wrap | run `train.py` (prints metric lines) so it is recorded as `toy/wrap` |
| instrument_edit | edit `train_silent.py` (prints nothing) to print the loss every 10 iterations, run it as `toy/edit` |

The two instrument tasks also expose `read_file`, `write_file` and
`run_command` (sandboxed to the task's project directory, `obs` first on
PATH), as a coding agent would have.

`tabulate.py label=a.jsonl,b.jsonl ...` prints per-task pass counts over
trials. It re-checks hard-suite answers with the current checkers against
each task's stored data, so a checker fix rescores old transcripts.

Each run writes `<data>/<label>.jsonl`. It holds one record per task: pass,
the reason for the result, the final answer, turns, wall time, and every tool
call as the model typed it (name, raw arguments, `isError`, output). The
summary counts tool errors and non-canonical tool names. Both show where
argument tolerance helped or was not enough.

`mcp_client.py` is a minimal stdio client and also works as a CLI for reading
tool output:

```
python3 bench/obs_agent_bench/mcp_client.py --obs target/debug/obs --dir obs summarize '{"run": "bkk07/small"}'
```

## Results

### Base suite (one-step tasks)

| date       | model / mode                      | pass  | tool calls | tool errors | wall  |
|------------|-----------------------------------|-------|------------|-------------|-------|
| 2026-10-11 | `--scripted`                      | 14/14 | 16         | 1 (by design: typo) | <1 s |
| 2026-10-11 | qwen3:30b-a3b before (the obs-mcp merge; same tool output as 0a52e7f) | 14/14 | 19 | 1 (typo, recovered next turn) | 883 s |
| 2026-10-11 | qwen3:30b-a3b at 4c1552f (after)  | 14/14 | 18 | 1 (typo, recovered) | 1549 s |

qwen3:8b, one trial each, before (0a52e7f) and after (4c1552f):

| task | baseline | iter3 |
|---|---|---|
| count_runs | 1/1 | 1/1 |
| last_value | 1/1 | 1/1 |
| min_value | 1/1 | 1/1 |
| best_size | 1/1 | 1/1 |
| best_at_shared | 1/1 | 1/1 |
| trend_improving | 1/1 | 1/1 |
| trend_noise | 1/1 | 1/1 |
| eta | 1/1 | 1/1 |
| seed_mean | 0/1 | 1/1 |
| typo_run | 0/1 | 0/1 |
| lr_alias | 1/1 | 1/1 |
| log_write | 1/1 | 1/1 |
| ingest_paste | 1/1 | 1/1 |
| board | 1/1 | 1/1 |
| total | **12/14** (86%) | **13/14** (93%) |

The 8B "before" run used the harness of that time (no `--max-tokens`, data
directly in the task dir); the base-suite checkers are unchanged, so the
numbers compare. The 8B misses `typo_run` both times: after the "No run named exp/meduim.
Closest: exp/mid, ..." error it asks the user which run was meant instead
of retrying. The base suite cannot separate tool designs at 30B scale.

### Hard suite: tool-output iterations

Setup: Ollama 0.35.1 on :11435 (RX 7600M XT 8 GB; qwen3:8b runs 93% on the
GPU, qwen3:30b-a3b is partly offloaded), `/no_think`, T=0.2, seed 0 + trial,
`--max-tokens 3000`. qwen3:8b: 3 trials per task. qwen3:30b-a3b: 1 trial (the
regression check). Each column is the `obs` binary built at that commit.
Scores are rescored with the final checkers (`tabulate.py`). `--scripted
--suite all` passes 27/27 at every commit listed. With `--remote` it also
passes 27/27, but the two instrument tasks are not meaningful there: their
`obs run --offline` writes files the server does not watch, so the
follow-up read_metric fails and only the file-level checks pass. Run the
hard suite locally.

| iteration | commit | change to tool output |
|---|---|---|
| baseline | 0a52e7f | tools as merged from obs-mcp, plus RemoteStore |
| iter1 | 0e3970c* | summarize Verdict line (DIVERGED / OVERFITTING / improving / no clear progress); NaN/inf warnings everywhere; spike report names the lr-like metric that changed just before; compare tests seed groups (Welch t, WITHIN/BEYOND seed noise); metric names by meaning ("validation loss" -> eval/nll); spike detector no longer flags the opening drop or float noise on clean curves |
| iter2 | b61ecd8 | the 0e3970c tool descriptions (see the note below); make_board splits panels at top-level commas and merges same-metric panels; glob misses list similar runs; summarize accepts a glob; read_metric also reports the biggest spike and its likely cause, OVERFITTING, and the run's other metrics |
| iter3 | 4c1552f | a run miss inside a board query lists the runs that log the panel's metric; run suggestions rank names sharing a word first |

\* Build slip: the iter1 binary was built from 0e3970c's tool code but
before its `mod.rs` edit, so it has iter1's new tool *output* but the old
tool *descriptions* and server instructions (the summarize description
naming the verdicts and "worth continuing / what went wrong" questions, the
compare description with the seed-group example). Those descriptions first
reach a model in iter2. iter1 therefore measures output changes alone, and
iter2 measures descriptions plus the iter2 output changes together.

qwen3:8b (passes / 3 trials):

| task | baseline 0a52e7f | iter1 0e3970c | iter2 b61ecd8 | iter3 4c1552f |
|---|---|---|---|---|
| diverged | 3/3 | 3/3 | 3/3 | 2/3 |
| continue_yes | 3/3 | 3/3 | 3/3 | 3/3 |
| continue_no | 3/3 | 3/3 | 3/3 | 3/3 |
| eta_unreachable | 3/3 | 3/3 | 3/3 | 3/3 |
| seeds_noise | 3/3 | 3/3 | 3/3 | 3/3 |
| seeds_real | 2/3 | 2/3 | 3/3 | 3/3 |
| spike_lr | 0/3 | 0/3 | 3/3 | 3/3 |
| wrong_metric | 3/3 | 3/3 | 3/3 | 3/3 |
| rl_board | 0/3 | 0/3 | 1/3 | 3/3 |
| overfit | 3/3 | 2/3 | 2/3 | 3/3 |
| paste_compare | 3/3 | 3/3 | 3/3 | 3/3 |
| instrument_wrap | 3/3 | 2/3 | 3/3 | 3/3 |
| instrument_edit | 0/3 | 0/3 | 0/3 | 0/3 |
| total | **29/39** (74%) | **27/39** (69%) | **33/39** (85%) | **35/39** (90%) |

qwen3:30b-a3b (passes / 1 trial):

| task | baseline 0a52e7f | iter1 0e3970c | iter2 b61ecd8 | iter3 4c1552f |
|---|---|---|---|---|
| diverged | 1/1 | 1/1 | 1/1 | 1/1 |
| continue_yes | 1/1 | 1/1 | 1/1 | 1/1 |
| continue_no | 1/1 | 1/1 | 1/1 | 1/1 |
| eta_unreachable | 1/1 | 1/1 | 1/1 | 1/1 |
| seeds_noise | 0/1 | 0/1 | 1/1 | 1/1 |
| seeds_real | 0/1 | 0/1 | 0/1 | 0/1 |
| spike_lr | 0/1 | 0/1 | 1/1 | 1/1 |
| wrong_metric | 1/1 | 1/1 | 1/1 | 1/1 |
| rl_board | 1/1 | 1/1 | 1/1 | 1/1 |
| overfit | 1/1 | 1/1 | 1/1 | 1/1 |
| paste_compare | 1/1 | 1/1 | 1/1 | 1/1 |
| instrument_wrap | 1/1 | 1/1 | 1/1 | 1/1 |
| instrument_edit | 0/1 | 0/1 | 0/1 | 0/1 |
| total | **9/13** (69%) | **9/13** (69%) | **11/13** (85%) | **11/13** (85%) |

What moved the 8B score, from the transcripts:

- **iter1 did not help (29 -> 27/39)**: with the old descriptions the 8B
  answered spike, overfit and seed questions with `read_metric` and
  per-run `summarize` calls (6-9 calls for a seed question), so the new
  verdict, spike-cause and seed-noise lines were mostly never seen. The
  losses themselves were single degenerate generations (`overfit`,
  `instrument_wrap` hitting the token cap), not worse tool use.
- **spike_lr 0/3 -> 3/3** (iter2). Two trials now call `summarize` (whose
  description says to use it for "what went wrong") and read the "Just
  before the spike at step 1250: lr changed 1e-4 -> 1e-3" line. The third
  still calls only `read_metric`, which now carries the same line. Before,
  it answered "the tools do not track changes". Lesson: put the conclusion
  in every tool a weak model might call, not only in the "right" one.
- **seeds_real 2/3 -> 3/3, seeds_noise 3/3 with 1 call instead of 6-9**
  (iter2). With the compare example `compare(runs="abl/base_s*,abl/wd_s*",
  ...)` in the description, the 8B makes one compare call and quotes the t
  and the WITHIN/BEYOND verdict.
- **rl_board 0/3 -> 1/3 -> 3/3** (iter2, iter3). Failures were query syntax
  (`a{...}, b{...}` with commas), a glob that matched nothing (`ppo/*`), and
  invented run names (`ppo_run1`), after which the model gave up on
  edit-distance suggestions led by `cls/run3`. Listing the runs that
  actually log "reward" made it retry correctly.
- qwen3:30b-a3b went 9 -> 11/13 with no task lost (spike_lr and seeds_noise
  gained), and stays 14/14 on the base suite; 8B base 12 -> 13/14.

Still failing: **instrument_edit** (0/3 and 0/1 in every column). Both
models overwrite `train_silent.py` without reading it (writing a stub loop
with `loss = 0.5` or `...`), or loop until the token cap. This is a
coding-agent skill the obs tools cannot supply. **seeds_real** at 30B: it
calls compare, gets "BEYOND seed noise", and answers with the two means
only, never saying whether the difference is real.

Compute: about 5.5 hours of local inference in total (8B 9-15 min per
13-task trial, 30B 17-25 min per hard trial, 26 min for the 30B base suite).
