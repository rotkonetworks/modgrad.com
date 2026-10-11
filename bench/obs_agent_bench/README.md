# obs-agent-bench

Can a local model answer questions about training runs through `obs mcp`?

The bench has 14 tasks. Each task gets a fresh data directory, filled through
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

# 2. a model behind an OpenAI-compatible endpoint (Ollama shown)
python3 bench/obs_agent_bench/bench.py --data /some/scratch \
    --endpoint http://127.0.0.1:11434/v1 --model qwen3:30b-a3b
```

Run `--scripted` first. It must pass 14/14; otherwise a model score says more
about the bench than about the model. qwen3 runs with `/no_think` unless you
pass `--think`. Other flags: `--only a,b`, `--max-turns`, `--temperature`,
`--seed`, `--timeout`.

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

| date       | model / mode                      | pass  | tool calls | tool errors | wall  |
|------------|-----------------------------------|-------|------------|-------------|-------|
| 2026-10-11 | `--scripted`                      | 14/14 | 16         | 1 (by design: typo) | <1 s |
| 2026-10-11 | qwen3:30b-a3b, `/no_think`, T=0.2, seed 0, Ollama 0.35.1, RX 7600M XT 8 GB | 14/14 | 19 | 1 (typo, recovered next turn) | 883 s |

These results are from one seed and one run each. qwen3-30b-a3b scores at
the ceiling, so smaller models or harder multi-step tasks are needed before
this bench can tell tool-design changes apart.
