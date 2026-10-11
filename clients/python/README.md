# modgrad (Python client)

The Python client for modgrad-obs. It implements the parts of the `wandb` API that
training scripts use, so for typical logging code you only change the import:

```python
import modgrad as wandb

wandb.init(project="my-project", config={"lr": 3e-4})
wandb.log({"loss": 0.52})
wandb.finish()
```

The package is pure Python with no required dependencies (it uses stdlib `urllib`). Its
syntax targets Python 3.8 and later; it has been tested only on Python 3.14.

## Install (from the repo)

```sh
pip install ./clients/python            # or: pip install -e ./clients/python
pip install "git+<repo-url>#subdirectory=clients/python"
```

The package is not on PyPI yet. The name `modgrad` may already be taken there, so check
before publishing.

## Local mode: no server

After `modgrad init` without `--server`, `.modgrad.json` is `{"data": "obs"}` and the client
writes each run straight to disk; nothing else needs to run:

```console
$ python train.py
modgrad: run exp1 -> obs/runs/exp1 (view: modgrad-server --root obs)
$ modgrad tail exp1
```

- **Where.** A run goes to `<data>/runs/<project>/<name>/`, or `<data>/runs/<name>/` without
  a project. `<data>` is `init(data=...)`, else `MODGRAD_DATA`, else the project file's
  `data` (relative to the file's directory), else `./obs`. A `/` in a name nests, as in
  `name="sweep/lr3e-4_s1"`. A name that is already taken gets a `-<4 hex>` suffix, so two
  scripts never write into one run. Without a name the run is `run-<date>-<time>-<4 hex>`.
  `id=` names the run exactly, and appends to it if it exists.
- **Format.** The directory is exactly what the Rust writer (`modgrad_obs::Run`, also used
  by `modgrad run --offline`) produces: `config.json`, `metrics.jsonl` (one line per `log`
  call with scalars), `arrays.jsonl` (histograms and tensors, f32 little-endian base64),
  `events.jsonl` (text, tables, alerts) and `summary.json`. `modgrad tail`, `modgrad
  export`, `modgrad mcp --dir` and `modgrad-server --root` read it unchanged.
- **Crash safety.** Each `log()` is appended with one `write()` to files opened for append,
  before `log()` returns. A killed process keeps everything it logged, except possibly a
  torn final line, which readers skip and the next open cuts. There is no fsync, so a power
  loss can still lose what the OS had not written yet. `config.json` and `summary.json`
  are replaced atomically. `summary.json` gets `finished: true` and `exit_code` at
  `finish()`, so a run without them was killed or is still running.
- **Resume.** `init(id="exp1", resume=True)` (or `name=` with `resume=`) continues after
  the run's last logged step, counting steps that hold only arrays, as the server does.
  `t` continues from the run's original start. `resume="must"` raises if the run does not
  exist.
- **Offline mode.** `init(mode="offline")`, `MODGRAD_MODE=offline` or `WANDB_MODE=offline`
  write locally the same way, whatever the project file says.
- `run.dir` is the run directory, `run.local` is True, `run.url` is None and `flush()`
  returns at once. There is no spool and no background thread.

## What happens with a server

- **Server or local.** Arguments beat environment variables, which beat `.modgrad.json`
  (written by `modgrad init`, looked up in the current directory and its parents). At each
  level, a server (`init(url=...)`, `MODGRAD_URL`, the file's `server`) means online and a
  data directory (`init(data=...)`, `MODGRAD_DATA`, a file without `server`) means local
  mode. With nothing set the client uses `modgrad.DEFAULT_URL` (`http://127.0.0.1:8790`,
  the same default as the `modgrad` CLI). The token and project in `.modgrad.json` are used
  the same way, and the token only goes to the server the file names. Run a local server
  with `modgrad-server --dev`.
- **No token and no project.** On first use the client creates an anonymous project
  (`POST /api/v1/projects`) and prints its board link. It caches the token in
  `~/.config/modgrad/anon.json`, so later runs on the same machine go to the same project.
  Anonymous projects expire after 30 days unless claimed, and they are not listed in the
  public index: open them through the board link.
- **A project but no token.** The client writes without a token, which `--dev` servers
  accept. On a run's first write the server returns a run token. The client keeps it in
  `~/.config/modgrad/runs.json`, so a resumed run can write again. If the server needs a
  token, the client falls back to an anonymous project.
- **Non-blocking `log()`.** Each `log()` call appends one line to an on-disk spool and
  returns. A background thread fsyncs the spool and posts it in batches to
  `POST /api/v1/runs/<id>/log`. It sends every second, or sooner once 500 items are
  waiting.
- **Crash safety.** A row is written to the spool before it is sent, and the server's
  acknowledgements are recorded as a byte offset. Anything not acknowledged is sent again,
  including after a restart that uses the same run id. A crash loses at most a torn final
  line. The spool format is the one `modgrad run` uses, so `modgrad push <dir>` can send it.
- **Errors never stop training.** If the server is unreachable, the client reports it once
  and keeps spooling, retrying with backoff. If the server rejects one item with a 400, the
  client drops that item with a warning and the rest of the queue keeps moving. If the
  server refuses the token, sending stops and the rows stay in the spool.
- **Exit.** `finish()` runs at exit through `atexit`. An uncaught exception gives
  `exit_code` 1, and Ctrl-C gives 130. `finish()` waits up to `MODGRAD_FINISH_TIMEOUT`
  seconds (default 30) for the queue to drain. If everything was acknowledged it marks the
  run finished on the server and deletes the spool. Otherwise it keeps the spool and prints
  the command that sends it later.
- **Forked processes.** In processes forked after `init`, such as DataLoader workers,
  `log()` does nothing.

### Sending a spool later

When the server stayed unreachable, `finish()` keeps the spool and prints how to send it:

```sh
python -m modgrad sync                    # send every spool under the spool root
python -m modgrad sync <dir>              # send one run (rows, events, config, summary, finish)
modgrad push <dir>                            # the Rust CLI: sends rows and events only
```

The spool root is `$XDG_CACHE_HOME/modgrad-obs/spool` (by default
`~/.cache/modgrad-obs/spool`), which is also where `modgrad run` spools. You can override it
with `MODGRAD_DIR` or `init(dir=...)`. A run with a project name and no token spools to
`<root>/<project>__<id>/`. A run with a token, or in an anonymous project, spools to
`<root>/<id>/`. The directory holds these files:

| file          | content                                                                              |
|---------------|--------------------------------------------------------------------------------------|
| `spool.jsonl` | one item per line: `{"row": …}`, `{"event": …}`, `{"config": …}`, `{"summary": …}`    |
| `acked`       | the byte offset of `spool.jsonl` that the server acknowledged                        |
| `meta.json`   | `{server, token, name, project, meta, run, dashboard, finish, config}` (mode 0600)    |
| `lock`        | an flock held while a process is writing; `sync` skips locked spools                 |

`modgrad push` ignores the `config` and `summary` lines. Use `python -m modgrad sync` when you
want those sent as well.

## Environment variables

Where a row lists several variables, the first one that is set wins.

| variable | meaning |
|---|---|
| `MODGRAD_URL`, `MODGRAD_URL` | server URL |
| `MODGRAD_TOKEN`, `MODGRAD_TOKEN` | bearer token (`obsp_…` for a project, `obsr_…` for a run) |
| `MODGRAD_PROJECT`, `WANDB_PROJECT`, `MODGRAD_PROJECT` | project (used when `init(project=)` is not given) |
| `MODGRAD_MODE`, `WANDB_MODE` | `online` (default), `offline` (write locally) or `disabled`; W&B's `dryrun` means offline and `run` means online |
| `MODGRAD_DATA` | obs directory for local runs; local mode unless `MODGRAD_URL` is set (beats the project file) |
| `WANDB_DISABLED=true` | same as `disabled` |
| `MODGRAD_NAME`, `WANDB_NAME` | run name |
| `MODGRAD_RUN_ID`, `WANDB_RUN_ID` | run id |
| `WANDB_RESUME`, `WANDB_TAGS`, `WANDB_NOTES`, `WANDB_RUN_GROUP`, `WANDB_JOB_TYPE` | as in W&B; tags, notes, group and job type are stored in the run's meta |
| `MODGRAD_DIR` | spool root |
| `MODGRAD_CONFIG_DIR` | where cached tokens live (default `~/.config/modgrad`) |
| `MODGRAD_FLUSH_INTERVAL`, `MODGRAD_BATCH_ROWS` | sender cadence (default 1 s and 500 items) |
| `MODGRAD_FINISH_TIMEOUT` | seconds `finish()` waits for the queue to drain (default 30) |
| `MODGRAD_RETRY_MAX` | maximum retry backoff in seconds (default 30) |
| `MODGRAD_SILENT=1` | no messages on stderr |

## API

### Supported as in W&B

- `init(project, name, config, id, resume, dir, mode, tags, notes, group, job_type, reinit)`
  returns a `Run`. With an existing run, `reinit=None/False` returns that run,
  `reinit=True` or `"finish_previous"` finishes it first, and `"create_new"` starts a run
  alongside it.
  - `init` also accepts `url=`, `token=` and `data=` (an obs directory: write locally),
    which are modgrad additions.
  - Other W&B arguments are accepted and ignored.
  - Run ids use only the characters `[A-Za-z0-9_.-]`. A `name` becomes `<name>-<4 hex>`,
    because W&B names are not unique. Pass `id=` for a stable id, and add `resume=` to
    continue its step count.
- `log(dict, step=None, commit=None)`, `run.log(...)`. This follows W&B's step rules:
  - Without `step`, a call writes at the run's step and then advances it, unless
    `commit=False`.
  - With `step`, the call writes at that step and does not advance unless `commit=True`.
  - Rows at the same step are merged on the server.
- `config` (`wandb.config` and `run.config`) supports attribute and item access,
  `update()` and `setdefaults()`. Every change is sent and merged into the run's config.
- `summary` (`wandb.summary` and `run.summary`) holds the last value of each scalar.
  Values set explicitly are sent to the server.
- `finish(exit_code=)`, `run.id`, `run.name`, `run.url`, `run.path`, `run.dir`,
  `run.step`, the context manager (`with wandb.init() as run:`), and `wandb.run`.
- `login()` does nothing and returns True. `Settings` and `setup` are accepted and do
  nothing.

### Logged values

| logged | stored as |
|---|---|
| int, float, bool, numpy scalar, 1-element tensor | a scalar; NaN, inf and None are stored as gaps |
| nested dict `{"train": {"loss": 1}}` | `train/loss` |
| `Histogram(seq, num_bins=64)`, `Histogram(np_histogram=(counts, edges))` | a histogram array, at most 512 bins |
| numpy array, torch tensor, list of numbers | a 64-bin histogram (as W&B does) |
| `modgrad.Tensor(array)` (a modgrad addition) | a raw f32 tensor, at most 262144 elements |
| `Image(array)` | a tensor; paths and PIL images are skipped for now |
| `Table(columns, data)` | a run event with text `table <key> {"columns": [...], "data": [[...]]}`, capped at 64 KiB |
| str | a run event with text `<key>: <value>` |

### Partial or stubbed

- `alert(title, text, level)` records a run event `alert[LEVEL] title: text`. No
  notification is sent.
- `define_metric(...)` is stored in `run.metric_defs`. The dashboard does not use it yet.
- `watch(model, log="gradients", log_freq=N)` maps to `watch_grads(model, every=N)`.
  It logs per-layer gradient norms, not W&B's gradient histograms. Parameter logging and
  graphs are not implemented.
- `save`, `log_code`, `Artifact`, `log_artifact`, `use_artifact` and `log_model` do
  nothing, because files are not uploaded yet.
- `Audio`, `Video`, `Html` and `Object3D` are skipped with a one-time warning.
- `wandb.Api` (the read API) and `wandb.sweep` are not implemented.
- HF's `report_to="wandb"` is not intercepted: pass `report_to="none"` and add the
  callback (see below).
- `run.path` and `run.url` are only final once the server has created the run, which
  happens in the background. Call `run.flush()` first if you need them right after `init`.
- Additions not in W&B: `event(text)`, `run.flush(timeout)` (blocks until the server has
  acknowledged everything), `Tensor`, and `watch_grads`.

## Per-layer metrics: `layers/<i>/<stat>`

The dashboard's layer×step heatmap reads this naming convention:

```
layers/<i>/grad_norm         L2 norm of layer i's gradients (i is 0-based, in model order)
params/<module>/grad_norm    parameters outside the numbered layers (embeddings, final norm, lm_head)
grad_norm/total              L2 norm over all watched parameters
```

Other per-layer statistics should use the same prefix, for example `layers/12/act_rms`
or `layers/3/update_ratio`.

`modgrad.watch_grads(model, every=100)` produces these keys:

- **What a layer is.** A layer is a numbered block: the first all-digit segment of a
  parameter name. `model.layers.3.mlp.up.weight` belongs to `model.layers.3`, and
  `transformer.h.3.attn.c_attn.weight` belongs to `transformer.h.3`.
- **Numbering.** Blocks are numbered in the order they first appear in
  `named_parameters()`. A single stack therefore keeps its own indices, and an
  encoder-decoder model gets encoder blocks 0..N-1 followed by decoder blocks N..2N-1.
- **Models without numbered blocks.** Each module that owns parameters is one layer.
- **The index mapping.** The list from index to module path is logged once, as the config
  key `modgrad/layers` and as an event `layers [...]`.
- **Sampling.** Gradients are captured on every `every`-th forward pass that runs with
  grad enabled in training mode. The capture uses post-accumulate-grad hooks: one
  reduction per parameter tensor and no host sync. The values are logged with the next
  `log()` call, so `optimizer.zero_grad()` before `log()` is fine. With gradient
  accumulation, the value is the norm of the accumulated gradient after the sampled
  micro-batch.
- **Stopping.** `watch_grads` returns a handle; call `.remove()` on it to stop.

## Integrations

### Hugging Face `transformers`

```python
from modgrad.integrations.huggingface import ModgradCallback

args = TrainingArguments(..., report_to="none")   # keep HF's own wandb callback off
trainer = Trainer(model=model, args=args, ...,
                  callbacks=[ModgradCallback(project="my-project", watch_grads_every=50)])
```

- **Logs.** Every `Trainer.log` call is logged at `state.global_step`. Keys are renamed as
  in W&B's integration: `loss` becomes `train/loss` and `eval_loss` becomes `eval/loss`.
- **Summary.** `train_runtime`, `train_loss` and the other end-of-training values go to
  the summary.
- **Config.** The config is the model config plus `TrainingArguments`.
- **Processes.** Only the world-zero process logs.
- **End of training.** `on_train_end` flushes but does not finish the run, so a
  `trainer.evaluate()` afterwards still logs to it. Pass `finish_on_train_end=True` to
  finish the run there.

### Lightning

```python
from modgrad.integrations.lightning import ModgradLogger

trainer = lightning.Trainer(logger=ModgradLogger(project="my-project"))
```

- **What it logs.** `log_hyperparams` updates the run config, and `log_metrics(metrics,
  step)` calls `run.log(metrics, step=step)`.
- **Packages.** The logger works with `lightning` or `pytorch_lightning`.
- **Finishing.** `finalize` flushes the run. It is finished at exit, or call
  `logger.finish()`.

### PyTorch

```python
modgrad.watch_grads(model, every=100)     # or wandb.watch(model, log_freq=100)
```

Both integration modules can be imported without their framework installed.

## Tests

```sh
cargo build --release -p modgrad-obs-server -p modgrad-obs    # builds target/release/modgrad-server and modgrad
cd clients/python && python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The fixture starts the built `modgrad-server --dev` on a free port (a debug build is enough;
release is preferred when both exist). Set `PYTEST_OBS_PORTS=18940-18949` to keep test servers
inside a port range. The local-mode tests read their runs back with the built `modgrad`. Tests that need `torch`,
`transformers` or `lightning` are skipped when those packages are missing. You can point
the tests at other binaries with `OBS_SERVER_BIN` and `OBS_BIN`.
