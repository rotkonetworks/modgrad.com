"""modgrad: training observability with a W&B-compatible API.

    import modgrad as wandb
    wandb.init(project="my-project", config={"lr": 3e-4})
    wandb.log({"loss": 0.5})
    wandb.finish()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._run import DEFAULT_URL, AlertLevel, Config, Error, Run, Summary, init
from ._types import Artifact, Audio, Histogram, Html, Image, Object3D, Table, Tensor, Video

__version__ = "0.1.0"

#: The current run (None before `init` and after `finish`), like `wandb.run`.
run: Optional[Run] = None
#: The current run's config / summary (placeholders before `init`).
config: Config = Config()
summary: Summary = Summary()


def _need_run(fn: str) -> Run:
    if run is None:
        raise Error(f"call modgrad.init() before modgrad.{fn}()")
    return run


def log(data: Dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None, sync: Optional[bool] = None) -> None:
    """Log a dict of metrics to the current run (see `Run.log`)."""
    _need_run("log").log(data, step=step, commit=commit)


def finish(exit_code: Optional[int] = None, quiet: Optional[bool] = None) -> None:
    """Finish the current run: flush, mark it finished. Safe to call twice."""
    if run is not None:
        run.finish(exit_code=exit_code, quiet=quiet)


def define_metric(name: str, *a: Any, **k: Any) -> Dict[str, Any]:
    """Stored on the run; the dashboard does not use it yet."""
    return _need_run("define_metric").define_metric(name, *a, **k)


def alert(title: str, text: str = "", level: Any = None, wait_duration: Any = None) -> None:
    """Recorded as a run event `alert[LEVEL] title: text` (no notification is sent)."""
    _need_run("alert").alert(title, text, level, wait_duration)


def event(text: str, step: Optional[int] = None) -> None:
    """A text event on the current run's timeline (modgrad extension)."""
    _need_run("event").event(text, step)


def watch(models: Any, criterion: Any = None, log: Optional[str] = "gradients", log_freq: int = 1000, idx: Any = None, log_graph: bool = False) -> None:
    """Logs per-layer gradient norms every `log_freq` steps (see `watch_grads`).
    W&B's gradient/parameter histograms and graphs are not implemented."""
    _need_run("watch").watch(models, criterion, log, log_freq, idx, log_graph)


def unwatch(models: Any = None) -> None:
    if run is not None:
        run.unwatch(models)


def watch_grads(model: Any, every: int = 100, run: Optional[Run] = None, prefix: str = "layers") -> Any:
    """Log per-layer gradient norms as `layers/<i>/grad_norm` (see README)."""
    from .torch import watch_grads as _wg

    return _wg(model, every=every, run=run, prefix=prefix)


def save(*a: Any, **k: Any) -> List[str]:
    """No-op: files are not uploaded yet."""
    return []


def login(*a: Any, **k: Any) -> bool:
    """No-op (returns True). Authentication is MODGRAD_TOKEN / OBS_TOKEN, or an
    anonymous project created on first use."""
    return True


def setup(*a: Any, **k: Any) -> None:
    return None


def teardown(*a: Any, **k: Any) -> None:
    finish()


def log_artifact(artifact: Any, *a: Any, **k: Any) -> Any:
    return artifact


def use_artifact(artifact: Any, *a: Any, **k: Any) -> Any:
    return artifact


def log_model(*a: Any, **k: Any) -> None:
    return None


def mark_preempting() -> None:
    if run is not None:
        run.mark_preempting()


def termlog(msg: str = "", *a: Any, **k: Any) -> None:
    from ._sender import say

    say(str(msg))


termwarn = termerror = termlog


class Settings(dict):
    """Accepted and ignored (`wandb.Settings(...)`)."""

    def __init__(self, **kw: Any):
        super().__init__(**kw)


__all__ = [
    "init", "log", "finish", "run", "config", "summary", "define_metric", "alert", "AlertLevel", "event", "watch", "unwatch",
    "watch_grads", "save", "login", "Histogram", "Table", "Tensor", "Image", "Audio", "Video", "Html", "Object3D", "Artifact",
    "Run", "Config", "Summary", "Settings", "Error", "DEFAULT_URL",
]
