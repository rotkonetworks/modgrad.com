"""Hugging Face `transformers` Trainer callback.

    from modgrad.integrations.huggingface import ModgradCallback
    trainer = Trainer(..., args=TrainingArguments(..., report_to="none"),
                      callbacks=[ModgradCallback(project="my-project")])

Logs every `Trainer.log` call at `state.global_step`, with keys rewritten
the way the W&B integration does (`loss` -> `train/loss`, `eval_loss` ->
`eval/loss`); the config is TrainingArguments plus the model config.
Only the world-zero process logs. Importable without transformers (it then
subclasses `object` with the same hook names).
"""

from __future__ import annotations

import numbers
from typing import Any, Dict, Optional

import modgrad

try:  # pragma: no cover - depends on the environment
    from transformers import TrainerCallback as _Base
except Exception:  # noqa: BLE001
    _Base = object  # type: ignore[assignment,misc]

# end-of-training values that belong in the summary, not the history
SUMMARY_KEYS = {"train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "total_flos"}


def rewrite_logs(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k.startswith("eval_"):
            out["eval/" + k[5:]] = v
        elif k.startswith("test_"):
            out["test/" + k[5:]] = v
        else:
            out["train/" + k] = v
    return out


def _world_zero(state: Any) -> bool:
    return bool(getattr(state, "is_world_process_zero", True))


class ModgradCallback(_Base):  # type: ignore[misc,valid-type]
    def __init__(self, project: Optional[str] = None, name: Optional[str] = None, finish_on_train_end: bool = False,
                 watch_grads_every: Optional[int] = None, **init_kwargs: Any):
        self.project = project
        self.name = name
        self.finish_on_train_end = finish_on_train_end
        self.watch_grads_every = watch_grads_every
        self.init_kwargs = init_kwargs
        self.run: Optional[modgrad.Run] = None
        self._owns_run = False
        self._watcher = None

    def setup(self, args: Any, state: Any, model: Any = None) -> None:
        if self.run is not None or not _world_zero(state):
            return
        config: Dict[str, Any] = {}
        mc = getattr(model, "config", None)
        if mc is not None and hasattr(mc, "to_dict"):
            try:
                config.update(mc.to_dict())
            except Exception:
                pass
        if args is not None:
            try:
                config.update(args.to_dict() if hasattr(args, "to_dict") else vars(args))
            except Exception:
                pass
        if model is not None and hasattr(model, "num_parameters"):
            try:
                config["model/num_parameters"] = model.num_parameters()
            except Exception:
                pass
        if modgrad.run is not None:
            self.run = modgrad.run
            self.run.config.update(config)
        else:
            run_name = self.name or getattr(args, "run_name", None)
            if run_name and run_name == getattr(args, "output_dir", None):
                run_name = None  # HF defaults run_name to output_dir
            self.run = modgrad.init(project=self.project, name=run_name, config=config, **self.init_kwargs)
            self._owns_run = True
        if self.watch_grads_every and model is not None:
            from modgrad.torch import watch_grads

            self._watcher = watch_grads(model, every=self.watch_grads_every, run=self.run)

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:
        self.setup(args, state, model)

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, model: Any = None, **kw: Any) -> None:
        if not _world_zero(state) or not logs:
            return
        if self.run is None:
            self.setup(args, state, model)
        if self.run is None:
            return
        summary = {k: v for k, v in logs.items() if k in SUMMARY_KEYS}
        hist = {k: v for k, v in logs.items() if k not in SUMMARY_KEYS}
        if summary:
            self.run.summary.update(summary)
        if hist:
            row = rewrite_logs(hist)
            row["train/global_step"] = state.global_step
            self.run.log(row, step=int(state.global_step))

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if self.run is None or not _world_zero(state):
            return
        best = getattr(state, "best_metric", None)
        if isinstance(best, numbers.Number):
            self.run.summary.update({"best_metric": float(best)})
        if self._watcher is not None:
            self._watcher.remove()
            self._watcher = None
        if self.finish_on_train_end and self._owns_run:
            self.run.finish()
            self.run = None
        else:
            self.run.flush(timeout=5.0)
