"""Lightning logger.

    from modgrad.integrations.lightning import ModgradLogger
    trainer = lightning.Trainer(logger=ModgradLogger(project="my-project"))

`log_hyperparams` -> run config, `log_metrics(metrics, step)` -> `run.log`
at that step. Works with `lightning` or `pytorch_lightning`; importable
without either (then a plain class with the same methods).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import modgrad

try:  # pragma: no cover - depends on the environment
    from lightning.pytorch.loggers.logger import Logger as _Base
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except Exception:  # noqa: BLE001
    try:
        from pytorch_lightning.loggers.logger import Logger as _Base  # type: ignore[no-redef]
        from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore[no-redef]
    except Exception:  # noqa: BLE001
        _Base = object  # type: ignore[assignment,misc]

        def rank_zero_only(fn):  # type: ignore[no-redef]
            return fn


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class ModgradLogger(_Base):  # type: ignore[misc,valid-type]
    def __init__(self, project: Optional[str] = None, name: Optional[str] = None, id: Optional[str] = None, prefix: str = "",
                 watch_grads_every: Optional[int] = None, **init_kwargs: Any):
        if _Base is not object:
            super().__init__()
        self._project = project
        self._name = name
        self._id = id
        self._prefix = prefix
        self._init_kwargs = init_kwargs
        self._run: Optional[modgrad.Run] = None
        self._watch_every = watch_grads_every

    @property
    def experiment(self) -> modgrad.Run:
        if self._run is None:
            self._run = modgrad.run if modgrad.run is not None else modgrad.init(project=self._project, name=self._name, id=self._id, **self._init_kwargs)
        return self._run

    @property
    def name(self) -> str:
        return self._project or "modgrad"

    @property
    def version(self) -> str:
        return self.experiment.id if self._run is not None or self._id is None else self._id

    @property
    def save_dir(self) -> Optional[str]:
        return self._run.dir if self._run is not None else None

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        if params is None:
            return
        if not isinstance(params, dict):
            params = vars(params) if hasattr(params, "__dict__") else dict(params)
        self.experiment.config.update(_flatten(dict(params)))

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        row = {(f"{self._prefix}/{k}" if self._prefix else k): v for k, v in metrics.items()}
        if step is not None:
            row["trainer/global_step"] = step
        self.experiment.log(row, step=step)

    @rank_zero_only
    def log_graph(self, model: Any, input_array: Any = None) -> None:
        if self._watch_every:
            from modgrad.torch import watch_grads

            watch_grads(model, every=self._watch_every, run=self.experiment)

    @rank_zero_only
    def save(self) -> None:
        if self._run is not None:
            self._run.flush(timeout=5.0)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run is not None:
            if status not in ("success", "finished"):
                self._run.event(f"lightning finalize: {status}")
            self._run.flush(timeout=5.0)

    def finish(self, exit_code: int = 0) -> None:
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
            self._run = None
