"""PyTorch helpers: per-layer gradient norms.

    modgrad.watch_grads(model, every=100)

Naming (shared with the dashboard's layer x step heatmap):

    layers/<i>/grad_norm      L2 norm of the gradients of layer i
    params/<module>/grad_norm parameters outside the numbered layers
                              (embeddings, final norm, lm_head, ...)
    grad_norm/total           L2 norm over all watched parameters

`i` is 0-based, in model order. A "layer" is a numbered block: the first
all-digit segment of a parameter name, e.g. `model.layers.3.mlp.up.weight`
-> block `model.layers.3`, `transformer.h.3.attn.c_attn.weight` ->
`transformer.h.3`. Blocks are numbered in the order they first appear in
`model.named_parameters()`, so a single stack keeps its own indices and an
encoder-decoder gets encoder 0..N-1 then decoder N..2N-1. A model without
numbered blocks uses each module that owns parameters as one layer.
The index -> module mapping is logged once as the config key
`modgrad/layers` and as an event `layers [...]`.

Sampling: gradients are captured on every `every`-th forward pass made with
grad enabled while the model is in training mode (one reduction per
parameter tensor, no host sync), and logged with the next `log()` call.
With gradient accumulation the norm is that of the accumulated gradient
after the sampled micro-batch's backward.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

_watchers: List["GradWatcher"] = []


def layer_assignment(names: List[str]) -> Tuple[List[str], Dict[str, Tuple[str, Any]]]:
    """Parameter names -> (layer module paths in index order,
    {param: ("layer", i) | ("params", module)})."""
    blocks: List[str] = []
    index: Dict[str, int] = {}
    out: Dict[str, Tuple[str, Any]] = {}
    for n in names:
        segs = n.split(".")
        for j, s in enumerate(segs[:-1]):
            if s.isdigit():
                key = ".".join(segs[: j + 1])
                if key not in index:
                    index[key] = len(blocks)
                    blocks.append(key)
                out[n] = ("layer", index[key])
                break
        else:
            out[n] = ("params", ".".join(segs[:-1]) or segs[-1])
    if not blocks:
        # no numbered blocks: every module that owns parameters is a layer
        for n in names:
            mod = out[n][1]
            if mod not in index:
                index[mod] = len(blocks)
                blocks.append(mod)
            out[n] = ("layer", index[mod])
    return blocks, out


class GradWatcher:
    def __init__(self, model: Any, every: int = 100, run: Any = None, prefix: str = "layers"):
        import torch

        self.torch = torch
        self.model = model
        self.every = max(1, int(every))
        self.run = run
        self.prefix = prefix
        self.count = 0
        self.armed = False
        self.sq: Dict[str, Any] = {}
        self.announced = False
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.layers, self.assign = layer_assignment([n for n, _ in params])
        self.handles: List[Any] = []
        self.handles.append(model.register_forward_pre_hook(self._pre_forward))
        for n, p in params:
            if hasattr(p, "register_post_accumulate_grad_hook"):
                self.handles.append(p.register_post_accumulate_grad_hook(self._make_post_hook(n)))
            else:  # torch < 2.1: the incoming gradient of this backward
                self.handles.append(p.register_hook(self._make_grad_hook(n)))

    def _pre_forward(self, module: Any, args: Any) -> None:
        if self.torch.is_grad_enabled() and module.training:
            self.armed = self.count % self.every == 0
            self.count += 1

    def _make_post_hook(self, name: str):
        def hook(p: Any) -> None:
            if self.armed and p.grad is not None:
                self.sq[name] = p.grad.detach().float().pow(2).sum()
        return hook

    def _make_grad_hook(self, name: str):
        def hook(g: Any) -> None:
            if self.armed and g is not None:
                self.sq[name] = g.detach().float().pow(2).sum()
        return hook

    def collect(self) -> Dict[str, float]:
        """The captured norms as metric keys (and clear them)."""
        if not self.sq:
            return {}
        names = list(self.sq)
        ts = [self.sq[n] for n in names]
        if len({t.device for t in ts}) > 1:  # model split across devices
            ts = [t.to("cpu") for t in ts]
        vals = self.torch.stack(ts).tolist()  # the one host sync
        self.sq = {}
        layer_sq: Dict[int, float] = {}
        other_sq: Dict[str, float] = {}
        for n, v in zip(names, vals):
            kind, k = self.assign.get(n, ("params", n))
            if kind == "layer":
                layer_sq[k] = layer_sq.get(k, 0.0) + v
            else:
                other_sq[k] = other_sq.get(k, 0.0) + v
        out: Dict[str, float] = {f"{self.prefix}/{i}/grad_norm": s ** 0.5 for i, s in sorted(layer_sq.items())}
        out.update({f"params/{m}/grad_norm": s ** 0.5 for m, s in other_sq.items()})
        out["grad_norm/total"] = sum(vals) ** 0.5
        return out

    def log_hook(self, run: Any, step: int) -> Dict[str, float]:
        if run is not self._run():
            return {}
        out = self.collect()
        if out and not self.announced:
            self.announced = True
            try:
                run.config.update({"modgrad/layers": self.layers})
                run.event("layers " + json.dumps(self.layers), step=step)
            except Exception:
                pass
        return out

    def _run(self) -> Any:
        if self.run is not None:
            return self.run
        import modgrad

        return modgrad.run

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        r = self._run()
        if r is not None and self.log_hook in r._log_hooks:
            r._log_hooks.remove(self.log_hook)
        if self in _watchers:
            _watchers.remove(self)


def watch_grads(model: Any, every: int = 100, run: Any = None, prefix: str = "layers") -> GradWatcher:
    """Log per-layer gradient norms every `every` training forward passes as
    `layers/<i>/grad_norm` (plus `params/<module>/grad_norm` and
    `grad_norm/total`). Returns a handle with `.remove()`."""
    import modgrad

    target = run if run is not None else modgrad.run
    if target is None:
        raise modgrad.Error("call modgrad.init() before watch_grads()")
    w = GradWatcher(model, every=every, run=target, prefix=prefix)
    target._log_hooks.append(w.log_hook)
    _watchers.append(w)
    return w


def unwatch_grads(run: Any = None, models: Any = None) -> None:
    ms = None if models is None else (models if isinstance(models, (list, tuple)) else [models])
    for w in list(_watchers):
        if (run is None or w.run is run) and (ms is None or any(w.model is m for m in ms)):
            w.remove()
