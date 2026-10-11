"""W&B-style data types and the conversion of logged values to the wire.

Wire values (see crates/modgrad-obs-server/src/v1_api.txt):
  scalar     a JSON number (null = gap)
  histogram  {"histogram": {"edges": [n+1], "counts": [n]}}   n <= 512
  tensor     {"tensor": {"shape": [...], "b64": <f32 little-endian>}}   <= 262144 elements
Anything else (tables, text) goes to the run's events.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MAX_BINS = 512
MAX_TENSOR_ELEMS = 256 * 1024
MAX_EVENT_BYTES = 64 * 1024

_warned: set = set()


def warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    from ._sender import say

    say(msg)


def _flat_numbers(x: Any) -> List[float]:
    """numpy / torch / nested sequences -> flat list of floats."""
    if hasattr(x, "detach"):  # torch
        x = x.detach().float().cpu().numpy()
    if hasattr(x, "ravel") and hasattr(x, "tolist"):  # numpy
        return [float(v) for v in x.ravel().tolist()]
    out: List[float] = []

    def walk(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for y in v:
                walk(y)
        else:
            out.append(float(v))

    walk(x)
    return out


def _shape(x: Any) -> List[int]:
    if hasattr(x, "shape"):
        return [int(s) for s in x.shape]
    shape: List[int] = []
    while isinstance(x, (list, tuple)):
        shape.append(len(x))
        x = x[0] if x else None
    return shape


def _b64_f32(data: Sequence[float]) -> str:
    return base64.b64encode(struct.pack("<%df" % len(data), *[v if math.isfinite(v) else float("nan") for v in data])).decode()


class Histogram:
    """`wandb.Histogram(sequence, num_bins=64)` or
    `wandb.Histogram(np_histogram=(counts, edges))`. At most 512 bins."""

    def __init__(self, sequence: Any = None, np_histogram: Optional[Tuple[Any, Any]] = None, num_bins: int = 64):
        if np_histogram is not None:
            counts, edges = np_histogram
            self.histogram = [float(c) for c in _flat_numbers(counts)]
            self.bins = [float(e) for e in _flat_numbers(edges)]
            if len(self.bins) != len(self.histogram) + 1:
                raise ValueError("np_histogram edges must have len(counts) + 1 entries")
            if len(self.histogram) > MAX_BINS:
                raise ValueError(f"histogram has {len(self.histogram)} bins; the limit is {MAX_BINS}")
            return
        if sequence is None:
            raise ValueError("Histogram needs a sequence or np_histogram")
        data = [v for v in _flat_numbers(sequence) if math.isfinite(v)]
        bins = max(1, min(int(num_bins), MAX_BINS))
        lo, hi = (min(data), max(data)) if data else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        w = (hi - lo) / bins
        counts = [0.0] * bins
        for v in data:
            counts[min(int((v - lo) / w), bins - 1)] += 1.0
        self.histogram = counts
        self.bins = [lo + w * i for i in range(bins + 1)]

    def to_wire(self) -> Dict[str, Any]:
        return {"histogram": {"edges": self.bins, "counts": self.histogram}}


class Tensor:
    """A raw array (modgrad extension; W&B has no direct equivalent).
    Row-major, stored as float32, at most 262144 elements."""

    def __init__(self, data: Any, shape: Optional[Sequence[int]] = None):
        self.shape = list(shape) if shape is not None else _shape(data)
        self.data = _flat_numbers(data)
        if math.prod(self.shape) != len(self.data):
            raise ValueError(f"shape {self.shape} does not match {len(self.data)} elements")
        if len(self.data) > MAX_TENSOR_ELEMS:
            raise ValueError(f"tensor has {len(self.data)} elements; the limit is {MAX_TENSOR_ELEMS}")

    def to_wire(self) -> Dict[str, Any]:
        return {"tensor": {"shape": self.shape, "b64": _b64_f32(self.data)}}


class Image:
    """`wandb.Image`. Only array data is kept (as a tensor, if it has at most
    262144 elements); file paths and PIL images are not uploaded yet."""

    def __init__(self, data_or_path: Any = None, mode: Any = None, caption: Optional[str] = None, **_: Any):
        self.caption = caption
        self.tensor: Optional[Tensor] = None
        if hasattr(data_or_path, "shape") and not isinstance(data_or_path, str):
            try:
                self.tensor = Tensor(data_or_path)
            except ValueError as e:
                warn_once("image-size", f"Image not logged: {e}")
        else:
            warn_once("image-kind", "Image: only numpy/torch arrays are stored for now; paths and PIL images are skipped")


class Table:
    """`wandb.Table(columns=, data=, dataframe=)`. Stored as a run event whose
    text is `table <key> <json>` with json = {"columns": [...], "data": [[...]]},
    capped at 64 KiB (rows beyond the cap are dropped and counted)."""

    def __init__(self, columns: Optional[List[str]] = None, data: Optional[List[List[Any]]] = None, rows: Optional[List[List[Any]]] = None, dataframe: Any = None, **_: Any):
        if dataframe is not None:
            columns = [str(c) for c in dataframe.columns]
            data = dataframe.values.tolist()
        self.columns: List[str] = list(columns or [])
        self.data: List[List[Any]] = [list(r) for r in (data if data is not None else rows or [])]

    def add_data(self, *row: Any) -> None:
        self.data.append(list(row))

    def add_column(self, name: str, data: Iterable[Any]) -> None:
        col = list(data)
        self.columns.append(name)
        if not self.data:
            self.data = [[v] for v in col]
        else:
            for r, v in zip(self.data, col):
                r.append(v)

    def get_column(self, name: str) -> List[Any]:
        i = self.columns.index(name)
        return [r[i] for r in self.data]

    def event_text(self, key: str) -> str:
        from ._net import clean

        rows = self.data
        while True:
            body = {"columns": self.columns, "data": rows}
            if len(rows) < len(self.data):
                body["truncated_rows"] = len(self.data) - len(rows)
            s = f"table {key} " + json.dumps(clean(body), default=str, allow_nan=False)
            if len(s) <= MAX_EVENT_BYTES or not rows:
                return s
            rows = rows[: len(rows) // 2]


class _Stub:
    def __init__(self, *a: Any, **k: Any):
        warn_once(type(self).__name__, f"{type(self).__name__} is not supported yet; values of this type are skipped")


class Audio(_Stub):
    pass


class Video(_Stub):
    pass


class Html(_Stub):
    pass


class Object3D(_Stub):
    pass


class Artifact:
    """No-op stand-in for `wandb.Artifact` (artifacts are not stored yet)."""

    def __init__(self, name: str = "", type: str = "", **_: Any):
        self.name, self.type = name, type
        warn_once("artifact", "Artifact is a no-op for now (nothing is uploaded)")

    def add_file(self, *a: Any, **k: Any) -> None:
        pass

    def add_dir(self, *a: Any, **k: Any) -> None:
        pass

    def add(self, *a: Any, **k: Any) -> None:
        pass

    def wait(self) -> "Artifact":
        return self


# ---------------------------------------------------------------- conversion


def _scalar(v: Any) -> Optional[float]:
    if v is None:
        return math.nan
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def convert(key: str, v: Any, out: Dict[str, Any], events: List[str], depth: int = 0) -> None:
    """Add one logged value to `out` (row values, flattened with '/') or
    `events` (text). Unknown types are skipped with a one-time warning."""
    s = _scalar(v)
    if s is not None:
        out[key] = s
        return
    if isinstance(v, Histogram) or isinstance(v, Tensor):
        out[key] = v.to_wire()
        return
    if isinstance(v, Image):
        if v.tensor is not None:
            out[key] = v.tensor.to_wire()
        return
    if isinstance(v, Table):
        events.append(v.event_text(key))
        return
    if isinstance(v, dict):
        if depth > 4:
            return
        for k, x in v.items():
            convert(f"{key}/{k}", x, out, events, depth + 1)
        return
    if isinstance(v, str):
        events.append(f"{key}: {v}")
        return
    if isinstance(v, _Stub):
        return
    # numpy / torch scalars and arrays
    size = None
    if hasattr(v, "numel") and callable(v.numel):
        size = int(v.numel())
    elif hasattr(v, "size") and hasattr(v, "shape"):
        size = int(v.size)
    if size is not None:
        if size == 1:
            try:
                out[key] = float(v.item() if hasattr(v, "item") else v)
            except (TypeError, ValueError):
                warn_once("type:" + type(v).__name__, f"cannot log {key!r} of type {type(v).__name__}; skipped")
            return
        if size == 0:
            return
        try:
            out[key] = Histogram(v).to_wire()
        except (TypeError, ValueError) as e:
            warn_once("arr:" + key, f"cannot log array {key!r}: {e}")
        return
    if isinstance(v, (list, tuple)):
        try:
            out[key] = Histogram(v).to_wire() if v else None
            if out[key] is None:
                del out[key]
        except (TypeError, ValueError):
            events.append(f"{key}: {json.dumps(v, default=str)}"[:MAX_EVENT_BYTES])
        return
    try:
        out[key] = float(v)
    except (TypeError, ValueError):
        warn_once("type:" + type(v).__name__, f"cannot log {key!r} of type {type(v).__name__}; skipped")
