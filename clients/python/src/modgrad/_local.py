"""Local writer: a run directory in the native modgrad-obs format, no server.

Used when the resolved config names no server (a local-mode `.modgrad.json`
from `modgrad init`, or `MODGRAD_DATA`) and for `mode="offline"`. It writes
exactly what the Rust `modgrad_obs::Run` writes (crates/modgrad-obs/src/lib.rs),
so `modgrad tail`, `modgrad export`, `modgrad mcp --dir` and
`modgrad-server --root` read it unchanged:

    <data>/runs/<project>/<id>/   (or <data>/runs/<id>/ without a project)
      config.json     {name, config, meta: {project?, git, hostname, start_unix, extra?}}
      metrics.jsonl   {"step", "t", key: value | null, ...} per log call with scalars
      arrays.jsonl    {"step", "t", "key", "kind", "shape", "dtype": "f32le", "b64"[, "edges"]}
      events.jsonl    {"step", "t", "text"}
      summary.json    {summary, elapsed_s[, finished, end_unix]}

Durability is the Rust contract: each record is one line handed to a single
write() on a file opened O_APPEND (no buffering, no thread), so a killed
process leaves at most one torn final line, which readers skip and the next
open cuts. config.json and summary.json are replaced atomically.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._net import clean
from ._spool import repair_tail

JSONL_FILES = ("metrics.jsonl", "arrays.jsonl", "events.jsonl")
MAX_HISTOGRAM_BINS = 512
MAX_ARRAY_ELEMS = 256 * 1024


def _line(v: Dict[str, Any]) -> bytes:
    # serde_json (no preserve_order) writes object keys sorted; match it
    return (json.dumps(clean(v), allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _pretty(v: Any) -> bytes:
    return json.dumps(clean(v), allow_nan=False, sort_keys=True, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def _replace(p: Path, data: bytes) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)


def _r6(x: float) -> float:
    return round(x * 1e6) / 1e6


def tail_step(p: Path) -> Optional[int]:
    """Step of the last complete line of a jsonl file that has one (reads the
    last 64 KiB), as the server's `tail_step` does."""
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            n = f.tell()
            f.seek(max(0, n - 64 * 1024))
            buf = f.read()
    except OSError:
        return None
    for line in reversed(buf.split(b"\n")):
        if not line:
            continue
        try:
            s = json.loads(line).get("step")
        except (ValueError, AttributeError):
            continue
        if isinstance(s, int) and not isinstance(s, bool) and s >= 0:
            return s
    return None


def last_logged_step(d: Path) -> Optional[int]:
    """The larger of the last steps of metrics.jsonl and arrays.jsonl (a step
    may hold only arrays): the server's rule for where a resumed run goes on."""
    steps = [s for s in (tail_step(d / f) for f in ("metrics.jsonl", "arrays.jsonl")) if s is not None]
    return max(steps) if steps else None


def _wire_array(key: str, v: Dict[str, Any]) -> Tuple[str, List[int], str, Optional[List[float]]]:
    """A `{"histogram"}` / `{"tensor"}` wire value -> (kind, shape, b64, edges),
    validated like `modgrad_obs::array_line`."""
    if "histogram" in v:
        h = v["histogram"]
        counts = [float(c) for c in h["counts"]]
        edges = [float(e) for e in h["edges"]]
        if len(edges) != len(counts) + 1:
            raise ValueError(f"histogram {key}: {len(edges)} edges for {len(counts)} bins")
        if len(counts) > MAX_HISTOGRAM_BINS:
            raise ValueError(f"histogram {key}: {len(counts)} bins exceeds {MAX_HISTOGRAM_BINS}")
        b64 = base64.b64encode(struct.pack("<%df" % len(counts), *counts)).decode()
        return "histogram", [len(counts)], b64, edges
    t = v["tensor"]
    shape = [int(s) for s in t["shape"]]
    n = len(base64.b64decode(t["b64"])) // 4
    prod = 1
    for s in shape:
        prod *= s
    if prod != n:
        raise ValueError(f"array {key}: shape {shape} != len {n}")
    if n > MAX_ARRAY_ELEMS:
        raise ValueError(f"array {key}: {n} elements exceeds cap {MAX_ARRAY_ELEMS}")
    return "tensor", shape, t["b64"], None


class LocalRun:
    """An open run directory. Thread-safe; every method may raise OSError,
    which the caller (`Run`) turns into a one-time warning."""

    def __init__(self, dir: Path, name: str, start_unix: float, summary: Dict[str, Any]):
        self.dir = dir
        #: the run's name: its path under runs/, as config.json records it
        self.name = name
        self.start_unix = start_unix
        self.summary: Dict[str, Any] = summary
        self._lock = threading.Lock()
        for f in JSONL_FILES:
            repair_tail(dir / f)
        self._fds = {f: os.open(dir / f, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644) for f in JSONL_FILES}

    @classmethod
    def create(cls, dir: Path, name: str, config: Dict[str, Any], meta: Dict[str, Any]) -> "LocalRun":
        """`Run::create_in`: a new run (or an existing one reopened, with its
        config.json rewritten)."""
        dir.mkdir(parents=True, exist_ok=True)
        _replace(dir / "config.json", _pretty({"name": name, "config": config, "meta": meta}))
        return cls(dir, name, float(meta["start_unix"]), {})

    @classmethod
    def open_existing(cls, dir: Path) -> "LocalRun":
        """`Run::open_existing`: append to a run without touching its config;
        `t` continues from its start and its summary is kept."""
        cfg = json.loads((dir / "config.json").read_text("utf-8"))
        cfg = cfg if isinstance(cfg, dict) else {}
        start = (cfg.get("meta") or {}).get("start_unix")
        try:
            sm = json.loads((dir / "summary.json").read_text("utf-8")).get("summary")
        except (OSError, ValueError, AttributeError):
            sm = None
        return cls(dir, str(cfg.get("name") or dir.name), float(start) if isinstance(start, (int, float)) else time.time(),
                   dict(sm) if isinstance(sm, dict) else {})

    def elapsed(self) -> float:
        return max(0.0, time.time() - self.start_unix)

    def _write(self, f: str, lines: List[bytes]) -> None:
        if lines:
            os.write(self._fds[f], b"".join(lines))

    def log(self, step: int, t: float, row: Dict[str, Any]) -> None:
        """One W&B log call: the scalars as one metrics line, each histogram or
        tensor as one arrays line. Invalid arrays raise ValueError and nothing
        of the call is written (as `Batch::push_log` rejects the whole log)."""
        scalars: Dict[str, Any] = {}
        arrays: List[bytes] = []
        for k, v in row.items():
            if isinstance(v, dict):
                kind, shape, b64, edges = _wire_array(k, v)
                rec: Dict[str, Any] = {"step": step, "t": _r6(t), "key": k, "kind": kind, "shape": shape, "dtype": "f32le", "b64": b64}
                if edges is not None:
                    rec["edges"] = edges
                arrays.append(_line(rec))
            else:
                scalars[k] = v
        with self._lock:
            if scalars:
                self._write("metrics.jsonl", [_line({**scalars, "step": step, "t": _r6(t)})])
                for k, v in scalars.items():
                    self.summary[k] = v
            self._write("arrays.jsonl", arrays)

    def event(self, step: int, t: float, text: str) -> None:
        with self._lock:
            self._write("events.jsonl", [_line({"step": step, "t": _r6(t), "text": text})])

    def update_config(self, patch: Dict[str, Any]) -> None:
        """`Run::update_config`: merge into config.json's `config`."""
        with self._lock:
            p = self.dir / "config.json"
            cfg = json.loads(p.read_text("utf-8"))
            if not isinstance(cfg.get("config"), dict):
                cfg["config"] = {}
            cfg["config"].update(patch)
            _replace(p, _pretty(cfg))

    def _write_summary(self, finished: bool) -> None:
        e = self.elapsed()
        v: Dict[str, Any] = {"summary": self.summary, "elapsed_s": _r6(e)}
        if finished:
            v["finished"] = True
            v["end_unix"] = self.start_unix + e
        _replace(self.dir / "summary.json", _pretty(v))

    def update_summary(self, patch: Dict[str, Any]) -> None:
        """`Run::update_summary`: merge and rewrite summary.json (not finished)."""
        with self._lock:
            self.summary.update(patch)
            self._write_summary(False)

    def finish(self, extra: Dict[str, Any]) -> None:
        """`Run::finish_with`: merge `extra` and write the final summary.json."""
        with self._lock:
            self.summary.update(extra)
            self._write_summary(True)

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds = {}
