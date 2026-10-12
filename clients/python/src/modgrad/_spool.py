"""Crash-safe on-disk spool, in the same format as `modgrad run --spool`.

A spool directory holds

    spool.jsonl   one item per line: {"row": {...}} | {"event": {...}}
                  | {"config": {...}} | {"summary": {...}}
    acked         byte offset of spool.jsonl that the server acknowledged
    meta.json     {server, token, name, project, meta, run, dashboard, finish}

`modgrad push <dir>` (the Rust CLI) understands the row and event lines and
skips the others; `python -m modgrad sync <dir>` sends all of them. Every
line is appended with one write() before it is sent, so a crash leaves at
most one torn final line, which is cut on reopen.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from ._net import dumps


def default_root() -> Path:
    env = os.environ.get("MODGRAD_DIR") or os.environ.get("MODGRAD_SPOOL")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "modgrad-obs" / "spool"


def dir_for(root: Path, name: str, project: Optional[str]) -> Path:
    """The directory `modgrad push` recomputes from meta.json: key = project/name
    when project is set and name has no '/', else name; '/' -> '__'."""
    key = f"{project}/{name}" if project and "/" not in name else name
    return root / key.replace("/", "__")


def _write_private(p: Path, data: bytes) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, p)


def repair_tail(p: Path) -> int:
    """Cut a torn final line (a crash mid-write). Returns bytes dropped."""
    try:
        b = p.read_bytes()
    except FileNotFoundError:
        return 0
    if not b or b.endswith(b"\n"):
        return 0
    keep = b.rfind(b"\n") + 1
    with open(p, "r+b") as f:
        f.truncate(keep)
    return len(b) - keep


class SpoolBusy(RuntimeError):
    pass


Item = Tuple[str, Any]  # ("row" | "event" | "config" | "summary", payload)


class Spool:
    """Append-before-send queue. Thread-safe: `append` is called from the
    training thread, `pending`/`ack`/`sync` from the sender thread."""

    def __init__(self, dir: Path, meta: Optional[Dict[str, Any]] = None):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lockfd = -1
        if not self._try_lock():
            raise SpoolBusy(f"{self.dir} is in use by another process")
        self.path = self.dir / "spool.jsonl"
        repair_tail(self.path)
        self._lock = threading.Lock()
        self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        self.offset = os.fstat(self._fd).st_size
        try:
            self.acked = min(int((self.dir / "acked").read_text().strip() or 0), self.offset)
        except (FileNotFoundError, ValueError):
            self.acked = 0
        self.unacked: Deque[Tuple[Item, int]] = deque()
        self._dirty = False
        self._load_unacked()
        self.meta: Dict[str, Any] = {"server": None, "token": None, "name": None, "project": None, "meta": {}, "run": None, "dashboard": None, "finish": None}
        try:
            self.meta.update(json.loads((self.dir / "meta.json").read_text()))
        except (FileNotFoundError, ValueError):
            pass
        if meta:
            self.meta.update(meta)
        self.save_meta()

    def _try_lock(self) -> bool:
        """Hold an exclusive flock on <dir>/lock while open, so `sync` never
        sends a spool that a live run is still writing."""
        try:
            import fcntl
        except ImportError:  # Windows: no locking
            return True
        fd = os.open(self.dir / "lock", os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._lockfd = fd
        return True

    @staticmethod
    def _parse(line: bytes) -> Optional[Item]:
        try:
            v = json.loads(line)
        except ValueError:
            return None
        if not isinstance(v, dict):
            return None
        for k in ("row", "event", "config", "summary"):
            if k in v:
                return (k, v[k])
        return None

    def _load_unacked(self) -> None:
        with open(self.path, "rb") as f:
            f.seek(self.acked)
            pos = self.acked
            for line in f:
                pos += len(line)
                it = self._parse(line)
                if it is not None:
                    self.unacked.append((it, pos))

    def save_meta(self) -> None:
        _write_private(self.dir / "meta.json", json.dumps(self.meta, indent=2, default=str).encode())

    def append(self, items: List[Item]) -> None:
        if not items:
            return
        lines = [(dumps({k: v}) + "\n").encode("utf-8") for k, v in items]
        with self._lock:
            os.write(self._fd, b"".join(lines))
            pos = self.offset
            for it, l in zip(items, lines):
                pos += len(l)
                self.unacked.append((it, pos))
            self.offset = pos
            self._dirty = True

    def sync(self) -> None:
        """fsync appended lines (done by the sender before each send)."""
        with self._lock:
            if self._dirty:
                os.fsync(self._fd)
                self._dirty = False

    def pending(self) -> int:
        return len(self.unacked)

    def head(self, n: int) -> List[Item]:
        with self._lock:
            return [it for it, _ in list(self.unacked)[:n]]

    def ack(self, n: int) -> None:
        with self._lock:
            if n <= 0:
                return
            end = 0
            for _ in range(min(n, len(self.unacked))):
                _, end = self.unacked.popleft()
            self.acked = end
        _write_private(self.dir / "acked", str(end).encode())

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                try:
                    os.fsync(self._fd)
                except OSError:
                    pass
                os.close(self._fd)
                self._fd = -1
            if self._lockfd >= 0:
                os.close(self._lockfd)
                self._lockfd = -1

    def remove(self) -> None:
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)
