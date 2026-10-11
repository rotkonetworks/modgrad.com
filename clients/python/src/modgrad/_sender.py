"""Background sender: spool -> server, batched, with backoff.

The training thread only appends to the spool (one write() per log call);
this thread fsyncs, creates the run, posts batches and records acks. It
never raises into the training job: network trouble is reported once and
the rows stay spooled until the server is back or `finish` gives up.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from . import _net
from ._spool import Item, Spool

MAX_POST_ITEMS = 1000


def say(msg: str) -> None:
    if os.environ.get("MODGRAD_SILENT", "").lower() in ("1", "true", "yes"):
        return
    try:
        print("modgrad: " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _envf(k: str, default: float) -> float:
    try:
        return float(os.environ.get(k, default))
    except ValueError:
        return default


def envelope(items: List[Item]) -> Dict[str, Any]:
    rows: List[Any] = []
    events: List[Any] = []
    config: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for kind, v in items:
        if kind == "row":
            rows.append(v)
        elif kind == "event":
            events.append(v)
        elif kind == "config" and isinstance(v, dict):
            config.update(v)
        elif kind == "summary" and isinstance(v, dict):
            summary.update(v)
    env: Dict[str, Any] = {}
    if rows:
        env["rows"] = rows
    if events:
        env["events"] = events
    if config:
        env["config"] = config
    if summary:
        env["summary"] = summary
    return env


class Sender:
    def __init__(self, spool: Spool, interval: float = 1.0, batch_rows: int = 500, retry_max: Optional[float] = None, on_created=None, auth=None):
        self.spool = spool
        self.interval = interval
        self.batch_rows = batch_rows
        self.retry_max = retry_max if retry_max is not None else _envf("MODGRAD_RETRY_MAX", 30.0)
        self.backoff = 0.5
        self.next_try = 0.0
        self.fatal: Optional[str] = None
        self.warned = False
        self.on_created = on_created
        # auth.prepare(meta) fills in a token before the create (anonymous
        # project); auth.unauthorized(meta) -> True to retry after a 401.
        self.auth = auth
        self._auth_retries = 0
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._stop = False
        self._finish: Optional[Dict[str, Any]] = None
        self._deadline = 0.0
        self.finished_ok = False
        self._thread = threading.Thread(target=self._loop, name="modgrad-sender", daemon=True)

    # ------------------------------------------------------------ control

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def maybe_wake(self) -> None:
        if self.spool.pending() >= self.batch_rows:
            self._wake.set()

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until everything spooled so far is acknowledged (or timeout)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.spool.pending() == 0 and self.spool.meta.get("run"):
                return True
            if self.fatal or not self._thread.is_alive():
                return False
            self._wake.set()
            time.sleep(0.02)
        return self.spool.pending() == 0

    def close(self, finish: Dict[str, Any], timeout: float) -> bool:
        """Send what is left, then finish the run. True when the server has
        everything (the spool is then deleted)."""
        self._deadline = time.monotonic() + timeout
        self._finish = finish
        self._stop = True
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout + 2.0)
        return self.finished_ok

    # ------------------------------------------------------------ network

    @property
    def server(self) -> str:
        return self.spool.meta["server"]

    @property
    def token(self) -> Optional[str]:
        return self.spool.meta.get("token")

    def _fail(self, err: str, retry: bool) -> None:
        if retry:
            if not self.warned:
                say(f"server {self.server} unreachable ({err}); spooling to {self.spool.dir} and retrying")
                self.warned = True
            self.next_try = time.monotonic() + self.backoff
            self.backoff = min(self.backoff * 2, self.retry_max)
        else:
            say(f"server refused: {err}. Training continues; rows stay in {self.spool.dir} "
                f"(send them later with `python -m modgrad sync {self.spool.dir}`)")
            self.fatal = err

    def _ok(self) -> None:
        self.backoff = 0.5
        if self.warned:
            say("server reachable again; sending spooled rows")
            self.warned = False

    def _ensure_created(self) -> bool:
        m = self.spool.meta
        if m.get("run"):
            return True
        if self.auth is not None:
            try:
                self.auth.prepare(m)
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status", None)
                self._fail(str(e), status is None or _net.retryable(status))
                return False
        body: Dict[str, Any] = {"name": m["name"], "meta": m.get("meta") or {}}
        if m.get("project"):
            body["project"] = m["project"]
        if m.get("config"):
            body["config"] = m["config"]
        try:
            r = _net.post(self.server, "/api/v1/runs", body, self.token)
        except Exception as e:  # noqa: BLE001 - never crash training
            self._fail(str(e), True)
            return False
        if not r.ok:
            if r.status == 401 and self.auth is not None and self._auth_retries < 2:
                self._auth_retries += 1
                try:
                    if self.auth.unauthorized(m, r.error_text()):
                        return self._ensure_created()
                except Exception:
                    pass
            self._fail(f"HTTP {r.status}: {r.error_text()}", _net.retryable(r.status))
            return False
        j = r.json() or {}
        m["run"] = j.get("run") or m["name"]
        m["dashboard"] = j.get("dashboard")
        if j.get("token") and not self.token:
            m["token"] = j["token"]
        self.spool.save_meta()
        self._ok()
        if self.on_created:
            try:
                self.on_created(m, j)
            except Exception:
                pass
        return True

    def _post_items(self, items: List[Item]) -> Optional[_net.Resp]:
        env = envelope(items)
        if not env:
            return _net.Resp(200, b"{}")
        return _net.post(self.server, f"/api/v1/runs/{self.spool.meta['run']}/log", env, self.token)

    def _send_some(self) -> bool:
        """Post unacked items. Returns True when nothing is left."""
        if self.fatal or time.monotonic() < self.next_try or not self._ensure_created():
            return False
        while self.spool.pending():
            if self.fatal or time.monotonic() < self.next_try:
                return False
            items = self.spool.head(MAX_POST_ITEMS)
            try:
                r = self._post_items(items)
            except Exception as e:  # noqa: BLE001
                self._fail(str(e), True)
                return False
            if r.ok:
                self.spool.ack(len(items))
                self._ok()
            elif r.status == 400:
                self._send_one_by_one(items)
            else:
                self._fail(f"HTTP {r.status}: {r.error_text()}", _net.retryable(r.status))
                return False
        return True

    def _send_one_by_one(self, items: List[Item]) -> None:
        """A batch got 400: send items singly and drop the ones the server
        rejects (with a warning), so one bad value cannot wedge the spool."""
        for it in items:
            try:
                r = self._post_items([it])
            except Exception as e:  # noqa: BLE001
                self._fail(str(e), True)
                return
            if r.ok:
                self.spool.ack(1)
            elif r.status == 400:
                say(f"server rejected a {it[0]} ({r.error_text()}); dropped")
                self.spool.ack(1)
            else:
                self._fail(f"HTTP {r.status}: {r.error_text()}", _net.retryable(r.status))
                return

    def _post_finish(self, finish: Dict[str, Any]) -> bool:
        try:
            r = _net.post(self.server, f"/api/v1/runs/{self.spool.meta['run']}/finish", finish, self.token)
        except Exception:
            return False
        return r.ok

    # ------------------------------------------------------------ loop

    def _loop(self) -> None:
        while True:
            self._wake.wait(self.interval)
            self._wake.clear()
            try:
                self.spool.sync()
                done = self._send_some()
            except Exception as e:  # noqa: BLE001 - keep the thread alive
                say(f"sender error: {e!r}")
                done = False
            if self._stop:
                while not done and not self.fatal and time.monotonic() < self._deadline:
                    time.sleep(max(0.05, min(0.5, self.next_try - time.monotonic())))
                    try:
                        done = self._send_some()
                    except Exception:
                        done = False
                if done and self.spool.meta.get("run") and self._post_finish(self._finish or {}):
                    self.finished_ok = True
                return
