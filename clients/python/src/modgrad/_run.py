"""Run, Config, Summary and `init` (the W&B-compatible surface)."""

from __future__ import annotations

import atexit
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from . import _auth, _net, _project
from ._local import LocalRun, last_logged_step
from ._sender import Sender, say
from ._spool import Spool, SpoolBusy, default_root, dir_for
from ._types import convert, warn_once

# The server used when neither `url`, MODGRAD_URL nor a project file (.modgrad.json from
# `modgrad init`) names one. It matches the `modgrad` CLI default; change it here once a hosted
# modgrad server exists.
DEFAULT_URL = "http://127.0.0.1:8790"


class Error(Exception):
    """Raised for usage errors (e.g. log() before init())."""


# ----------------------------------------------------------------- helpers


def _env(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


_ID_BAD = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_id(s: str) -> str:
    """A run id segment: [A-Za-z0-9_.-], at most 96 chars."""
    s = _ID_BAD.sub("-", str(s)).strip("-.") or "run"
    return s[:96]


def auto_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S", time.gmtime()) + "-%04x" % random.getrandbits(16)


def _git() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=2)
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def _to_dict(x: Any) -> Dict[str, Any]:
    """dict, argparse.Namespace, dataclass, or anything with __dict__."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "to_dict") and callable(x.to_dict):
        try:
            return dict(x.to_dict())
        except Exception:
            pass
    try:
        import dataclasses

        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return dataclasses.asdict(x)
    except Exception:
        pass
    if hasattr(x, "__dict__"):
        return {k: v for k, v in vars(x).items() if not k.startswith("_")}
    raise TypeError(f"config must be a dict-like object, got {type(x).__name__}")


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except Exception:
            pass
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            pass
    return str(v)


# ----------------------------------------------------------------- config / summary


class Config:
    """`run.config`: attribute and item access, `update`, `setdefaults`.
    Every change is sent to the server (merged into the run's config)."""

    def __init__(self, run: Optional["Run"] = None, data: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "_run", run)
        object.__setattr__(self, "_data", dict(data or {}))
        object.__setattr__(self, "_locked", set())

    def _emit(self, patch: Dict[str, Any]) -> None:
        run = self._run
        if run is not None and patch:
            run._emit_config(_jsonable(patch))

    def update(self, d: Any = None, allow_val_change: bool = True, **kw: Any) -> None:
        patch = _to_dict(d) if d is not None else {}
        patch.update(kw)
        self._data.update(patch)
        self._emit(patch)

    def setdefaults(self, d: Dict[str, Any]) -> None:
        patch = {k: v for k, v in _to_dict(d).items() if k not in self._data}
        self.update(patch)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def get(self, k: str, default: Any = None) -> Any:
        return self._data.get(k, default)

    def __getitem__(self, k: str) -> Any:
        return self._data[k]

    def __setitem__(self, k: str, v: Any) -> None:
        self.update({k: v})

    def __contains__(self, k: object) -> bool:
        return k in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, k: str) -> Any:
        try:
            return self._data[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k: str, v: Any) -> None:
        self.update({k: v})

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


class Summary:
    """`run.summary`: last value of every logged scalar, plus anything set
    explicitly (explicit values are sent to the server)."""

    def __init__(self, run: Optional["Run"] = None):
        object.__setattr__(self, "_run", run)
        object.__setattr__(self, "_data", {})

    def _auto(self, row: Dict[str, Any]) -> None:
        for k, v in row.items():
            if isinstance(v, float):
                self._data[k] = v

    def update(self, d: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        patch = dict(d or {})
        patch.update(kw)
        patch = _jsonable(patch)
        self._data.update(patch)
        if self._run is not None and patch:
            self._run._emit("summary", patch)

    def get(self, k: str, default: Any = None) -> Any:
        return self._data.get(k, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __getitem__(self, k: str) -> Any:
        return self._data[k]

    def __setitem__(self, k: str, v: Any) -> None:
        self.update({k: v})

    def __delitem__(self, k: str) -> None:
        self._data.pop(k, None)

    def __contains__(self, k: object) -> bool:
        return k in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, k: str) -> Any:
        try:
            return self._data[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k: str, v: Any) -> None:
        self.update({k: v})

    def __repr__(self) -> str:
        return f"Summary({self._data!r})"


class AlertLevel:
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


# ----------------------------------------------------------------- auth policy


class _AuthPolicy:
    """Decides the token before the run is created (in the sender thread, so
    `init` never blocks on the network)."""

    def __init__(self, server: str, anon: bool):
        self.server = server
        self.anon = anon

    def prepare(self, m: Dict[str, Any]) -> None:
        if m.get("token"):
            return
        run_full = f"{m['project']}/{m['name']}" if m.get("project") else None
        if run_full:
            t = _auth.run_token(self.server, run_full)
            if t:
                m["token"] = t
                return
        if self.anon:
            p = _auth.anon_project(self.server)
            if p and p.get("token"):
                m["token"] = p["token"]
                m.setdefault("meta", {})["anon_project"] = p.get("project")

    def unauthorized(self, m: Dict[str, Any], err: str) -> bool:
        if self.anon and m.get("token", "").startswith("obsp_"):
            # the cached anonymous project expired or the server was reset
            _auth.forget_anon(self.server)
            m["token"] = None
            return True
        if not m.get("token") and not self.anon:
            say(f"the server needs a token ({err}); falling back to an anonymous project")
            self.anon = True
            m["project"] = None
            return True
        return False


# ----------------------------------------------------------------- run


def _open_spool(sdir: Path, meta: Dict[str, Any]) -> Spool:
    """The run's spool; if another live process holds it (several ranks
    calling init with the same id), use a per-process sibling directory."""
    try:
        return Spool(sdir, meta)
    except SpoolBusy:
        alt = sdir.with_name(f"{sdir.name}.{os.getpid()}")
        warn_once("spool-busy", f"{sdir} is in use by another process; this process spools to {alt}")
        return Spool(alt, meta)


_active: List["Run"] = []
_lock = threading.RLock()
_exit_code: List[int] = [0]


class Run:
    """One run. Create with `modgrad.init(...)`."""

    def __init__(self, *, id: str, name: str, project: Optional[str], config: Dict[str, Any], mode: str, server: str, token: Optional[str],
                 anon: bool, root: Path, meta: Dict[str, Any], start_step: int = 0, tags: Sequence[str] = (), notes: Optional[str] = None,
                 group: Optional[str] = None, job_type: Optional[str] = None, interval: float = 1.0, batch_rows: int = 500,
                 finish_timeout: Optional[float] = None, retry_max: Optional[float] = None, local: Optional[LocalRun] = None):
        self.id = id
        self.name = name
        self.project = project
        self.entity = None
        self.mode = mode
        self.tags = tuple(tags or ())
        self.notes = notes
        self.group = group
        self.job_type = job_type
        self.server = server
        self.config = Config(None, config)
        self.summary = Summary(None)
        self.metric_defs: Dict[str, Dict[str, Any]] = {}
        self.step = start_step
        self._pid = os.getpid()
        self._t0 = time.time()
        self._finished = False
        self._log_hooks: List[Callable[["Run", int], Dict[str, Any]]] = []
        self._finish_timeout = finish_timeout if finish_timeout is not None else float(_env("MODGRAD_FINISH_TIMEOUT") or 30)
        self._spool: Optional[Spool] = None
        self._sender: Optional[Sender] = None
        self._dashboard: Optional[str] = None
        # local mode: a native run directory (no server, no spool)
        self._local = local
        self.dir = None
        if local is not None:
            self.dir = str(local.dir)
        elif mode != "disabled":
            sdir = dir_for(root, id, project)
            self._spool = _open_spool(sdir, {"server": server, "token": token, "name": id, "project": project, "meta": meta,
                                       "config": _jsonable(config), "display_name": name, "mode": mode})
            self.dir = str(sdir)
            if mode == "online":
                self._sender = Sender(self._spool, interval=interval, batch_rows=batch_rows, retry_max=retry_max,
                                      on_created=self._on_created, auth=_AuthPolicy(server, anon))
                self._sender.start()
        if self._spool is not None:
            # rows from an earlier attempt with this id that never reached the server
            for (kind, v), _ in list(self._spool.unacked):
                if kind == "row" and isinstance(v, dict) and isinstance(v.get("step"), int):
                    self.step = max(self.step, v["step"] + 1)
        object.__setattr__(self.config, "_run", self)
        object.__setattr__(self.summary, "_run", self)

    # -------------------------------------------------------------- identity

    @property
    def url(self) -> Optional[str]:
        if self._spool is not None and self._spool.meta.get("dashboard"):
            return self._spool.meta["dashboard"]
        return self._dashboard

    def get_url(self) -> Optional[str]:
        return self.url

    @property
    def path(self) -> str:
        if self._local is not None:
            return self._local.name
        if self._spool is not None and self._spool.meta.get("run"):
            return self._spool.meta["run"]
        return f"{self.project}/{self.id}" if self.project else self.id

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def offline(self) -> bool:
        """True when nothing goes to a server (offline mode or local mode)."""
        return self.mode == "offline" or self._local is not None

    @property
    def local(self) -> bool:
        """True when the run is written as a native run directory (`run.dir`)."""
        return self._local is not None

    @property
    def disabled(self) -> bool:
        return self.mode == "disabled"

    def _on_created(self, m: Dict[str, Any], resp: Dict[str, Any]) -> None:
        if resp.get("token") and m.get("token") == resp.get("token"):
            _auth.save_run_token(self.server, m["run"], resp["token"])
        if resp.get("created", True):
            say(f"run {m['run']} -> {m.get('dashboard')}")

    # -------------------------------------------------------------- writes

    def _live(self) -> bool:
        return not self._finished and os.getpid() == self._pid

    def _emit(self, kind: str, payload: Any) -> None:
        if self._local is not None:
            if self._live():
                self._local_write(kind, payload)
            return
        if self._spool is None or not self._live():
            return
        try:
            self._spool.append([(kind, payload)])
            if self._sender is not None:
                self._sender.maybe_wake()
        except Exception as e:  # noqa: BLE001 - never crash training (disk full, ...)
            warn_once("spool-write", f"cannot write the spool ({e}); logging continues in memory only")

    def _emit_config(self, patch: Dict[str, Any]) -> None:
        self._emit("config", patch)

    def _local_write(self, kind: str, payload: Any) -> None:
        lr = self._local
        try:
            if kind == "row":
                payload = dict(payload)
                lr.log(payload.pop("step"), payload.pop("t"), payload)
            elif kind == "event":
                lr.event(payload["step"], payload["t"], payload["text"])
            elif kind == "config":
                lr.update_config(payload)
            elif kind == "summary":
                lr.update_summary(payload)
        except ValueError as e:  # an invalid array: dropped, like the server's 400
            warn_once("local-bad:" + str(e)[:80], f"not logged: {e}")
        except Exception as e:  # noqa: BLE001 - never crash training (disk full, ...)
            warn_once("local-write", f"cannot write {lr.dir} ({e}); logging continues in memory only")

    def _t(self) -> float:
        if self._local is not None:
            return round(self._local.elapsed(), 6)
        return round(time.time() - self._t0, 6)

    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None, sync: Optional[bool] = None) -> None:
        """W&B `run.log`. Without `step`, rows go to the run's step counter,
        which advances after each call (unless commit=False). With `step`,
        the row is written at that step and the counter moves to it; it
        advances only on commit=True. Rows at the same step are merged."""
        if not self._live():
            return
        if not isinstance(data, dict):
            raise Error(f"log() takes a dict, got {type(data).__name__}")
        if step is not None:
            step = int(step)
            if step < 0:
                raise Error("step must be non-negative")
            if step > self.step:
                self.step = step
            at = step
            if commit is None:
                commit = False
        else:
            at = self.step
            if commit is None:
                commit = True
        for hook in list(self._log_hooks):
            try:
                extra = hook(self, at)
                if extra:
                    data = {**data, **extra}
            except Exception as e:  # noqa: BLE001
                warn_once("hook:" + repr(hook), f"a log hook failed ({e!r}); skipped")
        row: Dict[str, Any] = {}
        events: List[str] = []
        for k, v in data.items():
            k = str(k)
            if k in ("step", "_step", "t", "_timestamp", "_runtime"):
                continue
            convert(k, v, row, events)
        t = self._t()
        items = []
        if row:
            items.append(("row", {"step": at, "t": t, **row}))
            self.summary._auto(row)
        for e in events:
            items.append(("event", {"step": at, "t": t, "text": e}))
        if items and self._local is not None:
            for kind, v in items:
                self._local_write(kind, v)
        elif items and self._spool is not None:
            try:
                self._spool.append(items)
                if self._sender is not None:
                    self._sender.maybe_wake()
            except Exception as e:  # noqa: BLE001
                warn_once("spool-write", f"cannot write the spool ({e}); logging continues in memory only")
        if commit and at == self.step:
            self.step += 1

    def event(self, text: str, step: Optional[int] = None) -> None:
        """A text event on the run's timeline (modgrad extension)."""
        self._emit("event", {"step": self.step if step is None else int(step), "t": self._t(), "text": str(text)})

    def alert(self, title: str, text: str = "", level: Any = None, wait_duration: Any = None) -> None:
        lvl = getattr(level, "value", level) or AlertLevel.INFO
        self.event(f"alert[{lvl}] {title}" + (f": {text}" if text else ""))

    def define_metric(self, name: str, step_metric: Optional[str] = None, step_sync: Optional[bool] = None, hidden: Optional[bool] = None,
                      summary: Optional[str] = None, goal: Optional[str] = None, overwrite: Optional[bool] = None, **kw: Any) -> Dict[str, Any]:
        """Stored, not acted on: the dashboard does not use metric definitions yet."""
        d = {k: v for k, v in dict(step_metric=step_metric, step_sync=step_sync, hidden=hidden, summary=summary, goal=goal, **kw).items() if v is not None}
        self.metric_defs[name] = d
        return d

    def watch(self, models: Any, criterion: Any = None, log: Optional[str] = "gradients", log_freq: int = 1000, idx: Any = None, log_graph: bool = False) -> None:
        """Maps to `modgrad.watch_grads(model, every=log_freq)` when
        log is "gradients" or "all": per-layer gradient norms, not W&B's
        histograms. Parameter logging and graphs are not implemented."""
        if log not in ("gradients", "all"):
            return
        from .torch import watch_grads

        for m in models if isinstance(models, (list, tuple)) else [models]:
            watch_grads(m, every=log_freq, run=self)

    def unwatch(self, models: Any = None) -> None:
        from .torch import unwatch_grads

        unwatch_grads(self, models)

    def save(self, *a: Any, **k: Any) -> List[str]:
        """No-op: files are not uploaded yet."""
        return []

    def log_code(self, *a: Any, **k: Any) -> None:
        return None

    def log_artifact(self, artifact: Any, *a: Any, **k: Any) -> Any:
        return artifact

    def use_artifact(self, artifact: Any, *a: Any, **k: Any) -> Any:
        return artifact

    def log_model(self, *a: Any, **k: Any) -> None:
        return None

    def mark_preempting(self) -> None:
        self.event("preempting")

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until the server has acknowledged everything logged so far
        (modgrad extension; useful in tests and before checkpoints). In local
        mode every call is already written, so this returns True at once."""
        if self._local is not None:
            return True
        if self._sender is None:
            return self.mode == "disabled"
        return self._sender.flush(timeout)

    # -------------------------------------------------------------- finish

    def finish(self, exit_code: Optional[int] = None, quiet: Optional[bool] = None) -> None:
        if self._finished or os.getpid() != self._pid:
            return
        self._finished = True
        with _lock:
            if self in _active:
                _active.remove(self)
        _clear_globals(self)
        code = int(exit_code) if exit_code is not None else _exit_code[0]
        fin = {"exit_code": code}
        if self._local is not None:
            try:
                self._local.finish(fin)
            except Exception as e:  # noqa: BLE001
                say(f"cannot write the final summary of {self._local.dir}: {e}")
            self._local.close()
            return
        if self._spool is None:
            return
        self._spool.meta["finish"] = fin
        self._spool.save_meta()
        if self._sender is None:  # offline
            self._spool.close()
            if not quiet:
                say(f"offline run saved to {self._spool.dir}; send it with `python -m modgrad sync {self._spool.dir}`")
            return
        pending = self._spool.pending()
        ok = self._sender.close(fin, self._finish_timeout)
        if ok:
            self._spool.remove()
        else:
            self._spool.close()
            left = self._spool.pending()
            say(f"{left} of {max(left, pending)} items not sent; kept in {self._spool.dir}. "
                f"Send them later with `python -m modgrad sync {self._spool.dir}`")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, et: Any, ev: Any, tb: Any) -> bool:
        self.finish(exit_code=1 if et is not None else 0)
        return False

    def __repr__(self) -> str:
        return f"<modgrad.Run {self.path} mode={self.mode}>"


# ----------------------------------------------------------------- module state


def _clear_globals(run: Run) -> None:
    import modgrad

    if modgrad.run is run:
        modgrad.run = _active[-1] if _active else None
        cur = modgrad.run
        modgrad.config = cur.config if cur else Config()
        modgrad.summary = cur.summary if cur else Summary()


def _set_globals(run: Run) -> None:
    import modgrad

    modgrad.run = run
    modgrad.config = run.config
    modgrad.summary = run.summary


def _atexit() -> None:
    for r in list(_active):
        try:
            r.finish()
        except Exception:
            pass


_prev_hook = sys.excepthook


def _excepthook(et: Any, ev: Any, tb: Any) -> None:
    _exit_code[0] = 130 if et is KeyboardInterrupt else 1
    _prev_hook(et, ev, tb)


atexit.register(_atexit)
sys.excepthook = _excepthook


def _resume_step(server: str, run_full: str, token: Optional[str]) -> int:
    """One past the last step the server has for `run_full` (0 if none)."""
    try:
        if "/" not in run_full and token:
            # a project token places the run; the (idempotent) create names it
            r = _net.post(server, "/api/v1/runs", {"name": run_full}, token, timeout=3)
            if r.ok and isinstance(r.json(), dict) and r.json().get("run"):
                run_full = r.json()["run"]
        project = run_full.split("/")[0] if "/" in run_full else None
        status, idx = _net.get(server, "/api/index" + (f"?project={project}" if project else ""), token, timeout=3)
        if status == 200 and isinstance(idx, dict):
            tail = "/" + run_full
            for e in idx.get("runs", []):
                # with a project token and no project name, match <any project>/<id>
                if e.get("id") == run_full or ("/" not in run_full and str(e.get("id", "")).endswith(tail)):
                    ls = e.get("last_step")
                    return int(ls) + 1 if ls is not None else 0
    except Exception as e:  # noqa: BLE001
        warn_once("resume", f"cannot look up the resumed run's last step ({e}); continuing from step 0")
    return 0


def local_root(url: Optional[str], data: Optional[str], pf: Dict[str, Any], mode: str) -> Optional[Path]:
    """The obs directory to write a native run into, or None to use a server.

    Argument > environment > project file, the first that names either a
    server or a data directory decides: `url=` / `data=`, then `MODGRAD_URL`
    / `MODGRAD_DATA`, then `.modgrad.json` (`server`, or local mode `data`).
    Nothing named means the default server. Offline mode always writes
    locally, into the data directory resolved the same way (default ./obs)."""
    env_data = _env("MODGRAD_DATA", "OBS_DIR")
    file_data = Path(pf["_dir"]) / (pf.get("data") or "obs") if pf.get("_dir") else None
    root = Path(data) if data else Path(env_data) if env_data else file_data or Path("obs")
    if mode == "offline":
        local = True
    elif url or data:
        local = bool(data) and not url
    elif _env("MODGRAD_URL", "OBS_URL") or env_data:
        local = not _env("MODGRAD_URL", "OBS_URL")
    else:
        local = bool(pf) and not pf.get("server")
    return root.expanduser().absolute() if local else None


def _local_name(s: str) -> str:
    """A local run name: 1 to 3 `/`-separated parts of [A-Za-z0-9_.-]."""
    parts = [sanitize_id(p) for p in str(s).split("/") if p.strip("/")]
    return "/".join(parts[:3]) if parts else "run"


def _shown(p: Path) -> str:
    """`p` relative to the current directory when it is below it."""
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return str(p)


def _open_local(root: Path, project: Optional[str], id: Optional[str], name: Optional[str], resume: Any, cfg: Dict[str, Any],
                meta: Dict[str, Any]) -> Tuple[LocalRun, str, int]:
    """Create or reopen `<root>/runs/<project>/<id or name>`. Returns the run,
    its full name (path under runs/) and the step to continue from.

    - `id=` names the run: an existing directory is appended to, and with
      `resume` its steps continue after the last logged step.
    - `name=` without `id=`: with `resume` an existing run of that name is
      continued; otherwise a taken name gets a `-<4 hex>` suffix, so two
      scripts never write into one run.
    - neither: a fresh `run-<date>-<time>-<4 hex>`."""
    base = _local_name(id or name) if (id or name) else auto_id()
    full = f"{project}/{base}" if project and "/" not in base else base
    if full.count("/") > 2:
        full = sanitize_id(full)
    runs = root / "runs"
    resume_on = bool(resume) and str(resume).lower() not in ("never", "false", "0")
    d = runs / full
    exists = (d / "config.json").is_file()
    if str(resume).lower() == "must" and not exists:
        raise Error(f"resume='must' but there is no run {full} in {runs}")
    if exists and (id or resume_on):
        run = LocalRun.open_existing(d)
        if cfg:
            run.update_config(cfg)
        last = last_logged_step(d) if resume_on else None
        return run, full, (last + 1 if last is not None else 0)
    if d.exists():  # a taken name (or a directory of other runs)
        while True:
            alt = f"{full}-{random.getrandbits(16):04x}"
            if not (runs / alt).exists():
                full, d = alt, runs / alt
                break
    return LocalRun.create(d, full, cfg, meta), full, 0


def init(project: Optional[str] = None, entity: Optional[str] = None, config: Any = None, name: Optional[str] = None, id: Optional[str] = None,
         resume: Union[bool, str, None] = None, dir: Optional[str] = None, mode: Optional[str] = None, tags: Optional[Sequence[str]] = None,
         notes: Optional[str] = None, group: Optional[str] = None, job_type: Optional[str] = None, reinit: Union[bool, str, None] = None,
         config_exclude_keys: Optional[Sequence[str]] = None, config_include_keys: Optional[Sequence[str]] = None, *,
         url: Optional[str] = None, token: Optional[str] = None, data: Optional[str] = None, settings: Any = None, **ignored: Any) -> Run:
    """Start a run (W&B `wandb.init`). Extra modgrad arguments: `url`, `token`,
    and `data` (write a native run directory under this obs directory
    instead of sending to a server). Unknown W&B arguments are accepted and
    ignored."""
    import modgrad

    with _lock:
        if _active and reinit in (None, False, "return_previous", "default"):
            return _active[-1]
        if _active and reinit in (True, "finish_previous"):
            for r in list(_active):
                r.finish()

        mode = (mode or _env("MODGRAD_MODE", "WANDB_MODE") or "online").lower()
        if (_env("WANDB_DISABLED") or "").lower() in ("1", "true", "yes"):
            mode = "disabled"
        mode = {"run": "online", "dryrun": "offline", "shared": "online"}.get(mode, mode)
        if mode not in ("online", "offline", "disabled"):
            raise Error(f"mode must be online, offline or disabled, got {mode!r}")

        # argument > environment > project file (.modgrad.json) > default
        project_arg = project or _env("MODGRAD_PROJECT", "WANDB_PROJECT", "OBS_PROJECT")
        pf = _project.find()
        lroot = local_root(url, data, pf, mode) if mode != "disabled" else None
        env_url = _env("MODGRAD_URL", "OBS_URL")
        server = (url or env_url or pf.get("server") or DEFAULT_URL).rstrip("/")
        file_token = _project.token_for(pf, server)
        token = token or _env("MODGRAD_TOKEN", "OBS_TOKEN") or file_token
        project = project or _env("MODGRAD_PROJECT", "WANDB_PROJECT", "OBS_PROJECT") or (pf.get("project") if token and token == file_token else None)
        project = sanitize_id(project) if project else None
        name = name or _env("MODGRAD_NAME", "WANDB_NAME")
        id = id or _env("MODGRAD_RUN_ID", "WANDB_RUN_ID")
        resume = resume if resume is not None else _env("WANDB_RESUME")
        tags = list(tags or []) + [t for t in (_env("WANDB_TAGS") or "").split(",") if t]
        notes = notes or _env("WANDB_NOTES")
        group = group or _env("WANDB_RUN_GROUP")
        job_type = job_type or _env("WANDB_JOB_TYPE")

        run_id = sanitize_id(id) if id else (sanitize_id(name) + "-%04x" % random.getrandbits(16) if name else auto_id())
        cfg = _to_dict(config)
        if config_include_keys:
            cfg = {k: v for k, v in cfg.items() if k in config_include_keys}
        if config_exclude_keys:
            cfg = {k: v for k, v in cfg.items() if k not in config_exclude_keys}

        # no token: write tokenless when a project was named (dev servers),
        # else use (or create) this machine's anonymous project
        anon = token is None and project is None
        # a project token decides the project; don't send ours alongside it
        send_project = project if token is None else None

        meta: Dict[str, Any] = {"hostname": socket.gethostname(), "git": _git(), "client": "modgrad-python 0.1.0",
                                "python": sys.version.split()[0], "argv": sys.argv[:20], "display_name": name or run_id}
        for k, v in (("tags", tags), ("notes", notes), ("group", group), ("job_type", job_type), ("entity", entity), ("project_name", project)):
            if v:
                meta[k] = v

        if lroot is not None:
            # the project file's project belongs to its server, not to local runs
            lproject = sanitize_id(project_arg) if project_arg else None
            return _init_local(lroot, lproject, name, id, resume, cfg, meta, mode, tags, notes, group, job_type)

        start_step = 0
        if resume and id and mode == "online":
            run_full = f"{project}/{run_id}" if project else run_id
            if anon:
                cached = _auth._load("anon.json").get(server) or {}
                if cached.get("project"):
                    run_full = f"{cached['project']}/{run_id}"
            start_step = _resume_step(server, run_full, token)

        root = Path(dir).expanduser() / "modgrad" if dir else default_root()
        run = Run(id=run_id, name=name or run_id, project=send_project, config=cfg, mode=mode, server=server, token=token, anon=anon,
                  root=root, meta=meta, start_step=start_step, tags=tags, notes=notes, group=group, job_type=job_type,
                  interval=float(_env("MODGRAD_FLUSH_INTERVAL") or 1.0), batch_rows=int(_env("MODGRAD_BATCH_ROWS") or 500))
        run.project = project
        if cfg:
            run._emit_config(_jsonable(cfg))
        _active.append(run)
        _set_globals(run)
        return run


def _init_local(root: Path, project: Optional[str], name: Optional[str], id: Optional[str], resume: Any, cfg: Dict[str, Any],
                client_meta: Dict[str, Any], mode: str, tags: Sequence[str], notes: Optional[str], group: Optional[str], job_type: Optional[str]) -> Run:
    """`init` in local mode (the caller holds `_lock`)."""
    # config.json's meta as the server writes it: git and hostname as
    # fields, everything else the client knows under `extra`
    extra = {k: v for k, v in client_meta.items() if k not in ("git", "hostname", "project_name", "display_name")}
    if name:
        extra["display_name"] = name
    meta: Dict[str, Any] = {"git": client_meta.get("git"), "hostname": client_meta.get("hostname"), "start_unix": time.time(), "extra": _jsonable(extra)}
    if project:
        meta["project"] = project
    lrun, full, start_step = _open_local(root, project, id, name, resume, _jsonable(cfg), meta)
    run_id = full[len(project) + 1:] if project and full.startswith(project + "/") else full
    run = Run(id=run_id, name=name or run_id, project=project, config=cfg, mode=mode, server="", token=None, anon=False, root=root,
              meta=client_meta, start_step=start_step, tags=tags, notes=notes, group=group, job_type=job_type, local=lrun)
    _active.append(run)
    _set_globals(run)
    at = f" (resumed at step {start_step})" if start_step else ""
    say(f"run {full} -> {_shown(lrun.dir)}{at} (view: modgrad-server --root {_shown(root)})")
    return run


def _register_fork_guard() -> None:
    if hasattr(os, "register_at_fork"):
        def child() -> None:
            # DataLoader workers and other forks must not write the parent's run
            _active.clear()
            import modgrad

            modgrad.run = None

        os.register_at_fork(after_in_child=child)


_register_fork_guard()
