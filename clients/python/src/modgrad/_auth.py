"""Local token cache: the anonymous project per server, and run tokens
(needed to resume a run created without a token)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import _net
from ._sender import say


def config_dir() -> Path:
    env = os.environ.get("MODGRAD_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "modgrad"


def _load(name: str) -> Dict[str, Any]:
    try:
        v = json.loads((config_dir() / name).read_text())
        return v if isinstance(v, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _store(name: str, v: Dict[str, Any]) -> None:
    try:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        tmp = p.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(v, indent=2).encode())
        finally:
            os.close(fd)
        os.replace(tmp, p)
    except OSError:
        pass


def anon_project(server: str, fresh: bool = False) -> Optional[Dict[str, Any]]:
    """The cached anonymous project for `server`, or a new one
    (POST /api/v1/projects). Prints the board link when one is created."""
    server = server.rstrip("/")
    cache = _load("anon.json")
    p = cache.get(server)
    if not fresh and isinstance(p, dict) and p.get("token") and float(p.get("expires_unix") or 0) > time.time() + 3600:
        return p
    r = _net.post(server, "/api/v1/projects", None, None, timeout=10)
    if not r.ok:
        raise _net_error(r)
    p = r.json() or {}
    if not p.get("token"):
        return None
    cache[server] = {k: p.get(k) for k in ("project", "token", "board", "claim", "expires_unix")}
    _store("anon.json", cache)
    say(f"anonymous project {p.get('project')} created; dashboard: {p.get('board')}")
    if p.get("claim"):
        say(f"keep it beyond 30 days by claiming it: {p.get('claim')}")
    return cache[server]


def forget_anon(server: str) -> None:
    cache = _load("anon.json")
    if cache.pop(server.rstrip("/"), None) is not None:
        _store("anon.json", cache)


class HTTPStatusError(Exception):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text}")
        self.status = status


def _net_error(r: "_net.Resp") -> HTTPStatusError:
    return HTTPStatusError(r.status, r.error_text())


def run_token(server: str, run: str) -> Optional[str]:
    return _load("runs.json").get(f"{server.rstrip('/')} {run}")


def save_run_token(server: str, run: str, token: str) -> None:
    v = _load("runs.json")
    v[f"{server.rstrip('/')} {run}"] = token
    _store("runs.json", v)
