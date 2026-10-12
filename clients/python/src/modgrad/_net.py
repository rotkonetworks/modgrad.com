"""Tiny HTTP client on urllib (no dependencies)."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any, Optional, Tuple

USER_AGENT = "modgrad-python/0.1.0"


class Resp:
    __slots__ = ("status", "body")

    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8") or "null")
        except ValueError:
            return None

    def error_text(self) -> str:
        j = self.json()
        if isinstance(j, dict) and j.get("error"):
            return str(j["error"])
        return self.body[:300].decode("utf-8", "replace")


def clean(v: Any) -> Any:
    """Make `v` strict-JSON safe: non-finite floats become None, recursively."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def dumps(v: Any) -> str:
    """Strict JSON (no NaN/Infinity, which the server's parser rejects)."""
    return json.dumps(clean(v), allow_nan=False, separators=(",", ":"), default=str)


def request(method: str, url: str, body: Any = None, token: Optional[str] = None, timeout: float = 10.0) -> Resp:
    """One request. Raises OSError (URLError, timeouts) on transport failure;
    HTTP errors come back as a Resp with their status."""
    data = None if body is None else dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", USER_AGENT)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Resp(r.status, r.read())
    except urllib.error.HTTPError as e:
        try:
            b = e.read()
        except Exception:
            b = b""
        return Resp(e.code, b)


def retryable(status: int) -> bool:
    return status in (408, 425, 429) or status >= 500


def post(server: str, path: str, body: Any = None, token: Optional[str] = None, timeout: float = 10.0) -> Resp:
    return request("POST", server.rstrip("/") + path, body if body is not None else {}, token, timeout)


def get(server: str, path: str, token: Optional[str] = None, timeout: float = 10.0) -> Tuple[int, Any]:
    r = request("GET", server.rstrip("/") + path, None, token, timeout)
    return r.status, r.json()
