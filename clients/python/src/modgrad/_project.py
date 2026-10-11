"""The project file `.modgrad.json` written by `modgrad init` (the Rust CLI).

It is looked up in the current directory and its parents. With a server it
holds `{server, project, token, ...}`; in local mode `{data}` only. The
client reads it below explicit arguments and environment variables, and a
token from the file is only sent to the server the file names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

FILE = ".modgrad.json"


def find(start: Optional[Path] = None) -> Dict[str, Any]:
    """The nearest project file's content (plus `_dir`), or {} if none."""
    try:
        d = Path(start or os.getcwd()).resolve()
    except OSError:
        return {}
    for p in [d, *d.parents]:
        f = p / FILE
        try:
            v = json.loads(f.read_text())
        except FileNotFoundError:
            continue
        except (ValueError, OSError) as e:
            from ._sender import say

            say(f"ignoring {f}: {e}")
            return {}
        if not isinstance(v, dict):
            return {}
        v["_dir"] = str(p)
        return v
    return {}


def token_for(pf: Dict[str, Any], server: str) -> Optional[str]:
    """The file's token, only if `server` is the server it names."""
    s = pf.get("server")
    if isinstance(s, str) and s.rstrip("/") == server.rstrip("/"):
        t = pf.get("token")
        return t if isinstance(t, str) and t else None
    return None
