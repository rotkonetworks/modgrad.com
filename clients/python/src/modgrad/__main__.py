"""`python -m modgrad sync [DIR ...]`: send spooled or offline runs.

Without DIR, every run under the spool root is sent. Rows, events, config
and summary go to the server named in each run's meta.json (override with
--url / MODGRAD_URL); finished runs are marked finished and their spool
directory is removed once everything is acknowledged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ._run import _AuthPolicy
from ._sender import Sender, say
from ._spool import Spool, SpoolBusy, default_root


def sync_dir(d: Path, url: str = None, timeout: float = 60.0) -> bool:
    try:
        sp = Spool(d)
    except SpoolBusy:
        say(f"{d}: skipped, a running process is still writing it")
        return True
    m = sp.meta
    if url:
        m["server"] = url.rstrip("/")
    if not m.get("server") or not m.get("name"):
        say(f"{d}: not a spool directory (meta.json has no server/name)")
        return False
    n = sp.pending()
    anon = not m.get("token") and not m.get("project")
    s = Sender(sp, interval=0.05, retry_max=2.0, auth=_AuthPolicy(m["server"], anon))
    s.start()
    finish = m.get("finish") or {"pushed_later": True}
    ok = s.close(finish, timeout)
    if ok:
        sp.remove()
        say(f"sent {n} items -> {m.get('dashboard')}")
    else:
        sp.close()
        say(f"{sp.pending()} items still unsent; spool kept in {d}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m modgrad", description="modgrad client tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="send offline / spooled runs to the server")
    s.add_argument("dirs", nargs="*", help="spool directories (default: all under the spool root)")
    s.add_argument("--url", default=os.environ.get("MODGRAD_URL") or os.environ.get("OBS_URL"), help="server URL override")
    s.add_argument("--timeout", type=float, default=60.0)
    sub.add_parser("where", help="print the spool root")
    a = ap.parse_args(argv)
    if a.cmd == "where":
        print(default_root())
        return 0
    dirs = [Path(x) for x in a.dirs] or sorted(p for p in default_root().glob("*") if (p / "meta.json").is_file())
    if not dirs:
        say(f"nothing to sync under {default_root()}")
        return 0
    ok = all([sync_dir(d, a.url, a.timeout) for d in dirs])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
