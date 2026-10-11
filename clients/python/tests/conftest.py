"""Fixtures: a live obs-server (built binary, --dev, free port) and a clean
modgrad environment per test."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SERVER_BIN = Path(os.environ.get("OBS_SERVER_BIN", REPO / "target" / "release" / "obs-server"))
OBS_BIN = Path(os.environ.get("OBS_BIN", REPO / "target" / "release" / "obs"))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, root: Path, port: int):
        self.root = root
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None

    def start(self, timeout: float = 15.0) -> "Server":
        if not SERVER_BIN.exists():
            pytest.skip(f"{SERVER_BIN} not built (cargo build --release -p modgrad-obs-server)")
        self.log = open(self.root / "server.log", "ab")
        self.proc = subprocess.Popen([str(SERVER_BIN), "--root", str(self.root), "--listen", f"127.0.0.1:{self.port}", "--dev", "--no-static"],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        end = time.time() + timeout
        while time.time() < end:
            try:
                urllib.request.urlopen(self.url + "/api/index", timeout=1).read()
                return self
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"obs-server exited: {(self.root / 'server.log').read_text()[-2000:]}")
                time.sleep(0.05)
        raise RuntimeError("obs-server did not come up")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=5) as r:
            return json.loads(r.read())

    def runs(self, project: str = None):
        return {r["id"]: r for r in self.get("/api/index" + (f"?project={project}" if project else ""))["runs"]}

    def history(self, run: str):
        return self.get(f"/api/v1/runs/{run}/history")

    def config(self, run: str):
        return self.get(f"/api/v1/runs/{run}/config")["config"]

    def events(self, run: str):
        return self.get(f"/api/v1/runs/{run}/events")


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    s = Server(tmp_path_factory.mktemp("obs-root"), free_port()).start()
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def modgrad_env(tmp_path, monkeypatch, request):
    for k in list(os.environ):
        if k.startswith(("WANDB_", "MODGRAD_", "OBS_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODGRAD_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("MODGRAD_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("MODGRAD_RETRY_MAX", "0.5")
    monkeypatch.setenv("MODGRAD_FINISH_TIMEOUT", "15")
    if "server" in request.fixturenames:
        monkeypatch.setenv("MODGRAD_URL", request.getfixturevalue("server").url)
    yield
    import modgrad
    from modgrad import _run

    for r in list(_run._active):
        r.finish(quiet=True)
    _run._exit_code[0] = 0


def subprocess_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items()}
    env.update(extra)
    return env
