"""`.modgrad.json` (written by `modgrad init`): below arguments and env, above defaults."""

from __future__ import annotations

import json
import urllib.request

import modgrad
from modgrad import _project


def _anon(server):
    req = urllib.request.Request(server.url + "/api/v1/projects", method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _write(d, **v):
    (d / ".modgrad.json").write_text(json.dumps(v))


def test_project_file_supplies_server_and_token(server, tmp_path, monkeypatch):
    p = _anon(server)
    _write(tmp_path, server=server.url + "/", project=p["project"], token=p["token"])
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)  # found from a subdirectory
    monkeypatch.delenv("MODGRAD_URL")
    run = modgrad.init(name="from-file")
    assert run.server == server.url and run.project == p["project"]
    modgrad.log({"loss": 0.5})
    assert run.flush(10)
    path = run.path
    modgrad.finish()
    assert path.startswith(p["project"] + "/")
    assert server.runs(p["project"])[path]["finished"]


def test_env_beats_project_file_and_token_stays_with_its_server(server, tmp_path, monkeypatch):
    _write(tmp_path, server="http://127.0.0.1:9", project="anon-elsewhere", token="obsp_not_for_you")
    monkeypatch.chdir(tmp_path)
    run = modgrad.init(project="envwins", name="e")  # MODGRAD_URL (from the fixture) wins
    assert run.server == server.url
    assert run._spool.meta.get("token") != "obsp_not_for_you"
    modgrad.log({"loss": 1.0})
    path = run.path
    modgrad.finish()
    assert path in server.runs("envwins")


def test_find_and_token_for(tmp_path):
    _write(tmp_path, data="obs")
    assert _project.find(tmp_path / "missing" / "..")["data"] == "obs"
    pf = {"server": "http://a:1/", "token": "obsp_x"}
    assert _project.token_for(pf, "http://a:1") == "obsp_x"
    assert _project.token_for(pf, "http://b:1") is None
    assert _project.token_for({"data": "obs"}, "http://a:1") is None
    (tmp_path / ".modgrad.json").write_text("{broken")
    assert _project.find(tmp_path) == {}
