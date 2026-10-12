"""Local mode: runs written as native run directories, read by the Rust tools."""

from __future__ import annotations

import base64
import json
import math
import re
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import modgrad
from conftest import OBS_BIN, free_port, subprocess_env


def local_project(tmp_path: Path, data: str = "obs") -> Path:
    """What `modgrad init` writes in local mode (the tests run in tmp_path)."""
    (tmp_path / ".modgrad.json").write_text(json.dumps({"note": "written by `modgrad init`", "data": data}))
    return tmp_path / data


def lines(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines()]


def obs(*args: str, **kw):
    if not OBS_BIN.exists():
        pytest.skip(f"{OBS_BIN} not built (cargo build -p modgrad-obs)")
    out = subprocess.run([str(OBS_BIN), *args], capture_output=True, text=True, timeout=60, env=subprocess_env(), **kw)
    assert out.returncode == 0, out.stderr
    return out.stdout


def show(d: Path) -> dict:
    """`modgrad show <dir>` parsed: name, last step, counts, skipped."""
    first = obs("show", str(d)).splitlines()[0]
    return {k: int(re.search(rf"\b{k} (\d+)", first).group(1)) for k in ("rows", "arrays", "events", "skipped")}


def test_round_trip_through_the_rust_tools(tmp_path, capsys):
    data = local_project(tmp_path)
    run = modgrad.init(name="exp1", config={"lr": 3e-4, "layers": 2})
    assert run.local and run.offline and run.url is None and run.flush()
    d = data / "runs" / "exp1"
    assert Path(run.dir) == d
    assert "modgrad: run exp1 -> obs/runs/exp1 (view: modgrad-server --root obs)" in capsys.readouterr().err
    for i in range(5):
        modgrad.log({"loss": 1.0 / (i + 1), "gap": math.nan, "train": {"acc": i / 10}, "w": modgrad.Histogram([0.0, 0.5, 1.0, 1.0], num_bins=4)})
    modgrad.log({"act": modgrad.Tensor([[1.0, 2.0], [3.0, 4.0]])}, step=7)
    modgrad.log({"samples": modgrad.Table(columns=["p", "c"], data=[["2+2=", "4"]]), "note": "text"}, step=7)
    modgrad.config.update({"batch": 8})
    modgrad.summary["best"] = 0.2
    modgrad.alert("loss spike", "at step 3", level=modgrad.AlertLevel.WARN)
    modgrad.finish(exit_code=3)

    # the exact on-disk shape of modgrad_obs::Run
    assert sorted(p.name for p in d.iterdir()) == ["arrays.jsonl", "config.json", "events.jsonl", "metrics.jsonl", "summary.json"]
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["name"] == "exp1" and cfg["config"] == {"lr": 3e-4, "layers": 2, "batch": 8}
    assert set(cfg["meta"]) == {"git", "hostname", "start_unix", "extra"} and cfg["meta"]["extra"]["client"].startswith("modgrad-python")
    m = lines(d / "metrics.jsonl")
    assert [r["step"] for r in m] == [0, 1, 2, 3, 4]
    assert m[1] == {"step": 1, "t": m[1]["t"], "loss": 0.5, "gap": None, "train/acc": 0.1}
    assert list(m[0]) == sorted(m[0])  # serde_json's key order
    a = lines(d / "arrays.jsonl")
    h = a[0]
    assert (h["kind"], h["key"], h["shape"], h["dtype"], h["edges"]) == ("histogram", "w", [4], "f32le", [0.0, 0.25, 0.5, 0.75, 1.0])
    assert struct.unpack("<4f", base64.b64decode(h["b64"])) == (1.0, 0.0, 1.0, 2.0)
    t = a[-1]
    assert (t["kind"], t["key"], t["shape"], t["step"]) == ("tensor", "act", [2, 2], 7) and "edges" not in t
    assert struct.unpack("<4f", base64.b64decode(t["b64"])) == (1.0, 2.0, 3.0, 4.0)
    ev = [e["text"] for e in lines(d / "events.jsonl")]
    assert ev[0].startswith('table samples {"columns": ["p", "c"]') and "note: text" in ev and ev[-1] == "alert[WARN] loss spike: at step 3"
    sm = json.loads((d / "summary.json").read_text())
    assert sm["finished"] is True and sm["summary"]["exit_code"] == 3 and sm["summary"]["best"] == 0.2
    assert sm["summary"]["loss"] == 0.2 and sm["summary"]["gap"] is None
    assert sm["end_unix"] >= cfg["meta"]["start_unix"]

    # the Rust readers, unchanged
    assert show(d) == {"rows": 5, "arrays": 6, "events": 3, "skipped": 0}
    tj = json.loads(obs("tail", "exp1", "--json", cwd=tmp_path))  # a run id, resolved through .modgrad.json
    assert tj["step"] == 4 and tj["keys"]["loss"]["last"] == 0.2 and tj["keys"]["train/acc"]["n"] == 5
    assert json.loads(obs("tail", str(d), "--json"))["rows"] == 5
    obs("export", str(data))
    idx = json.loads((data / "index.json").read_text())
    e = next(r for r in idx["runs"] if r["id"] == "exp1")
    assert e["last_step"] == 7 and e["config"]["batch"] == 8 and not idx["errors"]


def test_matches_what_the_rust_writer_writes(tmp_path):
    """Same keys, same layout as a run `modgrad run --offline` writes."""
    data = local_project(tmp_path)
    obs("run", "--name", "rs", "--", sys.executable, "-c", "print('step 1: loss 0.5 acc 0.25', flush=True)", cwd=tmp_path)
    modgrad.init(name="py")
    modgrad.log({"loss": 0.5, "acc": 0.25}, step=1)
    modgrad.finish()
    rs, py = data / "runs" / "rs", data / "runs" / "py"
    assert {p.name for p in rs.iterdir() if not p.name.startswith(".")} >= {p.name for p in py.iterdir()}
    r_cfg, p_cfg = json.loads((rs / "config.json").read_text()), json.loads((py / "config.json").read_text())
    assert list(r_cfg) == list(p_cfg) and list(r_cfg["meta"]) == list(p_cfg["meta"])
    r_m, p_m = (rs / "metrics.jsonl").read_text().splitlines(), (py / "metrics.jsonl").read_text().splitlines()
    strip_t = lambda s: {k: v for k, v in json.loads(s).items() if k != "t"}
    assert strip_t(r_m[0]) == strip_t(p_m[0])
    assert [k for k in json.loads(r_m[0])] == [k for k in json.loads(p_m[0])]
    r_s, p_s = json.loads((rs / "summary.json").read_text()), json.loads((py / "summary.json").read_text())
    assert set(r_s) == set(p_s) and r_s["summary"]["loss"] == p_s["summary"]["loss"] == 0.5


def test_where_runs_go(tmp_path, monkeypatch):
    data = local_project(tmp_path)
    r = modgrad.init(project="sweep", name="lr3e-4_s1")
    assert Path(r.dir) == data / "runs" / "sweep" / "lr3e-4_s1" and r.path == "sweep/lr3e-4_s1"
    modgrad.finish()
    assert json.loads((Path(r.dir) / "config.json").read_text())["meta"]["project"] == "sweep"
    # a taken name is never written into twice
    r2 = modgrad.init(project="sweep", name="lr3e-4_s1")
    assert r2.dir != r.dir and Path(r2.dir).name.startswith("lr3e-4_s1-")
    modgrad.finish()
    # '/' in a name nests, like `modgrad run --name sweep/a`
    r3 = modgrad.init(name="abl/wd_s1")
    assert Path(r3.dir) == data / "runs" / "abl" / "wd_s1" and r3.path == "abl/wd_s1"
    modgrad.finish()
    # no name: a fresh id
    r4 = modgrad.init()
    assert Path(r4.dir).parent == data / "runs" and r4.id.startswith("run-")
    modgrad.finish()
    # the data dir is relative to the project file, not the working directory
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    r5 = modgrad.init(name="deep")
    assert Path(r5.dir) == data / "runs" / "deep"
    modgrad.finish()


def test_resolution_args_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODGRAD_FINISH_TIMEOUT", "0.2")
    dead = f"http://127.0.0.1:{free_port()}"
    local_project(tmp_path)
    # project file: local
    assert modgrad.init(name="f").local
    modgrad.finish()
    # environment beats the file
    monkeypatch.setenv("MODGRAD_URL", dead)
    r = modgrad.init(name="e")
    assert not r.local and r.server == dead
    modgrad.finish(quiet=True)
    # an argument beats the environment
    r = modgrad.init(name="a", data=str(tmp_path / "elsewhere"))
    assert r.local and Path(r.dir) == tmp_path / "elsewhere" / "runs" / "a"
    modgrad.finish()
    monkeypatch.delenv("MODGRAD_URL")
    r = modgrad.init(name="u", url=dead)
    assert not r.local
    modgrad.finish(quiet=True)
    # MODGRAD_DATA beats a project file that names a server
    (tmp_path / ".modgrad.json").write_text(json.dumps({"server": dead, "project": "p", "token": "obsp_x"}))
    assert not modgrad.init(name="s").local
    modgrad.finish(quiet=True)
    monkeypatch.setenv("MODGRAD_DATA", str(tmp_path / "envdata"))
    r = modgrad.init(name="d")
    assert r.local and Path(r.dir) == tmp_path / "envdata" / "runs" / "d"
    modgrad.finish()
    # disabled stays disabled
    r = modgrad.init(mode="disabled")
    assert r.dir is None and not r.local
    modgrad.finish()


def test_no_project_file_keeps_the_default_server(tmp_path, monkeypatch):
    monkeypatch.setenv("MODGRAD_FINISH_TIMEOUT", "0.2")
    r = modgrad.init(name="x")
    assert not r.local and r.server == modgrad.DEFAULT_URL
    modgrad.finish(quiet=True)


def test_offline_mode_writes_locally(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "offline")
    r = modgrad.init(project="offl", name="later", config={"lr": 1})
    assert r.offline and r.local and Path(r.dir) == tmp_path / "obs" / "runs" / "offl" / "later"
    modgrad.log({"loss": 1.0})
    modgrad.finish()
    assert "-> obs/runs/offl/later (view: modgrad-server --root obs)" in capsys.readouterr().err
    assert json.loads(obs("tail", r.dir, "--json"))["keys"]["loss"]["last"] == 1.0


def test_fork_from_records_the_parent(tmp_path):
    local_project(tmp_path)
    r = modgrad.init(id="sft", config={"lr": 2e-5, "stage": "sft"}, fork_from="pretrain?_step=3000")
    modgrad.log({"loss": 1.0})
    modgrad.finish()
    cfg = json.loads((Path(r.dir) / "config.json").read_text())["config"]
    assert cfg["parent_run"] == "pretrain" and cfg["parent_step"] == 3000 and cfg["stage"] == "sft"
    # config wins over fork_from; "<run>@<step>" works too, and no step means the parent's last
    assert modgrad.parse_parent("a/b@12.5") == {"parent_run": "a/b", "parent_step": 12.5}
    assert modgrad.parse_parent("a") == {"parent_run": "a"}
    r = modgrad.init(id="rl", config={"parent_step": 7}, fork_from="sft@9")
    modgrad.finish()
    cfg = json.loads((Path(r.dir) / "config.json").read_text())["config"]
    assert cfg["parent_run"] == "sft" and cfg["parent_step"] == 7


def test_resume_continues_after_the_last_step_arrays_count(tmp_path):
    local_project(tmp_path)
    modgrad.init(id="res", config={"lr": 1})
    for i in range(3):
        modgrad.log({"x": float(i)})
    modgrad.log({"w": modgrad.Tensor([1.0, 2.0])}, step=5)  # an array-only step
    modgrad.summary["best"] = 0.5
    modgrad.finish()
    t_before = lines(tmp_path / "obs" / "runs" / "res" / "metrics.jsonl")[-1]["t"]
    r = modgrad.init(id="res", resume=True, config={"bs": 2})
    assert r.step == 6
    modgrad.log({"x": 6.0})
    modgrad.finish()
    d = Path(r.dir)
    m = lines(d / "metrics.jsonl")
    assert [x["step"] for x in m] == [0, 1, 2, 6] and m[-1]["t"] >= t_before
    assert json.loads((d / "config.json").read_text())["config"] == {"lr": 1, "bs": 2}
    sm = json.loads((d / "summary.json").read_text())["summary"]
    assert sm["best"] == 0.5 and sm["x"] == 6.0
    # resume by name works too; resume="must" needs the run to exist
    r = modgrad.init(name="res", resume="allow")
    assert r.step == 7 and r.dir == str(d)
    modgrad.finish()
    with pytest.raises(modgrad.Error):
        modgrad.init(id="nope", resume="must")


KILLED = """
import itertools, modgrad as wandb
wandb.init(id="victim", config={"lr": 0.1})
for i in itertools.count():
    wandb.log({"loss": 1.0 / (i + 1), "h": wandb.Histogram([i, i + 1, i + 2], num_bins=3)})
"""


def test_killed_process_leaves_a_readable_run(tmp_path):
    data = local_project(tmp_path)
    script = tmp_path / "train.py"
    script.write_text(KILLED)
    p = subprocess.Popen([sys.executable, str(script)], cwd=tmp_path, env=subprocess_env(), stderr=subprocess.DEVNULL)
    d = data / "runs" / "victim"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if (d / "metrics.jsonl").exists() and (d / "metrics.jsonl").stat().st_size > 20_000:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the script never logged")
    finally:
        p.kill()  # SIGKILL: no atexit, no finish
        p.wait()
    assert not (d / "summary.json").exists() or not json.loads((d / "summary.json").read_text()).get("finished")
    s = show(d)
    assert s["rows"] > 100 and s["skipped"] <= 2  # at most one torn line per file
    tail = json.loads(obs("tail", str(d), "--json"))
    assert tail["rows"] == s["rows"]
    # resuming cuts any torn tail and continues after the last logged step
    m = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.endswith("}")]
    r = modgrad.init(id="victim", resume=True)
    assert r.step >= m[-1]["step"] + 1
    modgrad.log({"loss": 0.0})
    modgrad.finish()
    assert show(d)["skipped"] == 0
    assert json.loads((d / "summary.json").read_text())["finished"] is True


def test_exception_exit_code_in_local_mode(tmp_path):
    data = local_project(tmp_path)
    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent('''
        import modgrad as wandb
        wandb.init(name="boom")
        wandb.log({"loss": 1.0})
        raise RuntimeError("diverged")
    '''))
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=60, env=subprocess_env())
    assert out.returncode == 1 and "diverged" in out.stderr and "modgrad: run boom -> obs/runs/boom" in out.stderr
    sm = json.loads((data / "runs" / "boom" / "summary.json").read_text())
    assert sm["finished"] and sm["summary"]["exit_code"] == 1


def test_hf_callback_local(tmp_path):
    from modgrad.integrations.huggingface import ModgradCallback

    data = local_project(tmp_path)
    cb = ModgradCallback(project="hf", name="sft")
    args = SimpleNamespace(to_dict=lambda: {"learning_rate": 5e-5})
    state = SimpleNamespace(is_world_process_zero=True, global_step=0)
    cb.on_train_begin(args, state, None, model=None)
    for step in (2, 4):
        state.global_step = step
        cb.on_log(args, state, None, logs={"loss": 1.0 / step, "epoch": step / 4})
    cb.on_log(args, state, None, logs={"train_runtime": 1.5})
    cb.on_train_end(args, state, None)
    modgrad.finish()
    d = data / "runs" / "hf" / "sft"
    assert [r["step"] for r in lines(d / "metrics.jsonl")] == [2, 4]
    assert lines(d / "metrics.jsonl")[0]["train/loss"] == 0.5
    assert json.loads((d / "config.json").read_text())["config"]["learning_rate"] == 5e-5
    assert json.loads((d / "summary.json").read_text())["summary"]["train_runtime"] == 1.5


def test_lightning_logger_local(tmp_path):
    from modgrad.integrations.lightning import ModgradLogger

    data = local_project(tmp_path)
    lg = ModgradLogger(project="pl", name="tiny")
    lg.log_hyperparams({"lr": 0.1, "opt": {"name": "sgd"}})
    for s in range(3):
        lg.log_metrics({"train_loss": 1.0 / (s + 1)}, step=s)
    lg.finalize("success")
    d = Path(lg.experiment.dir)
    assert d.parent == data / "runs" / "pl" and d.name == lg.version
    lg.finish()
    assert [r["train_loss"] for r in lines(d / "metrics.jsonl")] == [1.0, 0.5, pytest.approx(1 / 3)]
    assert json.loads((d / "summary.json").read_text())["finished"] is True


def test_a_client_under_modgrad_run_logs_into_its_run(tmp_path, monkeypatch):
    """`modgrad run` around a script that itself uses the client: one run (the
    wrapper's), not a second run nested inside the wrapper's directory."""
    monkeypatch.chdir(tmp_path)
    data = local_project(tmp_path)
    (tmp_path / "train.py").write_text(textwrap.dedent("""
        import modgrad as wandb
        run = wandb.init(project="spirals", config={"lr": 0.01})
        for step in range(10, 60, 10):
            wandb.log({"train/loss": 1.0 / step}, step=step)
        run.finish()
    """))
    obs("run", "--name", "spirals", "--sys-interval", "0", "--", sys.executable, "train.py", cwd=tmp_path)
    d = data / "runs" / "spirals"
    assert [p.name for p in d.iterdir() if p.is_dir()] == []
    loss = [r["train/loss"] for r in lines(d / "metrics.jsonl") if "train/loss" in r]
    assert loss == [1.0 / s for s in range(10, 60, 10)]
    assert '"lr"' in (d / "config.json").read_text()
