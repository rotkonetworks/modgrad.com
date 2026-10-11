"""The W&B-compatible surface against a live dev server."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import modgrad
from conftest import OBS_BIN, Server, free_port, subprocess_env


def test_init_log_finish(server):
    run = modgrad.init(project="core", name="basic run", config={"lr": 3e-4, "layers": 2}, tags=["a"])
    assert modgrad.run is run and modgrad.config.lr == 3e-4 and modgrad.config["layers"] == 2
    modgrad.config.update({"batch": 32})
    modgrad.config.warmup = 10
    for i in range(10):
        modgrad.log({"loss": 1.0 / (i + 1), "train": {"acc": i / 10}, "gap": float("nan"), "flag": True})
    modgrad.log({"hist": modgrad.Histogram([1, 2, 2, 3], num_bins=3)})
    modgrad.summary["best_loss"] = 0.1
    modgrad.alert("done", "ten steps")
    run_id = run.path
    modgrad.finish()
    assert modgrad.run is None

    e = server.runs("core")[run_id]
    assert e["finished"] and e["last_step"] == 9  # last_step counts scalar rows only
    assert server.config(run_id) == {"lr": 3e-4, "layers": 2, "batch": 32, "warmup": 10}
    assert e["summary"]["best_loss"] == 0.1 and e["summary"]["exit_code"] == 0
    assert e["arrays"]["hist"]["kind"] == "histogram"
    h = server.history(run_id)
    assert h["steps"][:10] == list(range(10))
    assert h["columns"]["loss"][:3] == [1.0, 0.5, pytest.approx(1 / 3)]
    assert h["columns"]["train/acc"][9] == pytest.approx(0.9)
    assert all(v is None for v in h["columns"]["gap"][:10])
    assert h["columns"]["flag"][0] == 1.0
    assert any(ev["text"] == "alert[INFO] done: ten steps" for ev in server.events(run_id))
    # the spool is gone once everything is acknowledged
    assert not Path(run.dir).exists()


def test_step_semantics(server):
    run = modgrad.init(project="core", name="steps")
    modgrad.log({"a": 1}, commit=False)
    modgrad.log({"b": 2})  # same step as a, then advance
    modgrad.log({"a": 3}, step=10)
    modgrad.log({"b": 4})  # explicit step does not commit: still step 10
    modgrad.log({"a": 5})
    assert run.step == 12
    path = run.path
    modgrad.finish()
    h = server.history(path)
    assert h["steps"] == [0, 10, 11]
    assert h["columns"]["a"] == [1.0, 3.0, 5.0] and h["columns"]["b"] == [2.0, 4.0, None]


def test_numpy_table_text(server):
    np = pytest.importorskip("numpy")
    run = modgrad.init(project="core", name="types")
    t = modgrad.Table(columns=["x", "y"])
    t.add_data(1, "one")
    modgrad.log({"w": np.random.randn(500), "s": np.float32(2.5), "only_text": "hello", "tbl": t,
                 "img": modgrad.Image(np.zeros((4, 4))), "tensor": modgrad.Tensor([[1, 2], [3, 4]])})
    path = run.path
    modgrad.finish()
    e = server.runs("core")[path]
    assert e["arrays"]["w"]["kind"] == "histogram" and e["arrays"]["tensor"]["shape"] == [2, 2]
    assert e["arrays"]["img"]["shape"] == [4, 4]
    assert server.history(path)["columns"]["s"] == [2.5]
    texts = [ev["text"] for ev in server.events(path)]
    assert "only_text: hello" in texts
    tbl = next(x for x in texts if x.startswith("table tbl "))
    assert json.loads(tbl[len("table tbl "):]) == {"columns": ["x", "y"], "data": [[1, "one"]]}


def test_text_only_row_and_rejected_item_do_not_wedge(server):
    run = modgrad.init(project="core", name="wedge")
    modgrad.log({"note": "only text"})  # becomes an event, not an empty row
    run._spool.append([("row", {"step": -5, "x": 1})])  # the server rejects this (400)
    modgrad.log({"x": 2})
    assert run.flush(10)
    path = run.path
    modgrad.finish()
    assert server.history(path)["columns"]["x"] == [2.0]


def test_offline_then_obs_push(tmp_path, server, monkeypatch):
    monkeypatch.setenv("MODGRAD_MODE", "offline")
    run = modgrad.init(project="offl", name="later", config={"lr": 1})
    for i in range(5):
        modgrad.log({"loss": float(i)})
    d = Path(run.dir)
    modgrad.finish()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["finish"] == {"exit_code": 0} and meta["name"] == run.id and meta["project"] == "offl"
    assert "offl/" + run.id not in server.runs("offl")
    if not OBS_BIN.exists():
        pytest.skip("obs CLI not built")
    out = subprocess.run([str(OBS_BIN), "push", str(d), "--json"], capture_output=True, text=True, timeout=60,
                         env=subprocess_env(MODGRAD_URL=server.url))
    assert out.returncode == 0, out.stderr
    e = server.runs("offl")["offl/" + run.id]
    assert e["finished"] and e["last_step"] == 4
    assert server.history("offl/" + run.id)["columns"]["loss"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert not d.exists()


def test_offline_then_python_sync(tmp_path, server, monkeypatch):
    run = modgrad.init(project="offl", name="pysync", config={"lr": 1}, mode="offline")
    modgrad.log({"loss": 1.0})
    modgrad.summary["best"] = 1.0
    d = Path(run.dir)
    modgrad.finish()
    out = subprocess.run([sys.executable, "-m", "modgrad", "sync", str(d)], capture_output=True, text=True, timeout=60, env=subprocess_env())
    assert out.returncode == 0, out.stderr
    e = server.runs("offl")["offl/" + run.id]
    assert e["finished"] and server.config("offl/" + run.id) == {"lr": 1} and e["summary"]["best"] == 1.0


def test_server_down_then_up(tmp_path, monkeypatch):
    port = free_port()
    srv = Server(tmp_path / "late-root", port)
    srv.root.mkdir()
    monkeypatch.setenv("MODGRAD_URL", srv.url)
    run = modgrad.init(project="late", name="spooled")
    for i in range(20):
        modgrad.log({"loss": float(i)})
    time.sleep(1.5)  # the sender tries, fails, backs off
    assert run._spool.pending() >= 20 and run.url is None
    srv.start()
    try:
        assert run.flush(20), "spooled rows did not reach the server"
        for i in range(20, 25):
            modgrad.log({"loss": float(i)})
        path = run.path
        modgrad.finish()
        assert srv.runs("late")[path]["finished"]
        assert srv.history(path)["columns"]["loss"] == [float(i) for i in range(25)]
    finally:
        srv.stop()


def test_server_never_up_keeps_spool(tmp_path, monkeypatch):
    monkeypatch.setenv("MODGRAD_URL", f"http://127.0.0.1:{free_port()}")
    monkeypatch.setenv("MODGRAD_FINISH_TIMEOUT", "1")
    run = modgrad.init(project="gone", name="lost")
    modgrad.log({"loss": 1.0})
    d = Path(run.dir)
    t0 = time.time()
    modgrad.finish()  # must not raise or hang past the deadline
    assert time.time() - t0 < 6
    lines = (d / "spool.jsonl").read_text().splitlines()
    assert any(json.loads(l).get("row", {}).get("loss") == 1.0 for l in lines)
    assert json.loads((d / "meta.json").read_text())["finish"] == {"exit_code": 0}


QUICKSTART = '''
import wandb
import random

wandb.login()

# W&B quickstart (https://docs.wandb.ai/quickstart), unchanged below the import
epochs = 10
lr = 0.01

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
wandb.finish()
print("RUN", run.path)
'''


def test_wandb_quickstart_unchanged(tmp_path, server):
    script = tmp_path / "quickstart.py"
    script.write_text(QUICKSTART.replace("import wandb\n", "import modgrad as wandb\n", 1))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=subprocess_env())
    assert out.returncode == 0, out.stderr
    path = next(l.split()[1] for l in out.stdout.splitlines() if l.startswith("RUN "))
    e = server.runs("my-awesome-project")[path]
    assert e["finished"] and server.config(path) == {"learning_rate": 0.01, "epochs": 10}
    h = server.history(path)
    assert h["steps"] == list(range(8)) and len(h["columns"]["accuracy"]) == 8


def test_exception_sets_exit_code_and_finishes(tmp_path, server):
    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent('''
        import modgrad as wandb
        run = wandb.init(project="crash", name="boom")
        print("RUN", run.path, flush=True)
        wandb.log({"loss": 1.0})
        raise RuntimeError("diverged")
    '''))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=subprocess_env())
    assert out.returncode == 1 and "diverged" in out.stderr
    path = next(l.split()[1] for l in out.stdout.splitlines() if l.startswith("RUN "))
    e = server.runs("crash")[path]
    assert e["finished"] and e["summary"]["exit_code"] == 1


def test_anonymous_project(server, tmp_path):
    a = modgrad.init(name="anon one")
    modgrad.log({"x": 1.0})
    assert a.flush(10)
    pa = a.path
    modgrad.finish()
    project = pa.split("/")[0]
    assert project.startswith("anon-")
    assert pa in server.runs(project)
    assert pa not in server.runs()  # anonymous projects are not listed publicly
    b = modgrad.init(name="anon two")  # the cached project is reused
    modgrad.log({"x": 2.0})
    assert b.flush(10)
    assert b.path.split("/")[0] == project
    modgrad.finish()


def test_resume_continues_steps(server):
    r = modgrad.init(project="res", id="resumable")
    for i in range(3):
        modgrad.log({"x": float(i)})
    modgrad.finish()
    r2 = modgrad.init(project="res", id="resumable", resume="allow")
    assert r2.step == 3
    modgrad.log({"x": 3.0})
    modgrad.finish()
    assert server.history("res/resumable")["steps"] == [0, 1, 2, 3]


def test_env_vars_and_modes(server, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "fromwandb")
    monkeypatch.setenv("WANDB_NAME", "named")
    r = modgrad.init()
    assert r.project == "fromwandb" and r.name == "named" and r.id.startswith("named-")
    modgrad.finish()
    monkeypatch.setenv("MODGRAD_PROJECT", "ours")
    r = modgrad.init()
    assert r.project == "ours"
    modgrad.finish()
    monkeypatch.setenv("WANDB_MODE", "disabled")
    r = modgrad.init(config={"a": 1})
    assert r.disabled and r.dir is None and modgrad.config.a == 1
    modgrad.log({"x": 1.0})
    modgrad.finish()
    monkeypatch.setenv("MODGRAD_MODE", "offline")  # ours wins over WANDB_MODE
    r = modgrad.init()
    assert r.offline
    modgrad.finish(quiet=True)


def test_misc_surface(server):
    assert modgrad.login() is True
    with pytest.raises(modgrad.Error):
        modgrad.log({"x": 1})
    with modgrad.init(project="core", name="ctx") as run:
        assert modgrad.init() is run  # no reinit: the active run comes back
        assert modgrad.define_metric("val/*", step_metric="epoch") == {"step_metric": "epoch"}
        assert run.metric_defs["val/*"]["step_metric"] == "epoch"
        assert modgrad.save("*.pt") == []
        run.log({"x": 1})
        path = run.path
    assert modgrad.run is None
    assert server.runs("core")[path]["finished"]
    r1 = modgrad.init(project="core", name="re1")
    r2 = modgrad.init(project="core", name="re2", reinit="finish_previous")
    assert r1 is not r2 and r1._finished and modgrad.run is r2
    modgrad.finish()
