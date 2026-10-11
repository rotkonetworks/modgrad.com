"""obs-agent-bench: can a local model answer training-run questions through `obs mcp`?

Each task gets a fresh data directory populated *through the MCP tools*
(log / ingest_log) from the fixed tables below, so the ground truth lives
here, independent of the server's own analysis. A task is a user prompt plus
a checker on the final answer (and, for write tasks, on the stored data).

Modes:
  --scripted   run each task's canonical tool sequence and derive the answer
               from the tool text with a regex. Validates the bench and the
               tool output; every task must pass before a model score means
               anything.
  (default)    an agent loop against an OpenAI-compatible endpoint (Ollama's
               /v1/chat/completions) with the MCP tools exposed as functions.

Output: a JSONL transcript per run (every tool call as the model typed it,
raw arguments, isError, turns, wall time) and a summary table.
"""

import argparse
import json
import math
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_client import McpClient  # noqa: E402
import hard_tasks  # noqa: E402

CANON = {"log", "ingest_log", "list_runs", "read_metric", "summarize", "compare", "make_board"}

# ------------------------------------------------------------------ fixture

STEPS = [1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
FIX = {
    "exp/small": {"heldout_nll": [8.99, 5.44, 5.07, 5.31, 5.12, 5.02, 4.97, 4.95, 4.91], "train_loss": [3.91, 3.62, 3.55, 3.5, 3.47, 3.45, 3.44, 3.43, 3.42]},
    "exp/small_s2": {"heldout_nll": [8.98, 5.45, 5.12, 5.0, 4.93], "train_loss": [3.9, 3.63, 3.56, 3.51, 3.48]},
    "exp/small_s3": {"heldout_nll": [8.98, 5.32, 5.2, 5.12, 5.44], "train_loss": [3.92, 3.6, 3.54, 3.52, 3.49]},
    "exp/mid": {"heldout_nll": [9.0, 5.98, 5.65, 5.27, 5.04, 4.84, 4.79], "train_loss": [3.95, 3.5, 3.38, 3.3, 3.25, 3.21, 3.19]},
    "exp/large": {"heldout_nll": [8.99, 5.75, 5.11, 4.98, 4.71], "train_loss": [3.96, 3.41, 3.27, 3.18, 3.12]},
}
LIN = [(i * 10, round(10.0 - 0.1 * i + 0.002 * ((i * 7) % 5), 4)) for i in range(40)]


def _noise():
    x, out = 12345, []
    for i in range(40):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append((i * 10, round(3.0 + ((x >> 33) / (1 << 31) - 0.5), 4)))
    return out


NOISY = _noise()
PASTE = """Loading tokenizer...
step 100 | loss 2.84 | lr 3e-4 | acc 41.5%
step 200 | loss 2.51 | lr 3e-4 | acc 47.0%
step 300 | loss 2.33 | lr 2.5e-4 | acc 52.25%
saving checkpoint ckpt_300.pt
step 400 | loss 2.29 | lr 2e-4 | acc 53.0%"""


def build_fixture(c):
    for run, ms in FIX.items():
        n = len(ms["heldout_nll"])
        text = "\n".join(f"step {STEPS[i]} heldout_nll={ms['heldout_nll'][i]} train_loss={ms['train_loss'][i]}" for i in range(n))
        _must(c.call("ingest_log", {"run": run, "text": text}))
    _must(c.call("ingest_log", {"run": "lin_decay", "text": "\n".join(f"step {s} train_loss={v}" for s, v in LIN)}))
    _must(c.call("ingest_log", {"run": "noisy", "text": "\n".join(f"step {s} train_loss={v}" for s, v in NOISY)}))
    _must(c.call("log", {"run": "pilot1", "metrics": {"learning_rate": 0.0003, "train_loss": 4.1}, "step": 0}))
    _must(c.call("log", {"run": "pilot1", "metrics": {"learning_rate": 0.00015, "train_loss": 3.7}, "step": 100}))


def _must(r):
    text, err = r
    if err:
        raise RuntimeError(f"fixture build failed: {text}")


# ------------------------------------------------------------ ground truth


def seed_stats():
    vals = [FIX[r]["heldout_nll"][4] for r in ("exp/small", "exp/small_s2", "exp/small_s3")]  # step 1000
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, sd


def lin_eta(target):
    # least squares on the clean line: 10 - 0.01*step
    return (10.0 - target) / 0.01


def latest(run, key):
    return FIX[run][key][-1]


# ------------------------------------------------------------------ checks

NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def nums(text):
    out = []
    for t in NUM.findall(text.replace(",", "") if text.count(",") > 3 else text):
        try:
            out.append(float(t.replace(",", ".")))
        except ValueError:
            pass
    return out


def has_num(text, want, rel=0.01, abs_=1e-9):
    return any(abs(x - want) <= max(abs_, rel * abs(want)) for x in nums(text))


def mentions(text, name):
    return re.search(r"(?<![\w/])" + re.escape(name) + r"(?![\w])", text) is not None


def only_run(text, want, others):
    """`want` is named and no other candidate is named as the answer."""
    return mentions(text, want) and not any(mentions(text, o) for o in others if o != want)


def stored(data_dir, run):
    """{key: {step: value}} read straight from the run's metrics.jsonl."""
    out = {}
    p = os.path.join(data_dir, "runs", run, "metrics.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        step = row.get("step", row.get("_step"))
        for k, v in row.items():
            if isinstance(v, (int, float)) and k not in ("step", "_step", "t", "_t"):
                out.setdefault(k, {})[step] = v
    return out


# -------------------------------------------------------------------- tasks
# script: canonical tool calls; extract(outputs) -> answer text a careful
# reader of those outputs would give; check(answer, data_dir) -> (ok, why).

def T(id, prompt, script, extract, check):
    return {"id": id, "prompt": prompt, "script": script, "extract": extract, "check": check}


def rx(pat, flags=0):
    def f(outs):
        m = re.search(pat, "\n".join(outs), flags)
        return m.group(0) if m else ""
    return f


m_seed, sd_seed = seed_stats()
SIZES = ["exp/small", "exp/mid", "exp/large"]

TASKS = [
    T("count_runs", "How many training runs are stored? Answer with the number.",
      [("list_runs", {})], rx(r"^\d+ runs", re.M),
      lambda a, d: (has_num(a, 8, 0), "want 8")),
    T("last_value", "What is the latest train_loss of run exp/mid, and at which step?",
      [("read_metric", {"run": "exp/mid", "metric": "train_loss"})], rx(r"last \S+ at step \d+"),
      lambda a, d: (has_num(a, 3.19) and has_num(a, 1500, 0), "want 3.19 at step 1500")),
    T("min_value", "What was the lowest heldout_nll reached by exp/small, and at which step?",
      [("read_metric", {"run": "exp/small", "metric": "heldout_nll"})], rx(r"min \S+ at step \d+"),
      lambda a, d: (has_num(a, 4.91) and has_num(a, 2000, 0), "want 4.91 at step 2000")),
    T("best_size", "Of exp/small, exp/mid and exp/large, which run has the lowest heldout_nll at its own latest step? Name one run.",
      [("compare", {"runs": "exp/small,exp/mid,exp/large", "metric": "heldout_nll"})], rx(r"lowest is \S+ = \S+"),
      lambda a, d: (only_run(a, "exp/large", SIZES) or (mentions(a, "large") and not mentions(a, "mid") and not re.search(r"\bsmall\b", a)), "want exp/large (4.71)")),
    T("best_at_shared", "Compare heldout_nll of exp/small and exp/mid at the latest step both have logged. Which is lower there, and by how much?",
      [("compare", {"runs": "exp/small,exp/mid", "metric": "heldout_nll"})], rx(r"At the latest step all runs share.*"),
      lambda a, d: _check_shared(a)),
    T("trend_improving", "Is train_loss in run lin_decay still improving? Answer yes or no and give the rate per 10 steps.",
      [("summarize", {"run": "lin_decay", "metric": "train_loss"})], rx(r"Trend.*"),
      lambda a, d: _check_improving(a)),
    T("trend_noise", "Is train_loss in run noisy still going down? Be honest about uncertainty.",
      [("summarize", {"run": "noisy", "metric": "train_loss"})], rx(r"no clear trend[^.\n]*"),
      lambda a, d: (re.search(r"no clear|not clear|noise|noisy|flat|uncertain|not (?:really |clearly |significantly )?(?:going down|decreas|improv)|no (?:significant|meaningful|real)", a, re.I) is not None and re.search(r"^\s*yes\b", a, re.I) is None, "want: no clear trend")),
    T("eta", "Run lin_decay: at roughly which step will train_loss reach 5.5? Give a step number.",
      [("summarize", {"run": "lin_decay", "metric": "train_loss", "target": 5.5})], rx(r"Target 5\.5.*"),
      lambda a, d: (any(abs(x - lin_eta(5.5)) <= 0.15 * lin_eta(5.5) for x in nums(a) if x > 100), f"want ~{lin_eta(5.5):.0f} ±15%")),
    T("seed_mean", "Across the three seeds of exp/small, what is the mean ± sd of heldout_nll at the latest step all seeds share?",
      [("summarize", {"run": "exp/small", "metric": "heldout_nll"})], rx(r"at step \d+ \(latest step all seeds share\): \S+ ± \S+"),
      lambda a, d: (has_num(a, m_seed, 0.01) and has_num(a, sd_seed, 0.05), f"want {m_seed:.3f} ± {sd_seed:.3f}")),
    T("typo_run", "What is the latest heldout_nll of run exp/meduim?",
      [("read_metric", {"run": "exp/meduim", "metric": "heldout_nll"}), ("read_metric", {"run": "exp/mid", "metric": "heldout_nll"})], rx(r"last \S+ at step \d+"),
      lambda a, d: (has_num(a, 4.79), "want 4.79 (exp/mid)")),
    T("lr_alias", "What learning rate is run pilot1 using at its last step?",
      [("read_metric", {"run": "pilot1", "metric": "lr"})], rx(r"last \S+ at step \d+"),
      lambda a, d: (has_num(a, 0.00015, 0.01), "want 1.5e-4")),
    T("log_write", "Record these metrics for run demo_new at step 300: loss 2.1 and accuracy 0.62. Then confirm what was stored.",
      [("log", {"run": "demo_new", "metrics": {"loss": 2.1, "accuracy": 0.62}, "step": 300})], lambda outs: outs[-1],
      lambda a, d: _check_log(d)),
    T("ingest_paste", "Here is stdout from my training job. Store it in run pasted, then tell me the final accuracy.\n\n" + PASTE,
      [("ingest_log", {"run": "pasted", "text": PASTE}), ("read_metric", {"run": "pasted", "metric": "acc"})], rx(r"last \S+ at step \d+"),
      lambda a, d: _check_ingest(a, d)),
    T("board", "Make a dashboard of heldout_nll for exp/small and exp/mid and give me the link.",
      [("make_board", {"query": 'heldout_nll{run=~"exp/small|exp/mid"}'})], rx(r"URL: \S+"),
      lambda a, d: _check_board(a, d)),
]


def _check_shared(a):
    # latest step both share is 1500 (index 6): small 4.97, mid 4.79
    sm, md = FIX["exp/small"]["heldout_nll"][6], FIX["exp/mid"]["heldout_nll"][6]
    ok = "mid" in a and has_num(a, abs(sm - md), 0.05) and has_num(a, 1500, 0)
    return ok, f"want exp/mid lower at step 1500 by {abs(sm - md):.2f} ({md} vs {sm})"


def _check_improving(a):
    yes = re.search(r"\byes\b|improv", a, re.I) is not None
    no = re.search(r"\bno\b(?! clear)|not improving|no clear", a, re.I) is not None
    rate = has_num(a, -0.1, 0.1) or has_num(a, 0.1, 0.1)
    return yes and not no and rate, "want yes, about -0.1 per 10 steps"


def _check_log(d):
    s = stored(d, "demo_new")
    loss = s.get("loss", {})
    acc = next((v for k, v in s.items() if "acc" in k), {})
    ok = loss.get(300) == 2.1 and acc.get(300) == 0.62
    return ok, f"want loss=2.1, acc*=0.62 at step 300; stored {s}"


def _check_ingest(a, d):
    s = stored(d, "pasted")
    loss = s.get("loss", {})
    ok = all(loss.get(st) == v for st, v in [(100, 2.84), (200, 2.51), (300, 2.33), (400, 2.29)]) and has_num(a, 53.0, 0.001)
    return ok, f"want 4 loss rows and answer 53%; stored loss {loss}"


def _check_board(a, d):
    bdir = os.path.join(d, "boards")
    files = os.listdir(bdir) if os.path.isdir(bdir) else []
    good = False
    for f in files:
        b = json.load(open(os.path.join(bdir, f)))
        for p in b.get("panels", []):
            if p.get("metric") == "heldout_nll" and set(p.get("runs", [])) == {"exp/small", "exp/mid"}:
                good = True
    return good and "#board=" in a, f"want a heldout_nll board over exactly exp/small+exp/mid and its URL in the answer; boards {files}"


# ---------------------------------------------------------------- runners

SYSTEM = (
    "You are a helpful assistant for machine-learning experiments. You can read and record training-run metrics "
    "with the provided tools. Use the tools to look things up; do not guess numbers. When you have the answer, "
    "reply with a short final answer that states the numbers and run names."
)


def start_server(binary, root):
    """`obs-server --dev` on a free loopback port, storing under `root`."""
    import socket
    import subprocess
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    p = subprocess.Popen([binary, "--root", root, "--listen", f"127.0.0.1:{port}", "--dev", "--no-static"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            requests.get(url + "/api/index", timeout=1)
            return p, url
        except requests.RequestException:
            time.sleep(0.05)
    p.kill()
    raise RuntimeError(f"obs-server did not start on {url}")


def mcp_to_openai(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}} for t in tools]


def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()


def dispatch(c, task, ctx, name, args):
    if task.get("shell") and name in hard_tasks.SHELL_NAMES:
        return hard_tasks.shell_call(name, args, ctx["workdir"], ctx["env"])
    return c.call(name, args)


def run_scripted(task, c, ctx):
    calls, outs = [], []
    for name, args in task["script"]:
        text, err = dispatch(c, task, ctx, name, args)
        calls.append({"name": name, "args": args, "is_error": err, "text": text})
        outs.append(text)
    return task["extract"](outs), calls, 0


def run_agent(task, c, tools, a, ctx):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"] + ("" if a.think else " /no_think")}]
    calls = []
    for turn in range(1, a.max_turns + 1):
        body = {"model": a.model, "messages": msgs, "tools": tools, "temperature": a.temperature, "seed": a.seed + a.trial}
        r = requests.post(a.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=a.timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": tcs} if tcs else {})})
        if not tcs:
            return strip_think(msg.get("content")), calls, turn
        for tc in tcs:
            fn = tc.get("function", {})
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except ValueError:
                args = raw  # the server copes with a string
            text, err = dispatch(c, task, ctx, fn.get("name", ""), args)
            calls.append({"name": fn.get("name"), "args": raw, "is_error": err, "text": text})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": text})
    return "(no final answer within max turns)", calls, a.max_turns


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=os.path.join(root, "target", "debug", "obs"), help="obs binary")
    ap.add_argument("--data", default=os.environ.get("OBS_BENCH_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "obs-agent-bench")), help="scratch dir for per-task data")
    ap.add_argument("--scripted", action="store_true", help="run the canonical tool sequences instead of a model")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL")
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--think", action="store_true", help="let qwen3 think (default appends /no_think)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=900, help="seconds per model request")
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--suite", choices=["base", "hard", "all"], default="base", help="base = the 14 one-step tasks, hard = the multi-step tasks in hard_tasks.py")
    ap.add_argument("--trial", type=int, default=0, help="trial number; also offsets --seed, so trials differ")
    ap.add_argument("--remote", action="store_true", help="run each task against its own `obs-server --dev` (root = the task dir) through `obs mcp --url`")
    ap.add_argument("--obs-server", default=os.path.join(root, "target", "debug", "obs-server"), help="obs-server binary for --remote")
    ap.add_argument("--out", help="JSONL transcript path (default <data>/<label>.jsonl)")
    a = ap.parse_args()

    all_tasks = [dict(t, suite="base") for t in TASKS] + hard_tasks.HARD_TASKS
    tasks = [t for t in all_tasks if (a.suite == "all" or t["suite"] == a.suite) and (not a.only or t["id"] in a.only.split(","))]
    label = ("scripted" if a.scripted else re.sub(r"[^\w.-]", "_", a.model) + ("-think" if a.think else "")) + ("-remote" if a.remote else "") + (f"-{a.suite}" if a.suite != "base" else "") + (f"-t{a.trial}" if a.trial else "")
    os.makedirs(a.data, exist_ok=True)
    out_path = a.out or os.path.join(a.data, f"{label}.jsonl")
    out = open(out_path, "w")
    rows = []
    for t in tasks:
        # <workdir> is the "project": scripts live here, obs data in <workdir>/obs
        w = os.path.join(a.data, label, t["id"])
        if os.path.isdir(w):
            import shutil
            shutil.rmtree(w)
        d = os.path.join(w, "obs")
        os.makedirs(d)
        env = dict(os.environ, PATH=os.path.dirname(os.path.abspath(a.obs)) + os.pathsep + os.environ.get("PATH", ""))
        ctx = {"workdir": w, "env": env}
        srv = None
        if a.remote:
            srv, url = start_server(a.obs_server, d)
            env["OBS_URL"] = url
            c = McpClient([a.obs, "mcp", "--url", url])
        else:
            c = McpClient([a.obs, "mcp", "--dir", d])
        try:
            if t["suite"] == "hard":
                hard_tasks.build_hard_fixture(c, _must)
                hard_tasks.setup_scripts(w)
            else:
                build_fixture(c)
            mtools = c.tools() + (hard_tasks.SHELL_TOOLS if t.get("shell") else [])
            tools = mcp_to_openai(mtools)
            t0 = time.time()
            try:
                answer, calls, turns = run_scripted(t, c, ctx) if a.scripted else run_agent(t, c, tools, a, ctx)
            except Exception as e:  # network / model failure counts as a fail
                answer, calls, turns = f"(harness error: {e})", [], 0
            dt = time.time() - t0
        finally:
            c.close()
            if srv:
                srv.terminate()
                srv.wait(timeout=10)
        ok, why = t["check"](answer, d)
        ok = bool(ok)
        errs = sum(1 for x in calls if x["is_error"])
        renamed = sum(1 for x in calls if x["name"] not in CANON and x["name"] not in hard_tasks.SHELL_NAMES)
        rec = {"task": t["id"], "pass": ok, "why": why, "answer": answer, "turns": turns, "tool_calls": len(calls), "tool_errors": errs, "noncanonical_names": renamed, "seconds": round(dt, 1), "calls": calls, "model": None if a.scripted else a.model, "suite": t["suite"], "trial": a.trial, "obs": a.obs}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        rows.append(rec)
        print(f"{'PASS' if ok else 'FAIL'}  {t['id']:<16} calls={len(calls):<2} errors={errs} renamed={renamed} turns={turns:<2} {dt:6.1f}s" + ("" if ok else f"  [{why}] answer: {answer[:160]!r}"), flush=True)
    n = sum(r["pass"] for r in rows)
    print(f"\n{label}: {n}/{len(rows)} passed; {sum(r['tool_calls'] for r in rows)} tool calls, {sum(r['tool_errors'] for r in rows)} tool errors, "
          f"{sum(r['noncanonical_names'] for r in rows)} non-canonical tool names, {sum(r['seconds'] for r in rows):.0f}s. Transcript: {out_path}")
    sys.exit(0 if n == len(rows) else 1)


if __name__ == "__main__":
    main()
