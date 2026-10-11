"""Harder, multi-step tasks for obs-agent-bench (suite "hard").

The base suite (bench.py) asks one-tool questions; qwen3:30b-a3b scores 14/14
there. These tasks need several calls, judgement on top of the numbers, or
work outside the MCP tools (reading and running a training script), so they
separate tool designs. Every task shares one fixture project of ~25 runs,
built through the MCP tools (ingest_log) from deterministic generators below;
the ground truth is computed here, not taken from the tools.
"""

import json
import math
import os
import re
import subprocess

# ------------------------------------------------------------------ helpers


def lcg(seed):
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (x >> 33) / (1 << 31) - 0.5  # uniform in [-0.5, 0.5)


def r4(x):
    return round(x, 4)


# ------------------------------------------------------------------ fixture
# Each generator returns {key: [(step, value)]}; value may be "nan".

SWEEP_LRS = ["lr1e-4", "lr3e-4", "lr1e-3", "lr3e-3"]
DIVERGED = "sweep/lr3e-3"
DIVERGE_MIN_STEP = 700  # last good (lowest) point; rises from 750, NaN from 900


def gen_sweep(i, name):
    g = lcg(100 + i)
    rate = [2500, 1500, 900, 700][i]
    floor = [2.6, 2.4, 2.3, 2.3][i]
    tl, vl = [], []
    for s in range(0, 1501, 50):
        base = floor + 4.0 * math.exp(-s / rate)
        n = 0.01 * next(g)
        if name == DIVERGED and s > DIVERGE_MIN_STEP:
            if s >= 900:
                tl.append((s, "nan"))
                vl.append((s, "nan"))
                if s >= 1000:
                    break
                continue
            blow = {750: 1.6, 800: 3.5, 850: 9.0}[s]
            base *= blow
        tl.append((s, r4(base - 0.15 + n)))
        vl.append((s, r4(base + n)))
    return {"train_loss": tl, "val_loss": vl}


def gen_long(kind):
    g = lcg(7 if kind == "a" else 8)
    out = []
    for s in range(0, 3001, 100):
        if kind == "a":  # still clearly improving at step 3000
            v = 2.0 + 3.0 * math.exp(-s / 4000) + 0.004 * next(g)
        else:  # plateaued since ~step 1000; noise dominates
            v = 2.5 + 2.5 * math.exp(-s / 300) + 0.06 * next(g)
        out.append((s, r4(v)))
    return {"val_loss": out}


SEED_FINALS = {
    "abl/base": [3.012, 3.087, 2.968],
    "abl/wd": [2.991, 3.065, 3.030],
    "abl/warmup": [2.801, 2.832, 2.787],
}


def gen_seed(group, k):
    g = lcg(31 * (k + 1) + len(group))
    fin = SEED_FINALS[group][k]
    out = []
    for s in range(0, 2001, 100):
        v = fin + 1.5 * math.exp(-s / 400) + (0.003 * next(g) if s < 2000 else 0.0)
        out.append((s, r4(v)))
    return {"val_loss": out}


SPIKE_STEP, LR_CHANGE_STEP = 1250, 1200


def gen_spike():
    g = lcg(5)
    loss, lr = [], []
    for s in range(0, 2001, 50):
        v = 2.2 + 1.8 * math.exp(-s / 500) + 0.01 * next(g)
        v += {SPIKE_STEP: 2.0, SPIKE_STEP + 50: 0.6}.get(s, 0.12 if s > SPIKE_STEP + 50 else 0.0)
        loss.append((s, r4(v)))
        lr.append((s, 1e-4 if s < LR_CHANGE_STEP else 1e-3))
    return {"loss": loss, "lr": lr}


def gen_cls():
    g = lcg(9)
    m = {"train/loss": [], "eval/nll": [], "eval/acc_top1": [], "throughput": []}
    for s in range(0, 1001, 100):
        m["train/loss"].append((s, r4(0.9 + 1.4 * math.exp(-s / 300) + 0.005 * next(g))))
        m["eval/nll"].append((s, r4(1.15 + 1.2 * math.exp(-s / 350) + 0.005 * next(g))))
        m["eval/acc_top1"].append((s, r4(0.78 - 0.5 * math.exp(-s / 300))))
        m["throughput"].append((s, r4(1500 + 40 * next(g))))
    return m


def gen_rl(name):
    g = lcg(hash(name) % 1000 if False else {"rl/ppo_lr1": 1, "rl/ppo_lr2": 2, "rl/grpo": 3}[name])
    m = {"rollout/ep_reward_mean": [], "train/approx_kl": [], "train/entropy_loss": [], "train/value_loss": [], "time/fps": []}
    for s in range(0, 50001, 5000):
        m["rollout/ep_reward_mean"].append((s, r4(200 * (1 - math.exp(-s / 15000)) + 5 * next(g))))
        m["train/approx_kl"].append((s, r4(0.01 + 0.004 * next(g))))
        m["train/entropy_loss"].append((s, r4(-1.4 + s / 50000 * 0.8)))
        m["train/value_loss"].append((s, r4(30 * math.exp(-s / 20000) + next(g))))
        m["time/fps"].append((s, r4(900 + 30 * next(g))))
    return m


OVERFIT = "ovf/big"


def gen_ovf(name):
    g = lcg({"ovf/tiny": 11, "ovf/small": 12, "ovf/mid": 13, "ovf/big": 14}[name])
    tl, vl = [], []
    for s in range(0, 3001, 100):
        t = 1.0 + 2.0 * math.exp(-s / 700) + 0.005 * next(g)
        v = t + 0.3 + 0.005 * next(g)
        if name == OVERFIT and s > 1500:
            v += 0.0004 * (s - 1500)  # +0.6 by step 3000
            t -= 0.0001 * (s - 1500)
        tl.append((s, r4(t)))
        vl.append((s, r4(v)))
    return {"train_loss": tl, "val_loss": vl}


def hard_runs():
    runs = {}
    for i, lr in enumerate(SWEEP_LRS):
        runs[f"sweep/{lr}"] = gen_sweep(i, f"sweep/{lr}")
    runs["long/a"] = gen_long("a")
    runs["long/b"] = gen_long("b")
    for grp in SEED_FINALS:
        for k in range(3):
            runs[f"{grp}_s{k + 1}"] = gen_seed(grp, k)
    runs["ft/run7"] = gen_spike()
    runs["cls/run3"] = gen_cls()
    for n in ("rl/ppo_lr1", "rl/ppo_lr2", "rl/grpo"):
        runs[n] = gen_rl(n)
    for n in ("ovf/tiny", "ovf/small", "ovf/mid", OVERFIT):
        runs[n] = gen_ovf(n)
    return runs


def fmtv(v):
    return v if isinstance(v, str) else repr(v)


def build_hard_fixture(c, must):
    for run, ms in hard_runs().items():
        steps = sorted({s for col in ms.values() for s, _ in col})
        by = {k: dict(col) for k, col in ms.items()}
        lines = []
        for s in steps:
            kv = " ".join(f"{k}={fmtv(by[k][s])}" for k in ms if s in by[k])
            lines.append(f"step {s} {kv}")
        must(c.call("ingest_log", {"run": run, "text": "\n".join(lines)}))


def hard_values(run, key):
    return [(s, v) for s, v in hard_runs()[run][key] if not isinstance(v, str)]


# --------------------------------------------------------- toy scripts

TRAIN_WRAP = '''"""Toy training loop: fits w to 2.0 by gradient descent."""
w = 5.0
lr = 0.05
for it in range(101):
    loss = (w - 2.0) ** 2
    if it % 10 == 0:
        print(f"it {it} | loss {loss:.6f} | lr {lr}")
    w -= lr * 2 * (w - 2.0)
print("done")
'''

TRAIN_SILENT = '''"""Toy training loop: fits a line y = 3x + 1 with SGD. Prints nothing until the end."""
xs = [i / 10 for i in range(20)]
ys = [3 * x + 1 for x in xs]
a, b = 0.0, 0.0
lr = 0.02
for it in range(200):
    ga = gb = 0.0
    loss = 0.0
    for x, y in zip(xs, ys):
        e = a * x + b - y
        loss += e * e / len(xs)
        ga += 2 * e * x / len(xs)
        gb += 2 * e / len(xs)
    a -= lr * ga
    b -= lr * gb
print("training done")
'''


def wrap_truth():
    w, out = 5.0, {}
    for it in range(101):
        loss = (w - 2.0) ** 2
        if it % 10 == 0:
            out[it] = loss
        w -= 0.05 * 2 * (w - 2.0)
    return out


def silent_truth():
    xs = [i / 10 for i in range(20)]
    ys = [3 * x + 1 for x in xs]
    a = b = 0.0
    out = {}
    for it in range(200):
        ga = gb = loss = 0.0
        for x, y in zip(xs, ys):
            e = a * x + b - y
            loss += e * e / len(xs)
            ga += 2 * e * x / len(xs)
            gb += 2 * e / len(xs)
        out[it] = loss
        a -= 0.02 * ga
        b -= 0.02 * gb
    return out


def setup_scripts(workdir):
    open(os.path.join(workdir, "train.py"), "w").write(TRAIN_WRAP)
    open(os.path.join(workdir, "train_silent.py"), "w").write(TRAIN_SILENT)


# ------------------------------------------------- local "coding agent" tools
# Exposed only on tasks with shell=True: the agent must read a file, maybe
# edit it, and run commands, like a coding agent with obs available.

SHELL_TOOLS = [
    {"name": "read_file", "description": "Read a text file. Example: read_file(path=\"train.py\")",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "description": "File path (relative to the project directory or absolute)"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write (overwrite) a text file with the given content. Example: write_file(path=\"train.py\", content=\"print(1)\\n\")",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string", "description": "The full new file content"}}, "required": ["path", "content"]}},
    {"name": "run_command", "description": "Run a shell command in the project directory and return its exit code and output (last 3000 characters). Example: run_command(command=\"python3 train.py\")",
     "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]
SHELL_NAMES = {t["name"] for t in SHELL_TOOLS}


def shell_call(name, args, workdir, env):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    args = args or {}

    def path():
        p = str(args.get("path") or args.get("file") or args.get("filename") or "")
        return p if os.path.isabs(p) else os.path.join(workdir, p)

    try:
        if name == "read_file":
            return open(path()).read()[:20000], False
        if name == "write_file":
            p = path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            open(p, "w").write(str(args.get("content", "")))
            return f"wrote {len(str(args.get('content', '')))} bytes to {p}", False
        if name == "run_command":
            cmd = str(args.get("command") or args.get("cmd") or "")
            r = subprocess.run(cmd, shell=True, cwd=workdir, env=env, capture_output=True, text=True, timeout=120)
            out = (r.stdout + r.stderr)[-3000:]
            return f"exit code {r.returncode}\n{out}", r.returncode != 0
    except Exception as e:  # noqa: BLE001
        return f"error: {e}", True
    return f"unknown tool {name}", True


# ------------------------------------------------------------------ checks

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def nums(text):
    return [float(t) for t in NUM.findall(text.replace(",", ""))]


def has_num(text, want, rel=0.01, abs_=1e-9):
    return any(abs(x - want) <= max(abs_, rel * abs(want)) for x in nums(text))


def mentions(text, name):
    return re.search(r"(?<![\w/])" + re.escape(name) + r"(?![\w])", text) is not None


def mean_sd(v):
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def stored(data, run):
    out = {}
    p = os.path.join(data, "runs", run, "metrics.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        step = row.get("step")
        for k, v in row.items():
            if isinstance(v, (int, float)) and k not in ("step", "t"):
                out.setdefault(k, {})[step] = v
    return out


def all_stored_runs(data):
    root = os.path.join(data, "runs")
    out = []
    for dp, _, fs in os.walk(root):
        if "metrics.jsonl" in fs:
            out.append(os.path.relpath(dp, root))
    return out


HONEST = r"no clear|not clear|plateau|flat|stall|levell?ed|no skill|not (?:reliabl|predict)|unreliab|can(?:no|')t (?:reliably )?(?:predict|say|tell)|no reliable|uncertain|noise|not (?:really |clearly |significantly |meaningfully )?improv|no (?:longer|significant|meaningful|real) (?:improv|progress|gain)|diminish"
NOISE = r"within (?:the )?(?:seed )?noise|not (?:statistically )?significant|insignificant|no (?:significant|meaningful|real|clear) (?:difference|improvement|gain)|overlap|inconclusive|indistinguishable|within (?:one|1|the) (?:standard deviation|sd|std)|not (?:a )?(?:meaningful|real|clear)|comparable|negligible|too small"


def check_diverged(a, d):
    named = mentions(a, DIVERGED) or (re.search(r"\b3e-3\b|3e-03|0\.003\b", a) is not None)
    others = [f"sweep/{x}" for x in SWEEP_LRS if f"sweep/{x}" != DIVERGED]
    wrong = any(mentions(a, o) and re.search(re.escape(o) + r"[^.\n]{0,40}diverg", a) for o in others)
    when = any(DIVERGE_MIN_STEP <= x <= 900 for x in nums(a) if x == int(x))
    return named and when and not wrong, f"want {DIVERGED}, going wrong from step ~{DIVERGE_MIN_STEP + 50} (accept {DIVERGE_MIN_STEP}–900; NaN from 900)"


def check_continue_yes(a, d):
    yes = re.search(r"\byes\b|worth continuing|keep (?:it )?(?:going|training|running)|continue|still improving", a, re.I) is not None
    no = re.search(r"^\s*no\b|not worth|\bstop\b(?! improving)|(?<!no )(?<!not )(?<!without a )plateau", a, re.I) is not None
    return yes and not no, "want: yes, long/a is still clearly improving"


def check_continue_no(a, d):
    honest = re.search(HONEST, a, re.I) is not None
    yes = re.search(r"^\s*(?:\*\*)?yes\b|\bworth continuing\b(?! is doubtful)", a, re.I) is not None and re.search(r"not worth|isn't worth|is not worth|no longer worth", a, re.I) is None
    return honest and not yes, "want: no / plateaued, forecast has no skill"


def check_eta_unreachable(a, d):
    honest = re.search(HONEST + r"|not (?:going to |likely to )?reach|never|won'?t reach|unlikely|will not|doesn'?t reach|does not reach|no eta", a, re.I) is not None
    confident = re.search(r"(?:reach(?:es)?|hit) (?:2\.0|2)\b[^.\n]{0,30}(?:at|by|around) step \d+", a, re.I) is not None and re.search(r"optimistic|band|edge|unlikely|not", a, re.I) is None
    return honest and not confident, "want: no reliable ETA (plateau ~2.5 with noise)"


def _seed_stats(group):
    # the stored value at the last step (the generator adds a small decay term)
    return mean_sd([hard_values(f"{group}_s{k + 1}", "val_loss")[-1][1] for k in range(3)])


def check_seeds_noise(a, d):
    (mb, sb), (mw, sw) = _seed_stats("abl/base"), _seed_stats("abl/wd")
    means = has_num(a, mb, 0.002) and has_num(a, mw, 0.002)
    verdict = re.search(NOISE, a, re.I) is not None
    overclaim = re.search(r"\bwd\b[^.\n]{0,30}(?:clearly|significantly) (?:better|worse|lower|higher)", a, re.I) is not None
    return means and verdict and not overclaim, f"want base {mb:.3f}±{sb:.3f}, wd {mw:.3f}±{sw:.3f}, verdict: within seed noise"


def check_seeds_real(a, d):
    (mb, sb), (mw, sw) = _seed_stats("abl/base"), _seed_stats("abl/warmup")
    means = has_num(a, mb, 0.002) and has_num(a, mw, 0.002)
    yes = re.search(r"\byes\b|beyond|outside|significant|clearly|real (?:improvement|difference)|meaningful", a, re.I) is not None
    no = re.search(r"within (?:the )?(?:seed )?noise|not significant|inconclusive|not (?:clearly )?better", a, re.I) is not None
    return means and yes and not no, f"want warmup {mw:.3f}±{sw:.3f} better than base {mb:.3f}±{sb:.3f} beyond noise"


def check_spike(a, d):
    step = any(SPIKE_STEP - 50 <= x <= SPIKE_STEP + 50 for x in nums(a) if x == int(x) and x > 100)
    lr = re.search(r"\blr\b|learning[ _-]?rate", a, re.I) is not None
    up = re.search(r"1e-3|0\.001\b|10x|10 ?×|tenfold|increas|rais|jump|went up|higher|from 1e-4", a, re.I) is not None
    return step and lr and up, f"want spike at step {SPIKE_STEP} and the lr increase 1e-4 -> 1e-3 at step {LR_CHANGE_STEP}"


def check_wrong_metric(a, d):
    nll = hard_values("cls/run3", "eval/nll")[-1][1]
    tl = hard_values("cls/run3", "train/loss")[-1][1]
    return has_num(a, nll, 0.002) and not has_num(a, tl, 0.002), f"want eval/nll last = {nll} (not train/loss {tl})"


def check_rl_board(a, d):
    bdir = os.path.join(d, "boards")
    files = []
    for dp, _, fs in os.walk(bdir):
        files += [os.path.join(dp, f) for f in fs if f.endswith(".json")]
    want = {"rl/ppo_lr1", "rl/ppo_lr2"}
    covered = set()
    for f in files:
        b = json.load(open(f))
        for p in b.get("panels", []):
            if set(p.get("runs", [])) == want:
                m = p.get("metric", "")
                for k, pat in (("reward", "reward"), ("kl", "kl"), ("entropy", "entropy")):
                    if pat in m:
                        covered.add(k)
    ok = covered == {"reward", "kl", "entropy"} and "#board=" in a
    return ok, f"want board(s) with reward, kl, entropy panels over exactly the two PPO runs and the URL; covered {sorted(covered)} in {len(files)} board file(s)"


def check_overfit(a, d):
    others = ["ovf/tiny", "ovf/small", "ovf/mid"]
    named = mentions(a, OVERFIT) or re.search(r"\bbig\b", a) is not None
    wrong = any(mentions(a, o) for o in others) and re.search(r"(?:" + "|".join(re.escape(o) for o in others) + r")[^.\n]{0,40}overfit", a) is not None
    return named and not wrong, f"want {OVERFIT}"


PASTE_A = """epoch 1 step 100: loss=2.91 acc=0.31
epoch 1 step 200: loss=2.47 acc=0.39
epoch 2 step 300: loss=2.20 acc=0.45
epoch 2 step 400: loss=2.05 acc=0.48"""
PASTE_B = """[step   100] train/loss 2.88 | grad_norm 1.9
[step   200] train/loss 2.52 | grad_norm 1.4
[step   300] train/loss 2.31 | grad_norm 1.2"""


def check_paste_compare(a, d):
    sa, sb = stored(d, "paste/a"), stored(d, "paste/b")
    la = sa.get("loss", {})
    lb = next((v for k, v in sb.items() if "loss" in k), {})
    rows = la.get(300) == 2.20 and lb.get(300) == 2.31
    ans = (mentions(a, "paste/a") or re.search(r"job 1|first (?:job|run|log)", a, re.I) is not None) and re.search(r"paste/b[^.\n]{0,30}(?:is |has )?(?:the )?lower", a) is None
    diff = has_num(a, 0.11, 0.05)
    return rows and ans and diff, f"want both stored (a loss@300=2.2, b train/loss@300=2.31) and answer: paste/a lower by 0.11; stored a={la} b={lb}"


def check_instrument_wrap(a, d):
    truth = wrap_truth()
    best = None
    for run in all_stored_runs(d):
        s = stored(d, run)
        for k, col in s.items():
            if "loss" in k:
                hits = sum(1 for it, v in truth.items() if it in col and abs(col[it] - v) <= 1e-4 * max(1, abs(v)) + 1e-6)
                if best is None or hits > best[0]:
                    best = (hits, run)
    last = truth[100]
    ok = best is not None and best[0] >= 10 and best[1].endswith("toy/wrap") and has_num(a, last, 0.01, 1e-6)
    return ok, f"want run toy/wrap with the 11 printed loss values and the final loss {last:.6g} in the answer; best match {best}"


def check_instrument_edit(a, d):
    truth = silent_truth()
    best = None
    for run in all_stored_runs(d):
        s = stored(d, run)
        for k, col in s.items():
            if "loss" in k:
                hits = sum(1 for it, v in col.items() if it in truth and abs(truth[it] - v) <= 1e-3 * max(1e-3, abs(truth[it])))
                if best is None or hits > best[0]:
                    best = (hits, run, max(col) if col else None)
    ok = best is not None and best[0] >= 10 and best[1].endswith("toy/edit")
    if ok:
        ok = has_num(a, truth[best[2]], 0.02, 1e-6)
    return ok, f"want run toy/edit with >=10 loss values matching the script's loss at those iterations, and the last logged loss in the answer; best {best}"


# -------------------------------------------------------------------- tasks
# script: canonical calls (MCP tools or shell tools); extract(outs) -> answer.


def T(id, prompt, script, extract, check, shell=False):
    return {"id": id, "prompt": prompt, "script": script, "extract": extract, "check": check, "shell": shell, "suite": "hard"}


def joined(outs):
    return "\n".join(outs)


def _seeds_extract(outs):
    t = joined(outs)
    ms = re.findall(r"latest step all seeds share\): (\S+) ± (\S+)", t)
    return f"{' and '.join(f'{m} ± {s}' for m, s in ms)}; the difference is within seed noise" if ms else t


def _seeds_real_extract(outs):
    t = joined(outs)
    ms = re.findall(r"latest step all seeds share\): (\S+) ± (\S+)", t)
    return f"{' and '.join(f'{m} ± {s}' for m, s in ms)}; yes, warmup is better beyond seed noise" if ms else t


def _wrap_extract(outs):
    m = re.search(r"last (\S+) at step 100", joined(outs))
    return f"final loss {m.group(1)}" if m else joined(outs)


def _edit_extract(outs):
    m = re.search(r"last (\S+) at step (\d+)", joined(outs))
    return f"last logged loss {m.group(1)} at iteration {m.group(2)}" if m else joined(outs)


EDIT_SCRIPT = TRAIN_SILENT.replace("    a -= lr * ga", "    if it % 10 == 0:\n        print(f\"step {it} loss={loss:.8f}\")\n    a -= lr * ga")

HARD_TASKS = [
    T("diverged", "Four learning-rate sweep runs are stored under sweep/. One of them diverged. Which run, and at roughly which step did it start to go wrong?",
      [("list_runs", {"filter": "sweep"}), ("compare", {"runs": "sweep/*", "metric": "val_loss"}), ("read_metric", {"run": "sweep/lr3e-3", "metric": "val_loss", "last": 40})],
      lambda outs: "sweep/lr3e-3 diverged; val_loss was lowest at step 700 then rose from step 750 and became NaN by 900", check_diverged),
    T("continue_yes", "Is it worth continuing run long/a for another 1000 steps? Answer yes or no, with your evidence.",
      [("summarize", {"run": "long/a"})], lambda outs: "Yes, continue: val_loss is still improving and the forecast (with skill) keeps falling.", check_continue_yes),
    T("continue_no", "Is it worth continuing run long/b for another 1000 steps? What val_loss would it reach? Answer yes or no, with your evidence.",
      [("summarize", {"run": "long/b"})], lambda outs: "No: val_loss has plateaued (no clear trend) and the forecast has no skill, so expect about the current value.", check_continue_no),
    T("eta_unreachable", "When will run long/b reach a val_loss of 2.0? Give a step number if you can.",
      [("summarize", {"run": "long/b", "metric": "val_loss", "target": 2.0})], lambda outs: "No reliable ETA: the curve has plateaued and the forecast has no skill.", check_eta_unreachable),
    T("seeds_noise", "The abl/ runs are an ablation with 3 seeds per config. Does weight decay (abl/wd_s*) beat the baseline (abl/base_s*) on val_loss at the end of training? Give mean ± sd over seeds for both, and say whether the difference is within seed noise.",
      [("summarize", {"run": "abl/base_s1", "metric": "val_loss"}), ("summarize", {"run": "abl/wd_s1", "metric": "val_loss"})], _seeds_extract, check_seeds_noise),
    T("seeds_real", "The abl/ runs are an ablation with 3 seeds per config. Is warmup (abl/warmup_s*) better than the baseline (abl/base_s*) on final val_loss beyond seed noise? Give mean ± sd for both.",
      [("summarize", {"run": "abl/base_s1", "metric": "val_loss"}), ("summarize", {"run": "abl/warmup_s1", "metric": "val_loss"})], _seeds_real_extract, check_seeds_real),
    T("spike_lr", "Run ft/run7 had a loss spike. At which step, and what changed in the run right before it?",
      [("summarize", {"run": "ft/run7", "metric": "loss"}), ("read_metric", {"run": "ft/run7", "metric": "lr", "last": 40})],
      lambda outs: "The spike is at step 1250; the learning rate went up from 1e-4 to 1e-3 at step 1200.", check_spike),
    T("wrong_metric", "What is the latest validation loss of run cls/run3?",
      [("read_metric", {"run": "cls/run3", "metric": "validation loss"}), ("read_metric", {"run": "cls/run3", "metric": "eval/nll"})],
      lambda outs: re.search(r"last \S+ at step \d+", outs[-1]).group(0), check_wrong_metric),
    T("rl_board", "Make a dashboard for the two PPO runs showing the RL metrics reward, KL and entropy, and give me the link.",
      [("make_board", {"query": 'rollout/ep_reward_mean{run=~"rl/ppo_lr1|rl/ppo_lr2"}; train/approx_kl{run=~"rl/ppo_lr1|rl/ppo_lr2"}; train/entropy_loss{run=~"rl/ppo_lr1|rl/ppo_lr2"}'})],
      lambda outs: re.search(r"URL: \S+", outs[-1]).group(0), check_rl_board),
    T("overfit", "One of the ovf/ runs is overfitting: its validation loss is getting worse while its training loss still falls. Which one?",
      [("compare", {"runs": "ovf/*", "metric": "val_loss"}), ("summarize", {"run": "ovf/big"})], lambda outs: "ovf/big", check_overfit),
    T("paste_compare", "Here are logs from two jobs. Store the first as run paste/a and the second as run paste/b, then tell me which has the lower training loss at step 300 and by how much.\n\nJob 1:\n" + PASTE_A + "\n\nJob 2:\n" + PASTE_B,
      [("ingest_log", {"run": "paste/a", "text": PASTE_A}), ("ingest_log", {"run": "paste/b", "text": PASTE_B})],
      lambda outs: "paste/a is lower at step 300 by 0.11 (2.20 vs 2.31)", check_paste_compare),
    T("instrument_wrap", "My training script is train.py in the project directory. Run it so that its metrics are recorded in obs as run toy/wrap, then tell me the final loss. "
      "The obs CLI is installed: `obs run --offline --name NAME -- <command>` runs a command and records the metric lines it prints into ./obs.",
      [("read_file", {"path": "train.py"}), ("run_command", {"command": "obs run --offline --name toy/wrap -- python3 train.py"}), ("read_metric", {"run": "toy/wrap", "metric": "loss"})],
      _wrap_extract, check_instrument_wrap, shell=True),
    T("instrument_edit", "My training script train_silent.py prints nothing while it trains. Change it so the loss is printed every 10 iterations, run it so those metrics are recorded in obs as run toy/edit, then tell me the loss at the last logged iteration. "
      "The obs CLI is installed: `obs run --offline --name NAME -- <command>` runs a command and records the metric lines it prints (like `step 10 loss=0.5`) into ./obs.",
      [("read_file", {"path": "train_silent.py"}), ("write_file", {"path": "train_silent.py", "content": EDIT_SCRIPT}),
       ("run_command", {"command": "obs run --offline --name toy/edit -- python3 train_silent.py"}), ("read_metric", {"run": "toy/edit", "metric": "loss"})],
      _edit_extract, check_instrument_edit, shell=True),
]
