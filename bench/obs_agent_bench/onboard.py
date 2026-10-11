"""obs-agent-bench, suite "onboard": can a local coding agent take a fresh repo
to a live modgrad dashboard, unaided?

Each task is a fresh git repo (one of four training scripts, onboard_repos.py)
plus a prompt a user would type. The agent gets three coding tools
(read_file, write_file, run_command), the prompt, and obs/llms.txt in its
system prompt. It never sees modgrad's source.

Environment the agent works in (everything through bubblewrap):
  - its task dir T, read-write: T/repo (the repo, the working directory),
    T/home (HOME), T/tmp (TMPDIR). Nothing else is writable: / is a
    read-only tmpfs with /usr, /etc, /sys, /opt/intel, /opt/rocm bound
    read-only; /home, /steam, /tmp of the host are not mounted at all.
  - `modgrad` and `modgrad-server` (copied from target/release) at
    /opt/modgrad/bin, first on PATH. The built dashboard (obs/web/dist) is
    bound read-only at the path compiled into modgrad-server, as after
    `cargo install --path` from a checkout.
  - the Python client installed as a user would install it: the wheel built
    from clients/python, `pip install --target T/home/.local/lib/python3.X/
    site-packages`, i.e. the result of `pip install --user modgrad-*.whl`.
    System torch is available; wandb is not installed.
  - network is shared (loopback reaches the task's modgrad-server).

Server variant: the harness starts `modgrad-server --root <outside T>
--listen 127.0.0.1:19010` (not --dev: writes need a token; anonymous
projects allowed), and the prompt names its URL.

Checker (all must hold; see check()):
  ran        a stored run has the script's metrics with the original's final
             values (local: any runs/**/metrics.jsonl under T/repo; server: the
             server's root on disk)
  link       local mode: the answer has a `modgrad-server [--root DIR]`
             command, and that command, started by the harness from the repo,
             serves the run (/api/index lists it). Server mode: the answer
             has a URL on the server that answers 200 and whose #run= /
             #project= / #board= names the run or its project.
  no_token   no `obsp_` token in git history, the index, or any file
             `git add -A` would pick up; .modgrad.json (if any) is ignored.
  original   the repo's script, run plainly (`python <script>`, no MODGRAD_*
             env) in the sandbox after the agent is done, exits 0 and still
             prints its final metric line (wandb variant: exits 0).

Usage:
  python3 onboard.py --sandbox-selftest
  python3 onboard.py --scripted [--negative]
  python3 onboard.py --endpoint http://127.0.0.1:11435/v1 --model qwen3:30b-a3b --trial 0
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import onboard_repos as R  # noqa: E402

SERVER_PORT = 19010
VIEW_PORT = 19011
BIN_MOUNT = "/opt/modgrad/bin"
DIST = os.path.join(ROOT, "obs", "web", "dist")
CMD_TIMEOUT = 120
OUT_CAP = 3000
READ_CAP = 12000
PYVER = f"python{sys.version_info.major}.{sys.version_info.minor}"

# ------------------------------------------------------------------ tasks

LOCAL_WATCH = "set up modgrad for this repo so I can watch training"
LOCAL_LINK = "log this training to modgrad and give me the dashboard link"
SWAP = "we're moving off wandb. switch this repo to modgrad, run the training and give me the dashboard link"
SERVER_LINK = "use our modgrad server at {url} - log this training there and give me the dashboard link"
SERVER_WATCH = "set up modgrad for this repo so I can watch training. use our modgrad server at {url}"

TASKS = [
    {"id": "plain_watch", "variant": "plain", "server": False, "prompt": LOCAL_WATCH},
    {"id": "hf_link", "variant": "hf", "server": False, "prompt": LOCAL_LINK},
    {"id": "rl_watch", "variant": "rl", "server": False, "prompt": LOCAL_WATCH},
    {"id": "wandb_swap", "variant": "wandb", "server": False, "prompt": SWAP},
    {"id": "plain_server", "variant": "plain", "server": True, "prompt": SERVER_LINK},
    {"id": "wandb_server", "variant": "wandb", "server": True, "prompt": SERVER_WATCH},
]

# ------------------------------------------------------------- environment


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def build_env(envdir, obs_bin, server_bin):
    """Binaries, the client wheel and the ground truth, once per --data dir."""
    os.makedirs(os.path.join(envdir, "bin"), exist_ok=True)
    for b in (obs_bin, server_bin):
        shutil.copy2(b, os.path.join(envdir, "bin", os.path.basename(b)))
    wheels = os.path.join(envdir, "wheels")
    shutil.rmtree(wheels, ignore_errors=True)
    hatch = os.path.join(envdir, "hatch")
    if not os.path.isdir(hatch):
        sh([sys.executable, "-m", "pip", "install", "-q", "--target", hatch, "hatchling"])
    src = os.path.join(envdir, "client-src")  # build from a copy: no build/ litter in the checkout
    shutil.rmtree(src, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "clients", "python"), src, ignore=shutil.ignore_patterns("tests", "__pycache__", "*.egg-info", "build", "dist"))
    sh([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation", "-w", wheels, src],
       env=dict(os.environ, PYTHONPATH=hatch))
    golden = {}
    for v, (files, script) in R.VARIANTS.items():
        g = os.path.join(envdir, "golden", v)
        shutil.rmtree(g, ignore_errors=True)
        os.makedirs(os.path.join(g, "repo"))
        os.makedirs(os.path.join(g, "home"))
        os.makedirs(os.path.join(g, "tmp"))
        for p, c in files.items():
            open(os.path.join(g, "repo", p), "w").write(c)
        extra = {}
        if v == "wandb":
            os.makedirs(os.path.join(g, "rec"))
            open(os.path.join(g, "rec", "wandb.py"), "w").write(R.WANDB_RECORDER)
            extra = {"PYTHONPATH": os.path.join(g, "rec"), "WANDB_RECORD": os.path.join(g, "rec", "log.jsonl")}
        code, out = sandbox_run(g, envdir, f"python {script}", extra_env=extra)
        if code != 0:
            raise RuntimeError(f"golden run of {v} failed:\n{out}")
        golden[v] = truth(v, out, os.path.join(g, "rec", "log.jsonl"))
    json.dump(golden, open(os.path.join(envdir, "golden.json"), "w"), indent=1)
    return golden


def truth(v, out, rec):
    """Final values the original script reports, and its last metric line."""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if v == "plain":
        last = [ln for ln in lines if ln.startswith("step ")][-1]
        return {"metrics": {"loss": float(last.split()[-1])}, "n": 30, "last_line": last}
    if v == "rl":
        last = [ln for ln in lines if ln.startswith("step ")][-1]
        m = dict(re.findall(r"(\w+) (-?[\d.]+)", last.split("|", 1)[1]))
        return {"metrics": {"reward": float(m["reward"]), "kl": float(m["kl"])}, "n": 30, "last_line": last}
    if v == "hf":
        logs = [ln for ln in lines if ln.startswith("{'loss'")]
        return {"metrics": {"loss": float(re.search(r"'loss': ([\d.]+)", logs[-1]).group(1))}, "n": len(logs), "last_line": logs[-1]}
    if v == "wandb":
        rows = [json.loads(x) for x in open(rec)]
        losses = [r["train/loss"] for r in rows if "train/loss" in r]
        return {"metrics": {"loss": losses[-1]}, "n": len(losses), "last_line": None}
    raise ValueError(v)


def bwrap_argv(T, envdir):
    a = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
         "--symlink", "usr/lib", "/lib64", "--symlink", "usr/bin", "/sbin", "--ro-bind", "/etc", "/etc",
         "--ro-bind", "/sys", "/sys", "--proc", "/proc", "--dev", "/dev", "--dir", "/tmp", "--dir", "/var/tmp"]
    for opt in ("/opt/intel", "/opt/rocm"):  # MKL for system torch; rocm-smi for `modgrad run`'s GPU sampling
        if os.path.isdir(opt):
            a += ["--ro-bind", opt, opt]
    if os.path.isdir("/run/systemd/resolve"):
        a += ["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]
    a += ["--ro-bind", os.path.join(envdir, "bin"), BIN_MOUNT]
    if os.path.isdir(DIST):
        # modgrad-server looks for <its crate dir>/../../obs/web/dist, so the crate dir must exist too
        a += ["--ro-bind", DIST, DIST, "--dir", os.path.join(ROOT, "crates", "modgrad-obs-server")]
    a += ["--bind", T, T, "--remount-ro", "/", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
          "--die-with-parent", "--new-session", "--chdir", os.path.join(T, "repo"), "--clearenv"]
    env = {"PATH": f"{BIN_MOUNT}:{T}/home/.local/bin:/usr/local/bin:/usr/bin", "HOME": f"{T}/home", "TMPDIR": f"{T}/tmp",
           "LANG": "C.UTF-8", "TERM": "dumb", "USER": "dev", "PYTHONUNBUFFERED": "1",
           "GIT_AUTHOR_NAME": "dev", "GIT_AUTHOR_EMAIL": "dev@example.org", "GIT_COMMITTER_NAME": "dev", "GIT_COMMITTER_EMAIL": "dev@example.org"}
    return a, env


def sandbox_run(T, envdir, cmd, stdin=None, extra_env=None, timeout=CMD_TIMEOUT, argv=None):
    """Run `sh -c cmd` (or argv) in the sandbox; (exit code, merged output)."""
    a, env = bwrap_argv(T, envdir)
    env.update(extra_env or {})
    for k, v in env.items():
        a += ["--setenv", k, v]
    a += argv or ["sh", "-c", cmd]
    p = subprocess.Popen(a, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = p.communicate(stdin.encode() if stdin is not None else None, timeout=timeout)
        return p.returncode, out.decode(errors="replace")
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return 124, out.decode(errors="replace") + f"\n(killed: the command ran longer than {timeout} s)"


def setup_task(T, envdir, variant):
    shutil.rmtree(T, ignore_errors=True)
    repo = os.path.join(T, "repo")
    for d in ("repo", "home", "tmp"):
        os.makedirs(os.path.join(T, d))
    files, _ = R.VARIANTS[variant]
    for p, c in files.items():
        open(os.path.join(repo, p), "w").write(c)
    g = ["git", "-c", "user.name=dev", "-c", "user.email=dev@example.org", "-c", "init.defaultBranch=main"]
    sh(g + ["init", "-q"], cwd=repo)
    sh(g + ["add", "-A"], cwd=repo)
    sh(g + ["commit", "-q", "-m", "initial commit"], cwd=repo)
    site = os.path.join(T, "home", ".local", "lib", PYVER, "site-packages")
    wheel = glob.glob(os.path.join(envdir, "wheels", "*.whl"))[0]
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", "--target", site, wheel])


# ------------------------------------------------------- the agent's tools

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file. Example: read_file(path=\"train.py\")",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path, relative to the repository or absolute"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (create or overwrite) a text file with the given content. Example: write_file(path=\"notes.txt\", content=\"hello\\n\")",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string", "description": "The full new file content"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": f"Run a shell command (sh) in the repository and return its exit code and output (last {OUT_CAP} characters). Commands time out after {CMD_TIMEOUT} s. Example: run_command(command=\"ls -la\")",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def tool_call(T, envdir, name, args):
    """(text, is_error). File tools run inside the sandbox too, so they see and
    may write exactly what run_command does."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    args = args if isinstance(args, dict) else {}
    path = str(args.get("path") or args.get("file") or args.get("filename") or args.get("file_path") or "")
    if name == "read_file":
        if not path:
            return "error: read_file needs path", True
        code, out = sandbox_run(T, envdir, None, argv=["cat", "--", path])
        if code:
            return f"error: {out.strip()}", True
        return out[:READ_CAP] + (f"\n... (truncated at {READ_CAP} characters)" if len(out) > READ_CAP else ""), False
    if name == "write_file":
        if not path:
            return "error: write_file needs path", True
        content = str(args.get("content", ""))
        code, out = sandbox_run(T, envdir, None, stdin=content,
                                argv=["sh", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "sh", path])
        if code:
            return f"error: {out.strip()}", True
        return f"wrote {len(content)} bytes to {path}", False
    if name == "run_command":
        cmd = str(args.get("command") or args.get("cmd") or "")
        if not cmd:
            return "error: run_command needs command", True
        code, out = sandbox_run(T, envdir, cmd)
        return f"exit code {code}\n{out[-OUT_CAP:]}", code != 0
    return f"unknown tool {name!r}; the tools are read_file, write_file, run_command", True


# --------------------------------------------------------------- checker


def runs_under(root):
    """{run id: {key: [(step, value)]}} for every runs/**/metrics.jsonl below root."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "metrics.jsonl"), recursive=True):
        if "/.git/" in p:
            continue
        rel = os.path.relpath(os.path.dirname(p), root)
        parts = rel.split(os.sep)
        if "runs" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("runs")
        rid = "/".join(parts[i + 1:])
        series = {}
        for line in open(p, errors="replace"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for k, v in row.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("step", "_step", "t", "_t", "_runtime", "_timestamp"):
                    series.setdefault(k, []).append((row.get("step", row.get("_step")), v))
        out[rid] = {"series": series}
    return out


def close(a, b):
    return abs(a - b) <= 1e-4 + 1e-3 * abs(b)  # the scripts print 4-5 decimals


def run_matches(series, gt):
    """The run carries every ground-truth metric: a key naming it (any prefix,
    e.g. train/loss) with at least half the original's points, whose last
    value is the original's final one."""
    for want, val in gt["metrics"].items():
        def named(k):
            last = k.lower().split("/")[-1]
            if want == "loss":
                return "loss" in last and not any(x in k.lower() for x in ("val", "eval", "test"))
            return want in last
        if not any(named(k) and len(pts) >= max(2, gt["n"] // 2) and close(pts[-1][1], val) for k, pts in series.items()):
            return False
    return True


def good_runs(root, gt):
    return {rid: r for rid, r in runs_under(root).items() if run_matches(r["series"], gt)}


def view_commands(answer):
    """`modgrad-server ...` commands in the answer, as argument lists."""
    out = []
    for m in re.finditer(r"modgrad-server\b([^\n`'\"|&;)]*)", answer):
        out.append(m.group(1).split())
    return out


def free_port(port):
    with socket.socket() as so:
        so.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # like the server; TIME_WAIT is not busy
        try:
            so.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def check_view_command(args, repo, server_bin, good):
    """Start the command the answer gives (root resolved from the repo), see whether it serves a good run."""
    root = "obs"
    for i, x in enumerate(args):
        if x == "--root" and i + 1 < len(args):
            root = args[i + 1]
        elif x.startswith("--root="):
            root = x.split("=", 1)[1]
    root = os.path.expanduser(root.strip("'\""))
    if not free_port(VIEW_PORT):
        return False, f"port {VIEW_PORT} busy"
    p = subprocess.Popen([server_bin, "--root", root, "--listen", f"127.0.0.1:{VIEW_PORT}", "--no-static"], cwd=repo,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                ids = [r["id"] for r in requests.get(f"http://127.0.0.1:{VIEW_PORT}/api/index", timeout=1).json()["runs"]]
                hit = [i for i in ids if i in good]
                return bool(hit), f"modgrad-server --root {root} lists {ids[:6]}"
            except (requests.RequestException, ValueError, KeyError):
                if p.poll() is not None:
                    return False, f"modgrad-server --root {root} exited"
                time.sleep(0.05)
        return False, "view server did not start"
    finally:
        p.terminate()
        p.wait(timeout=10)


def check_server_url(answer, url, good, sroot):
    base = url.rstrip("/")
    seen = []
    for m in re.finditer(r"https?://[^\s)\]`'\"<>,]+", answer):
        u = m.group(0).rstrip(".:;")
        n = u.replace("://localhost:", "://127.0.0.1:")
        if not n.startswith(base):
            continue
        seen.append(u)
        path, _, frag = n[len(base):].partition("#")
        try:
            st = requests.get(base + (path or "/"), timeout=5).status_code
        except requests.RequestException as e:
            st = str(e)
        if st != 200:
            continue
        kv = dict(x.split("=", 1) for x in frag.split("&") if "=" in x)
        if "run" in kv and kv["run"] in good:
            return True, u
        if "project" in kv and any(r.startswith(kv["project"].rstrip("/") + "/") for r in good):
            return True, u
        if "board" in kv:
            for f in glob.glob(os.path.join(sroot, "boards", "**", "*.json"), recursive=True):
                if os.path.splitext(os.path.relpath(f, os.path.join(sroot, "boards")))[0] == kv["board"] and any(r in open(f).read() for r in good):
                    return True, u
    return False, f"server URLs in the answer: {seen or 'none'} (need one that answers 200 and names the run or its project)"


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout


def check_git(repo):
    if "obsp_" in git(repo, "log", "--all", "-p"):
        return False, "a token is in git history"
    if "obsp_" in git(repo, "diff", "--cached"):
        return False, "a token is staged"
    for f in git(repo, "ls-files", "-co", "--exclude-standard").split("\n"):
        p = os.path.join(repo, f)
        if f and os.path.isfile(p) and "obsp_" in open(p, errors="replace").read():
            return False, f"{f} holds a token and is not ignored (git add -A would commit it)"
    if os.path.exists(os.path.join(repo, ".modgrad.json")) and subprocess.run(["git", "check-ignore", "-q", ".modgrad.json"], cwd=repo).returncode != 0:
        return False, ".modgrad.json is not gitignored"
    return True, "no token reachable by git"


def check_original(T, envdir, variant, gt):
    script = R.VARIANTS[variant][1]
    code, out = sandbox_run(T, envdir, f"python {script}", timeout=180)
    if code != 0:
        return False, f"`python {script}` exits {code}: {out[-300:]!r}"
    if gt["last_line"] and gt["last_line"] not in out:
        return False, f"`python {script}` no longer prints {gt['last_line']!r}"
    return True, f"`python {script}` exits 0"


def check(task, T, envdir, answer, gt, server_bin, sroot=None, url=None):
    """{criterion: (ok, why)}. Order matters: the original-script rerun comes
    last, since it may add a run of its own."""
    repo = os.path.join(T, "repo")
    res = {}
    good = good_runs(sroot if task["server"] else repo, gt)
    allruns = list(runs_under(sroot if task["server"] else repo))
    res["ran"] = (bool(good), f"matching runs {sorted(good)}; all runs {allruns}; want {gt['metrics']}")
    if task["server"]:
        res["link"] = check_server_url(answer, url, good, sroot) if good else (False, "no run to link to")
    else:
        cmds = view_commands(answer)
        res["link"] = (False, "no `modgrad-server` command in the answer")
        if good:
            for c in cmds:
                ok, why = check_view_command(c, repo, server_bin, good)
                res["link"] = (ok, why)
                if ok:
                    break
        elif cmds:
            res["link"] = (False, "no run to view")
    res["no_token"] = check_git(repo)
    res["original"] = check_original(T, envdir, task["variant"], gt)
    return res


# ---------------------------------------------------------------- servers


def start_server(server_bin, root):
    if not free_port(SERVER_PORT):
        raise RuntimeError(f"port {SERVER_PORT} is busy")
    os.makedirs(root, exist_ok=True)
    p = subprocess.Popen([server_bin, "--root", root, "--listen", f"127.0.0.1:{SERVER_PORT}"] + (["--static", DIST] if os.path.isdir(DIST) else []),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{SERVER_PORT}"
    for _ in range(200):
        try:
            requests.get(url + "/api/index", timeout=1)
            return p, url
        except requests.RequestException:
            time.sleep(0.05)
    p.kill()
    raise RuntimeError("modgrad-server did not start")


# -------------------------------------------------------------- solutions


def scripted(task, T, envdir, url, negative=None):
    """The canonical solution a careful agent would follow from llms.txt.
    negative: 'no_link' (answer without a view command / URL), 'bare_url'
    (the server's front page, which does not list an anonymous project),
    'leak' (commit .modgrad.json), 'stub' (overwrite the script), 'fake' (log
    made-up lines instead of training) — each must fail."""
    v = task["variant"]
    script = R.VARIANTS[v][1]
    calls = []

    def run(cmd):
        text, err = tool_call(T, envdir, "run_command", {"command": cmd})
        calls.append({"name": "run_command", "args": {"command": cmd}, "is_error": err, "text": text})
        return text

    init = run(f"modgrad init --server {url}" if task["server"] else "modgrad init")
    name = f"{v}-demo"  # no '/': with a project token a run id must start with the project
    if negative == "fake":
        run(f"modgrad run --name {name} -- python -c \"[print(f'step {{i}} loss {{1/i:.4f}}') for i in range(10, 310, 10)]\"")
    elif v == "wandb":
        run("sed -i 's/^import wandb$/import modgrad as wandb/' train.py")
        run(f"python {script}")
    else:
        run(f"modgrad run --name {name} -- python {script}")
    if negative == "leak":
        run("git add -f .modgrad.json && git commit -qm 'add modgrad config'")
    if negative == "stub":
        tool_call(T, envdir, "write_file", {"path": script, "content": "print('step 300 loss 0.0')\n"})
    if task["server"]:
        m = re.search(r"Dashboard: (\S+)", init)
        answer = f"Training is logged to the server. Dashboard: {m.group(1) if m else url}"
    else:
        answer = f"Logged run {name}. View it: `modgrad-server --root obs`, then open http://127.0.0.1:8790/#run={name}"
    if negative == "no_link":
        answer = "Done: training is logged to modgrad."
    if negative == "bare_url":
        answer = f"Training is logged. Dashboard: {url}/"
    return answer, calls, len(calls)


SYSTEM = (
    "You are a coding agent working in the user's git repository. Your tools act on that repository: "
    "read_file and write_file take paths relative to it, and run_command runs a shell command (sh) in it. "
    "Work step by step: look at the repository, make the change, run it, check the result. "
    "When the task is done, reply to the user with a short final message.\n\n"
    "The user has given you the documentation that modgrad publishes for coding agents (its llms.txt):\n\n"
    "<llms.txt>\n{llms}\n</llms.txt>"
)


def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()


def run_agent(task, T, envdir, prompt, a, llms):
    msgs = [{"role": "system", "content": SYSTEM.format(llms=llms)}, {"role": "user", "content": prompt + ("" if a.think else " /no_think")}]
    calls, max_prompt = [], 0
    for turn in range(1, a.max_turns + 1):
        body = {"model": a.model, "messages": msgs, "tools": TOOLS, "temperature": a.temperature, "seed": a.seed + a.trial, "max_tokens": a.max_tokens}
        r = requests.post(a.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=a.timeout)
        r.raise_for_status()
        j = r.json()
        max_prompt = max(max_prompt, (j.get("usage") or {}).get("prompt_tokens", 0))
        choice = j["choices"][0]
        msg = choice["message"]
        tcs = msg.get("tool_calls") or []
        if choice.get("finish_reason") == "length" and not tcs:
            return f"(no answer: the reply hit --max-tokens {a.max_tokens}) " + strip_think(msg.get("content"))[:200], calls, turn, max_prompt
        msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": tcs} if tcs else {})})
        if not tcs:
            return strip_think(msg.get("content")), calls, turn, max_prompt
        for tc in tcs:
            fn = tc.get("function", {})
            raw = fn.get("arguments")
            t0 = time.time()
            text, err = tool_call(T, envdir, fn.get("name", ""), raw)
            calls.append({"name": fn.get("name"), "args": raw, "is_error": err, "text": text, "seconds": round(time.time() - t0, 1)})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": text})
    return "(no final answer within max turns)", calls, a.max_turns, max_prompt


# ------------------------------------------------------------ self-test


def selftest(envdir):
    """The sandbox may write only inside its task dir and cannot see the source."""
    T = os.path.join(envdir, "selftest")
    setup_task(T, envdir, "plain")
    probe = os.path.join(os.path.dirname(T), "outside-probe")
    targets = ["/x", "/tmp/x", "/var/tmp/x", "/etc/x", "/usr/x", BIN_MOUNT + "/x", os.path.join(ROOT, "x"),
               os.path.expanduser("~/x"), probe, os.path.join(envdir, "bin", "x"), os.path.join(T, "..", "x")]
    rows, ok = [], True
    for t in targets:
        _, e1 = tool_call(T, envdir, "write_file", {"path": t, "content": "x"})
        _, e2 = tool_call(T, envdir, "run_command", {"command": f"echo x > '{t}'"})
        on_host = os.path.exists(t)
        good = e1 and e2 and not on_host
        ok &= good
        rows.append(f"{'ok  ' if good else 'FAIL'} write {t}: write_file {'refused' if e1 else 'ALLOWED'}, shell {'refused' if e2 else 'ALLOWED'}, on host: {on_host}")
    for t in [os.path.join(ROOT, "Cargo.toml"), os.path.join(ROOT, "obs", "llms.txt"), os.path.expanduser("~/.bashrc"),
              os.path.join(ROOT, "crates", "modgrad-obs", "src", "lib.rs")]:
        _, e = tool_call(T, envdir, "read_file", {"path": t})
        ok &= e
        rows.append(f"{'ok  ' if e else 'FAIL'} read {t}: {'hidden' if e else 'VISIBLE'}")
    for t in ["repo/inside.txt", "home/inside.txt", "tmp/inside.txt"]:
        _, e = tool_call(T, envdir, "write_file", {"path": os.path.join(T, t), "content": "x"})
        good = not e and os.path.exists(os.path.join(T, t))
        ok &= good
        rows.append(f"{'ok  ' if good else 'FAIL'} write {T}/{t}: {'allowed' if good else 'REFUSED'}")
    code, out = sandbox_run(T, envdir, "command -v modgrad modgrad-server; python -c 'import modgrad, torch; print(modgrad.__file__)'; python -c 'import wandb' 2>&1 | tail -1")
    rows.append("tools inside: " + " | ".join(out.split("\n")))
    ok &= code == 0 and modgrad_ok(out, T)
    print("\n".join(rows))
    print("sandbox self-test:", "PASS" if ok else "FAIL")
    return ok


def modgrad_ok(out, T):
    return f"{BIN_MOUNT}/modgrad" in out and os.path.join(T, "home", ".local") in out


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=os.path.join(ROOT, "target", "release", "modgrad"))
    ap.add_argument("--obs-server", default=os.path.join(ROOT, "target", "release", "modgrad-server"))
    ap.add_argument("--data", default=os.environ.get("OBS_BENCH_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "obs-agent-bench")))
    ap.add_argument("--llms", default=os.path.join(ROOT, "obs", "llms.txt"))
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--negative", action="store_true", help="with --scripted: also run broken solutions that must fail")
    ap.add_argument("--sandbox-selftest", action="store_true")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11435/v1")
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--only")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--tag", default="", help="appended to the label (e.g. the commit)")
    ap.add_argument("--out")
    a = ap.parse_args()

    os.makedirs(a.data, exist_ok=True)
    data = os.path.abspath(a.data)
    envdir = os.path.join(data, "_env")
    golden = build_env(envdir, a.obs, a.obs_server)
    if a.sandbox_selftest:
        sys.exit(0 if selftest(envdir) else 1)
    llms = open(a.llms).read()
    label = ("scripted" if a.scripted else re.sub(r"[^\w.-]", "_", a.model) + ("-think" if a.think else "")) + "-onboard" + (f"-{a.tag}" if a.tag else "") + (f"-t{a.trial}" if a.trial else "")
    out_path = a.out or os.path.join(data, f"{label}.jsonl")
    out = open(out_path, "w")
    runs = [(t, None) for t in TASKS if not a.only or t["id"] in a.only.split(",")]
    if a.scripted and a.negative:
        runs += [(t, "no_link") for t in TASKS[:1]] + [(t, n) for t in TASKS if t["id"] == "plain_server" for n in ("leak", "bare_url")] + [(t, n) for t in TASKS[:1] for n in ("stub", "fake")]
    rows = []
    for t, neg in runs:
        tid = t["id"] + (f"!{neg}" if neg else "")
        T = os.path.join(data, label, tid)
        setup_task(T, envdir, t["variant"])
        srv, url, sroot = None, None, None
        if t["server"]:
            sroot = os.path.join(data, label, tid + ".server")
            shutil.rmtree(sroot, ignore_errors=True)
            srv, url = start_server(a.obs_server, sroot)
        prompt = t["prompt"].format(url=url)
        t0 = time.time()
        maxp = 0
        try:
            if a.scripted:
                answer, calls, turns = scripted(t, T, envdir, url, neg)
            else:
                answer, calls, turns, maxp = run_agent(t, T, envdir, prompt, a, llms)
        except Exception as e:  # noqa: BLE001
            answer, calls, turns = f"(harness error: {e})", [], 0
        dt = time.time() - t0
        try:
            res = check(t, T, envdir, answer, golden[t["variant"]], a.obs_server, sroot, url)
        finally:
            if srv:
                srv.terminate()
                srv.wait(timeout=10)
        ok = all(v[0] for v in res.values())
        rec = {"task": tid, "suite": "onboard", "pass": ok, "why": {k: list(v) for k, v in res.items()}, "answer": answer, "prompt": prompt,
               "turns": turns, "tool_calls": len(calls), "tool_errors": sum(1 for c in calls if c["is_error"]), "seconds": round(dt, 1),
               "max_prompt_tokens": maxp, "calls": calls, "model": None if a.scripted else a.model, "trial": a.trial, "negative": neg}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        rows.append(rec)
        crit = " ".join(f"{k}={'y' if v[0] else 'N'}" for k, v in res.items())
        expect = "" if not neg else ("  (negative: fails as it should)" if not ok else "  (NEGATIVE PASSED: checker bug)")
        print(f"{'PASS' if ok else 'FAIL'}  {tid:<20} {crit}  calls={len(calls):<2} turns={turns:<2} {dt:6.1f}s ctx={maxp}{expect}", flush=True)
        if not ok:
            for k, v in res.items():
                if not v[0]:
                    print(f"        {k}: {v[1][:300]}")
            print(f"        answer: {answer[:300]!r}")
    pos = [r for r in rows if not r["negative"]]
    n = sum(r["pass"] for r in pos)
    print(f"\n{label}: {n}/{len(pos)} passed, {sum(r['seconds'] for r in pos):.0f}s. Transcript: {out_path}")
    negs = [r for r in rows if r["negative"]]
    if negs:
        print(f"negative controls: {sum(not r['pass'] for r in negs)}/{len(negs)} failed as they should")
    sys.exit(0 if n == len(pos) and not any(r["pass"] for r in negs) else 1)


if __name__ == "__main__":
    main()
