//! `modgrad run [opts] -- <command...>`: wrap a training command.
//!
//! Threads: one reader per child pipe (read -> write through unchanged ->
//! hand complete lines to a channel; nothing else, so a slow network can
//! never block the child on a full pipe), one parser (lines -> rows via
//! `textlog::parse_line`), one system sampler, and one sender.
//!
//! The sender appends every row to a spool file and fsyncs it before
//! sending, and records the acknowledged byte offset after each successful
//! POST. Unacknowledged rows are resent with backoff while the server is
//! away, and if the run ends before they get through the spool stays on
//! disk for `modgrad push <dir>`. Steps are resolved here (not by the server),
//! so a resend is idempotent.
//!
//! POSTs go through `net::push_request`, which moves to HTTP/3 when the
//! server offers it (or `--h3`) and falls back to TCP on its own; the spool
//! sits in front of either transport and does not know which one is used.

use crate::net;
use crate::sys::Sampler;
use modgrad_obs::textlog;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

pub struct Opts {
    pub server: String,
    pub token: Option<String>,
    pub name: Option<String>,
    pub project: Option<String>,
    pub offline: bool,
    pub out: PathBuf,
    pub sys_interval: f64,
    pub gpu: bool,
    pub json: bool,
    pub spool_root: PathBuf,
    /// When to try HTTP/3 (`--h3`, `--no-h3`).
    pub h3: net::H3,
    /// Lineage config from `--parent <run>[@step]` and `--stage <s>`
    /// (`parent_run`, `parent_step`, `stage`); an empty object without them.
    pub config: Value,
    pub cmd: Vec<String>,
}

/// `--parent <run>[@step]` and `--stage <s>` as config keys (see "Lineage"
/// in obs/llms.txt). `Err` for a step that is not a number.
pub fn lineage_config(parent: Option<&str>, stage: Option<&str>) -> Result<Value, String> {
    let mut c = serde_json::Map::new();
    if let Some(p) = parent.map(str::trim).filter(|p| !p.is_empty()) {
        let (run, step) = match p.rsplit_once('@') {
            Some((r, s)) => (r, Some(s)),
            None => (p, None),
        };
        c.insert("parent_run".into(), json!(run));
        if let Some(s) = step {
            let v: f64 = s.parse().map_err(|_| format!("--parent {p}: '{s}' is not a step (use <run>@<step>)"))?;
            c.insert("parent_step".into(), if v.fract() == 0.0 && v.abs() < 9e15 { json!(v as i64) } else { json!(v) });
        }
    }
    if let Some(s) = stage.map(str::trim).filter(|s| !s.is_empty()) {
        c.insert("stage".into(), json!(s));
    }
    Ok(Value::Object(c))
}

/// Messages from the CLI to stderr: a line of text, or one JSON object per
/// line with `--json` (machine-readable for agents and scripts).
pub fn say(json_mode: bool, v: Value, text: impl AsRef<str>) {
    if json_mode {
        eprintln!("{v}");
    } else {
        eprintln!("modgrad: {}", text.as_ref());
    }
}

/// Step policy for parsed lines: a line's own step wins; a line without
/// one belongs to the last step seen; if the script never prints steps,
/// each metric line is the next step (0, 1, 2, ...).
#[derive(Debug, Default)]
pub struct Stepper {
    seen_step: bool,
    next_auto: u64,
    last: Option<u64>,
}

impl Stepper {
    pub fn step_for(&mut self, parsed: Option<u64>) -> u64 {
        let s = match parsed {
            Some(s) => {
                self.seen_step = true;
                s
            }
            None if self.seen_step => self.last.unwrap_or(0),
            None => {
                self.next_auto += 1;
                self.next_auto - 1
            }
        };
        self.last = Some(s);
        s
    }
}

#[derive(Debug, Clone)]
pub enum Item {
    Row(Value),
    Event(Value),
}

impl Item {
    fn to_line(&self) -> String {
        match self {
            Item::Row(r) => json!({ "row": r }).to_string(),
            Item::Event(e) => json!({ "event": e }).to_string(),
        }
    }
    fn from_line(l: &str) -> Option<Item> {
        let v: Value = serde_json::from_str(l).ok()?;
        v.get("row").map(|r| Item::Row(r.clone())).or_else(|| v.get("event").map(|e| Item::Event(e.clone())))
    }
}

enum Msg {
    Item(Item),
    Done(Value),
}

/// `run-<UTC yyyymmdd-hhmmss>-<pid hex>`.
pub fn auto_name() -> String {
    let s = modgrad_obs::now_unix() as i64;
    let (days, rem) = (s.div_euclid(86400), s.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("run-{y:04}{m:02}{d:02}-{:02}{:02}{:02}-{:04x}", rem / 3600, rem / 60 % 60, rem % 60, std::process::id() & 0xffff)
}

fn write_private(p: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = p.with_extension("tmp");
    {
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        o.open(&tmp)?.write_all(bytes)?;
    }
    std::fs::rename(tmp, p)
}

// ------------------------------------------------------------ remote sink

/// The server side of a run: create, spool, send, finish.
pub struct Remote {
    pub server: String,
    token: Option<String>,
    name: String,
    project: Option<String>,
    meta: Value,
    pub run: Option<String>,
    pub dashboard: Option<String>,
    pub dir: PathBuf,
    spool: std::fs::File,
    offset: u64,
    acked: u64,
    unacked: VecDeque<(Item, u64)>,
    backoff: Duration,
    next_try: Instant,
    /// A non-retryable error (bad token, ...): stop sending, keep spooling.
    pub fatal: Option<String>,
    json: bool,
    warned: bool,
    /// When to try HTTP/3 (`--h3`, `--no-h3`).
    pub h3: net::H3,
    /// Config sent with the create (`--parent`, `--stage`); `null` sends none.
    pub config: Value,
}

const MAX_POST_ITEMS: usize = 1000;

impl Remote {
    pub fn new(server: &str, token: Option<String>, name: &str, project: Option<String>, meta: Value, spool_root: &Path, json_mode: bool) -> std::io::Result<Remote> {
        let key = match &project {
            Some(p) if !name.contains('/') => format!("{p}/{name}"),
            _ => name.to_string(),
        };
        let dir = spool_root.join(key.replace('/', "__"));
        std::fs::create_dir_all(&dir)?;
        // a crash mid-write leaves a partial last line: cut it so appends start clean
        modgrad_obs::repair_tail(&dir.join("spool.jsonl"))?;
        let spool = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("spool.jsonl"))?;
        let offset = spool.metadata()?.len();
        let acked = std::fs::read_to_string(dir.join("acked")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0u64).min(offset);
        let mut r = Remote {
            server: server.trim_end_matches('/').to_string(),
            token,
            name: name.to_string(),
            project,
            meta,
            run: None,
            dashboard: None,
            dir,
            spool,
            offset,
            acked,
            unacked: VecDeque::new(),
            backoff: Duration::from_millis(500),
            next_try: Instant::now(),
            fatal: None,
            json: json_mode,
            warned: false,
            h3: net::H3::Auto,
            config: Value::Null,
        };
        // a previous attempt's unsent tail (same run name) goes first
        r.load_unacked()?;
        r.save_meta(None)?;
        Ok(r)
    }

    /// Reopen a spool directory left behind (for `modgrad push`).
    pub fn reopen(dir: &Path, json_mode: bool) -> std::io::Result<(Remote, Option<Value>)> {
        let m: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        let root = dir.parent().unwrap_or(Path::new("."));
        let mut r = Remote::new(
            m["server"].as_str().unwrap_or_default(),
            m["token"].as_str().map(String::from),
            m["name"].as_str().unwrap_or_default(),
            m["project"].as_str().map(String::from),
            m["meta"].clone(),
            root,
            json_mode,
        )?;
        r.run = m["run"].as_str().map(String::from);
        r.dashboard = m["dashboard"].as_str().map(String::from);
        Ok((r, m.get("finish").cloned().filter(|f| !f.is_null())))
    }

    fn load_unacked(&mut self) -> std::io::Result<()> {
        let mut f = std::fs::File::open(self.dir.join("spool.jsonl"))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut pos = self.acked as usize;
        for line in buf[self.acked as usize..].split_inclusive(|&b| b == b'\n') {
            pos += line.len();
            if let Some(it) = Item::from_line(String::from_utf8_lossy(line).trim()) {
                self.unacked.push_back((it, pos as u64));
            }
        }
        Ok(())
    }

    fn save_meta(&self, finish: Option<&Value>) -> std::io::Result<()> {
        let m = json!({ "server": self.server, "token": self.token, "name": self.name, "project": self.project, "meta": self.meta, "run": self.run, "dashboard": self.dashboard, "finish": finish });
        write_private(&self.dir.join("meta.json"), &serde_json::to_vec_pretty(&m)?)
    }

    pub fn pending(&self) -> usize {
        self.unacked.len()
    }

    /// Append items to the spool (durable before any send).
    pub fn spool(&mut self, items: Vec<Item>) -> std::io::Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let mut text = String::new();
        let mut ends = Vec::with_capacity(items.len());
        for it in &items {
            text.push_str(&it.to_line());
            text.push('\n');
            ends.push(self.offset + text.len() as u64);
        }
        self.spool.write_all(text.as_bytes())?;
        self.spool.sync_data()?;
        self.offset += text.len() as u64;
        self.unacked.extend(items.into_iter().zip(ends));
        Ok(())
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.server)
    }

    fn post(&self, path: &str, body: &Value, token: Option<&str>) -> std::io::Result<net::Resp> {
        let (r, notice) = net::push_request("POST", &self.url(path), token, Some(body.to_string().as_bytes()), "application/json", Duration::from_secs(10), self.h3);
        if let Some(n) = notice {
            say(self.json, json!({"event": "transport", "transport": n.transport, "fallback": n.fallback, "server": self.server}), n.text);
        }
        r
    }

    fn retryable(status: u16) -> bool {
        status == 408 || status == 429 || status >= 500
    }

    fn fail(&mut self, err: String, retry: bool) {
        if retry {
            if !self.warned {
                say(self.json, json!({"event": "server_unreachable", "server": self.server, "error": err, "spool": self.dir}), format!("server {} unreachable ({err}); spooling to {} and retrying", self.server, self.dir.display()));
                self.warned = true;
            }
            self.next_try = Instant::now() + self.backoff;
            self.backoff = (self.backoff * 2).min(Duration::from_secs(30));
        } else {
            say(self.json, json!({"event": "server_error", "error": err}), format!("server refused: {err}"));
            self.fatal = Some(err);
        }
    }

    /// Create (or resume) the run; idempotent. Returns true once created.
    pub fn ensure_created(&mut self) -> bool {
        if self.run.is_some() {
            return true;
        }
        if self.fatal.is_some() || Instant::now() < self.next_try {
            return false;
        }
        let mut body = json!({ "name": self.name, "project": self.project, "meta": self.meta });
        if self.config.is_object() {
            body["config"] = self.config.clone();
        }
        match self.post("/api/v1/runs", &body, self.token.as_deref()) {
            Ok(r) if r.ok() => {
                let j = r.json();
                self.run = j["run"].as_str().map(String::from);
                self.dashboard = j["dashboard"].as_str().map(String::from);
                if let Some(t) = j["token"].as_str() {
                    self.token = Some(t.to_string());
                }
                if self.warned {
                    say(self.json, json!({"event": "server_back", "server": self.server}), "server reachable again; sending spooled rows");
                    self.warned = false;
                }
                self.backoff = Duration::from_millis(500);
                let _ = self.save_meta(None);
                true
            }
            Ok(r) => {
                let retry = Self::retryable(r.status);
                self.fail(format!("HTTP {}: {}", r.status, r.error_text()), retry);
                false
            }
            Err(e) => {
                self.fail(e.to_string(), true);
                false
            }
        }
    }

    /// Send what is unacknowledged (up to a few posts). Returns true when
    /// nothing is left.
    pub fn flush(&mut self) -> bool {
        while !self.unacked.is_empty() {
            if !self.ensure_created() || Instant::now() < self.next_try || self.fatal.is_some() {
                return false;
            }
            let n = self.unacked.len().min(MAX_POST_ITEMS);
            let (mut rows, mut events) = (Vec::new(), Vec::new());
            for (it, _) in self.unacked.iter().take(n) {
                match it {
                    Item::Row(r) => rows.push(r.clone()),
                    Item::Event(e) => events.push(e.clone()),
                }
            }
            let run = self.run.clone().unwrap();
            match self.post(&format!("/api/v1/runs/{run}/log"), &json!({ "rows": rows, "events": events }), self.token.as_deref()) {
                Ok(r) if r.ok() => {
                    let end = self.unacked[n - 1].1;
                    self.unacked.drain(..n);
                    self.acked = end;
                    let _ = write_private(&self.dir.join("acked"), end.to_string().as_bytes());
                    self.backoff = Duration::from_millis(500);
                    if self.warned {
                        say(self.json, json!({"event": "server_back", "server": self.server}), "server reachable again; spooled rows sent");
                        self.warned = false;
                    }
                }
                Ok(r) => {
                    let retry = Self::retryable(r.status);
                    self.fail(format!("HTTP {}: {}", r.status, r.error_text()), retry);
                    return false;
                }
                Err(e) => {
                    self.fail(e.to_string(), true);
                    return false;
                }
            }
        }
        true
    }

    /// Flush everything (retrying until `deadline`), then finish. Removes
    /// the spool when all of it got through.
    pub fn close(&mut self, summary: &Value, deadline: Instant) -> bool {
        let _ = self.save_meta(Some(summary));
        while !self.flush() && self.fatal.is_none() && Instant::now() < deadline {
            std::thread::sleep(self.next_try.saturating_duration_since(Instant::now()).min(Duration::from_millis(500)).max(Duration::from_millis(50)));
        }
        if !self.unacked.is_empty() || self.run.is_none() {
            return false;
        }
        let run = self.run.clone().unwrap();
        match self.post(&format!("/api/v1/runs/{run}/finish"), summary, self.token.as_deref()) {
            Ok(r) if r.ok() => {
                let _ = std::fs::remove_dir_all(&self.dir);
                true
            }
            Ok(r) => {
                say(self.json, json!({"event": "server_error", "error": r.error_text()}), format!("finish refused: {}", r.error_text()));
                false
            }
            Err(_) => false,
        }
    }
}

// ------------------------------------------------------------ local sink

struct Local {
    run: modgrad_obs::Run,
    dir: PathBuf,
}

impl Local {
    fn write(&mut self, items: Vec<Item>) {
        for it in items {
            let r = match &it {
                Item::Row(r) => {
                    let step = r["step"].as_u64().unwrap_or(0);
                    let t = r["t"].as_f64().unwrap_or(0.0);
                    let kv: Vec<(&str, f64)> = r.as_object().into_iter().flatten().filter(|(k, _)| *k != "step" && *k != "t").map(|(k, v)| (k.as_str(), v.as_f64().unwrap_or(f64::NAN))).collect();
                    self.run.log_with_time(step, t, &kv)
                }
                Item::Event(e) => self.run.event_with_time(e["step"].as_u64().unwrap_or(0), e["t"].as_f64().unwrap_or(0.0), e["text"].as_str().unwrap_or("")),
            };
            if let Err(e) = r {
                eprintln!("modgrad: local write: {e}");
            }
        }
    }
}

// ------------------------------------------------------------ the wrapper

fn reader<R: Read + Send + 'static, W: Write + Send + 'static>(mut src: R, mut dst: W, tx: mpsc::Sender<String>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        let mut line: Vec<u8> = Vec::new();
        loop {
            let n = match src.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            // pass through unchanged, first
            let _ = dst.write_all(&buf[..n]);
            let _ = dst.flush();
            for &b in &buf[..n] {
                if b == b'\n' {
                    let _ = tx.send(String::from_utf8_lossy(&line).into_owned());
                    line.clear();
                } else {
                    line.push(b);
                }
            }
            // progress bars redraw with '\r' and may never print '\n'
            if line.len() > 64 * 1024 {
                let cut = line.iter().rposition(|&b| b == b'\r').map_or(line.len() - 4096, |i| i + 1);
                line.drain(..cut);
            }
        }
        if !line.is_empty() {
            let _ = tx.send(String::from_utf8_lossy(&line).into_owned());
        }
    })
}

#[cfg(unix)]
fn ignore_sigint() {
    // Ctrl-C reaches the whole foreground process group: let the child
    // handle it and exit, then record its exit and finish the run.
    unsafe extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
    }
    const SIGINT: i32 = 2;
    const SIG_IGN: usize = 1;
    unsafe {
        signal(SIGINT, SIG_IGN);
    }
}
#[cfg(not(unix))]
fn ignore_sigint() {}

fn exit_code(st: &std::process::ExitStatus) -> (i32, Option<i32>) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = st.signal() {
            return (128 + sig, Some(sig));
        }
    }
    (st.code().unwrap_or(1), None)
}

pub fn run(o: Opts) -> i32 {
    if o.cmd.is_empty() {
        eprintln!("usage: modgrad run [--server URL] [--token T] [--name N] [--project P] [--parent RUN[@STEP]] [--stage S] [--offline] -- <command...>");
        return 2;
    }
    let name = o.name.clone().unwrap_or_else(auto_name);
    let detected = modgrad_obs::Meta::detect();
    let command = o.cmd.join(" ");
    let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default();
    let meta = json!({ "git": detected.git, "hostname": detected.hostname, "command": command, "cwd": cwd, "client": "modgrad run" });

    // the sink: a local run directory, or the server through the spool
    let mut local = None;
    let mut remote = None;
    // what the child sees: a Python client in it logs into this run instead of making a second one
    let mut wrapped: Vec<(&str, String)>;
    let mut view_hint = None;
    if o.offline {
        let full = match &o.project {
            Some(p) if !name.contains('/') => format!("{p}/{name}"),
            _ => name.clone(),
        };
        let mut m = detected.clone();
        m.project = o.project.clone();
        m.extra = Some(json!({ "command": command, "cwd": cwd, "client": "modgrad run" }));
        match modgrad_obs::Run::create_with_meta(&o.out, &full, &o.config, m) {
            Ok(run) => {
                let dir = run.dir().to_path_buf();
                say(o.json, json!({"event": "start", "mode": "offline", "run": full, "dir": dir}), format!("offline run {full} -> {} (view: modgrad-server --root {})", dir.display(), o.out.display()));
                let out = std::path::absolute(&o.out).unwrap_or_else(|_| o.out.clone());
                wrapped = vec![("MODGRAD_WRAPPED_RUN", full.clone()), ("MODGRAD_DATA", out.display().to_string())];
                view_hint = Some(format!("view: `modgrad-server --root {}` (serves until Ctrl-C), then open http://127.0.0.1:8790/#run={}", o.out.display(), full.replace('/', "%2F")));
                local = Some(Local { run, dir });
            }
            Err(e) => {
                eprintln!("modgrad: cannot create local run: {e}");
                return 1;
            }
        }
    } else {
        match Remote::new(&o.server, o.token.clone(), &name, o.project.clone(), meta, &o.spool_root, o.json) {
            Ok(mut r) => {
                r.h3 = o.h3;
                r.config = o.config.clone();
                r.ensure_created();
                match (&r.run, &r.dashboard) {
                    (Some(run), Some(d)) => say(o.json, json!({"event": "start", "mode": "server", "run": run, "dashboard": d, "server": r.server, "spool": r.dir}), format!("run {run} -> {d}")),
                    _ if r.fatal.is_some() => {
                        eprintln!("modgrad: not starting the command; fix the error above (or use --offline)");
                        return 1;
                    }
                    _ => say(o.json, json!({"event": "start", "mode": "spooling", "run": name, "server": r.server, "spool": r.dir}), format!("run {name}: waiting for {}", r.server)),
                }
                // the server prefixes the token's project to `name`, for the client as for us
                wrapped = vec![("MODGRAD_WRAPPED_RUN", name.clone()), ("MODGRAD_URL", r.server.clone())];
                if let Some(t) = &o.token {
                    wrapped.push(("MODGRAD_TOKEN", t.clone()));
                }
                if let Some(p) = &o.project {
                    wrapped.push(("MODGRAD_PROJECT", p.clone()));
                }
                remote = Some(r);
            }
            Err(e) => {
                eprintln!("modgrad: cannot open spool under {}: {e}", o.spool_root.display());
                return 1;
            }
        }
    }

    let mut cmd = std::process::Command::new(&o.cmd[0]);
    cmd.args(&o.cmd[1..]).stdin(std::process::Stdio::inherit()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    if std::env::var_os("PYTHONUNBUFFERED").is_none() {
        // a piped stdout makes Python block-buffer; metrics would arrive in bursts
        cmd.env("PYTHONUNBUFFERED", "1");
    }
    for (k, v) in &wrapped {
        cmd.env(k, v);
    }
    let t0 = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("modgrad: cannot start {:?}: {e}", o.cmd[0]);
            if let Some(mut r) = remote {
                r.close(&json!({ "exit_code": 127, "error": e.to_string() }), Instant::now() + Duration::from_secs(5));
            }
            return 127;
        }
    };
    ignore_sigint();
    let pid = child.id();

    let (ltx, lrx) = mpsc::channel::<String>();
    let r_out = reader(child.stdout.take().unwrap(), std::io::stdout(), ltx.clone());
    let r_err = reader(child.stderr.take().unwrap(), std::io::stderr(), ltx);
    let (stx, srx) = mpsc::channel::<Msg>();
    let last_step = Arc::new(AtomicU64::new(0));

    // parser: lines -> rows
    let parser = {
        let (stx, last_step) = (stx.clone(), last_step.clone());
        std::thread::spawn(move || {
            let mut st = Stepper::default();
            let mut rows = 0u64;
            for line in lrx {
                let Some(r) = textlog::parse_line(&line) else { continue };
                if r.values.is_empty() {
                    if let Some(s) = r.step {
                        st.step_for(Some(s));
                        last_step.store(s, Ordering::Relaxed);
                    }
                    continue;
                }
                let step = st.step_for(r.step);
                last_step.store(step, Ordering::Relaxed);
                let mut row = serde_json::Map::new();
                row.insert("step".into(), json!(step));
                row.insert("t".into(), json!(t0.elapsed().as_secs_f64()));
                for (k, v) in r.values {
                    row.insert(k, if v.is_finite() { json!(v) } else { Value::Null });
                }
                rows += 1;
                let _ = stx.send(Msg::Item(Item::Row(Value::Object(row))));
            }
            rows
        })
    };

    // system sampler
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = (o.sys_interval > 0.0).then(|| {
        let (stx, stop, last_step) = (stx.clone(), stop.clone(), last_step.clone());
        let (interval, gpu, json_mode) = (Duration::from_secs_f64(o.sys_interval), o.gpu, o.json);
        std::thread::spawn(move || {
            let mut s = Sampler::new(pid, gpu);
            if s.gpu_name() != "none" && !json_mode {
                eprintln!("modgrad: sampling GPU via {} every {:.0}s", s.gpu_name(), interval.as_secs_f64());
            }
            while !stop.load(Ordering::Relaxed) {
                let v = s.sample();
                if v.is_empty() {
                    break;
                }
                let mut row = serde_json::Map::new();
                row.insert("step".into(), json!(last_step.load(Ordering::Relaxed)));
                row.insert("t".into(), json!(t0.elapsed().as_secs_f64()));
                for (k, x) in v {
                    row.insert(k, json!((x * 100.0).round() / 100.0));
                }
                let _ = stx.send(Msg::Item(Item::Row(Value::Object(row))));
                let until = Instant::now() + interval;
                while Instant::now() < until && !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        })
    });

    let _ = stx.send(Msg::Item(Item::Event(json!({ "step": 0, "t": 0.0, "text": format!("modgrad run: {command}") }))));

    // sender: spool + post (or local writes), every 250 ms
    let json_mode = o.json;
    let sender = std::thread::spawn(move || -> (bool, Option<String>) {
        let tick = Duration::from_millis(250);
        let mut batch: Vec<Item> = Vec::new();
        let mut last_flush = Instant::now();
        let mut done = None;
        loop {
            match srx.recv_timeout(tick) {
                Ok(Msg::Item(it)) => batch.push(it),
                Ok(Msg::Done(s)) => done = Some(s),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {}
            }
            let due = last_flush.elapsed() >= tick || batch.len() >= 256 || done.is_some();
            if due {
                let items = std::mem::take(&mut batch);
                if let Some(l) = local.as_mut() {
                    l.write(items);
                } else if let Some(r) = remote.as_mut() {
                    if let Err(e) = r.spool(items) {
                        eprintln!("modgrad: spool write failed: {e}");
                    }
                    r.flush();
                }
                last_flush = Instant::now();
            }
            if let Some(summary) = done.take() {
                if let Some(l) = local.take() {
                    let wall = summary["wall_s"].as_f64().unwrap_or(0.0);
                    let dir = l.dir.clone();
                    let ok = l.run.finish_with_time(&summary, wall).is_ok();
                    return (ok, Some(dir.display().to_string()));
                }
                let mut r = remote.take().unwrap();
                let ok = r.close(&summary, Instant::now() + Duration::from_secs(10));
                if !ok {
                    say(json_mode, json!({"event": "unsent", "rows": r.pending(), "spool": r.dir, "resend": format!("modgrad push {}", r.dir.display())}), format!("{} rows not yet on the server; kept in {} (resend: modgrad push {})", r.pending(), r.dir.display(), r.dir.display()));
                }
                return (ok, r.dashboard.clone());
            }
        }
    });

    let status = child.wait();
    let wall = t0.elapsed().as_secs_f64();
    let _ = r_out.join();
    let _ = r_err.join();
    let rows = parser.join().unwrap_or(0);
    stop.store(true, Ordering::Relaxed);
    if let Some(s) = sampler {
        let _ = s.join();
    }
    let (code, signal) = match &status {
        Ok(st) => exit_code(st),
        Err(_) => (1, None),
    };
    let mut summary = json!({ "exit_code": code, "wall_s": (wall * 1000.0).round() / 1000.0, "command": command, "metric_rows": rows });
    if let Some(s) = signal {
        summary["signal"] = json!(s);
    }
    let text = match signal {
        Some(s) => format!("killed by signal {s}"),
        None => format!("exit {code}"),
    };
    let _ = stx.send(Msg::Item(Item::Event(json!({ "step": last_step.load(Ordering::Relaxed), "t": wall, "text": format!("modgrad run: {text} after {wall:.1}s") }))));
    let _ = stx.send(Msg::Done(summary));
    drop(stx);
    let (ok, where_) = sender.join().unwrap_or((false, None));
    say(o.json, json!({"event": "end", "exit_code": code, "signal": signal, "wall_s": wall, "metric_rows": rows, "synced": ok, "dashboard": where_}), format!("{text} after {wall:.1}s, {rows} metric rows{}", where_.map(|w| format!(" -> {w}")).unwrap_or_default()));
    if let Some(v) = view_hint.filter(|_| !o.json) {
        eprintln!("modgrad: {v}");
    }
    code
}

/// `modgrad push <spool dir>`: resend what a previous `modgrad run` left behind.
pub fn push(dir: &Path, json_mode: bool, h3: net::H3) -> i32 {
    let (mut r, finish) = match Remote::reopen(dir, json_mode) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("modgrad push: {}: {e}", dir.display());
            return 1;
        }
    };
    r.h3 = h3;
    let n = r.pending();
    let summary = finish.unwrap_or(json!({ "pushed_later": true }));
    let ok = r.close(&summary, Instant::now() + Duration::from_secs(60));
    say(json_mode, json!({"event": "push", "rows": n, "synced": ok, "run": r.run, "dashboard": r.dashboard}), if ok { format!("sent {n} rows -> {}", r.dashboard.clone().unwrap_or_default()) } else { format!("{} rows still unsent; spool kept in {}", r.pending(), dir.display()) });
    if ok { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineage_flags() {
        assert_eq!(lineage_config(Some("pt/base@3000"), Some("sft")).unwrap(), json!({"parent_run": "pt/base", "parent_step": 3000, "stage": "sft"}));
        assert_eq!(lineage_config(Some("base"), None).unwrap(), json!({"parent_run": "base"}));
        assert_eq!(lineage_config(Some("base@1.5"), None).unwrap(), json!({"parent_run": "base", "parent_step": 1.5}));
        assert_eq!(lineage_config(None, None).unwrap(), json!({}));
        assert!(lineage_config(Some("base@x"), None).is_err());
    }

    #[test]
    fn stepper_policy() {
        let mut s = Stepper::default();
        // no steps printed: each metric line is the next step
        assert_eq!((s.step_for(None), s.step_for(None), s.step_for(None)), (0, 1, 2));
        // once steps appear, they win and stepless lines join the last one
        assert_eq!(s.step_for(Some(100)), 100);
        assert_eq!(s.step_for(None), 100);
        assert_eq!(s.step_for(Some(200)), 200);
    }

    #[test]
    fn spool_survives_restart_and_tracks_acks() {
        let root = std::env::temp_dir().join(format!("obs-spool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // nothing listens on port 9: every send fails and stays spooled
        let mut r = Remote::new("http://127.0.0.1:9", None, "p/r", None, json!({}), &root, true).unwrap();
        r.spool(vec![Item::Row(json!({"step": 0, "x": 1})), Item::Event(json!({"step": 0, "text": "hi"}))]).unwrap();
        assert!(!r.flush());
        assert_eq!(r.pending(), 2);
        // a torn tail (crash mid-write) is ignored on reopen
        std::fs::OpenOptions::new().append(true).open(r.dir.join("spool.jsonl")).unwrap().write_all(b"{\"row\": {\"st").unwrap();
        let (r2, _) = Remote::reopen(&root.join("p__r"), true).unwrap();
        assert_eq!(r2.pending(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
