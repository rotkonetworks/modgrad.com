//! The project file (`.modgrad.json`) and `modgrad init`.
//!
//! `modgrad init` writes the file in the current directory; `run`, `tail`
//! and `mcp` (and the Python client) find it by walking up from the
//! current directory. It holds either a server, its project and a write
//! token, or (local mode, no server) the data directory:
//!
//! ```text
//! {"server": "https://obs.example.org", "project": "anon-…", "token": "obsp_…", "expires_unix": …, "board": "…"}
//! {"data": "obs"}
//! ```
//!
//! Every setting resolves as flag > environment > project file > default.
//! A token from the file is only sent to the server the file names.

use modgrad_obs::net;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const FILE: &str = ".modgrad.json";
pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8790";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Project {
    pub server: Option<String>,
    pub project: Option<String>,
    pub token: Option<String>,
    pub expires_unix: Option<f64>,
    pub board: Option<String>,
    /// Local mode: the obs directory, relative to the file's directory.
    pub data: Option<String>,
}

fn s(v: &Value, k: &str) -> Option<String> {
    v[k].as_str().filter(|x| !x.is_empty()).map(String::from)
}

pub fn same_server(a: &str, b: &str) -> bool {
    a.trim_end_matches('/') == b.trim_end_matches('/')
}

impl Project {
    pub fn from_json(v: &Value) -> Project {
        Project { server: s(v, "server").map(|x| x.trim_end_matches('/').to_string()), project: s(v, "project"), token: s(v, "token"), expires_unix: v["expires_unix"].as_f64(), board: s(v, "board"), data: s(v, "data") }
    }

    pub fn to_json(&self) -> Value {
        let mut v = json!({ "note": if self.token.is_some() { "written by `modgrad init`; holds a write token: keep it out of git" } else { "written by `modgrad init`" } });
        for (k, x) in [("server", &self.server), ("project", &self.project), ("token", &self.token), ("board", &self.board), ("data", &self.data)] {
            if let Some(x) = x {
                v[k] = json!(x);
            }
        }
        if let Some(e) = self.expires_unix {
            v["expires_unix"] = json!(e);
        }
        v
    }

    /// The token, but only for `server` (never sent to a server it was not issued by).
    pub fn token_for(&self, server: &str) -> Option<String> {
        self.server.as_deref().filter(|s| same_server(s, server)).and(self.token.clone())
    }

    /// Local mode: no server named.
    pub fn local(&self) -> bool {
        self.server.is_none()
    }
}

/// A project file found on disk: the directory holding it, and its content.
#[derive(Debug, Clone)]
pub struct Found {
    pub dir: PathBuf,
    pub p: Project,
}

impl Found {
    /// The local data directory (`data`, default `obs`), resolved against the file's directory.
    pub fn data_dir(&self) -> PathBuf {
        self.dir.join(self.p.data.as_deref().unwrap_or("obs"))
    }
}

/// Read `<dir>/.modgrad.json`. `Ok(None)` if there is none.
pub fn read(dir: &Path) -> Result<Option<Project>, String> {
    let path = dir.join(FILE);
    match std::fs::read(&path) {
        Ok(b) => serde_json::from_slice::<Value>(&b).map(|v| Some(Project::from_json(&v))).map_err(|e| format!("{}: not valid JSON ({e}); fix or delete it", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// The nearest project file at or above `start`. An unreadable file is
/// reported on stderr and skipped, so a broken file never stops training.
pub fn find(start: &Path) -> Option<Found> {
    for d in start.ancestors() {
        match read(d) {
            Ok(Some(p)) => return Some(Found { dir: d.to_path_buf(), p }),
            Ok(None) => {}
            Err(e) => {
                eprintln!("modgrad: ignoring {e}");
                return None;
            }
        }
    }
    None
}

/// `find` from the current directory.
pub fn find_cwd() -> Option<Found> {
    std::env::current_dir().ok().and_then(|d| find(&d))
}

// ------------------------------------------------------------------ init

#[derive(Debug, Default)]
pub struct InitOpts {
    /// `--server` or `MODGRAD_URL`.
    pub server: Option<String>,
    pub token: Option<String>,
    pub project: Option<String>,
    /// `--data` or `MODGRAD_DATA` (local mode).
    pub data: Option<String>,
    /// `--local`: local mode even if the existing file names a server.
    pub local: bool,
    /// `--mcp claude|cursor`.
    pub mcp: Option<String>,
    /// The `command` written into the MCP config.
    pub exe: String,
}

#[derive(Debug)]
pub struct Report {
    pub p: Project,
    /// "created", "updated" or "unchanged".
    pub file: &'static str,
    /// "created", "added" or "present".
    pub gitignore: &'static str,
    pub gitignore_lines: Vec<String>,
    /// (config file, "added" | "updated" | "unchanged") for `--mcp`.
    pub mcp: Option<(String, &'static str)>,
    /// Training scripts found in the directory, for the next steps.
    pub scripts: Vec<Script>,
}

/// A training script `init` found, and how to get its metrics into a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub file: String,
    /// It imports wandb: swap the import (the client then logs), don't wrap it.
    pub wandb: bool,
}

/// Training scripts at the top of `dir`: `*.py` whose name starts with
/// `train` or `finetune`, or with a `__main__` block that calls `.backward(`
/// or builds a `Trainer(` (a library like `minitrainer.py` has neither).
/// Those names first, at most 3.
pub fn find_scripts(dir: &Path) -> Vec<Script> {
    let mut v: Vec<(bool, String, bool)> = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.ends_with(".py") || name.starts_with("test_") || name == "setup.py" || !e.path().is_file() {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(e.path()) else { continue };
        let lower = name.to_lowercase();
        let named = lower.starts_with("train") || lower.starts_with("finetune");
        let main = src.contains("__main__") && (src.contains(".backward(") || src.contains("Trainer("));
        if !named && !main {
            continue;
        }
        let wandb = src.lines().map(str::trim).any(|l| l == "import wandb" || l.starts_with("import wandb ") || l.starts_with("from wandb "));
        v.push((!named, name, wandb));
    }
    v.sort();
    v.truncate(3);
    v.into_iter().map(|(_, file, wandb)| Script { file, wandb }).collect()
}

fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        std::io::Write::write_all(&mut o.open(&tmp)?, bytes)?;
    }
    std::fs::rename(&tmp, path)
}

/// `POST /api/v1/projects`: a new anonymous project.
fn new_project(server: &str) -> Result<Project, String> {
    let r = net::request("POST", &format!("{server}/api/v1/projects"), None, None, "application/json", Duration::from_secs(10))
        .map_err(|e| format!("cannot reach {server}: {e}. Start a server (`modgrad-server --root obs --dev`), or run `modgrad init` without --server for local mode"))?;
    if !r.ok() {
        return Err(format!("{server} refused to create a project (HTTP {}: {}). Pass a token instead: modgrad init --server {server} --token obsp_…", r.status, r.error_text()));
    }
    let j = r.json();
    let (Some(project), Some(token)) = (s(&j, "project"), s(&j, "token")) else {
        return Err(format!("{server}: unexpected reply to POST /api/v1/projects: {j}"));
    };
    Ok(Project { server: Some(server.to_string()), board: s(&j, "board").or_else(|| Some(format!("{server}/#project={project}"))), project: Some(project), token: Some(token), expires_unix: j["expires_unix"].as_f64(), data: None })
}

/// Add `lines` to `<dir>/.gitignore` unless already there.
fn gitignore(dir: &Path, lines: &[String]) -> Result<&'static str, String> {
    let path = dir.join(".gitignore");
    let old = std::fs::read_to_string(&path).ok();
    let have: Vec<&str> = old.as_deref().unwrap_or("").lines().map(str::trim).collect();
    let missing: Vec<&String> = lines.iter().filter(|l| !have.contains(&l.as_str()) && !have.contains(&l.trim_start_matches('/'))).collect();
    if missing.is_empty() {
        return Ok("present");
    }
    let mut text = old.clone().unwrap_or_default();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str("# modgrad: project file (may hold a write token) and local runs\n");
    for l in missing {
        text.push_str(l);
        text.push('\n');
    }
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(if old.is_none() { "created" } else { "added" })
}

/// Merge `mcpServers.modgrad` into a JSON MCP config, keeping every other entry.
fn register_mcp(path: &Path, exe: &str) -> Result<&'static str, String> {
    let mut v = match std::fs::read(path) {
        Ok(b) => serde_json::from_slice::<Value>(&b).map_err(|e| format!("{}: not valid JSON ({e}); not touching it", path.display()))?,
        Err(_) => json!({}),
    };
    if !v.is_object() || !(v["mcpServers"].is_null() || v["mcpServers"].is_object()) {
        return Err(format!("{}: expected {{\"mcpServers\": {{…}}}}; not touching it", path.display()));
    }
    let entry = json!({ "command": exe, "args": ["mcp"] });
    let had = v["mcpServers"].get("modgrad").cloned();
    if had.as_ref() == Some(&entry) {
        return Ok("unchanged");
    }
    v["mcpServers"]["modgrad"] = entry;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut text = serde_json::to_string_pretty(&v).unwrap_or_default();
    text.push('\n');
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(if had.is_some() { "updated" } else { "added" })
}

/// Set up `dir` as a modgrad project. Idempotent: an existing file for the
/// same server keeps its token (no new project) unless the token expires
/// within the hour.
pub fn init(dir: &Path, o: &InitOpts) -> Result<Report, String> {
    let mcp_rel = match o.mcp.as_deref() {
        None => None,
        Some("claude" | "claude-code") => Some(".mcp.json"),
        Some("cursor") => Some(".cursor/mcp.json"),
        Some(other) => return Err(format!("--mcp {other:?}: supported clients are claude (writes .mcp.json) and cursor (.cursor/mcp.json)")),
    };
    let old = read(dir)?;
    let server = o.server.clone().map(|s| s.trim_end_matches('/').to_string()).or_else(|| if o.local { None } else { old.as_ref().and_then(|p| p.server.clone()) });
    let p = match server {
        Some(server) => {
            if !server.starts_with("http://") && !server.starts_with("https://") {
                return Err(format!("--server {server:?}: want a URL like http://127.0.0.1:8790"));
            }
            let reuse = old.as_ref().filter(|p| p.token_for(&server).is_some() && p.expires_unix.is_none_or(|e| e > modgrad_obs::now_unix() + 3600.0));
            match (&o.token, reuse) {
                (Some(t), _) => Project { server: Some(server.clone()), project: o.project.clone(), token: Some(t.clone()), board: o.project.as_ref().map(|p| format!("{server}/#project={p}")), ..Default::default() },
                (None, Some(p)) => Project { data: None, ..p.clone() },
                (None, None) => new_project(&server)?,
            }
        }
        None => Project { data: Some(o.data.clone().or_else(|| old.as_ref().and_then(|p| p.data.clone())).unwrap_or_else(|| "obs".into())), ..Default::default() },
    };
    let file = match &old {
        Some(x) if *x == p => "unchanged",
        Some(_) => "updated",
        None => "created",
    };
    if file != "unchanged" {
        let mut text = serde_json::to_string_pretty(&p.to_json()).unwrap_or_default();
        text.push('\n');
        write_private(&dir.join(FILE), text.as_bytes()).map_err(|e| format!("{}: {e}", dir.join(FILE).display()))?;
    }
    // the token file always; local data too (runs are not source)
    let mut lines = vec![format!("/{FILE}")];
    if let Some(d) = p.data.as_deref().filter(|d| !Path::new(d).is_absolute() && !d.starts_with("..")) {
        lines.push(format!("/{}/", d.trim_start_matches("./").trim_end_matches('/')));
    }
    let gi = gitignore(dir, &lines)?;
    let mcp = match mcp_rel {
        None => None,
        Some(rel) => Some((rel.to_string(), register_mcp(&dir.join(rel), &o.exe)?)),
    };
    Ok(Report { p, file, gitignore: gi, gitignore_lines: lines, mcp, scripts: find_scripts(dir) })
}

/// A one-line, in-place edit of `import wandb` (perl: the same flags on Linux and macOS, unlike `sed -i`).
pub fn swap_command(file: &str) -> String {
    format!("perl -pi -e 's/^import wandb$/import modgrad as wandb/' {file}")
}

/// What to do next, as shown by `modgrad init`: `{run, python, dashboard,
/// view?, scripts}`. `scripts` has one step per training script found.
pub fn next_steps(p: &Project, scripts: &[Script]) -> Value {
    let d = p.data.as_deref().unwrap_or("obs");
    let steps: Vec<Value> = scripts
        .iter()
        .map(|s| {
            let stem = s.file.trim_end_matches(".py");
            if s.wandb {
                json!({"file": s.file, "how": "swap", "edit": swap_command(&s.file), "command": format!("python {}", s.file)})
            } else {
                json!({"file": s.file, "how": "wrap", "command": format!("modgrad run --name {stem} -- python {}", s.file), "run": stem})
            }
        })
        .collect();
    let run = steps.iter().find_map(|x| x["how"].eq("wrap").then(|| x["command"].as_str().unwrap_or("").to_string())).unwrap_or_else(|| "modgrad run --name exp1 -- python train.py".into());
    match &p.server {
        Some(server) => json!({
            "run": run,
            "scripts": steps,
            "python": "import modgrad as wandb\nwandb.init(name=\"exp1\", config={\"lr\": 3e-4})\nwandb.log({\"loss\": 0.52})",
            "dashboard": p.board.clone().unwrap_or_else(|| format!("{server}/")),
        }),
        None => json!({
            "run": run,
            "scripts": steps,
            "python": format!("import modgrad as wandb   # no server: writes {d}/runs/exp1\nwandb.init(name=\"exp1\", config={{\"lr\": 3e-4}})\nwandb.log({{\"loss\": 0.52}})"),
            "view": format!("modgrad-server --root {d}"),
            "dashboard": format!("{DEFAULT_SERVER}/"),
        }),
    }
}

pub fn render(r: &Report) -> String {
    let mut out = String::new();
    let what = match &r.p.server {
        Some(s) => format!("project {} on {s}", r.p.project.as_deref().unwrap_or("(from token)")),
        None => format!("local mode, runs in {}/", r.p.data.as_deref().unwrap_or("obs")),
    };
    out.push_str(&format!("modgrad: {FILE} {}: {what}\n", r.file));
    if let Some(e) = r.p.expires_unix {
        let days = ((e - modgrad_obs::now_unix()) / 86400.0).round();
        out.push_str(&format!("modgrad: anonymous project; expires in {days} days unless claimed\n"));
    }
    match r.gitignore {
        "present" => {}
        g => out.push_str(&format!("modgrad: .gitignore {g}: {}\n", r.gitignore_lines.join(" "))),
    }
    if let Some((f, st)) = &r.mcp {
        out.push_str(&format!("modgrad: MCP server \"modgrad\" {st} in {f}{}\n", if *st == "unchanged" { "" } else { " (restart the agent to load it)" }));
    }
    let n = next_steps(&r.p, &r.scripts);
    out.push_str("\nNext: nothing is recorded until the training runs through modgrad.\n");
    let steps = n["scripts"].as_array().cloned().unwrap_or_default();
    let mut tail = None;
    for st in &steps {
        let (f, c) = (st["file"].as_str().unwrap_or(""), st["command"].as_str().unwrap_or(""));
        if st["how"] == "swap" {
            out.push_str(&format!("  {f} uses wandb: change `import wandb` to `import modgrad as wandb` (only that line), then run it as usual:\n    {}\n    {c}\n", st["edit"].as_str().unwrap_or("")));
        } else {
            out.push_str(&format!("  {f}: if it prints metric lines (`step 10 loss 0.52`, Trainer dicts), run it through modgrad, no code change:\n    {c}\n"));
            tail = tail.or(st["run"].as_str().map(String::from));
        }
    }
    if steps.is_empty() {
        out.push_str(&format!("  {}\n", n["run"].as_str().unwrap_or("")));
        tail = Some("exp1".into());
    }
    out.push_str(&format!("Then check the run has rows: modgrad tail {}\n", tail.as_deref().unwrap_or("<run name>")));
    out.push_str("\nPython (to log from code instead):\n");
    for l in n["python"].as_str().unwrap_or("").lines() {
        out.push_str(&format!("  {l}\n"));
    }
    out.push('\n');
    if let Some(v) = n["view"].as_str() {
        out.push_str(&format!("Dashboard (local mode, no server is running yet): start `{v}`, which serves until Ctrl-C, then open {}\n", n["dashboard"].as_str().unwrap_or("")));
    } else {
        // agents copied this line as proof of a finished setup; the run's own link comes from `modgrad run`
        out.push_str("Not done yet: nothing is recorded until the training runs. The link to give is the one `modgrad run` prints when it ends (…/#run=…).\n");
        out.push_str(&format!("Project page (empty until a run logs): {}\n", n["dashboard"].as_str().unwrap_or("")));
    }
    out
}

pub fn report_json(r: &Report) -> Value {
    let mut v = json!({
        "ok": true,
        "file": FILE,
        "file_status": r.file,
        "mode": if r.p.local() { "local" } else { "server" },
        "gitignore": r.gitignore,
        "next": next_steps(&r.p, &r.scripts),
    });
    for (k, x) in [("server", &r.p.server), ("project", &r.p.project), ("board", &r.p.board), ("data", &r.p.data)] {
        if let Some(x) = x {
            v[k] = json!(x);
        }
    }
    if let Some((f, st)) = &r.mcp {
        v["mcp"] = json!({ "file": f, "status": st });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("modgrad-init-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A one-shot fake server answering `POST /api/v1/projects`; returns its URL.
    fn fake_server(n: usize) -> String {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        let u = url.clone();
        std::thread::spawn(move || {
            for (i, c) in l.incoming().take(n).enumerate() {
                let mut c = c.unwrap();
                let mut buf = [0u8; 4096];
                let _ = c.read(&mut buf);
                let body = json!({"project": format!("anon-{i}"), "token": format!("obsp_t{i}"), "board": format!("{u}/#project=anon-{i}"), "expires_unix": modgrad_obs::now_unix() + 30.0 * 86400.0}).to_string();
                let _ = write!(c, "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        url
    }

    #[test]
    fn local_init_is_idempotent_and_ignores_the_file() {
        let d = tmp("local");
        let o = InitOpts { exe: "modgrad".into(), ..Default::default() };
        let r = init(&d, &o).unwrap();
        assert_eq!((r.file, r.gitignore, r.p.data.as_deref()), ("created", "created", Some("obs")));
        // the Python client writes local runs itself: the snippet is the plain W&B one
        let py = next_steps(&r.p, &[])["python"].as_str().unwrap().to_string();
        assert!(py.starts_with("import modgrad as wandb") && py.contains("obs/runs/exp1"), "{py}");
        let gi = std::fs::read_to_string(d.join(".gitignore")).unwrap();
        assert!(gi.contains("/.modgrad.json\n") && gi.contains("/obs/\n"), "{gi}");
        let r = init(&d, &o).unwrap();
        assert_eq!((r.file, r.gitignore), ("unchanged", "present"));
        assert_eq!(std::fs::read_to_string(d.join(".gitignore")).unwrap(), gi);
        // found from a subdirectory; data resolves against the file's directory
        let sub = d.join("a/b");
        std::fs::create_dir_all(&sub).unwrap();
        let f = find(&sub).unwrap();
        assert_eq!((f.dir.clone(), f.data_dir()), (d.clone(), d.join("obs")));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn server_init_reuses_its_token() {
        let d = tmp("server");
        std::fs::write(d.join(".gitignore"), "target").unwrap();
        let url = fake_server(2);
        let o = InitOpts { server: Some(url.clone()), exe: "modgrad".into(), ..Default::default() };
        let r = init(&d, &o).unwrap();
        assert_eq!((r.file, r.gitignore, r.p.token.as_deref(), r.p.project.as_deref()), ("created", "added", Some("obsp_t0"), Some("anon-0")));
        assert_eq!(std::fs::read_to_string(d.join(".gitignore")).unwrap(), "target\n# modgrad: project file (may hold a write token) and local runs\n/.modgrad.json\n");
        #[cfg(unix)]
        assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(d.join(FILE)).unwrap().permissions()) & 0o777, 0o600);
        // a second init (no flags at all) keeps the server and the token
        let r = init(&d, &InitOpts { exe: "modgrad".into(), ..Default::default() }).unwrap();
        assert_eq!((r.file, r.p.token.as_deref()), ("unchanged", Some("obsp_t0")));
        // the token is only offered to its own server
        assert_eq!(r.p.token_for(&format!("{url}/")).as_deref(), Some("obsp_t0"));
        assert_eq!(r.p.token_for("http://elsewhere:1"), None);
        // --local switches modes
        let r = init(&d, &InitOpts { local: true, exe: "modgrad".into(), ..Default::default() }).unwrap();
        assert_eq!((r.file, r.p.server.clone(), r.p.token.clone()), ("updated", None, None));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn unreachable_server_writes_nothing() {
        let d = tmp("down");
        let e = init(&d, &InitOpts { server: Some("http://127.0.0.1:9".into()), exe: "modgrad".into(), ..Default::default() }).unwrap_err();
        assert!(e.contains("cannot reach") && e.contains("without --server"), "{e}");
        assert!(!d.join(FILE).exists() && !d.join(".gitignore").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn mcp_json_is_merged_not_clobbered() {
        let d = tmp("mcp");
        std::fs::write(d.join(".mcp.json"), r#"{"mcpServers": {"other": {"command": "x"}}}"#).unwrap();
        let o = InitOpts { mcp: Some("claude".into()), exe: "/bin/modgrad".into(), ..Default::default() };
        assert_eq!(init(&d, &o).unwrap().mcp, Some((".mcp.json".into(), "added")));
        let v: Value = serde_json::from_slice(&std::fs::read(d.join(".mcp.json")).unwrap()).unwrap();
        assert_eq!(v["mcpServers"]["other"]["command"], "x");
        assert_eq!(v["mcpServers"]["modgrad"], json!({"command": "/bin/modgrad", "args": ["mcp"]}));
        assert_eq!(init(&d, &o).unwrap().mcp, Some((".mcp.json".into(), "unchanged")));
        // a broken config is left alone
        std::fs::write(d.join(".mcp.json"), "{oops").unwrap();
        assert!(init(&d, &o).unwrap_err().contains("not touching it"));
        assert_eq!(std::fs::read_to_string(d.join(".mcp.json")).unwrap(), "{oops");
        assert!(init(&d, &InitOpts { mcp: Some("vim".into()), ..Default::default() }).unwrap_err().contains("supported clients"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn next_steps_name_the_training_scripts() {
        let d = tmp("scripts");
        std::fs::write(d.join("train_grpo.py"), "for i in range(3):\n    print(f'step {i} reward 1')\n").unwrap();
        std::fs::write(d.join("finetune.py"), "import wandb\nif __name__ == '__main__':\n    loss.backward()\n").unwrap();
        std::fs::write(d.join("minitrainer.py"), "class Trainer:\n    def train(self):\n        loss.backward()\n").unwrap();
        std::fs::write(d.join("utils.py"), "x = 1\n").unwrap();
        let r = init(&d, &InitOpts { exe: "modgrad".into(), ..Default::default() }).unwrap();
        let files: Vec<&str> = r.scripts.iter().map(|s| s.file.as_str()).collect();
        assert_eq!(files, ["finetune.py", "train_grpo.py"]);
        assert!(r.scripts[0].wandb && !r.scripts[1].wandb);
        let text = render(&r);
        assert!(text.contains("modgrad run --name train_grpo -- python train_grpo.py"), "{text}");
        assert!(text.contains("finetune.py uses wandb: change `import wandb` to `import modgrad as wandb`"), "{text}");
        assert!(text.contains("perl -pi -e 's/^import wandb$/import modgrad as wandb/' finetune.py"), "{text}");
        assert!(text.contains("modgrad tail train_grpo") && text.contains("serves until Ctrl-C"), "{text}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
