//! The SSH door over real sockets: `modgrad-server --ssh` on loopback,
//! driven by the system `ssh` client with throwaway ed25519 keys (temp dir,
//! no agent, no ~/.ssh, no known_hosts). Ports 18960-18963. Skipped when
//! `ssh` or `ssh-keygen` is not installed.
#![cfg(feature = "ssh")]

use modgrad_obs::net;
use serde_json::Value;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("obs-ssh-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn have(cmd: &str) -> bool {
    Command::new(cmd).arg("-V").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok()
}

/// A modgrad-server child, killed (by its own handle) on drop.
struct Server {
    child: Child,
    http: String,
    ssh_port: u16,
    root: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn server(tag: &str, http_port: u16, ssh_port: u16, extra: &[&str]) -> Server {
    server_at(tmp(tag), http_port, ssh_port, extra)
}

fn server_at(root: PathBuf, http_port: u16, ssh_port: u16, extra: &[&str]) -> Server {
    for p in [http_port, ssh_port] {
        // or the readiness check below would find someone else's server
        drop(std::net::TcpListener::bind(("127.0.0.1", p)).unwrap_or_else(|e| panic!("port {p} is busy: {e}")));
    }
    let child = Command::new(env!("CARGO_BIN_EXE_modgrad-server"))
        .args(["--root", root.to_str().unwrap(), "--no-static", "--listen", &format!("127.0.0.1:{http_port}"), "--ssh", &format!("127.0.0.1:{ssh_port}")])
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let s = Server { child, http: format!("http://127.0.0.1:{http_port}"), ssh_port, root };
    let t0 = Instant::now();
    while std::net::TcpStream::connect(("127.0.0.1", ssh_port)).is_err() || std::net::TcpStream::connect(("127.0.0.1", http_port)).is_err() {
        assert!(t0.elapsed() < Duration::from_secs(20), "server did not start");
        std::thread::sleep(Duration::from_millis(50));
    }
    s
}

fn keygen(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    let st = Command::new("ssh-keygen").args(["-q", "-t", "ed25519", "-N", "", "-C", name, "-f"]).arg(&p).status().unwrap();
    assert!(st.success());
    p
}

/// `ssh` isolated from the user's config, agent and known_hosts.
fn ssh(s: &Server, key: Option<&Path>, extra: &[&str], args: &[&str]) -> Command {
    let mut c = Command::new("ssh");
    c.args(["-F", "/dev/null", "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]);
    c.args(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10"]);
    if let Some(k) = key {
        c.arg("-i").arg(k);
    }
    c.args(["-p", &s.ssh_port.to_string()]).args(extra).arg("obs@127.0.0.1").args(args);
    c
}

/// Run with `input` on stdin; (output, stdout, stderr).
fn run(mut c: Command, input: &[u8]) -> (Output, String, String) {
    let mut ch = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut stdin = ch.stdin.take().unwrap();
    let input = input.to_vec();
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let o = ch.wait_with_output().unwrap();
    w.join().unwrap();
    let (out, err) = (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned());
    (o, out, err)
}

/// The run id in the dashboard URL the door prints.
fn run_of(stderr: &str) -> String {
    let i = stderr.find("#run=").unwrap_or_else(|| panic!("no dashboard URL in {stderr:?}"));
    stderr[i + 5..].split_whitespace().next().unwrap().replace("%2F", "/")
}

fn project_of(run: &str) -> &str {
    run.split('/').next().unwrap()
}

fn get(s: &Server, path: &str) -> Value {
    let r = net::request("GET", &format!("{}{path}", s.http), None, None, "application/json", Duration::from_secs(10)).unwrap();
    assert!(r.ok(), "GET {path}: {} {}", r.status, String::from_utf8_lossy(&r.body));
    r.json()
}

fn ssh_keys(s: &Server) -> usize {
    let a: Value = serde_json::from_slice(&std::fs::read(s.root.join(".obs-auth.json")).unwrap_or_default()).unwrap_or(Value::Null);
    a["ssh_keys"].as_object().map_or(0, |m| m.len())
}

#[test]
fn ssh_door() {
    if !have("ssh") || !have("ssh-keygen") {
        eprintln!("skipped: no ssh / ssh-keygen");
        return;
    }
    let s = server("door", 18960, 18961, &[]);
    let keys = tmp("door-keys");
    let (k1, k2) = (keygen(&keys, "k1"), keygen(&keys, "k2"));
    assert!(s.root.join("ssh_host_ed25519").is_file(), "host key made on first start");

    // a piped training log: echoed unchanged, parsed into rows, finished on EOF
    let log = "loading data\nstep 1: loss 2.5 acc 10%\nstep 2: loss 2.0\n  val_loss: 1.9\nstep 3 | loss=1.5\nno newline at the end";
    let (o, out, err) = run(ssh(&s, Some(&k1), &[], &["run", "demo"]), log.as_bytes());
    assert!(o.status.success(), "{o:?}");
    assert_eq!(out, log, "output passes through unchanged");
    assert_eq!(err.lines().count(), 1, "one stderr line, the URL: {err:?}");
    let run1 = run_of(&err);
    assert!(run1.starts_with("anon-") && run1.ends_with("/demo"), "{run1}");
    assert!(err.contains(&format!("{}/#run=", s.http)), "links use the server URL: {err}");
    let h = get(&s, &format!("/api/v1/runs/{run1}/history?keys=loss,val_loss"));
    assert_eq!(h["steps"], serde_json::json!([1, 2, 3]));
    assert_eq!(h["columns"]["loss"], serde_json::json!([2.5, 2.0, 1.5]));
    assert_eq!(h["columns"]["val_loss"], serde_json::json!([null, 1.9, null]), "a stepless line takes the last step");
    let sm = get(&s, &format!("/api/v1/runs/{run1}/summary"));
    assert_eq!((sm["finished"].clone(), sm["summary"]["exit_code"].clone()), (Value::Bool(true), serde_json::json!(0)), "{sm}");

    // the same key lands in the same project (auto-named run); another key elsewhere
    let (o, _, err) = run(ssh(&s, Some(&k1), &[], &[]), b"step 7: loss 0.7\n");
    assert!(o.status.success(), "{o:?}");
    let run2 = run_of(&err);
    assert_eq!(project_of(&run2), project_of(&run1));
    assert!(run2[project_of(&run1).len() + 1..].starts_with("run-"), "{run2}");
    let (o, _, err) = run(ssh(&s, Some(&k2), &[], &["run", "other"]), b"loss 1\n");
    assert!(o.status.success(), "{o:?}");
    let run3 = run_of(&err);
    assert_ne!(project_of(&run3), project_of(&run1));
    assert_eq!(ssh_keys(&s), 2);

    // runs: this key's runs only
    let (o, out, _) = run(ssh(&s, Some(&k1), &[], &["runs"]), b"");
    assert!(o.status.success(), "{o:?}");
    assert!(out.contains("demo") && out.contains(&run2[project_of(&run2).len() + 1..]) && !out.contains("other"), "{out}");
    assert!(out.contains("finished"), "{out}");

    // tail: latest values
    let (o, out, _) = run(ssh(&s, Some(&k1), &[], &["tail", "demo"]), b"");
    assert!(o.status.success(), "{o:?}");
    assert!(out.starts_with("demo  step 3") && out.lines().any(|l| l.starts_with("loss") && l.contains("1.5")), "{out}");
    let (o, _, err) = run(ssh(&s, Some(&k2), &[], &["tail", "demo"]), b"");
    assert!(!o.status.success() && err.contains("unknown run"), "another key's run is not visible by name: {err}");

    // token: a project token that works with the HTTP API, for the same project
    let (o, out, err) = run(ssh(&s, Some(&k1), &[], &["token"]), b"");
    assert!(o.status.success(), "{o:?}");
    let token = out.trim();
    assert!(token.starts_with("obsp_") && !token.contains('\n'), "stdout is just the token: {out:?}");
    assert!(err.contains(&format!("modgrad init --server {} --token {token}", s.http)) && err.contains("curl"), "{err}");
    let r = net::request("POST", &format!("{}/api/v1/runs/via-http/log", s.http), Some(token), Some(br#"{"step": 1, "loss": 0.5}"#), "application/json", Duration::from_secs(10)).unwrap();
    assert!(r.ok(), "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.json()["run"], Value::String(format!("{}/via-http", project_of(&run1))));

    // help, unknown verbs, an interactive login: usage, never a shell
    let (o, out, _) = run(ssh(&s, Some(&k1), &[], &["help"]), b"");
    assert!(o.status.success() && out.contains("ssh <server> token"), "{out}");
    let (o, _, err) = run(ssh(&s, Some(&k1), &[], &["sh", "-c", "id"]), b"");
    assert_eq!(o.status.code(), Some(2));
    assert!(err.contains("unknown command") && err.contains("ssh <server> runs"), "{err}");
    let (o, out, err) = run(ssh(&s, Some(&k1), &["-tt"], &[]), b"");
    assert!(o.status.success(), "{o:?}");
    assert!(format!("{out}{err}").contains("ssh <server> token"), "a pty login prints usage: {out:?} {err:?}");
    let (o, _, _) = run(ssh(&s, Some(&k1), &[], &["runs"]), b"");
    assert!(o.status.success());
    assert_eq!(ssh_keys(&s), 2, "help and logins made no new projects");

    // password / keyboard-interactive: not offered, refused, no project made
    let (o, _, err) = run(ssh(&s, None, &["-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive"], &["runs"]), b"");
    assert_eq!(o.status.code(), Some(255));
    assert!(err.contains("Permission denied (publickey)"), "{err}");
    assert_eq!(ssh_keys(&s), 2);

    // an oversized line (4 MiB, no newline, no '\r'): echoed in full, the
    // server keeps going, and the next metric line still lands
    let mut big = vec![b'x'; 4 << 20];
    big.extend_from_slice(b"\nstep 9: loss 0.09\n");
    let (o, out, _) = run(ssh(&s, Some(&k1), &[], &["run", "big"]), &big);
    assert!(o.status.success(), "{:?}", o.status);
    assert_eq!(out.len(), big.len(), "echo is complete");
    let h = get(&s, &format!("/api/v1/runs/{}/big/history?keys=loss", project_of(&run1)));
    assert_eq!((h["steps"].clone(), h["columns"]["loss"].clone()), (serde_json::json!([9]), serde_json::json!([0.09])));

    // a dropped connection (ssh killed mid-run) leaves the run unfinished
    let mut c = ssh(&s, Some(&k1), &[], &["run", "dropped"]);
    let mut ch = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    ch.stdin.as_mut().unwrap().write_all(b"step 1: loss 1\n").unwrap();
    let mut echoed = [0u8; 15];
    ch.stdout.as_mut().unwrap().read_exact(&mut echoed).unwrap();
    let id = format!("{}/dropped", project_of(&run1));
    let t0 = Instant::now();
    while get(&s, &format!("/api/v1/runs/{id}/history?keys=loss"))["steps"] != serde_json::json!([1]) {
        assert!(t0.elapsed() < Duration::from_secs(10), "row did not land");
        std::thread::sleep(Duration::from_millis(50));
    }
    ch.kill().unwrap();
    ch.wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let sm = get(&s, &format!("/api/v1/runs/{id}/summary"));
    assert_ne!(sm["finished"], Value::Bool(true), "{sm}");

    // the key -> project map survives a restart
    let root = s.root.clone();
    drop(s);
    let s3 = server_at(root, 18960, 18961, &[]);
    let (o, _, err) = run(ssh(&s3, Some(&k1), &[], &["run", "again"]), b"loss 1\n");
    assert!(o.status.success(), "{o:?}");
    assert_eq!(project_of(&run_of(&err)), project_of(&run1));
}

#[test]
fn ssh_limits() {
    if !have("ssh") || !have("ssh-keygen") {
        eprintln!("skipped: no ssh / ssh-keygen");
        return;
    }
    let s = server("limits", 18962, 18963, &["--ssh-max-per-key", "1", "--public-url", "https://obs.example.org/"]);
    let keys = tmp("limits-keys");
    let (k1, k2) = (keygen(&keys, "k1"), keygen(&keys, "k2"));
    // one open pipe holds k1's only session
    let mut c = ssh(&s, Some(&k1), &[], &["run", "long"]);
    let mut ch = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    ch.stdin.as_mut().unwrap().write_all(b"x\n").unwrap();
    let mut b = [0u8; 2];
    ch.stdout.as_mut().unwrap().read_exact(&mut b).unwrap();
    let (o, _, err) = run(ssh(&s, Some(&k1), &[], &["runs"]), b"");
    assert_eq!(o.status.code(), Some(1));
    assert!(err.contains("too many concurrent ssh sessions"), "{err}");
    // other keys are not affected
    let (o, out, _) = run(ssh(&s, Some(&k2), &[], &["runs"]), b"");
    assert!(o.status.success(), "{out}");
    // EOF ends the first session and frees the slot
    drop(ch.stdin.take());
    let o = ch.wait_with_output().unwrap();
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("https://obs.example.org/#run=anon-"), "--public-url is the link base: {:?}", String::from_utf8_lossy(&o.stderr));
    let (o, out, _) = run(ssh(&s, Some(&k1), &[], &["runs"]), b"");
    assert!(o.status.success() && out.contains("long"), "{out}");
}
