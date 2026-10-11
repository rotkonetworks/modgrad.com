//! `modgrad init`, then `run` / `tail` with no flags: the project file
//! decides where runs go.

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("modgrad-init-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The binary in `cwd`, with no MODGRAD_* / OBS_* settings inherited.
fn modgrad(cwd: &Path, args: &[&str]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_modgrad"));
    c.current_dir(cwd).args(args);
    for (k, _) in std::env::vars() {
        if k.starts_with("MODGRAD_") || k.starts_with("OBS_") {
            c.env_remove(k);
        }
    }
    c.env("XDG_CACHE_HOME", cwd.join(".cache"));
    c.output().unwrap()
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn local_init_then_run_and_tail_without_flags() {
    let d = tmp("local");
    let o = modgrad(&d, &["init", "--json"]);
    assert!(o.status.success(), "{}", text(&o));
    let v: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!((v["mode"].clone(), v["data"].clone(), v["gitignore"].clone()), (json!("local"), json!("obs"), json!("created")), "{v}");
    assert!(v["next"]["run"].as_str().unwrap().starts_with("modgrad run "), "{v}");
    // from a subdirectory: the file is found, data lands next to it
    let sub = d.join("src");
    std::fs::create_dir_all(&sub).unwrap();
    let o = modgrad(&sub, &["run", "--name", "exp1", "--no-gpu", "--", "sh", "-c", "echo 'step 1 loss 0.5'; echo 'step 2 loss 0.25'"]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(d.join("obs/runs/exp1/metrics.jsonl").is_file(), "{}", text(&o));
    let o = modgrad(&sub, &["tail", "exp1", "--json", "--keys", "loss"]);
    assert!(o.status.success(), "{}", text(&o));
    let v: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["keys"]["loss"]["last"], json!(0.25), "{v}");
    let o = modgrad(&sub, &["tail", "nope"]);
    assert!(!o.status.success() && text(&o).contains("no run nope"), "{}", text(&o));
    // re-running init changes nothing
    let o = modgrad(&d, &["init"]);
    assert!(text(&o).contains(".modgrad.json unchanged") && !text(&o).contains(".gitignore"), "{}", text(&o));
    let _ = std::fs::remove_dir_all(&d);
}

/// A fake modgrad-server: issues one anonymous project, accepts anything
/// else, and records each request's first line and Authorization header.
fn fake_server() -> (String, Arc<Mutex<Vec<(String, String)>>>) {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { continue };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            // headers, then Content-Length bytes of body
            loop {
                let n = c.read(&mut chunk).unwrap_or(0);
                buf.extend_from_slice(&chunk[..n]);
                let t = String::from_utf8_lossy(&buf).to_string();
                if let Some(i) = t.find("\r\n\r\n") {
                    let len = t.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|x| x.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                    if buf.len() >= i + 4 + len || n == 0 {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let t = String::from_utf8_lossy(&buf).to_string();
            let first = t.lines().next().unwrap_or("").to_string();
            let auth = t.lines().find_map(|l| l.strip_prefix("Authorization: ")).unwrap_or("").to_string();
            let body = if first.starts_with("POST /api/v1/projects ") {
                json!({"project": "anon-abc", "token": "obsp_secret", "board": "http://x/#project=anon-abc", "expires_unix": 4e9})
            } else {
                json!({"run": "anon-abc/exp1", "created": true})
            }
            .to_string();
            s.lock().unwrap().push((first, auth));
            let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        }
    });
    (url, seen)
}

#[test]
fn server_init_then_run_sends_the_file_token() {
    let d = tmp("server");
    let (url, seen) = fake_server();
    let o = modgrad(&d, &["init", "--server", &url, "--mcp", "claude"]);
    assert!(o.status.success(), "{}", text(&o));
    let out = text(&o);
    for want in ["project anon-abc", ".gitignore created", "MCP server \"modgrad\" added in .mcp.json", "modgrad run --name exp1 -- python train.py", "import modgrad as wandb", "Dashboard (runs appear once they log): http://x/#project=anon-abc"] {
        assert!(out.contains(want), "missing {want:?} in\n{out}");
    }
    let mcp: Value = serde_json::from_slice(&std::fs::read(d.join(".mcp.json")).unwrap()).unwrap();
    assert_eq!(mcp["mcpServers"]["modgrad"]["args"], json!(["mcp"]));
    assert!(!std::fs::read_to_string(d.join(".mcp.json")).unwrap().contains("obsp_"), "the token stays out of .mcp.json");
    let o = modgrad(&d, &["run", "--name", "exp1", "--no-gpu", "--", "sh", "-c", "echo 'step 1 loss 0.5'"]);
    assert!(o.status.success(), "{}", text(&o));
    let reqs = seen.lock().unwrap().clone();
    assert_eq!(reqs.iter().filter(|(l, _)| l.starts_with("POST /api/v1/projects ")).count(), 1, "{reqs:?}");
    let writes: Vec<_> = reqs.iter().filter(|(l, _)| l.starts_with("POST /api/v1/runs")).collect();
    assert!(!writes.is_empty() && writes.iter().all(|(_, a)| a == "Bearer obsp_secret"), "{reqs:?}");
    // a flag naming another server never gets the file's token
    let (other, seen2) = fake_server();
    let o = modgrad(&d, &["run", "--server", &other, "--name", "exp2", "--no-gpu", "--", "true"]);
    assert!(o.status.success(), "{}", text(&o));
    let reqs = seen2.lock().unwrap().clone();
    assert!(!reqs.is_empty() && reqs.iter().all(|(_, a)| a.is_empty()), "{reqs:?}");
    let _ = std::fs::remove_dir_all(&d);
}
