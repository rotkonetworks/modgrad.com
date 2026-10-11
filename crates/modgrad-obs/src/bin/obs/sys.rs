//! System samples for `obs run`: RSS and CPU of the wrapped process tree
//! (from /proc), and GPU utilisation and VRAM from rocm-smi or nvidia-smi
//! when present. Keys: `sys/rss_mb`, `sys/cpu_pct`, `sys/gpu<i>/util_pct`,
//! `sys/gpu<i>/vram_mb`. GPU numbers are whole-device, not per-process.

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Clock ticks per second for /proc stat times (USER_HZ; 100 on Linux).
const CLK_TCK: f64 = 100.0;

/// (ppid, utime + stime ticks, rss kB) of one process.
fn proc_info(pid: u32) -> Option<(u32, u64, u64)> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // comm may contain spaces and parens: fields start after the last ')'
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest[0] = state (field 3); ppid = field 4, utime = 14, stime = 15, rss pages = 24
    let ppid = f.get(1)?.parse().ok()?;
    let ticks = f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?;
    let rss_kb = f.get(21)?.parse::<u64>().ok()? * 4;
    Some((ppid, ticks, rss_kb))
}

/// `pid` and all its descendants.
fn tree(root: u32) -> Vec<(u32, u64, u64)> {
    let mut all: HashMap<u32, (u32, u64, u64)> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for e in rd.flatten() {
            if let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok())
                && let Some(i) = proc_info(pid)
            {
                all.insert(pid, i);
            }
        }
    }
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        if let Some(&(_, t, r)) = all.get(&p) {
            out.push((p, t, r));
        }
        stack.extend(all.iter().filter(|(_, (pp, _, _))| *pp == p).map(|(c, _)| *c));
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Gpu {
    None,
    Rocm,
    Nvidia,
}

/// Run a command with a hard timeout; stdout on success.
fn run_timeout(cmd: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut ch = Command::new(cmd).args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let deadline = Instant::now() + timeout;
    loop {
        match ch.try_wait() {
            Ok(Some(st)) if st.success() => break,
            Ok(Some(_)) => return None,
            Ok(None) if Instant::now() > deadline => {
                let _ = ch.kill();
                let _ = ch.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
    let mut s = String::new();
    std::io::Read::read_to_string(&mut ch.stdout.take()?, &mut s).ok()?;
    Some(s)
}

/// rocm-smi JSON: `{"card0": {"GPU use (%)": "12", "VRAM Total Used Memory (B)": "123", ...}}`.
/// Field names vary across versions, so keys are matched loosely.
pub fn parse_rocm(json: &str) -> Vec<(String, f64)> {
    // some versions print a warning line before the JSON
    let json = json.find('{').map_or("", |i| &json[i..]);
    let v: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
    let mut out = Vec::new();
    let Some(cards) = v.as_object() else { return out };
    let mut names: Vec<&String> = cards.keys().filter(|k| k.starts_with("card")).collect();
    names.sort_by_key(|k| k[4..].parse::<u32>().unwrap_or(0));
    for (i, card) in names.into_iter().enumerate() {
        let num = |pred: &dyn Fn(&str) -> bool| -> Option<f64> {
            cards[card].as_object()?.iter().find(|(k, _)| pred(&k.to_ascii_lowercase())).and_then(|(_, v)| v.as_str().map(|s| s.trim().to_string()).or_else(|| v.as_f64().map(|x| x.to_string()))?.parse().ok())
        };
        if let Some(u) = num(&|k| k.contains("gpu use")) {
            out.push((format!("sys/gpu{i}/util_pct"), u));
        }
        if let Some(b) = num(&|k| k.contains("vram") && k.contains("used") && k.contains("(b)")) {
            out.push((format!("sys/gpu{i}/vram_mb"), b / 1048576.0));
        }
    }
    out
}

/// nvidia-smi `--query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`.
pub fn parse_nvidia(csv: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for (i, l) in csv.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let f: Vec<f64> = l.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if let [u, m] = f[..] {
            out.push((format!("sys/gpu{i}/util_pct"), u));
            out.push((format!("sys/gpu{i}/vram_mb"), m));
        }
    }
    out
}

const ROCM_ARGS: [&str; 4] = ["--showuse", "--showmeminfo", "vram", "--json"];
const NV_ARGS: [&str; 2] = ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"];

pub struct Sampler {
    pid: u32,
    gpu: Gpu,
    last: Option<(Instant, u64)>,
}

impl Sampler {
    /// Probe for a GPU tool once (bounded by a timeout).
    pub fn new(pid: u32, gpu: bool) -> Sampler {
        let t = Duration::from_secs(5);
        let gpu = if !gpu {
            Gpu::None
        } else if run_timeout("rocm-smi", &ROCM_ARGS, t).is_some_and(|s| !parse_rocm(&s).is_empty()) {
            Gpu::Rocm
        } else if run_timeout("nvidia-smi", &NV_ARGS, t).is_some_and(|s| !parse_nvidia(&s).is_empty()) {
            Gpu::Nvidia
        } else {
            Gpu::None
        };
        Sampler { pid, gpu, last: None }
    }

    pub fn gpu_name(&self) -> &'static str {
        match self.gpu {
            Gpu::None => "none",
            Gpu::Rocm => "rocm-smi",
            Gpu::Nvidia => "nvidia-smi",
        }
    }

    /// One sample; empty once the process tree is gone.
    pub fn sample(&mut self) -> Vec<(String, f64)> {
        let procs = tree(self.pid);
        if procs.is_empty() {
            return Vec::new();
        }
        let ticks: u64 = procs.iter().map(|p| p.1).sum();
        let rss_kb: u64 = procs.iter().map(|p| p.2).sum();
        let now = Instant::now();
        let mut out = vec![("sys/rss_mb".to_string(), rss_kb as f64 / 1024.0)];
        if let Some((t0, k0)) = self.last {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.0 {
                out.push(("sys/cpu_pct".into(), (ticks.saturating_sub(k0)) as f64 / CLK_TCK / dt * 100.0));
            }
        }
        self.last = Some((now, ticks));
        let t = Duration::from_secs(5);
        let g = match self.gpu {
            Gpu::Rocm => run_timeout("rocm-smi", &ROCM_ARGS, t).map(|s| parse_rocm(&s)),
            Gpu::Nvidia => run_timeout("nvidia-smi", &NV_ARGS, t).map(|s| parse_nvidia(&s)),
            Gpu::None => Some(Vec::new()),
        };
        match g {
            Some(g) => out.extend(g),
            None => self.gpu = Gpu::None, // hung or failed: stop asking
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_and_nvidia_parsing() {
        let r = parse_rocm(r#"WARNING: low-power state

{"card1": {"GPU use (%)": "7", "VRAM Total Memory (B)": "8573157376", "VRAM Total Used Memory (B)": "1048576000"}, "card0": {"GPU use (%)": "93"}, "system": {}}"#);
        assert_eq!(r, vec![("sys/gpu0/util_pct".into(), 93.0), ("sys/gpu1/util_pct".into(), 7.0), ("sys/gpu1/vram_mb".into(), 1000.0)]);
        assert_eq!(parse_nvidia("45, 1234\n0, 10\n"), vec![("sys/gpu0/util_pct".into(), 45.0), ("sys/gpu0/vram_mb".into(), 1234.0), ("sys/gpu1/util_pct".into(), 0.0), ("sys/gpu1/vram_mb".into(), 10.0)]);
        assert!(parse_rocm("garbage").is_empty());
    }

    #[test]
    fn own_process_is_visible() {
        let mut s = Sampler { pid: std::process::id(), gpu: Gpu::None, last: None };
        let a = s.sample();
        assert!(a[0].0 == "sys/rss_mb" && a[0].1 > 0.0, "{a:?}");
        let b = s.sample();
        assert!(b.iter().any(|(k, _)| k == "sys/cpu_pct"));
    }
}
