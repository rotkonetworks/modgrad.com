//! The SSH door (feature `ssh`): `python train.py 2>&1 | ssh <server>`.
//!
//! Zero install, no signup: the client's public key is the identity. Its
//! SHA256 fingerprint maps to an anonymous project
//! ([`Auth::ssh_key_project`]); the same key always lands there. Any key
//! is accepted; password and keyboard-interactive auth are not offered.
//!
//! Each session channel runs one fixed verb, never a shell or a command:
//!
//! * no command (stdin piped, no pty) or `run [<name>]`: stdin is a
//!   training log. A run is created and its dashboard URL printed on
//!   stderr; every byte is echoed back on stdout; metric lines go through
//!   [`TextParser`] into rows. EOF finishes the run (exit_code 0); a
//!   dropped connection leaves it unfinished.
//! * `token`: a fresh (expiring) project token on stdout, and how to use it
//!   with `modgrad init` and curl on stderr.
//! * `runs`: this key's runs. `tail <run>`: latest values ([`spark::table`]).
//! * `help`, an unknown verb, or an interactive login (a pty): usage.
//!
//! Bounds: [`SshConfig`] caps connections (globally, over a semaphore taken
//! before the handshake), concurrent sessions per key, channels per
//! connection, the line buffer, and idle time. Port forwarding, agent and
//! X11 requests and subsystems use russh's defaults, which refuse them.

use crate::auth::Cred;
use crate::http::AppState;
use crate::ingest::IngestRequest;
use crate::read::{ReadRequest, ReadResponse};
use crate::state::{Record, valid_run_name};
use crate::v1::{TextLine, TextParser, run_url};
use crate::{AuthedRequest, AuthedResponse, IngestError};
use modgrad_obs::spark;
use russh::keys::{HashAlg, PrivateKey, PublicKey, ssh_key};
use russh::server::{Auth as SshAuth, Handler, Msg, Session};
use russh::{Channel, ChannelId, ChannelMsg, ChannelOpenFailure, MethodKind, MethodSet};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Semaphore;
use tower::ServiceExt;

/// The host key file under the obs root.
pub const HOST_KEY_FILE: &str = "ssh_host_ed25519";

#[derive(Debug, Clone)]
pub struct SshConfig {
    /// Close a connection that has seen no traffic this long (training can
    /// be quiet for a while: default one hour).
    pub idle: Duration,
    /// Open connections, all keys together (also caps unauthenticated ones).
    pub max_conns: usize,
    /// Concurrent sessions (pipes and commands) per key.
    pub max_per_key: usize,
    /// Session channels per connection (`ControlMaster` multiplexing).
    pub max_channels: usize,
    /// Longest line kept for parsing; longer ones are still echoed, and
    /// only their tail (after the last `\r`, else the last 4 KiB) is parsed.
    pub max_line: usize,
    /// Time to finish authentication after connecting.
    pub auth_timeout: Duration,
}

impl Default for SshConfig {
    fn default() -> Self {
        SshConfig { idle: Duration::from_secs(3600), max_conns: 256, max_per_key: 8, max_channels: 4, max_line: 64 * 1024, auth_timeout: Duration::from_secs(30) }
    }
}

/// Load `<root>/ssh_host_ed25519`, or make one (mode 0600) on first start.
pub fn host_key(root: &Path) -> std::io::Result<PrivateKey> {
    let p = root.join(HOST_KEY_FILE);
    let err = |e: ssh_key::Error| std::io::Error::other(format!("{}: {e}", p.display()));
    if p.is_file() {
        return PrivateKey::read_openssh_file(&p).map_err(err);
    }
    let mut seed = [0u8; 32];
    std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom")?, &mut seed)?;
    let key = PrivateKey::from(ssh_key::private::Ed25519Keypair::from_seed(&seed));
    let pem = key.to_openssh(ssh_key::LineEnding::LF).map_err(err)?;
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    std::io::Write::write_all(&mut o.open(&p)?, pem.as_bytes())?;
    Ok(key)
}

/// `SHA256:<base64>`, as `ssh-keygen -l` prints it.
pub fn fingerprint(k: &PublicKey) -> String {
    k.fingerprint(HashAlg::Sha256).to_string()
}

/// Shared by every connection.
struct Door {
    st: AppState,
    cfg: SshConfig,
    /// Base of printed links (`--public-url`).
    public: String,
    /// Live sessions per key fingerprint.
    per_key: Mutex<HashMap<String, usize>>,
}

/// Serve SSH on `lis` until the process exits.
pub async fn serve(lis: tokio::net::TcpListener, key: PrivateKey, st: AppState, public: String, cfg: SshConfig) {
    let rcfg = Arc::new(russh::server::Config {
        server_id: russh::SshId::Standard("SSH-2.0-modgrad".into()),
        methods: MethodSet::from(&[MethodKind::PublicKey][..]),
        keys: vec![key],
        inactivity_timeout: Some(cfg.idle),
        keepalive_interval: None,
        auth_rejection_time: Duration::from_millis(500),
        auth_rejection_time_initial: Some(Duration::ZERO),
        max_auth_attempts: 6,
        // incoming data waits in at most channel_buffer_size packets before
        // the session stops reading the socket (back-pressure to the client)
        window_size: 256 * 1024,
        channel_buffer_size: 16,
        nodelay: true,
        ..Default::default()
    });
    let conns = Arc::new(Semaphore::new(cfg.max_conns));
    let door = Arc::new(Door { st, cfg, public: public.trim_end_matches('/').to_string(), per_key: Mutex::default() });
    loop {
        let (sock, peer) = match lis.accept().await {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!("ssh accept: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let Ok(permit) = conns.clone().try_acquire_owned() else {
            tracing::warn!("ssh: {peer}: connection limit ({}) reached; dropped", door.cfg.max_conns);
            continue;
        };
        let (door, rcfg) = (door.clone(), rcfg.clone());
        tokio::spawn(async move {
            let _permit = permit;
            let authed = Arc::new(AtomicBool::new(false));
            let h = Conn { door: door.clone(), fp: None, project: None, channels: Arc::new(AtomicUsize::new(0)), authed: authed.clone() };
            // the version exchange happens in here: bound it like auth
            let session = match tokio::time::timeout(door.cfg.auth_timeout, russh::server::run_stream(rcfg, sock, h)).await {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    tracing::debug!("ssh: {peer}: {e:?}");
                    return;
                }
                Err(_) => return,
            };
            let handle = session.handle();
            let timeout = door.cfg.auth_timeout;
            let watchdog = tokio::spawn(async move {
                tokio::time::sleep(timeout).await;
                if !authed.load(Ordering::Relaxed) {
                    let _ = handle.disconnect(russh::Disconnect::ByApplication, "authentication timed out".into(), String::new()).await;
                }
            });
            if let Err(e) = session.await {
                tracing::debug!("ssh: {peer}: {e:?}");
            }
            watchdog.abort();
        });
    }
}

/// One connection.
struct Conn {
    door: Arc<Door>,
    fp: Option<String>,
    project: Option<String>,
    channels: Arc<AtomicUsize>,
    authed: Arc<AtomicBool>,
}

impl Handler for Conn {
    type Error = russh::Error;

    async fn auth_publickey_offered(&mut self, _user: &str, _key: &PublicKey) -> Result<SshAuth, Self::Error> {
        Ok(SshAuth::Accept)
    }

    /// Called after the signature proved the client holds the key.
    async fn auth_publickey(&mut self, _user: &str, key: &PublicKey) -> Result<SshAuth, Self::Error> {
        let fp = fingerprint(key);
        match self.door.st.auth.ssh_key_project(&fp) {
            Ok((project, new)) => {
                if new {
                    tracing::info!("ssh: new key {fp} -> project {project}");
                }
                self.fp = Some(fp);
                self.project = Some(project);
                self.authed.store(true, Ordering::Relaxed);
                Ok(SshAuth::Accept)
            }
            Err(e) => {
                tracing::warn!("ssh: key {fp} refused: {e}");
                Ok(SshAuth::reject())
            }
        }
    }

    async fn channel_open_session(&mut self, channel: Channel<Msg>, reply: russh::server::ChannelOpenHandle, _s: &mut Session) -> Result<(), Self::Error> {
        let (Some(fp), Some(project)) = (self.fp.clone(), self.project.clone()) else {
            reply.reject(ChannelOpenFailure::AdministrativelyProhibited).await;
            return Ok(());
        };
        if self.channels.fetch_add(1, Ordering::Relaxed) >= self.door.cfg.max_channels {
            self.channels.fetch_sub(1, Ordering::Relaxed);
            reply.reject(ChannelOpenFailure::ResourceShortage).await;
            return Ok(());
        }
        reply.accept().await;
        let (door, channels) = (self.door.clone(), self.channels.clone());
        tokio::spawn(async move {
            run_channel(&door, channel, &fp, &project).await;
            channels.fetch_sub(1, Ordering::Relaxed);
        });
        Ok(())
    }

    // The requests below are acted on by the channel's task (it receives
    // them as ChannelMsg); here they are only acknowledged.
    async fn pty_request(&mut self, channel: ChannelId, _term: &str, _w: u32, _h: u32, _pw: u32, _ph: u32, _modes: &[(russh::Pty, u32)], s: &mut Session) -> Result<(), Self::Error> {
        s.channel_success(channel)
    }

    async fn shell_request(&mut self, channel: ChannelId, s: &mut Session) -> Result<(), Self::Error> {
        s.channel_success(channel)
    }

    async fn exec_request(&mut self, channel: ChannelId, _data: &[u8], s: &mut Session) -> Result<(), Self::Error> {
        s.channel_success(channel)
    }

    async fn subsystem_request(&mut self, channel: ChannelId, _name: &str, s: &mut Session) -> Result<(), Self::Error> {
        s.channel_failure(channel)
    }
}

/// A session slot for one key, released on drop.
struct KeySlot<'a> {
    door: &'a Door,
    fp: String,
}

impl Drop for KeySlot<'_> {
    fn drop(&mut self) {
        let mut m = self.door.per_key.lock().unwrap();
        if let Some(n) = m.get_mut(&self.fp) {
            *n -= 1;
            if *n == 0 {
                m.remove(&self.fp);
            }
        }
    }
}

impl Door {
    fn slot(&self, fp: &str) -> Option<KeySlot<'_>> {
        let mut m = self.per_key.lock().unwrap();
        let n = m.entry(fp.to_string()).or_insert(0);
        if *n >= self.cfg.max_per_key {
            return None;
        }
        *n += 1;
        Some(KeySlot { door: self, fp: fp.to_string() })
    }

    /// Through [`crate::Authed`] as the key's project. Overload and timeouts
    /// are retried a few times: an HTTP client retries a 503 itself, but
    /// nobody is there to retry for a pipe.
    async fn ingest_retry(&self, project: &str, req: IngestRequest) -> Result<AuthedResponse, String> {
        use crate::http::ApiError;
        let mut wait = Duration::from_millis(200);
        for attempt in 1.. {
            let e = match self.st.authed().oneshot(AuthedRequest { token: None, cred: Some(Cred::Project(project.to_string())), req: req.clone() }).await {
                Ok(a) => return Ok(a),
                Err(e) => ApiError::from_box(e),
            };
            match e {
                ApiError::Ingest(IngestError::Overloaded | IngestError::Timeout) if attempt < 4 => {
                    tokio::time::sleep(wait).await;
                    wait *= 2;
                }
                ApiError::Auth(e) => return Err(e.to_string()),
                ApiError::Ingest(e) => return Err(e.to_string()),
                e => return Err(format!("{e:?}")),
            }
        }
        unreachable!()
    }

    fn usage(&self) -> String {
        format!(
            "modgrad over ssh: your key is your account (project per key, no signup). Dashboard: {base}/\n\
             \n\
             \x20 python train.py 2>&1 | ssh <server>             stream a training log into a new run\n\
             \x20 python train.py 2>&1 | ssh <server> run <name>  ... with a run name\n\
             \x20 ssh <server> token                              a project token for curl, `modgrad init` and Python\n\
             \x20 ssh <server> runs                               this key's runs\n\
             \x20 ssh <server> tail <run>                         latest values of a run\n\
             \n\
             Metric lines like \"step 100: loss 2.31 acc 71%\" become rows; your output is echoed back unchanged.\n",
            base = self.public
        )
    }
}

/// The channel is gone, or the client stopped reading for `idle`.
struct Gone;

/// Every write is bounded by the idle time. A client that stops reading
/// stdout (but keeps the connection open) stalls our writes; our task then
/// stops draining the channel, and russh's session loop, which waits for
/// that inbox inline, can stop reading the socket, its idle timer
/// included. Giving up drops the channel, which unblocks it.
async fn send(idle: Duration, f: impl Future<Output = Result<(), russh::Error>>) -> Result<(), Gone> {
    match tokio::time::timeout(idle, f).await {
        Ok(Ok(())) => Ok(()),
        _ => Err(Gone),
    }
}

/// Bytes to stdout.
async fn out(door: &Door, ch: &Channel<Msg>, s: impl Into<bytes::Bytes>) -> Result<(), Gone> {
    send(door.cfg.idle, ch.data_bytes(s.into())).await
}

/// Bytes to stderr.
async fn err(door: &Door, ch: &Channel<Msg>, s: impl Into<bytes::Bytes>) -> Result<(), Gone> {
    send(door.cfg.idle, ch.extended_data_bytes(1, s.into())).await
}

async fn exit(door: &Door, ch: &Channel<Msg>, code: u32) {
    let _ = send(door.cfg.idle, async {
        ch.exit_status(code).await?;
        ch.eof().await?;
        ch.close().await
    })
    .await;
}

/// Wait for shell/exec on a fresh channel, then run the verb.
async fn run_channel(door: &Door, mut ch: Channel<Msg>, fp: &str, project: &str) {
    let mut pty = false;
    let command = loop {
        match ch.wait().await {
            Some(ChannelMsg::RequestPty { .. }) => pty = true,
            Some(ChannelMsg::RequestShell { .. }) => break None,
            Some(ChannelMsg::Exec { command, .. }) => break Some(String::from_utf8_lossy(&command).into_owned()),
            Some(ChannelMsg::Eof | ChannelMsg::Close) | None => return,
            _ => {}
        }
    };
    let Some(_slot) = door.slot(fp) else {
        if err(door, &ch, format!("modgrad: too many concurrent ssh sessions for this key (limit {}); end one and retry\n", door.cfg.max_per_key)).await.is_ok() {
            exit(door, &ch, 1).await;
        }
        return;
    };
    let words: Vec<&str> = command.as_deref().unwrap_or("").split_whitespace().collect();
    let code = match (command.is_none(), words.as_slice()) {
        // an interactive login (`ssh <server>` from a terminal) is not a log
        (true, _) if pty => err(door, &ch, door.usage().replace('\n', "\r\n")).await.map(|_| 0),
        (_, []) => ingest(door, &mut ch, project, None).await,
        (_, ["run"]) => ingest(door, &mut ch, project, None).await,
        (_, ["run", name]) => ingest(door, &mut ch, project, Some(name)).await,
        (_, ["token"]) => token(door, &ch, project).await,
        (_, ["runs"]) => runs(door, &ch, project).await,
        (_, ["tail", run]) => tail(door, &ch, project, run).await,
        (_, ["help" | "-h" | "--help" | "usage"]) => out(door, &ch, door.usage()).await.map(|_| 0),
        (_, w) => err(door, &ch, format!("modgrad: unknown command {:?}\n\n{}", w.join(" "), door.usage())).await.map(|_| 2),
    };
    // a gone channel just drops (no exit status to deliver)
    if let Ok(code) = code {
        exit(door, &ch, code).await;
    }
}

/// The id of run `name` in `project` (always inside it), if valid.
fn run_id(project: &str, name: &str) -> Option<String> {
    let name = name.strip_prefix(&format!("{project}/")).unwrap_or(name);
    let id = format!("{project}/{name}");
    valid_run_name(&id).then_some(id)
}

/// Splits a byte stream into lines for the parser, with a bounded buffer.
struct Lines {
    buf: Vec<u8>,
    max: usize,
    parser: TextParser,
}

impl Lines {
    /// Feed bytes; returns the rows of the lines they completed.
    fn feed(&mut self, data: &[u8], recs: &mut Vec<Record>) {
        for &b in data {
            if b == b'\n' {
                self.flush(recs);
            } else {
                self.buf.push(b);
            }
        }
        // progress bars redraw with '\r' and may never print '\n' (as `modgrad run`)
        if self.buf.len() > self.max {
            let cut = self.buf.iter().rposition(|&b| b == b'\r').map_or(self.buf.len() - self.max.min(4096), |i| i + 1);
            self.buf.drain(..cut);
            if self.buf.len() > self.max {
                self.buf.clear();
            }
        }
    }

    fn flush(&mut self, recs: &mut Vec<Record>) {
        if let TextLine::Row(r) = self.parser.line(&String::from_utf8_lossy(&self.buf)) {
            recs.push(r);
        }
        self.buf.clear();
    }
}

/// A verb's exit status, or [`Gone`].
type Exit = Result<u32, Gone>;

async fn ingest(door: &Door, ch: &mut Channel<Msg>, project: &str, name: Option<&str>) -> Exit {
    let name = name.map(String::from).unwrap_or_else(|| crate::v1::auto_name(modgrad_obs::now_unix(), &crate::v1::suffix()));
    let Some(id) = run_id(project, &name) else {
        err(door, ch, format!("modgrad: bad run name {name:?}: use 1-2 '/'-separated parts of [A-Za-z0-9_.-]\n")).await?;
        return Ok(2);
    };
    let create = IngestRequest::CreateRun { name: id.clone(), project: None, config: json!({}), meta: Some(json!({ "via": "ssh" })) };
    let run = match door.ingest_retry(project, create).await {
        Ok(a) => a.run,
        Err(e) => {
            err(door, ch, format!("modgrad: cannot create run {id}: {e}\n")).await?;
            return Ok(1);
        }
    };
    err(door, ch, format!("modgrad: run {run} -> {}\n", run_url(&door.public, &run))).await?;
    let mut lines = Lines { buf: Vec::new(), max: door.cfg.max_line, parser: TextParser::default() };
    let mut warned = false;
    loop {
        let mut recs = Vec::new();
        match ch.wait().await {
            Some(ChannelMsg::Data { data }) => {
                // pass through unchanged, first
                out(door, ch, data.clone()).await?;
                lines.feed(&data, &mut recs);
            }
            Some(ChannelMsg::Eof) => {
                lines.flush(&mut recs);
                send_rows(door, ch, project, &run, recs, &mut warned).await?;
                if let Err(e) = door.ingest_retry(project, IngestRequest::Finish { run: run.clone(), summary: json!({ "exit_code": 0 }) }).await {
                    err(door, ch, format!("modgrad: could not finish {run}: {e}\n")).await?;
                }
                return Ok(0);
            }
            // the connection dropped without EOF: the run stays unfinished
            Some(ChannelMsg::Close) | None => return Err(Gone),
            _ => {}
        }
        send_rows(door, ch, project, &run, recs, &mut warned).await?;
    }
}

async fn send_rows(door: &Door, ch: &Channel<Msg>, project: &str, run: &str, records: Vec<Record>, warned: &mut bool) -> Result<(), Gone> {
    if records.is_empty() {
        return Ok(());
    }
    if let Err(e) = door.ingest_retry(project, IngestRequest::Rows { run: run.to_string(), records }).await {
        // tell the user once; their output keeps flowing either way
        if !std::mem::replace(warned, true) {
            err(door, ch, format!("modgrad: rows not stored: {e}\n")).await?;
        }
    }
    Ok(())
}

async fn token(door: &Door, ch: &Channel<Msg>, project: &str) -> Exit {
    let (t, expires) = match door.st.auth.ssh_project_token(project) {
        Ok(x) => x,
        Err(e) => {
            err(door, ch, format!("modgrad: {e}\n")).await?;
            return Ok(1);
        }
    };
    let base = &door.public;
    let exp = match expires {
        Some(e) => format!("expires in {:.0} days; `ssh <server> token` again for a new one", ((e - modgrad_obs::now_unix()) / 86400.0).max(0.0)),
        None => "does not expire".into(),
    };
    out(door, ch, format!("{t}\n")).await?;
    err(
        door,
        ch,
        format!(
            "modgrad: project token for {project} ({exp}). Use it with:\n\
             \x20 modgrad init --server {base} --token {t}\n\
             \x20 curl -X POST {base}/api/v1/runs/my-run/log -H \"Authorization: Bearer {t}\" -d '{{\"loss\": 0.5}}'\n\
             \x20 board: {base}/#project={project}\n"
        ),
    )
    .await?;
    Ok(0)
}

async fn runs(door: &Door, ch: &Channel<Msg>, project: &str) -> Exit {
    let prefix = format!("{project}/");
    let list = match door.st.read.clone().oneshot(ReadRequest::ListRuns { filter: Some(prefix.clone()) }).await {
        Ok(ReadResponse::Runs(r)) => r,
        Ok(_) => Vec::new(),
        Err(e) => {
            err(door, ch, format!("modgrad: {e}\n")).await?;
            return Ok(1);
        }
    };
    let mut rows: Vec<&Value> = list.iter().filter(|r| r["id"].as_str().is_some_and(|id| id.starts_with(&prefix))).collect();
    rows.sort_by(|a, b| a["meta"]["start_unix"].as_f64().unwrap_or(0.0).total_cmp(&b["meta"]["start_unix"].as_f64().unwrap_or(0.0)));
    if rows.is_empty() {
        out(door, ch, format!("no runs yet in {project}; start one with: python train.py 2>&1 | ssh <server>\n")).await?;
        return Ok(0);
    }
    let w = rows.iter().filter_map(|r| r["id"].as_str()).map(|id| id.len() - prefix.len()).max().unwrap_or(3).max(3);
    let mut s = format!("{:w$}  {:>8}  {:>7}  {:8}  dashboard\n", "run", "step", "rows", "status");
    for r in rows {
        let id = r["id"].as_str().unwrap_or("");
        let step = r["last_step"].as_u64().map_or("-".into(), |s| s.to_string());
        let status = if r["finished"] == true { "finished" } else { "running" };
        s.push_str(&format!("{:w$}  {step:>8}  {:>7}  {status:8}  {}\n", &id[prefix.len()..], r["n_rows"].as_u64().unwrap_or(0), run_url(&door.public, id)));
    }
    s.push_str(&format!("project board: {}/#project={project}\n", door.public));
    out(door, ch, s).await?;
    Ok(0)
}

async fn tail(door: &Door, ch: &Channel<Msg>, project: &str, name: &str) -> Exit {
    let Some(id) = run_id(project, name) else {
        err(door, ch, format!("modgrad: {name:?} is not a run name in this key's project\n")).await?;
        return Ok(2);
    };
    match door.st.read.clone().oneshot(ReadRequest::History { run: id.clone(), keys: Vec::new(), range: 0..u64::MAX, stride: 1 }).await {
        Ok(ReadResponse::History(h)) => {
            let cols = h.columns.into_iter().collect();
            out(door, ch, spark::table(&id[project.len() + 1..], &h.steps, &cols, 30)).await?;
            out(door, ch, format!("{}\n", run_url(&door.public, &id))).await?;
            Ok(0)
        }
        Ok(_) => Ok(1),
        Err(e) => {
            err(door, ch, format!("modgrad: {e} (`ssh <server> runs` lists them)\n")).await?;
            Ok(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_split_carry_and_bound() {
        let mut l = Lines { buf: Vec::new(), max: 64, parser: TextParser::default() };
        let mut recs = Vec::new();
        l.feed(b"step 3: loss 1.5\nval", &mut recs);
        l.feed(b"_loss: 0.25\n", &mut recs);
        assert_eq!(recs.len(), 2);
        assert!(matches!(&recs[1], Record::Log { step: 3, .. }), "the step carries across chunks: {recs:?}");
        recs.clear();
        l.feed(&[b'x'; 1000], &mut recs);
        assert!(l.buf.len() <= 64);
        l.feed(b"\rloss 0.5\n", &mut recs);
        assert_eq!(recs.len(), 1, "the tail after the last \\r is still parsed");
    }

    #[test]
    fn run_ids_stay_in_the_project() {
        assert_eq!(run_id("anon-1", "exp1").as_deref(), Some("anon-1/exp1"));
        assert_eq!(run_id("anon-1", "anon-1/exp1").as_deref(), Some("anon-1/exp1"));
        assert_eq!(run_id("anon-1", "a/b").as_deref(), Some("anon-1/a/b"));
        assert_eq!(run_id("anon-1", "anon-2/x"), Some("anon-1/anon-2/x".into()), "another project's name is just a name here");
        assert_eq!(run_id("anon-1", "../x"), None);
        assert_eq!(run_id("anon-1", "a/b/c"), None, "at most 3 parts with the project");
    }
}
