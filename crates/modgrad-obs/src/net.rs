//! Minimal HTTP client for the obs CLI and `modgrad mcp --url`: plain `http://` over a
//! `TcpStream` (HTTP/1.1, `Connection: close`), and `https://` through the
//! `curl` binary so no TLS stack is linked for TCP.
//!
//! [`push_request`] (what `modgrad run` / `push` send rows with) can also
//! use HTTP/3 (feature `h3`): after the first TCP answer it tries QUIC when
//! the URL is https and the server sent `Alt-Svc: h3=":<port>"`, or always
//! when asked to (`--h3`). The UDP handshake gets [`H3_HANDSHAKE`]; if it
//! fails, or the connection dies later, that server stays on TCP for the
//! rest of the process. Every request is safe to resend either way (the
//! API is idempotent), so a request lost with the QUIC connection is
//! simply retried over TCP.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[cfg(feature = "h3")]
mod h3;

pub struct Resp {
    pub status: u16,
    pub body: Vec<u8>,
    /// The `Alt-Svc` header of a TCP response, if any.
    pub alt_svc: Option<String>,
}

impl Resp {
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
    /// The server's teaching error message, or the raw body.
    pub fn error_text(&self) -> String {
        let j = self.json();
        j["message"].as_str().or(j["error"].as_str()).map(String::from).unwrap_or_else(|| String::from_utf8_lossy(&self.body).into_owned())
    }
}

fn split_url(url: &str) -> io::Result<(&str, &str, String)> {
    let (scheme, rest) = url.split_once("://").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("bad URL {url:?} (want http://host:port)")))?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, "/".to_string()),
    };
    Ok((scheme, hostport, path))
}

/// One request. `body` is sent as JSON (or text if `ctype` says so).
pub fn request(method: &str, url: &str, token: Option<&str>, body: Option<&[u8]>, ctype: &str, timeout: Duration) -> io::Result<Resp> {
    let (scheme, hostport, path) = split_url(url)?;
    match scheme {
        "http" => http(method, hostport, &path, token, body, ctype, timeout),
        "https" => curl(method, url, token, body, ctype, timeout),
        s => Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unsupported scheme {s:?}"))),
    }
}

fn http(method: &str, hostport: &str, path: &str, token: Option<&str>, body: Option<&[u8]>, ctype: &str, timeout: Duration) -> io::Result<Resp> {
    let addr_str = if hostport.contains(':') { hostport.to_string() } else { format!("{hostport}:80") };
    let addrs: Vec<_> = addr_str.to_socket_addrs()?.collect();
    let mut last = io::Error::new(io::ErrorKind::NotFound, format!("{hostport}: no address"));
    let mut stream = None;
    for a in addrs {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => last = e,
        }
    }
    let mut s = stream.ok_or(last)?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    let body = body.unwrap_or(&[]);
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {hostport}\r\nConnection: close\r\nUser-Agent: obs-cli\r\nContent-Length: {}\r\n", body.len());
    if !body.is_empty() {
        req.push_str(&format!("Content-Type: {ctype}\r\n"));
    }
    if let Some(t) = token {
        req.push_str(&format!("Authorization: Bearer {t}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes())?;
    s.write_all(body)?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw)?;
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> io::Result<Resp> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    let end = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or_else(|| bad("truncated HTTP response"))?;
    let head = String::from_utf8_lossy(&raw[..end]);
    let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or_else(|| bad("bad status line"))?;
    let mut body = raw[end + 4..].to_vec();
    if head.lines().any(|l| l.to_ascii_lowercase().starts_with("transfer-encoding:") && l.to_ascii_lowercase().contains("chunked")) {
        body = dechunk(&body).ok_or_else(|| bad("bad chunked body"))?;
    }
    let alt_svc = head.lines().find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("alt-svc")).map(|(_, v)| v.trim().to_string()));
    Ok(Resp { status, body, alt_svc })
}

fn dechunk(mut b: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let i = b.windows(2).position(|w| w == b"\r\n")?;
        let n = usize::from_str_radix(std::str::from_utf8(&b[..i]).ok()?.split(';').next()?.trim(), 16).ok()?;
        b = &b[i + 2..];
        if n == 0 {
            return Some(out);
        }
        out.extend_from_slice(b.get(..n)?);
        b = b.get(n + 2..)?;
    }
}

/// `https://` via the curl binary. The token goes in on stdin (`-H @-`)
/// so it never appears in `ps`; the body goes through a 0600 temp file.
fn curl(method: &str, url: &str, token: Option<&str>, body: Option<&[u8]>, ctype: &str, timeout: Duration) -> io::Result<Resp> {
    use std::process::{Command, Stdio};
    let mut c = Command::new("curl");
    // the last two lines of stdout: the Alt-Svc header (curl >= 7.84), then the status
    c.args(["-sS", "-X", method, "--max-time", &timeout.as_secs().max(1).to_string(), "-w", "\n%header{alt-svc}\n%{http_code}", "-H", &format!("Content-Type: {ctype}")]);
    if let Some(ca) = TLS_CA.get() {
        c.arg("--cacert").arg(ca);
    }
    if token.is_some() {
        c.args(["-H", "@-"]);
    }
    let tmp = match body {
        Some(b) => {
            let p = std::env::temp_dir().join(format!("obs-curl-{}-{:x}", std::process::id(), crate::now_unix().to_bits()));
            let mut o = std::fs::OpenOptions::new();
            o.write(true).create_new(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
            o.open(&p)?.write_all(b)?;
            c.arg("--data-binary").arg(format!("@{}", p.display()));
            Some(p)
        }
        None => None,
    };
    c.arg(url).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let r = (|| {
        let mut ch = c.spawn().map_err(|e| io::Error::new(e.kind(), format!("https needs the curl binary: {e}")))?;
        if let Some(mut si) = ch.stdin.take()
            && let Some(t) = token
        {
            si.write_all(format!("Authorization: Bearer {t}\n").as_bytes())?;
        }
        let out = ch.wait_with_output()?;
        let text = out.stdout;
        let nl = text.iter().rposition(|&b| b == b'\n').unwrap_or(0);
        let status: u16 = std::str::from_utf8(&text[nl..]).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        if status == 0 {
            return Err(io::Error::other(format!("curl: {}", String::from_utf8_lossy(&out.stderr).trim())));
        }
        let nl2 = text[..nl].iter().rposition(|&b| b == b'\n').unwrap_or(0);
        let alt = String::from_utf8_lossy(&text[(nl2 + 1).min(nl)..nl]).trim().to_string();
        Ok(Resp { status, body: text[..nl2].to_vec(), alt_svc: (!alt.is_empty() && !alt.contains("%header")).then_some(alt) })
    })();
    if let Some(p) = tmp {
        let _ = std::fs::remove_file(p);
    }
    r
}

// ------------------------------------------------------------ transport choice

static TLS_CA: OnceLock<PathBuf> = OnceLock::new();

/// Trust this PEM CA as well as the system's, for curl (`https://`) and
/// HTTP/3 alike (e.g. a dev server's `--h3-self-signed` certificate).
/// First call wins.
pub fn set_tls_ca(pem: PathBuf) {
    let _ = TLS_CA.set(pem);
}

/// How long the UDP handshake may take before a server is marked TCP-only.
pub const H3_HANDSHAKE: Duration = Duration::from_millis(1500);

/// A one-time report from [`push_request`]: the transport it settled on
/// for a server. Callers print `text` once.
#[derive(Debug, Clone)]
pub struct Notice {
    /// "h3", "https" or "http".
    pub transport: &'static str,
    /// Why it is not (or no longer) HTTP/3, when it was tried.
    pub fallback: Option<String>,
    pub text: String,
}

enum Via {
    #[cfg(feature = "h3")]
    H3(std::sync::Arc<h3::Conn>),
    Tcp,
}

fn origins() -> &'static Mutex<HashMap<String, Via>> {
    static O: OnceLock<Mutex<HashMap<String, Via>>> = OnceLock::new();
    O.get_or_init(Default::default)
}

/// The UDP port in an `Alt-Svc` value offering `h3`, e.g. `h3=":443"; ma=86400`.
/// (An alternative host is not followed: the port is tried on the URL's host.)
pub fn alt_svc_h3_port(v: &str) -> Option<u16> {
    v.split(',').find_map(|alt| {
        let (proto, rest) = alt.trim().split_once('=')?;
        if proto.trim() != "h3" {
            return None;
        }
        let authority = rest.split(';').next()?.trim().trim_matches('"');
        authority.rsplit_once(':')?.1.parse().ok()
    })
}

/// Try HTTP/3? On an https URL whose server advertised it, or when forced.
/// (A plain http:// URL is a choice of no TLS; it only goes to QUIC on request.)
fn want_h3(scheme: &str, advertised: Option<u16>, force: bool) -> bool {
    force || (scheme == "https" && advertised.is_some())
}

fn tcp_name(scheme: &str) -> &'static str {
    if scheme == "https" { "HTTPS" } else { "HTTP" }
}

/// [`request`] for pushing rows: over HTTP/3 once a server is known to
/// speak it (see the module docs), else TCP. `force_h3` tries HTTP/3 even
/// for an `http://` URL or without `Alt-Svc` (on the advertised port, else
/// the URL's port). The returned [`Notice`] is set the one time the
/// transport for a server is decided or changes.
pub fn push_request(method: &str, url: &str, token: Option<&str>, body: Option<&[u8]>, ctype: &str, timeout: Duration, force_h3: bool) -> (io::Result<Resp>, Option<Notice>) {
    let (scheme, hostport, _) = match split_url(url) {
        Ok(x) => x,
        Err(e) => return (Err(e), None),
    };
    let origin = format!("{scheme}://{hostport}");
    let mut notice = None;
    #[cfg(feature = "h3")]
    {
        let conn = match origins().lock().unwrap().get(&origin) {
            Some(Via::H3(c)) => Some(c.clone()),
            _ => None,
        };
        if let Some(c) = conn {
            match c.request(method, url, token, body, ctype, timeout) {
                Ok(r) => return (Ok(r), None),
                Err(e) => {
                    origins().lock().unwrap().insert(origin.clone(), Via::Tcp);
                    let why = format!("HTTP/3 connection lost: {e}");
                    notice = Some(Notice { transport: if scheme == "https" { "https" } else { "http" }, text: format!("{why}; continuing over {}", tcp_name(scheme)), fallback: Some(why) });
                }
            }
        }
    }
    let r = request(method, url, token, body, ctype, timeout);
    let Ok(resp) = &r else { return (r, notice) };
    if origins().lock().unwrap().contains_key(&origin) {
        return (r, notice);
    }
    // first answer from this server: decide the transport for the rest of the process
    let advertised = resp.alt_svc.as_deref().and_then(alt_svc_h3_port);
    let wanted = want_h3(scheme, advertised, force_h3);
    #[cfg(feature = "h3")]
    if wanted {
        let host = hostport.rsplit_once(':').filter(|(_, p)| p.parse::<u16>().is_ok() && !hostport.ends_with(']')).map_or(hostport, |(h, _)| h);
        let url_port = hostport.rsplit_once(':').and_then(|(_, p)| p.parse().ok()).unwrap_or(if scheme == "https" { 443 } else { 80 });
        let port = advertised.unwrap_or(url_port);
        let ca = TLS_CA.get().map(PathBuf::as_path);
        let (via, n) = match h3::connect(host, port, H3_HANDSHAKE, ca) {
            Ok(c) => {
                let text = format!("pushing over HTTP/3 (udp {})", c.addr);
                (Via::H3(std::sync::Arc::new(c)), Notice { transport: "h3", fallback: None, text })
            }
            Err(e) => {
                // a timeout is what a firewall dropping UDP looks like
                let what = if e.kind() == io::ErrorKind::TimedOut { "UDP blocked, fell back" } else { "HTTP/3 failed, fell back" };
                let why = format!("HTTP/3 to udp {host}:{port}: {e}");
                let text = format!("pushing over {} ({what}: {why})", tcp_name(scheme));
                (Via::Tcp, Notice { transport: if scheme == "https" { "https" } else { "http" }, fallback: Some(why), text })
            }
        };
        origins().lock().unwrap().insert(origin, via);
        return (r, Some(n));
    }
    origins().lock().unwrap().insert(origin, Via::Tcp);
    if wanted {
        // built without the h3 feature
        let why = "this modgrad was built without HTTP/3 support (feature h3)".to_string();
        notice = Some(Notice { transport: if scheme == "https" { "https" } else { "http" }, text: format!("pushing over {} ({why})", tcp_name(scheme)), fallback: Some(why) });
    }
    (r, notice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses() {
        let r = parse_response(b"HTTP/1.1 201 Created\r\ncontent-length: 2\r\n\r\n{}").unwrap();
        assert_eq!((r.status, r.body.as_slice()), (201, &b"{}"[..]));
        let r = parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n").unwrap();
        assert_eq!(r.body, b"abcde");
        assert!(parse_response(b"HTTP/1.1 200").is_err());
        assert_eq!(split_url("http://h:1/a/b?x").unwrap(), ("http", "h:1", "/a/b?x".to_string()));
        let r = parse_response(b"HTTP/1.1 200 OK\r\nalt-svc: h3=\":8443\"; ma=86400\r\n\r\n").unwrap();
        assert_eq!(r.alt_svc.as_deref().and_then(alt_svc_h3_port), Some(8443));
    }

    #[test]
    fn alt_svc() {
        assert_eq!(alt_svc_h3_port(r#"h3-29=":1", h3=":443"; ma=86400"#), Some(443));
        assert_eq!(alt_svc_h3_port(r#"h3="obs.example.org:8443""#), Some(8443));
        assert_eq!(alt_svc_h3_port("clear"), None);
        assert_eq!(alt_svc_h3_port(r#"h2=":443""#), None);
        assert!(want_h3("https", Some(443), false) && want_h3("http", None, true));
        assert!(!want_h3("https", None, false) && !want_h3("http", Some(443), false));
    }
}
