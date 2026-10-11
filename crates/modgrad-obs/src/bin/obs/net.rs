//! Minimal HTTP client for the obs CLI, std only: plain `http://` over a
//! `TcpStream` (HTTP/1.1, `Connection: close`), and `https://` through the
//! `curl` binary so no TLS stack is linked. (A QUIC/HTTP3 transport with
//! a long-lived stream per run is the planned replacement; see
//! docs/design/obs-product.md.)

use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

pub struct Resp {
    pub status: u16,
    pub body: Vec<u8>,
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
    Ok(Resp { status, body })
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
    c.args(["-sS", "-X", method, "--max-time", &timeout.as_secs().max(1).to_string(), "-w", "\n%{http_code}", "-H", &format!("Content-Type: {ctype}")]);
    if token.is_some() {
        c.args(["-H", "@-"]);
    }
    let tmp = match body {
        Some(b) => {
            let p = std::env::temp_dir().join(format!("obs-curl-{}-{:x}", std::process::id(), modgrad_obs::now_unix().to_bits()));
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
        Ok(Resp { status, body: text[..nl].to_vec() })
    })();
    if let Some(p) = tmp {
        let _ = std::fs::remove_file(p);
    }
    r
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
    }
}
