//! HTTP/3 client (feature `h3`): one QUIC connection per server, kept open
//! across requests, driven by a small background tokio runtime so the
//! callers (the CLI's sender thread) stay synchronous. TLS roots are the
//! system's plus an optional extra CA ([`super::set_tls_ca`]).

use super::Resp;
use bytes::{Buf, Bytes};
use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

fn runtime() -> io::Result<&'static tokio::runtime::Runtime> {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    if let Some(rt) = RT.get() {
        return Ok(rt);
    }
    // one worker: it drives the connection (acks, keep-alives) between requests
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).thread_name("modgrad-h3").enable_all().build()?;
    Ok(RT.get_or_init(|| rt))
}

pub struct Conn {
    // kept so the socket outlives every request
    _endpoint: quinn::Endpoint,
    send: h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>,
    pub addr: SocketAddr,
}

fn tls(ca: Option<&Path>) -> io::Result<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    for c in rustls_native_certs::load_native_certs().certs {
        let _ = roots.add(c);
    }
    if let Some(p) = ca {
        use rustls::pki_types::{CertificateDer, pem::PemObject};
        let bad = |e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", p.display()));
        for c in CertificateDer::pem_file_iter(p).map_err(bad)? {
            roots.add(c.map_err(bad)?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", p.display())))?;
        }
    }
    let mut cfg = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(io::Error::other)?
        .with_root_certificates(roots)
        .with_no_client_auth();
    cfg.alpn_protocols = vec![b"h3".to_vec()];
    Ok(cfg)
}

/// Handshake with `host:port` over UDP, giving up after `timeout` (the
/// whole QUIC + TLS + HTTP/3 setup). `host` is also the TLS server name.
pub fn connect(host: &str, port: u16, timeout: Duration, ca: Option<&Path>) -> io::Result<Conn> {
    let addr = (host.trim_matches(['[', ']']), port).to_socket_addrs()?.next().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{host}: no address")))?;
    let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls(ca)?).map_err(io::Error::other)?;
    let mut cfg = quinn::ClientConfig::new(Arc::new(crypto));
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(10)));
    transport.max_idle_timeout(Some(Duration::from_secs(60).try_into().expect("idle timeout")));
    cfg.transport_config(Arc::new(transport));
    let rt = runtime()?;
    let name = host.trim_matches(['[', ']']).to_string();
    rt.block_on(async move {
        let bind: SocketAddr = if addr.is_ipv4() { "0.0.0.0:0".parse().unwrap() } else { "[::]:0".parse().unwrap() };
        let mut ep = quinn::Endpoint::client(bind)?;
        ep.set_default_client_config(cfg);
        let setup = async {
            let conn = ep.connect(addr, &name).map_err(io::Error::other)?.await.map_err(io::Error::other)?;
            let (mut driver, send) = h3::client::new(h3_quinn::Connection::new(conn)).await.map_err(io::Error::other)?;
            tokio::spawn(async move {
                let _ = std::future::poll_fn(|cx| driver.poll_close(cx)).await;
            });
            Ok::<_, io::Error>(send)
        };
        match tokio::time::timeout(timeout, setup).await {
            Ok(Ok(send)) => Ok(Conn { _endpoint: ep, send, addr }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(io::Error::new(io::ErrorKind::TimedOut, format!("no HTTP/3 answer from udp {addr} within {:.1}s", timeout.as_secs_f64()))),
        }
    })
}

impl Conn {
    /// One request on its own stream. `url` keeps the scheme the caller
    /// used for TCP: it goes out as `X-Forwarded-Proto` so links the
    /// server builds (dashboards) point at the TCP origin.
    pub fn request(&self, method: &str, url: &str, token: Option<&str>, body: Option<&[u8]>, ctype: &str, timeout: Duration) -> io::Result<Resp> {
        let (scheme, rest) = url.split_once("://").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("bad URL {url:?}")))?;
        let mut req = http::Request::builder().method(method).uri(format!("https://{rest}")).header("user-agent", "obs-cli").header("x-forwarded-proto", scheme);
        let body = Bytes::copy_from_slice(body.unwrap_or(&[]));
        if !body.is_empty() {
            req = req.header("content-type", ctype);
        }
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let req = req.body(()).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let mut send = self.send.clone();
        let exchange = async move {
            let mut s = send.send_request(req).await.map_err(io::Error::other)?;
            if !body.is_empty() {
                s.send_data(body).await.map_err(io::Error::other)?;
            }
            s.finish().await.map_err(io::Error::other)?;
            let resp = s.recv_response().await.map_err(io::Error::other)?;
            let mut out = Vec::new();
            while let Some(mut c) = s.recv_data().await.map_err(io::Error::other)? {
                while c.has_remaining() {
                    let b = c.chunk();
                    out.extend_from_slice(b);
                    let n = b.len();
                    c.advance(n);
                }
            }
            Ok(Resp { status: resp.status().as_u16(), body: out, alt_svc: None })
        };
        runtime()?.block_on(async move { tokio::time::timeout(timeout, exchange).await.unwrap_or_else(|_| Err(io::Error::new(io::ErrorKind::TimedOut, "HTTP/3 request timed out"))) })
    }
}
