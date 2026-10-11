//! HTTP/3 adapter (feature `h3`): a QUIC listener (quinn + h3) in front of
//! the same axum [`Router`] the TCP listener serves. Each h3 request is
//! turned into an ordinary `http::Request` (body buffered, capped at
//! [`MAX_BODY`]), handed to the router, and the response body is streamed
//! back frame by frame, so SSE works too. Nothing here knows the API: the
//! v1 routes see the same requests as over TCP.
//!
//! QUIC needs TLS. [`Tls::from_pem`] loads a real certificate;
//! [`Tls::self_signed`] is for local testing only (clients must be told to
//! trust it). [`advertise`] adds `Alt-Svc: h3=":<port>"` to the TCP
//! responses so clients know where to find the UDP listener.

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, Request, Response, StatusCode, header};
use bytes::{Buf, Bytes, BytesMut};
use http_body_util::BodyExt;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

/// Request bodies over h3 are buffered before routing; larger ones get 413.
/// Same as axum's default body limit, which the TCP path enforces: buffering
/// more would only hold memory for requests the router refuses anyway.
pub const MAX_BODY: usize = 2 << 20;

pub struct Tls {
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
}

impl Tls {
    /// A certificate chain and private key from PEM files (e.g. the ones the
    /// TLS proxy in front of the TCP listener uses for the same hostname).
    pub fn from_pem(cert: &Path, key: &Path) -> io::Result<Tls> {
        let bad = |what: &Path, e: rustls::pki_types::pem::Error| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", what.display()));
        let chain = CertificateDer::pem_file_iter(cert).map_err(|e| bad(cert, e))?.collect::<Result<Vec<_>, _>>().map_err(|e| bad(cert, e))?;
        if chain.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: no certificate", cert.display())));
        }
        let key = PrivateKeyDer::from_pem_file(key).map_err(|e| bad(key, e))?;
        Ok(Tls { chain, key })
    }

    /// A fresh self-signed certificate for `names` (DNS names or IPs), for
    /// local testing. Returns the certificate as PEM so it can be written
    /// where clients can be pointed at it (`modgrad run --tls-ca <pem>`).
    pub fn self_signed(names: Vec<String>) -> io::Result<(Tls, String)> {
        let c = rcgen::generate_simple_self_signed(names).map_err(io::Error::other)?;
        let pem = c.cert.pem();
        let key = PrivateKeyDer::try_from(c.key_pair.serialize_der()).map_err(io::Error::other)?;
        Ok((Tls { chain: vec![c.cert.der().clone()], key }, pem))
    }
}

/// Bind the UDP listener. Must run inside a tokio runtime.
pub fn endpoint(addr: SocketAddr, tls: Tls) -> io::Result<quinn::Endpoint> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut cfg = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(io::Error::other)?
        .with_no_client_auth()
        .with_single_cert(tls.chain, tls.key)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    cfg.alpn_protocols = vec![b"h3".to_vec()];
    let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(cfg).map_err(io::Error::other)?;
    let mut server = quinn::ServerConfig::with_crypto(Arc::new(crypto));
    let mut transport = quinn::TransportConfig::default();
    // a run that logs rarely keeps its connection: clients send keep-alives
    // every 10 s, so 60 s of silence means the client is gone
    transport.max_idle_timeout(Some(Duration::from_secs(60).try_into().expect("idle timeout")));
    server.transport_config(Arc::new(transport));
    quinn::Endpoint::server(server, addr)
}

/// `Alt-Svc` on every response of `app`: "HTTP/3 is on UDP `port` of this
/// same host". Put this on the TCP router only.
pub fn advertise(app: Router, port: u16) -> Router {
    let v = HeaderValue::from_str(&format!("h3=\":{port}\"; ma=86400")).expect("alt-svc value");
    app.layer(axum::middleware::map_response(move |mut r: Response<Body>| {
        let v = v.clone();
        async move {
            r.headers_mut().insert(header::ALT_SVC, v);
            r
        }
    }))
}

/// Accept connections until the endpoint is closed; every connection and
/// every request on it runs in its own task.
pub async fn serve(ep: quinn::Endpoint, app: Router) {
    while let Some(incoming) = ep.accept().await {
        let app = app.clone();
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => return tracing::debug!("h3 handshake: {e}"),
            };
            let remote = conn.remote_address();
            let mut h3c = match h3::server::builder().build::<_, Bytes>(h3_quinn::Connection::new(conn)).await {
                Ok(c) => c,
                Err(e) => return tracing::debug!(%remote, "h3 setup: {e}"),
            };
            loop {
                match h3c.accept().await {
                    Ok(Some(resolver)) => {
                        let app = app.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle(resolver, app).await {
                                tracing::debug!(%remote, "h3 request: {e}");
                            }
                        });
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::debug!(%remote, "h3 connection: {e}");
                        break;
                    }
                }
            }
        });
    }
}

type Resolver = h3::server::RequestResolver<h3_quinn::Connection, Bytes>;

async fn handle(resolver: Resolver, app: Router) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (req, mut stream) = resolver.resolve_request().await?;
    let (mut parts, ()) = req.into_parts();
    let mut body = BytesMut::new();
    while let Some(mut chunk) = stream.recv_data().await? {
        if body.len() + chunk.remaining() > MAX_BODY {
            let r = Response::builder().status(StatusCode::PAYLOAD_TOO_LARGE).body(()).expect("response");
            stream.send_response(r).await?;
            stream.finish().await?;
            return Ok(());
        }
        while chunk.has_remaining() {
            let c = chunk.chunk();
            body.extend_from_slice(c);
            let n = c.len();
            chunk.advance(n);
        }
    }
    // HTTP/3 has no Host header (it is :authority); the v1 handlers build
    // dashboard links from Host and X-Forwarded-Proto
    if !parts.headers.contains_key(header::HOST)
        && let Some(a) = parts.uri.authority().and_then(|a| HeaderValue::from_str(a.as_str()).ok())
    {
        parts.headers.insert(header::HOST, a);
    }
    parts.headers.entry("x-forwarded-proto").or_insert(HeaderValue::from_static("https"));
    let resp = app.oneshot(Request::from_parts(parts, Body::from(body.freeze()))).await.unwrap_or_else(|e| match e {});
    let (mut rparts, mut rbody) = resp.into_parts();
    for h in [header::CONNECTION, header::TRANSFER_ENCODING, header::UPGRADE] {
        rparts.headers.remove(h);
    }
    rparts.headers.remove("keep-alive");
    stream.send_response(Response::from_parts(rparts, ())).await?;
    // frame by frame: an SSE body never ends, and each event goes out as it comes
    while let Some(frame) = rbody.frame().await {
        if let Ok(data) = frame?.into_data()
            && !data.is_empty()
        {
            stream.send_data(data).await?;
        }
    }
    stream.finish().await?;
    Ok(())
}
