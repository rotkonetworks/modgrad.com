//! Write authorisation for the v1 ingest API, and run auto-creation.
//!
//! Bearer tokens, stored only as SHA-256 hashes in `<root>/.obs-auth.json`
//! (mode 0600; dotfiles are never served):
//!
//! * a **run token** (`obsr_…`), minted when a run is created, writes to
//!   that run only;
//! * a **project token** (`obsp_…`) creates and writes any run in its
//!   project. Operators provision them (`modgrad-server token add <project>`);
//!   anyone can get an **anonymous** one (`POST /api/v1/projects`) for a
//!   fresh `anon-<random>` project, which expires after
//!   [`AuthConfig::anon_expiry_days`] unless claimed.
//!
//! An **SSH key** (the `ssh` feature) is an identity of its own: its SHA256
//! fingerprint maps to one anonymous project, made on first connect and
//! kept for good (the key is the claim). `ssh <server> token` mints
//! expiring project tokens for that project.
//!
//! Run ids: with a project token, an id without `/` is put in the token's
//! project (`exp1` -> `anon-3f9a…/exp1`). Writing to a run that does not
//! exist creates it (no mandatory create call).
//!
//! Dev mode (`modgrad-server --dev`): requests without a token may create and
//! write runs that have no run token. A run created with a token stays
//! protected. Reads are open; anonymous projects are hidden from the
//! unfiltered index and reachable by their unguessable board URL.
//!
//! [`Authed`] is the tower service in front of [`IngestService`]; the HTTP
//! adapter only extracts the bearer header.

use crate::error::BoxError;
use crate::ingest::{IngestRequest, IngestResponse, IngestService};
use crate::read::ReadState;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};
use tower::{Service, ServiceExt};

pub const AUTH_FILE: &str = ".obs-auth.json";
/// Prefix of anonymous project names.
pub const ANON_PREFIX: &str = "anon-";

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AuthError {
    /// No usable credential (HTTP 401).
    #[error("this write needs a bearer token: missing or not recognised")]
    Unauthorized,
    #[error("this token expired; get a new one with POST /api/v1/projects")]
    Expired,
    /// A valid credential that does not cover this request (HTTP 403).
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// The run exists and the caller may not resume it (HTTP 409).
    #[error("run {0:?} already exists and has its own token; resume it with that token or pick another name")]
    Conflict(String),
    #[error("auth store: {0}")]
    Store(String),
}

#[derive(Debug, Clone, Copy)]
pub struct AuthConfig {
    /// Allow tokenless creates and writes to unprotected runs.
    pub dev: bool,
    /// Allow `POST /api/v1/projects` (anonymous project tokens).
    pub anon_projects: bool,
    /// Anonymous tokens expire this many days after issue; 0 = never.
    pub anon_expiry_days: f64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        AuthConfig { dev: false, anon_projects: true, anon_expiry_days: 30.0 }
    }
}

#[derive(Debug, Clone)]
struct ProjectTok {
    project: String,
    expires_unix: Option<f64>,
}

#[derive(Default)]
struct Db {
    /// run id -> run-token hash (None = created without a token, dev mode)
    runs: BTreeMap<String, Option<String>>,
    /// run-token hash -> run id
    run_tokens: BTreeMap<String, String>,
    /// project-token hash -> project
    projects: BTreeMap<String, ProjectTok>,
    /// claim-token hash -> project
    claims: BTreeMap<String, String>,
    /// SSH key fingerprint (`SHA256:…`) -> project
    ssh_keys: BTreeMap<String, String>,
}

/// The token store. Cheap to clone (shared).
#[derive(Clone)]
pub struct Auth {
    db: Arc<RwLock<Db>>,
    path: Option<PathBuf>,
    pub cfg: AuthConfig,
}

/// What a bearer token is.
#[derive(Debug, Clone, PartialEq)]
pub enum Cred {
    None,
    Run(String),
    Project(String),
}

/// A freshly issued anonymous project.
#[derive(Debug, Clone)]
pub struct AnonProject {
    pub project: String,
    pub token: String,
    pub claim: String,
    pub expires_unix: Option<f64>,
}

fn hash(token: &str) -> String {
    Sha256::digest(token.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

fn random_hex(bytes: usize) -> io::Result<String> {
    use std::io::Read;
    let mut b = vec![0u8; bytes];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

fn store_err(e: io::Error) -> AuthError {
    AuthError::Store(e.to_string())
}

impl Auth {
    /// In-memory store (tests).
    pub fn memory(cfg: AuthConfig) -> Auth {
        Auth { db: Arc::default(), path: None, cfg }
    }

    /// Load `<root>/.obs-auth.json` (missing = empty).
    pub fn open(root: &Path, cfg: AuthConfig) -> io::Result<Auth> {
        let path = root.join(AUTH_FILE);
        let mut db = Db::default();
        match std::fs::read(&path) {
            Ok(b) => {
                let v: Value = serde_json::from_slice(&b)?;
                for (run, h) in v["runs"].as_object().into_iter().flatten() {
                    let h = h.as_str().map(String::from);
                    if let Some(h) = &h {
                        db.run_tokens.insert(h.clone(), run.clone());
                    }
                    db.runs.insert(run.clone(), h);
                }
                for (h, p) in v["projects"].as_object().into_iter().flatten() {
                    if let Some(project) = p["project"].as_str() {
                        db.projects.insert(h.clone(), ProjectTok { project: project.into(), expires_unix: p["expires_unix"].as_f64() });
                    }
                }
                for (h, p) in v["claims"].as_object().into_iter().flatten() {
                    if let Some(p) = p.as_str() {
                        db.claims.insert(h.clone(), p.into());
                    }
                }
                for (fp, p) in v["ssh_keys"].as_object().into_iter().flatten() {
                    if let Some(p) = p.as_str() {
                        db.ssh_keys.insert(fp.clone(), p.into());
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        Ok(Auth { db: Arc::new(RwLock::new(db)), path: Some(path), cfg })
    }

    pub fn dev(&self) -> bool {
        self.cfg.dev
    }

    fn persist(&self, db: &Db) -> io::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        let v = json!({
            "runs": db.runs,
            "projects": db.projects.iter().map(|(h, p)| (h.clone(), json!({"project": p.project, "expires_unix": p.expires_unix}))).collect::<serde_json::Map<_, _>>(),
            "claims": db.claims,
            "ssh_keys": db.ssh_keys,
        });
        let tmp = path.with_extension("json.tmp");
        {
            use std::io::Write;
            let mut o = std::fs::OpenOptions::new();
            o.write(true).create(true).truncate(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
            let mut f = o.open(&tmp)?;
            f.write_all(&serde_json::to_vec_pretty(&v)?)?;
        }
        std::fs::rename(tmp, path)
    }

    /// Provision a (non-expiring) project token; returns the plaintext.
    pub fn add_project_token(&self, project: &str) -> io::Result<String> {
        let t = format!("obsp_{}", random_hex(24)?);
        let mut db = self.db.write().unwrap();
        db.projects.insert(hash(&t), ProjectTok { project: project.into(), expires_unix: None });
        self.persist(&db)?;
        Ok(t)
    }

    /// A fresh `anon-<16 hex>` name, if anonymous projects are allowed.
    fn anon_name(&self) -> Result<String, AuthError> {
        if !self.cfg.anon_projects {
            return Err(AuthError::Forbidden("anonymous projects are disabled on this server".into()));
        }
        Ok(format!("{ANON_PREFIX}{}", random_hex(8).map_err(store_err)?))
    }

    /// When an anonymous token issued now expires (None = never).
    fn anon_expiry(&self) -> Option<f64> {
        let d = self.cfg.anon_expiry_days;
        (d > 0.0).then(|| modgrad_obs::now_unix() + d * 86400.0)
    }

    /// Issue an anonymous project: `anon-<16 hex>`, a project token, and a
    /// claim token (for attaching an SSH key or email later).
    pub fn new_anon_project(&self) -> Result<AnonProject, AuthError> {
        let project = self.anon_name()?;
        let token = format!("obsp_{}", random_hex(24).map_err(store_err)?);
        let claim = format!("obsc_{}", random_hex(24).map_err(store_err)?);
        let expires_unix = self.anon_expiry();
        let mut db = self.db.write().unwrap();
        db.projects.insert(hash(&token), ProjectTok { project: project.clone(), expires_unix });
        db.claims.insert(hash(&claim), project.clone());
        self.persist(&db).map_err(store_err)?;
        Ok(AnonProject { project, token, claim, expires_unix })
    }

    /// The project of an SSH key (by its `SHA256:…` fingerprint), made on
    /// first sight. Returns (project, new). With anonymous projects
    /// disabled, known keys still work and new ones are refused.
    pub fn ssh_key_project(&self, fingerprint: &str) -> Result<(String, bool), AuthError> {
        if let Some(p) = self.db.read().unwrap().ssh_keys.get(fingerprint) {
            return Ok((p.clone(), false));
        }
        let project = self.anon_name()?;
        let mut db = self.db.write().unwrap();
        // a concurrent first connect of the same key may have won the race
        if let Some(p) = db.ssh_keys.get(fingerprint) {
            return Ok((p.clone(), false));
        }
        db.ssh_keys.insert(fingerprint.to_string(), project.clone());
        self.persist(&db).map_err(store_err)?;
        Ok((project, true))
    }

    /// A new project token for an SSH key's project, expiring like an
    /// anonymous one. Returns (token, expires_unix).
    pub fn ssh_project_token(&self, project: &str) -> Result<(String, Option<f64>), AuthError> {
        let token = format!("obsp_{}", random_hex(24).map_err(store_err)?);
        let expires_unix = self.anon_expiry();
        let mut db = self.db.write().unwrap();
        db.projects.insert(hash(&token), ProjectTok { project: project.into(), expires_unix });
        self.persist(&db).map_err(store_err)?;
        Ok((token, expires_unix))
    }

    /// The project a claim token is for.
    pub fn claim_project(&self, claim: &str) -> Option<String> {
        self.db.read().unwrap().claims.get(&hash(claim)).cloned()
    }

    /// Classify a bearer token.
    pub fn cred(&self, token: Option<&str>) -> Result<Cred, AuthError> {
        let Some(t) = token else { return Ok(Cred::None) };
        let h = hash(t);
        let db = self.db.read().unwrap();
        if let Some(r) = db.run_tokens.get(&h) {
            return Ok(Cred::Run(r.clone()));
        }
        match db.projects.get(&h) {
            Some(p) if p.expires_unix.is_some_and(|e| modgrad_obs::now_unix() > e) => Err(AuthError::Expired),
            Some(p) => Ok(Cred::Project(p.project.clone())),
            None => Err(AuthError::Unauthorized),
        }
    }

    /// The full run id for `id` under `cred`: a project token puts an id
    /// without `/` into its project.
    pub fn resolve(id: &str, cred: &Cred) -> String {
        match cred {
            Cred::Project(p) if !id.contains('/') => format!("{p}/{id}"),
            _ => id.to_string(),
        }
    }

    /// May `cred` write to (or create) run `id`? `exists` = in the store.
    pub fn check(&self, id: &str, cred: &Cred) -> Result<(), AuthError> {
        let db = self.db.read().unwrap();
        let protected = matches!(db.runs.get(id), Some(Some(_)));
        match cred {
            Cred::Run(r) if r == id => Ok(()),
            Cred::Run(r) => Err(AuthError::Forbidden(format!("this run token is for {r:?}, not {id:?}"))),
            Cred::Project(p) if id.strip_prefix(p.as_str()).is_some_and(|rest| rest.starts_with('/')) => Ok(()),
            Cred::Project(p) => Err(AuthError::Forbidden(format!("this project token is for project {p:?}; run ids must start with \"{p}/\" (or have no '/')"))),
            Cred::None if self.cfg.dev && !protected => Ok(()),
            Cred::None if protected => Err(AuthError::Conflict(id.to_string())),
            Cred::None => Err(AuthError::Unauthorized),
        }
    }

    fn known(&self, id: &str) -> bool {
        self.db.read().unwrap().runs.contains_key(id)
    }

    /// Record a new run; mints a run token unless `anonymous`.
    pub fn register_run(&self, id: &str, anonymous: bool) -> Result<Option<String>, AuthError> {
        let t = if anonymous { None } else { Some(format!("obsr_{}", random_hex(24).map_err(store_err)?)) };
        let mut db = self.db.write().unwrap();
        if let Some(t) = &t {
            db.run_tokens.insert(hash(t), id.to_string());
        }
        db.runs.insert(id.to_string(), t.as_deref().map(hash));
        self.persist(&db).map_err(store_err)?;
        Ok(t)
    }
}

/// An ingest request plus the caller's bearer token. The request's run id
/// is resolved against the token (see [`Auth::resolve`]).
#[derive(Debug, Clone)]
pub struct AuthedRequest {
    pub token: Option<String>,
    /// A credential the transport already established (SSH: the key's
    /// project); takes the place of `token`.
    pub cred: Option<Cred>,
    pub req: IngestRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuthedResponse {
    /// The resolved run id.
    pub run: String,
    /// The run did not exist before this request.
    pub created: bool,
    /// A run token minted for a new run (shown once).
    pub token: Option<String>,
    pub resp: IngestResponse,
}

/// Authorisation and auto-create in front of the ingest stack.
#[derive(Clone)]
pub struct Authed {
    pub auth: Auth,
    pub read: ReadState,
    pub inner: IngestService,
}

fn set_run(req: &mut IngestRequest, id: String) {
    match req {
        IngestRequest::CreateRun { name, .. } => *name = id,
        IngestRequest::Log { run, .. }
        | IngestRequest::LogArray { run, .. }
        | IngestRequest::Event { run, .. }
        | IngestRequest::Rows { run, .. }
        | IngestRequest::Finish { run, .. }
        | IngestRequest::Summary { run, .. }
        | IngestRequest::Config { run, .. } => *run = id,
    }
}

impl Service<AuthedRequest> for Authed {
    type Response = AuthedResponse;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<AuthedResponse, BoxError>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, r: AuthedRequest) -> Self::Future {
        let (auth, read, inner) = (self.auth.clone(), self.read.clone(), self.inner.clone());
        Box::pin(async move {
            let cred = match r.cred {
                Some(c) => c,
                None => auth.cred(r.token.as_deref())?,
            };
            let mut req = r.req;
            let id = Auth::resolve(req.run(), &cred);
            set_run(&mut req, id.clone());
            auth.check(&id, &cred)?;
            if !crate::state::valid_run_name(&id) {
                return Err(Box::new(crate::error::StateError::InvalidRunName(id)) as BoxError);
            }
            let project = match &cred {
                Cred::Project(p) => Some(p.clone()),
                _ => id.split_once('/').map(|(p, _)| p.to_string()),
            };
            let exists = read.snapshot().runs.contains_key(&id);
            let anonymous = cred == Cred::None;
            let mut token = None;
            match req {
                IngestRequest::CreateRun { name, project: body_project, config, meta } => {
                    let project = body_project.filter(|_| !matches!(cred, Cred::Project(_))).or(project);
                    let resp = inner.oneshot(IngestRequest::CreateRun { name, project, config, meta }).await?;
                    // an explicit create always gets a run token (protecting the run);
                    // only implicit tokenless auto-creates stay open
                    if !auth.known(&id) || (anonymous && !exists) {
                        token = auth.register_run(&id, false)?;
                    }
                    Ok(AuthedResponse { run: id, created: !exists, token, resp })
                }
                other => {
                    if !exists {
                        inner.clone().oneshot(IngestRequest::CreateRun { name: id.clone(), project, config: json!({}), meta: None }).await?;
                    }
                    if !auth.known(&id) {
                        token = auth.register_run(&id, anonymous)?;
                    }
                    let resp = inner.oneshot(other).await?;
                    Ok(AuthedResponse { run: id, created: !exists, token, resp })
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dev: bool) -> AuthConfig {
        AuthConfig { dev, ..Default::default() }
    }

    #[test]
    fn run_and_project_tokens() {
        let a = Auth::memory(cfg(false));
        assert_eq!(a.check("p/r", &Cred::None), Err(AuthError::Unauthorized));
        let pt = a.add_project_token("p").unwrap();
        let c = a.cred(Some(&pt)).unwrap();
        assert_eq!(c, Cred::Project("p".into()));
        assert_eq!(Auth::resolve("r", &c), "p/r");
        assert_eq!(Auth::resolve("q/r", &c), "q/r");
        assert_eq!(a.check("p/r", &c), Ok(()));
        assert!(matches!(a.check("q/r", &c), Err(AuthError::Forbidden(_))));
        assert!(matches!(a.check("px/r", &c), Err(AuthError::Forbidden(_))), "prefix must end at a '/'");
        let rt = a.register_run("p/r", false).unwrap().unwrap();
        let rc = a.cred(Some(&rt)).unwrap();
        assert_eq!(rc, Cred::Run("p/r".into()));
        assert_eq!(a.check("p/r", &rc), Ok(()));
        assert!(matches!(a.check("p/s", &rc), Err(AuthError::Forbidden(_))));
        assert_eq!(a.cred(Some("nope")), Err(AuthError::Unauthorized));
        assert_eq!(a.check("p/r", &Cred::None), Err(AuthError::Conflict("p/r".into())));
    }

    #[test]
    fn dev_mode_tokenless_only_for_unprotected_runs() {
        let a = Auth::memory(cfg(true));
        assert_eq!(a.check("r", &Cred::None), Ok(()));
        assert_eq!(a.register_run("r", true).unwrap(), None);
        assert_eq!(a.check("r", &Cred::None), Ok(()));
        a.register_run("s", false).unwrap();
        assert_eq!(a.check("s", &Cred::None), Err(AuthError::Conflict("s".into())));
    }

    #[test]
    fn anonymous_projects_expire() {
        let a = Auth::memory(AuthConfig { anon_expiry_days: -1.0 / 86400.0, ..cfg(false) });
        // negative expiry is "never" (only > 0 sets one)
        let p = a.new_anon_project().unwrap();
        assert!(p.project.starts_with(ANON_PREFIX) && p.expires_unix.is_none());
        assert_eq!(a.cred(Some(&p.token)).unwrap(), Cred::Project(p.project.clone()));
        assert_eq!(a.claim_project(&p.claim), Some(p.project.clone()));
        let b = Auth::memory(AuthConfig { anon_expiry_days: 1e-9, ..cfg(false) });
        let q = b.new_anon_project().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(b.cred(Some(&q.token)), Err(AuthError::Expired));
        let off = Auth::memory(AuthConfig { anon_projects: false, ..cfg(false) });
        assert!(off.new_anon_project().is_err());
    }

    #[test]
    fn persists_hashes_only() {
        let dir = std::env::temp_dir().join(format!("obs-auth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = Auth::open(&dir, cfg(false)).unwrap();
        let pt = a.add_project_token("p").unwrap();
        let rt = a.register_run("p/r", false).unwrap().unwrap();
        let anon = a.new_anon_project().unwrap();
        let text = std::fs::read_to_string(dir.join(AUTH_FILE)).unwrap();
        for secret in [&pt, &rt, &anon.token, &anon.claim] {
            assert!(!text.contains(secret.as_str()));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join(AUTH_FILE)).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let b = Auth::open(&dir, cfg(false)).unwrap();
        assert_eq!(b.cred(Some(&rt)).unwrap(), Cred::Run("p/r".into()));
        assert_eq!(b.cred(Some(&pt)).unwrap(), Cred::Project("p".into()));
        assert_eq!(b.cred(Some(&anon.token)).unwrap(), Cred::Project(anon.project.clone()));
        assert_eq!(b.claim_project(&anon.claim), Some(anon.project));
        let (kp, new) = b.ssh_key_project("SHA256:abc").unwrap();
        assert!(new && kp.starts_with(ANON_PREFIX));
        let c = Auth::open(&dir, AuthConfig { anon_projects: false, ..cfg(false) }).unwrap();
        assert_eq!(c.ssh_key_project("SHA256:abc").unwrap(), (kp, false), "a known key survives a restart and --no-anon");
        assert!(c.ssh_key_project("SHA256:other").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
