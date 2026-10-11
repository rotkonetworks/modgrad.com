//! Write authorisation for the v1 ingest API, run auto-creation, and the
//! account model behind the portal ([`crate::portal`]).
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
//!   [`AuthConfig::anon_expiry_days`] unless claimed. Signed-in owners mint
//!   them for their projects (`POST /api/v1/me/tokens`).
//!
//! An **SSH key** (the `ssh` feature) is an identity of its own: its SHA256
//! fingerprint maps to one anonymous project, made on first connect and
//! kept for good (the key is the claim). `ssh <server> token` mints
//! expiring project tokens for that project.
//!
//! **Accounts** (schema v2) group credentials: SSH keys, email addresses,
//! zafu keys and passkeys ([`CredKind`]). A key's first connect makes an
//! *auto* account holding just that key and owning its project; signing in
//! on the web makes it a real one. Owned projects ([`Owned`]) are private
//! or public; private ones are read only by the owner's session or a token
//! for the project ([`Auth::readable`]). Browser sessions are random cookie
//! values stored as hashes, like every other secret. The account operations
//! live in [`crate::account`].
//!
//! Run ids: with a project token, an id without `/` is put in the token's
//! project (`exp1` -> `anon-3f9a…/exp1`). Writing to a run that does not
//! exist creates it (no mandatory create call).
//!
//! Dev mode (`modgrad-server --dev`): requests without a token may create and
//! write runs that have no run token. A run created with a token stays
//! protected. Reads are open except private owned projects; anonymous
//! projects are hidden from the unfiltered index and reachable by their
//! unguessable board URL.
//!
//! [`Authed`] is the tower service in front of [`IngestService`]; the HTTP
//! adapter only extracts the bearer header.

use crate::error::BoxError;
use crate::ingest::{IngestRequest, IngestResponse, IngestService};
use crate::read::ReadState;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
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
/// The `version` this build writes into [`AUTH_FILE`]. A file without one
/// is v1 (tokens and SSH keys only) and is migrated on open.
pub const SCHEMA_VERSION: u64 = 2;

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
    /// An SSH key added on the web that has not proven itself over SSH yet.
    #[error("this key was added to a modgrad account on the web but not confirmed; run `ssh <server> login` and open the link signed in to that account")]
    Unconfirmed,
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

/// Lifetimes and limits of the sign-in secrets (seconds unless noted).
#[derive(Debug, Clone, Copy)]
pub struct PortalConfig {
    /// A session expires this many days after it was last seen.
    pub session_days: f64,
    /// `last_seen` is written to disk at most this often.
    pub seen_every: f64,
    pub email_ttl: f64,
    pub ssh_ttl: f64,
    /// zafu and passkey challenges.
    pub challenge_ttl: f64,
    /// Sign-in emails: (at most, per window) per address and per client IP.
    pub email_per_address: (usize, f64),
    pub email_per_ip: (usize, f64),
    /// Outstanding challenges kept in memory; beyond this new ones are refused.
    pub max_challenges: usize,
}

impl Default for PortalConfig {
    fn default() -> Self {
        PortalConfig { session_days: 30.0, seen_every: 600.0, email_ttl: 900.0, ssh_ttl: 600.0, challenge_ttl: 300.0, email_per_address: (3, 900.0), email_per_ip: (10, 3600.0), max_challenges: 10_000 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ProjectTok {
    pub project: String,
    #[serde(default)]
    pub expires_unix: Option<f64>,
    /// Set on tokens minted on the web (`POST /api/v1/me/tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<f64>,
    /// Kept in memory on every use, written with the next change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used: Option<f64>,
}

/// One way to sign in. Serialised with a `kind` tag (`ssh`, `email`,
/// `zafu`, `passkey`); unknown fields are ignored so the enum can grow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CredKind {
    Ssh {
        /// `SHA256:…`, exactly as [`crate::ssh::fingerprint`] prints it.
        fingerprint: String,
        /// The openssh public key line; None for keys migrated from v1,
        /// filled on the key's next connect.
        #[serde(default)]
        public_key: Option<String>,
        #[serde(default)]
        label: Option<String>,
        /// False for a key added on the web (`POST /api/v1/me/creds/ssh`)
        /// that has not connected yet: anyone can paste a public key, so it
        /// does not route that key's pushes until `ssh <server> login`
        /// proves the holder agrees.
        #[serde(default = "yes")]
        verified: bool,
    },
    Email {
        /// Lowercased and trimmed.
        address: String,
    },
    Zafu {
        /// Hex of the 32-byte ed25519 key (zafu's per-site ZID key).
        public_key: String,
    },
    Passkey {
        /// b64url, as the authenticator reports it.
        cred_id: String,
        /// COSE key, b64url.
        public_key: String,
        alg: i64,
        sign_count: u32,
        #[serde(default)]
        label: Option<String>,
    },
}

fn yes() -> bool {
    true
}

impl CredKind {
    pub fn kind(&self) -> &'static str {
        match self {
            CredKind::Ssh { .. } => "ssh",
            CredKind::Email { .. } => "email",
            CredKind::Zafu { .. } => "zafu",
            CredKind::Passkey { .. } => "passkey",
        }
    }
    /// What makes two credentials the same one (one account may hold it).
    pub fn key(&self) -> String {
        match self {
            CredKind::Ssh { fingerprint, .. } => format!("ssh:{fingerprint}"),
            CredKind::Email { address } => format!("email:{address}"),
            CredKind::Zafu { public_key } => format!("zafu:{}", public_key.to_ascii_lowercase()),
            CredKind::Passkey { cred_id, .. } => format!("passkey:{cred_id}"),
        }
    }
    /// Stable public id (`cred_<16 hex>`), for listing and removal.
    pub fn id(&self) -> String {
        format!("cred_{}", &hash(&self.key())[..16])
    }
    /// What the account page shows.
    pub fn label(&self) -> String {
        match self {
            CredKind::Ssh { fingerprint, label, .. } => label.clone().unwrap_or_else(|| fingerprint.clone()),
            CredKind::Email { address } => address.clone(),
            CredKind::Zafu { public_key } => format!("zafu {}…", &public_key[..public_key.len().min(12)]),
            CredKind::Passkey { label, .. } => label.clone().unwrap_or_else(|| "passkey".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Credential {
    #[serde(flatten)]
    pub kind: CredKind,
    pub added: f64,
    #[serde(default)]
    pub last_used: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Account {
    /// `acct_<22 base32>`.
    pub id: String,
    pub created: f64,
    pub creds: Vec<Credential>,
    /// Made implicitly (an SSH key's first connect, or the v1 migration)
    /// and never signed in on the web. Its single key, proven again, may be
    /// moved into another account (see [`crate::account`]).
    #[serde(default)]
    pub auto: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Private,
    Public,
}

impl Visibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Visibility::Private => "private",
            Visibility::Public => "public",
        }
    }
    pub fn parse(s: &str) -> Option<Visibility> {
        match s {
            "private" => Some(Visibility::Private),
            "public" => Some(Visibility::Public),
            _ => None,
        }
    }
}

/// A project with an owner. Unowned projects (operator tokens, anonymous
/// ones nobody claimed) read as they always did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Owned {
    pub owner: String,
    pub visibility: Visibility,
    pub created: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub account: String,
    pub created: f64,
    pub last_seen: f64,
    pub expires: f64,
    #[serde(default)]
    pub ua: String,
    /// `last_seen` as last written to disk.
    #[serde(skip)]
    pub(crate) written: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct EmailTok {
    pub address: String,
    pub expires: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SshTok {
    pub fingerprint: String,
    pub public_key: String,
    pub expires: f64,
}

/// What an outstanding challenge was issued for (kept in memory only).
#[derive(Debug, Clone, PartialEq)]
pub enum Challenge {
    Zafu,
    /// `account`: the signed-in account it adds to; `user`: the WebAuthn
    /// user handle (the account id, or a fresh one for a new account).
    PasskeyRegister { account: Option<String>, user: String },
    PasskeyLogin,
}

#[derive(Default)]
pub(crate) struct Db {
    /// run id -> run-token hash (None = created without a token, dev mode)
    pub runs: BTreeMap<String, Option<String>>,
    /// run-token hash -> run id
    pub run_tokens: BTreeMap<String, String>,
    /// project-token hash -> project
    pub projects: BTreeMap<String, ProjectTok>,
    /// claim-token hash -> project
    pub claims: BTreeMap<String, String>,
    /// SSH key fingerprint (`SHA256:…`) -> project
    pub ssh_keys: BTreeMap<String, String>,
    /// account id -> account
    pub accounts: BTreeMap<String, Account>,
    /// project -> owner
    pub owned: BTreeMap<String, Owned>,
    /// session-cookie hash -> session
    pub sessions: BTreeMap<String, Session>,
    /// email sign-in token hash -> address
    pub email_tokens: BTreeMap<String, EmailTok>,
    /// `ssh login` token hash -> key
    pub ssh_tokens: BTreeMap<String, SshTok>,
    /// challenge hash -> (what for, expires); never written to disk
    pub challenges: HashMap<String, (Challenge, f64)>,
}

impl Db {
    /// Drop expired sessions and one-time secrets.
    pub(crate) fn prune(&mut self, now: f64) {
        self.sessions.retain(|_, s| s.expires > now);
        self.email_tokens.retain(|_, t| t.expires > now);
        self.ssh_tokens.retain(|_, t| t.expires > now);
        self.challenges.retain(|_, (_, e)| *e > now);
    }

    /// The account holding a credential with this [`CredKind::key`].
    pub(crate) fn holder(&self, key: &str) -> Option<(&Account, &Credential)> {
        self.accounts.values().find_map(|a| a.creds.iter().find(|c| c.kind.key() == key).map(|c| (a, c)))
    }
}

/// The token store. Cheap to clone (shared).
#[derive(Clone)]
pub struct Auth {
    pub(crate) db: Arc<RwLock<Db>>,
    path: Option<PathBuf>,
    pub cfg: AuthConfig,
    pub portal: PortalConfig,
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

/// Lowercase hex SHA-256: how every secret is stored and looked up (so
/// lookups compare digests, never the secrets themselves).
pub fn hash(token: &str) -> String {
    hash_bytes(token.as_bytes())
}

pub fn hash_bytes(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn random_bytes(n: usize) -> io::Result<Vec<u8>> {
    use std::io::Read;
    let mut b = vec![0u8; n];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(b)
}

fn random_hex(bytes: usize) -> io::Result<String> {
    Ok(random_bytes(bytes)?.iter().map(|x| format!("{x:02x}")).collect())
}

/// `n` random bytes as unpadded base64url (cookies, email and SSH links).
pub(crate) fn random_b64url(n: usize) -> io::Result<String> {
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random_bytes(n)?))
}

/// `acct_` + 22 lowercase base32 characters (110 random bits).
pub(crate) fn new_account_id() -> io::Result<String> {
    const A: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let b = random_bytes(14)?;
    let mut bits: u128 = 0;
    for x in &b {
        bits = (bits << 8) | *x as u128;
    }
    Ok(format!("acct_{}", (0..22).map(|i| A[((bits >> (107 - 5 * i)) & 31) as usize] as char).collect::<String>()))
}

pub(crate) fn store_err(e: io::Error) -> AuthError {
    AuthError::Store(e.to_string())
}

/// A v2 section, or empty when absent.
fn section<T: serde::de::DeserializeOwned + Default>(v: &Value, k: &str) -> io::Result<T> {
    match v.get(k) {
        None | Some(Value::Null) => Ok(T::default()),
        Some(x) => serde_json::from_value(x.clone()).map_err(|e| io::Error::other(format!("{AUTH_FILE}: {k}: {e}"))),
    }
}

/// Read a store file of any known version; returns the db and the version read.
fn load(v: &Value) -> io::Result<(Db, u64)> {
    let version = v.get("version").and_then(Value::as_u64).unwrap_or(1);
    if version > SCHEMA_VERSION {
        return Err(io::Error::other(format!("{AUTH_FILE} has schema version {version}; this modgrad-server reads up to {SCHEMA_VERSION}: upgrade it")));
    }
    let mut db = Db::default();
    for (run, h) in v["runs"].as_object().into_iter().flatten() {
        let h = h.as_str().map(String::from);
        if let Some(h) = &h {
            db.run_tokens.insert(h.clone(), run.clone());
        }
        db.runs.insert(run.clone(), h);
    }
    for (h, p) in v["projects"].as_object().into_iter().flatten() {
        if let Ok(t) = serde_json::from_value::<ProjectTok>(p.clone()) {
            db.projects.insert(h.clone(), t);
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
    db.accounts = section(v, "accounts")?;
    db.owned = section(v, "owned")?;
    db.sessions = section(v, "sessions")?;
    for s in db.sessions.values_mut() {
        s.written = s.last_seen;
    }
    db.email_tokens = section(v, "email_tokens")?;
    db.ssh_tokens = section(v, "ssh_tokens")?;
    if version < 2 {
        migrate_v1(&mut db, modgrad_obs::now_unix())?;
    }
    Ok((db, version))
}

/// v1 -> v2: every SSH key becomes an auto account holding that key and
/// owning the key's project. The project stays public (as it read before);
/// its owner can make it private.
fn migrate_v1(db: &mut Db, now: f64) -> io::Result<()> {
    for (fp, project) in db.ssh_keys.clone() {
        if db.holder(&format!("ssh:{fp}")).is_some() {
            continue;
        }
        let id = new_account_id()?;
        let cred = Credential { kind: CredKind::Ssh { fingerprint: fp, public_key: None, label: None, verified: true }, added: now, last_used: None };
        db.accounts.insert(id.clone(), Account { id: id.clone(), created: now, creds: vec![cred], auto: true });
        db.owned.entry(project).or_insert(Owned { owner: id, visibility: Visibility::Public, created: now });
    }
    Ok(())
}

impl Auth {
    /// In-memory store (tests).
    pub fn memory(cfg: AuthConfig) -> Auth {
        Auth { db: Arc::default(), path: None, cfg, portal: PortalConfig::default() }
    }

    /// Load `<root>/.obs-auth.json` (missing = empty). A v1 file is
    /// migrated and rewritten as v2 (the original is kept next to it as
    /// `.obs-auth.json.v1`).
    pub fn open(root: &Path, cfg: AuthConfig) -> io::Result<Auth> {
        let path = root.join(AUTH_FILE);
        let (db, version) = match std::fs::read(&path) {
            Ok(b) => {
                let v: Value = serde_json::from_slice(&b)?;
                let (db, version) = load(&v)?;
                if version < SCHEMA_VERSION {
                    write_private(&path.with_extension("json.v1"), &b)?;
                }
                (db, version)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => (Db::default(), SCHEMA_VERSION),
            Err(e) => return Err(e),
        };
        let a = Auth { db: Arc::new(RwLock::new(db)), path: Some(path), cfg, portal: PortalConfig::default() };
        if version < SCHEMA_VERSION {
            let mut db = a.db.write().unwrap();
            a.persist(&mut db)?;
        }
        Ok(a)
    }

    /// Replace the sign-in lifetimes and limits (tests shorten them).
    pub fn with_portal(mut self, p: PortalConfig) -> Auth {
        self.portal = p;
        self
    }

    pub fn dev(&self) -> bool {
        self.cfg.dev
    }

    /// Prune, then write the whole store (tmp + rename, mode 0600).
    pub(crate) fn persist(&self, db: &mut Db) -> io::Result<()> {
        db.prune(modgrad_obs::now_unix());
        let Some(path) = &self.path else { return Ok(()) };
        let v = json!({
            "version": SCHEMA_VERSION,
            "runs": db.runs,
            "projects": db.projects,
            "claims": db.claims,
            "ssh_keys": db.ssh_keys,
            "accounts": db.accounts,
            "owned": db.owned,
            "sessions": db.sessions,
            "email_tokens": db.email_tokens,
            "ssh_tokens": db.ssh_tokens,
        });
        let tmp = path.with_extension("json.tmp");
        write_private(&tmp, &serde_json::to_vec_pretty(&v)?)?;
        std::fs::rename(tmp, path)
    }

    /// Provision a (non-expiring) project token; returns the plaintext.
    pub fn add_project_token(&self, project: &str) -> io::Result<String> {
        let t = format!("obsp_{}", random_hex(24)?);
        let mut db = self.db.write().unwrap();
        db.projects.insert(hash(&t), ProjectTok { project: project.into(), created: Some(modgrad_obs::now_unix()), ..Default::default() });
        self.persist(&mut db)?;
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
    /// claim token (for attaching it to an account later).
    pub fn new_anon_project(&self) -> Result<AnonProject, AuthError> {
        let project = self.anon_name()?;
        let token = format!("obsp_{}", random_hex(24).map_err(store_err)?);
        let claim = format!("obsc_{}", random_hex(24).map_err(store_err)?);
        let expires_unix = self.anon_expiry();
        let mut db = self.db.write().unwrap();
        db.projects.insert(hash(&token), ProjectTok { project: project.clone(), expires_unix, created: Some(modgrad_obs::now_unix()), ..Default::default() });
        db.claims.insert(hash(&claim), project.clone());
        self.persist(&mut db).map_err(store_err)?;
        Ok(AnonProject { project, token, claim, expires_unix })
    }

    /// The project of an SSH key (by its `SHA256:…` fingerprint), made on
    /// first sight. Returns (project, new). With anonymous projects
    /// disabled, known keys still work and new ones are refused.
    pub fn ssh_key_project(&self, fingerprint: &str) -> Result<(String, bool), AuthError> {
        self.ssh_key_project_with(fingerprint, None)
    }

    /// [`Auth::ssh_key_project`], also recording the key's openssh line. A
    /// new key gets an auto account that holds it and owns its project
    /// (public, like every anonymous project). A key added on the web and
    /// not yet confirmed is refused ([`AuthError::Unconfirmed`]).
    pub fn ssh_key_project_with(&self, fingerprint: &str, public_key: Option<&str>) -> Result<(String, bool), AuthError> {
        let key = format!("ssh:{fingerprint}");
        {
            let db = self.db.read().unwrap();
            let unconfirmed = matches!(db.holder(&key), Some((_, Credential { kind: CredKind::Ssh { verified: false, .. }, .. })));
            if unconfirmed {
                return Err(AuthError::Unconfirmed);
            }
            let missing_line = public_key.is_some() && matches!(db.holder(&key), Some((_, Credential { kind: CredKind::Ssh { public_key: None, .. }, .. })));
            if let (Some(p), false) = (db.ssh_keys.get(fingerprint), missing_line) {
                return Ok((p.clone(), false));
            }
        }
        let project = self.anon_name();
        let mut db = self.db.write().unwrap();
        // fill in the line of a key migrated from v1
        if let Some(line) = public_key {
            for a in db.accounts.values_mut() {
                for c in &mut a.creds {
                    if let CredKind::Ssh { fingerprint: f, public_key: p @ None, .. } = &mut c.kind
                        && f == fingerprint
                    {
                        *p = Some(line.to_string());
                    }
                }
            }
        }
        // a concurrent first connect of the same key may have won the race
        if let Some(p) = db.ssh_keys.get(fingerprint).cloned() {
            self.persist(&mut db).map_err(store_err)?;
            return Ok((p, false));
        }
        let project = project?;
        let now = modgrad_obs::now_unix();
        db.ssh_keys.insert(fingerprint.to_string(), project.clone());
        let owner = match db.holder(&key) {
            Some((a, _)) => a.id.clone(),
            None => {
                let id = new_account_id().map_err(store_err)?;
                let cred = Credential { kind: CredKind::Ssh { fingerprint: fingerprint.into(), public_key: public_key.map(String::from), label: None, verified: true }, added: now, last_used: Some(now) };
                db.accounts.insert(id.clone(), Account { id: id.clone(), created: now, creds: vec![cred], auto: true });
                id
            }
        };
        db.owned.insert(project.clone(), Owned { owner, visibility: Visibility::Public, created: now });
        self.persist(&mut db).map_err(store_err)?;
        Ok((project, true))
    }

    /// A new project token for an SSH key's project, expiring like an
    /// anonymous one. Returns (token, expires_unix).
    pub fn ssh_project_token(&self, project: &str) -> Result<(String, Option<f64>), AuthError> {
        let token = format!("obsp_{}", random_hex(24).map_err(store_err)?);
        let expires_unix = self.anon_expiry();
        let mut db = self.db.write().unwrap();
        db.projects.insert(hash(&token), ProjectTok { project: project.into(), expires_unix, created: Some(modgrad_obs::now_unix()), label: Some("ssh token".into()), ..Default::default() });
        self.persist(&mut db).map_err(store_err)?;
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
        let now = modgrad_obs::now_unix();
        {
            let db = self.db.read().unwrap();
            if let Some(r) = db.run_tokens.get(&h) {
                return Ok(Cred::Run(r.clone()));
            }
            match db.projects.get(&h) {
                Some(p) if p.expires_unix.is_some_and(|e| now > e) => return Err(AuthError::Expired),
                // last_used to the minute: no write lock on every push
                Some(p) if p.last_used.is_some_and(|u| now - u < 60.0) => return Ok(Cred::Project(p.project.clone())),
                Some(_) => {}
                None => return Err(AuthError::Unauthorized),
            }
        }
        // in memory only; written with the next change to the store
        let mut db = self.db.write().unwrap();
        let p = db.projects.get_mut(&h).ok_or(AuthError::Unauthorized)?;
        p.last_used = Some(now);
        Ok(Cred::Project(p.project.clone()))
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
        self.persist(&mut db).map_err(store_err)?;
        Ok(t)
    }
}

/// Create (or truncate) a file readable by its owner only, and fill it.
fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    let mut f = o.open(path)?;
    f.write_all(bytes)?;
    f.sync_all()
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
        IngestRequest::DeleteProject { .. } => {}
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
            if matches!(r.req, IngestRequest::DeleteProject { .. }) {
                return Err(Box::new(AuthError::Forbidden("projects are deleted with their account".into())) as BoxError);
            }
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

    #[test]
    fn v1_file_opens_and_migrates() {
        let dir = std::env::temp_dir().join(format!("obs-auth-v1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // what a v1 server wrote: no version key
        let (pt, rt) = ("obsp_aaaa", "obsr_bbbb");
        let v1 = json!({
            "runs": { "anon-1/r": hash(rt), "dev-run": null },
            "projects": { hash(pt): { "project": "anon-1", "expires_unix": null } },
            "claims": { hash("obsc_cccc"): "anon-1" },
            "ssh_keys": { "SHA256:key1": "anon-1", "SHA256:key2": "anon-2" },
        });
        std::fs::write(dir.join(AUTH_FILE), serde_json::to_vec(&v1).unwrap()).unwrap();
        let a = Auth::open(&dir, cfg(false)).unwrap();
        assert_eq!(a.cred(Some(pt)).unwrap(), Cred::Project("anon-1".into()));
        assert_eq!(a.cred(Some(rt)).unwrap(), Cred::Run("anon-1/r".into()));
        assert_eq!(a.claim_project("obsc_cccc"), Some("anon-1".into()));
        for (fp, project) in [("SHA256:key1", "anon-1"), ("SHA256:key2", "anon-2")] {
            let o = a.owned(project).expect("the key's project is owned");
            let acct = a.account(&o.owner).unwrap();
            assert!(acct.id.starts_with("acct_") && acct.id.len() == 27 && acct.auto);
            assert!(matches!(&acct.creds[..], [Credential { kind: CredKind::Ssh { fingerprint, verified: true, .. }, .. }] if fingerprint == fp));
            assert_eq!(o.visibility, Visibility::Public, "migrated projects read as before");
            assert_eq!(a.ssh_key_project(fp).unwrap(), (project.to_string(), false));
        }
        // rewritten as v2, the v1 original kept; reopening does not migrate twice
        let text: Value = serde_json::from_slice(&std::fs::read(dir.join(AUTH_FILE)).unwrap()).unwrap();
        assert_eq!(text["version"], SCHEMA_VERSION);
        assert_eq!(text["accounts"].as_object().unwrap().len(), 2);
        assert!(dir.join(".obs-auth.json.v1").is_file());
        let b = Auth::open(&dir, cfg(false)).unwrap();
        assert_eq!(b.owned("anon-1"), a.owned("anon-1"));
        assert_eq!(b.db.read().unwrap().accounts.len(), 2);
        // the line of a migrated key is filled on its next connect
        b.ssh_key_project_with("SHA256:key1", Some("ssh-ed25519 AAAA k1")).unwrap();
        let acct = b.account(&b.owned("anon-1").unwrap().owner).unwrap();
        assert!(matches!(&acct.creds[0].kind, CredKind::Ssh { public_key: Some(l), .. } if l == "ssh-ed25519 AAAA k1"));
        // a newer file is refused rather than rewritten
        std::fs::write(dir.join(AUTH_FILE), br#"{"version": 99}"#).unwrap();
        assert!(Auth::open(&dir, cfg(false)).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
