//! Account operations on the [`Auth`] store: signing in and linking
//! credentials, browser sessions, the one-time sign-in secrets (email
//! links, `ssh login` links, zafu and passkey challenges), owned projects
//! and the tokens minted for them. The HTTP side is [`crate::portal`].
//!
//! Linking rule, for every kind of credential: signed in and the credential
//! is unknown -> it joins the current account; held by another account ->
//! [`AccountError::CredTaken`] (accounts are never merged silently);
//! signed out and unknown -> a new account. Two exceptions, both decided by
//! proof of possession:
//!
//! * an **auto** account (made by an SSH key's first connect, never signed
//!   in on the web) whose only credential is the one presented is folded
//!   into the current account, projects included. Whoever proves that key
//!   already controls everything in it.
//! * an SSH key added on the web is unconfirmed (anyone can paste a public
//!   key). Proving the key (`ssh <server> login`) confirms it in the
//!   account that added it, or, from anyone else, drops that claim.
//!
//! Every secret is stored as its SHA-256 ([`hash`]); lookups compare
//! digests. Sessions slide: each use pushes `expires` out, and `last_seen`
//! reaches the disk at most every [`PortalConfig::seen_every`] seconds.

use crate::auth::{ANON_PREFIX, Account, Auth, Challenge, Cred, CredKind, Credential, EmailTok, Owned, ProjectTok, Session, SshTok, Visibility, hash, hash_bytes, new_account_id, random_b64url, random_bytes};
use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AccountError {
    #[error("this credential belongs to another account; sign in with it and remove it there first")]
    CredTaken,
    #[error("this is the account's only credential; add another before removing it")]
    LastCred,
    #[error("that project name is taken")]
    NameTaken,
    #[error("{0}")]
    BadName(String),
    #[error("{0}")]
    AlreadyOwned(String),
    #[error("{0} not found")]
    NotFound(&'static str),
    #[error("{0}")]
    Forbidden(String),
    /// A one-time secret that is unknown, used or expired.
    #[error("this sign-in link or challenge is invalid, used or expired")]
    BadToken,
    #[error("too many sign-in attempts in flight; retry shortly")]
    Busy,
    #[error("auth store: {0}")]
    Store(String),
}

impl AccountError {
    /// The stable `code` of the JSON error.
    pub fn code(&self) -> &'static str {
        match self {
            AccountError::CredTaken => "cred_taken",
            AccountError::LastCred => "last_cred",
            AccountError::NameTaken => "name_taken",
            AccountError::BadName(_) => "bad_name",
            AccountError::AlreadyOwned(_) => "already_owned",
            AccountError::NotFound(_) => "not_found",
            AccountError::Forbidden(_) => "forbidden",
            AccountError::BadToken => "bad_token",
            AccountError::Busy => "busy",
            AccountError::Store(_) => "store",
        }
    }
    pub fn status(&self) -> u16 {
        match self {
            AccountError::CredTaken | AccountError::LastCred | AccountError::NameTaken | AccountError::AlreadyOwned(_) => 409,
            AccountError::BadName(_) => 400,
            AccountError::NotFound(_) => 404,
            AccountError::Forbidden(_) => 403,
            AccountError::BadToken => 401,
            AccountError::Busy => 503,
            AccountError::Store(_) => 500,
        }
    }
}

fn store(e: std::io::Error) -> AccountError {
    AccountError::Store(e.to_string())
}

/// The outcome of presenting a credential.
#[derive(Debug, Clone, PartialEq)]
pub struct SignIn {
    pub account: String,
    /// The credential joined an account that was already signed in.
    pub linked: bool,
    /// A new account was made for it.
    pub created: bool,
}

/// Who is reading: the session's account and the bearer token, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewer {
    pub account: Option<String>,
    pub cred: Cred,
}

impl Viewer {
    pub fn nobody() -> Viewer {
        Viewer { account: None, cred: Cred::None }
    }
}

/// May a viewer read a project?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Unowned or public: anyone.
    Open,
    /// Private, and the viewer is the owner or holds a token for it.
    Allowed,
    /// Private, and the viewer brought no session and no token.
    SignedOut,
    /// Private, and what the viewer brought does not cover it.
    Denied,
}

impl Access {
    pub fn ok(self) -> bool {
        matches!(self, Access::Open | Access::Allowed)
    }
}

/// Names that would shadow a route or read as official.
pub const RESERVED: &[&str] = &["api", "login", "account", "admin", "lab", "docs", "new", "me", "static", "assets"];

/// `^[a-z0-9][a-z0-9-]{2,39}$`, not reserved, not `anon-…`.
pub fn check_project_name(name: &str) -> Result<(), AccountError> {
    let b = name.as_bytes();
    let shape = (3..=40).contains(&b.len()) && b[0].is_ascii_alphanumeric() && !b[0].is_ascii_uppercase() && b.iter().all(|c| c.is_ascii_digit() || c.is_ascii_lowercase() || *c == b'-');
    if !shape {
        return Err(AccountError::BadName(format!("{name:?}: a project name is 3-40 characters of a-z, 0-9 and '-', starting with a letter or digit")));
    }
    if name.starts_with(ANON_PREFIX) || RESERVED.contains(&name) {
        return Err(AccountError::BadName(format!("{name:?} is reserved")));
    }
    Ok(())
}

/// Lowercased, trimmed address, if it looks like one. Strict on purpose:
/// it goes into a mail header.
pub fn normalize_email(s: &str) -> Option<String> {
    let a = s.trim().to_ascii_lowercase();
    let (local, domain) = a.split_once('@')?;
    let ok = (3..=254).contains(&a.len())
        && !local.is_empty()
        && local.len() <= 64
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
        && a.chars().all(|c| c.is_ascii_alphanumeric() || "@.!#$%&'*+/=?^_`{|}~-".contains(c))
        && a.matches('@').count() == 1;
    ok.then_some(a)
}

/// The project of a run id (its first segment).
pub fn project_of(run: &str) -> &str {
    run.split('/').next().unwrap_or(run)
}

/// `ses_` / `tok_` + 16 hex of a stored hash: a public handle on a secret.
fn handle(prefix: &str, h: &str) -> String {
    format!("{prefix}_{}", &h[..16])
}

impl Auth {
    fn now() -> f64 {
        modgrad_obs::now_unix()
    }

    // ------------------------------------------------------------ sign-in

    /// Present a proven credential (the caller verified it), signed in as
    /// `current` or not. See the module docs for the linking rule.
    pub fn sign_in(&self, current: Option<&str>, kind: CredKind) -> Result<SignIn, AccountError> {
        let now = Self::now();
        let key = kind.key();
        let mut db = self.db.write().unwrap();
        if let Some(cur) = current
            && !db.accounts.contains_key(cur)
        {
            return Err(AccountError::NotFound("account"));
        }
        let mut holder = db.holder(&key).map(|(a, c)| (a.id.clone(), a.auto, a.creds.len(), c.kind.clone()));
        // a web-added SSH key that is now proven
        if let Some((hid, _, _, CredKind::Ssh { verified: false, .. })) = &holder {
            let hid = hid.clone();
            let acct = db.accounts.get_mut(&hid).unwrap();
            if current == Some(hid.as_str()) {
                for c in acct.creds.iter_mut().filter(|c| c.kind.key() == key) {
                    c.kind = kind.clone();
                    c.last_used = Some(now);
                }
                own_key_project(&mut db, &kind, &hid);
                self.persist(&mut db).map_err(store)?;
                return Ok(SignIn { account: hid, linked: true, created: false });
            }
            acct.creds.retain(|c| c.kind.key() != key);
            holder = None;
        }
        let out = match (holder, current) {
            (Some((hid, ..)), None) => {
                touch(db.accounts.get_mut(&hid).unwrap(), &kind, now);
                SignIn { account: hid, linked: false, created: false }
            }
            (Some((hid, ..)), Some(cur)) if cur == hid => {
                touch(db.accounts.get_mut(&hid).unwrap(), &kind, now);
                SignIn { account: hid, linked: false, created: false }
            }
            (Some((hid, true, 1, _)), Some(cur)) => {
                // fold the key's auto account into this one
                let gone = db.accounts.remove(&hid).unwrap();
                let acct = db.accounts.get_mut(cur).unwrap();
                for mut c in gone.creds {
                    c.last_used = Some(now);
                    acct.creds.push(c);
                }
                for o in db.owned.values_mut().filter(|o| o.owner == hid) {
                    o.owner = cur.to_string();
                }
                db.sessions.retain(|_, s| s.account != hid);
                SignIn { account: cur.to_string(), linked: true, created: false }
            }
            (Some(_), Some(_)) => return Err(AccountError::CredTaken),
            (None, Some(cur)) => {
                db.accounts.get_mut(cur).unwrap().creds.push(Credential { kind: kind.clone(), added: now, last_used: Some(now) });
                SignIn { account: cur.to_string(), linked: true, created: false }
            }
            (None, None) => {
                let id = new_account_id().map_err(store)?;
                db.accounts.insert(id.clone(), Account { id: id.clone(), created: now, creds: vec![Credential { kind: kind.clone(), added: now, last_used: Some(now) }], auto: false });
                SignIn { account: id, linked: false, created: true }
            }
        };
        own_key_project(&mut db, &kind, &out.account);
        self.persist(&mut db).map_err(store)?;
        Ok(out)
    }

    /// Add an SSH key to an account from its public key alone (no proof):
    /// stored unconfirmed until `ssh <server> login` from that key.
    pub fn add_ssh_key(&self, account: &str, fingerprint: &str, public_key: &str, label: Option<String>) -> Result<String, AccountError> {
        let kind = CredKind::Ssh { fingerprint: fingerprint.into(), public_key: Some(public_key.into()), label, verified: false };
        let id = kind.id();
        let mut db = self.db.write().unwrap();
        match db.holder(&kind.key()) {
            Some((a, _)) if a.id == account => return Ok(id),
            Some(_) => return Err(AccountError::CredTaken),
            None => {}
        }
        let acct = db.accounts.get_mut(account).ok_or(AccountError::NotFound("account"))?;
        acct.creds.push(Credential { kind, added: Self::now(), last_used: None });
        self.persist(&mut db).map_err(store)?;
        Ok(id)
    }

    pub fn account(&self, id: &str) -> Option<Account> {
        self.db.read().unwrap().accounts.get(id).cloned()
    }

    /// The account and stored key of a passkey: (account, COSE key b64url, alg, sign count).
    pub fn passkey(&self, cred_id: &str) -> Option<(String, String, i64, u32)> {
        let db = self.db.read().unwrap();
        match db.holder(&format!("passkey:{cred_id}")) {
            Some((a, Credential { kind: CredKind::Passkey { public_key, alg, sign_count, .. }, .. })) => Some((a.id.clone(), public_key.clone(), *alg, *sign_count)),
            _ => None,
        }
    }

    /// Remove one credential (by [`CredKind::id`]); never the last.
    pub fn remove_cred(&self, account: &str, id: &str) -> Result<(), AccountError> {
        let mut db = self.db.write().unwrap();
        let acct = db.accounts.get_mut(account).ok_or(AccountError::NotFound("account"))?;
        let i = acct.creds.iter().position(|c| c.kind.id() == id).ok_or(AccountError::NotFound("credential"))?;
        if acct.creds.len() == 1 {
            return Err(AccountError::LastCred);
        }
        let c = acct.creds.remove(i);
        // the key no longer routes to the account's project
        if let CredKind::Ssh { fingerprint, .. } = &c.kind {
            db.ssh_keys.remove(fingerprint);
        }
        self.persist(&mut db).map_err(store)?;
        Ok(())
    }

    // ------------------------------------------------------------ sessions

    /// A new browser session; returns the cookie value (32 random bytes,
    /// b64url), which is stored only as its hash.
    pub fn new_session(&self, account: &str, ua: &str) -> Result<String, AccountError> {
        let cookie = random_b64url(32).map_err(store)?;
        let now = Self::now();
        let mut db = self.db.write().unwrap();
        let acct = db.accounts.get_mut(account).ok_or(AccountError::NotFound("account"))?;
        acct.auto = false;
        let ua: String = ua.chars().filter(|c| !c.is_control()).take(200).collect();
        db.sessions.insert(hash(&cookie), Session { account: account.into(), created: now, last_seen: now, expires: now + self.portal.session_days * 86400.0, ua, written: now });
        self.persist(&mut db).map_err(store)?;
        Ok(cookie)
    }

    /// The account of a live session cookie; slides its expiry.
    pub fn session(&self, cookie: &str) -> Option<String> {
        let h = hash(cookie);
        let now = Self::now();
        {
            let db = self.db.read().unwrap();
            let s = db.sessions.get(&h)?;
            if s.expires <= now || !db.accounts.contains_key(&s.account) {
                return None;
            }
            if now - s.written < self.portal.seen_every {
                return Some(s.account.clone());
            }
        }
        let mut db = self.db.write().unwrap();
        let days = self.portal.session_days;
        let s = db.sessions.get_mut(&h)?;
        s.last_seen = now;
        s.expires = now + days * 86400.0;
        s.written = now;
        let account = s.account.clone();
        if let Err(e) = self.persist(&mut db) {
            tracing::warn!("auth store: {e}");
        }
        Some(account)
    }

    /// Sign out the session behind a cookie.
    pub fn end_session(&self, cookie: &str) -> Result<(), AccountError> {
        let mut db = self.db.write().unwrap();
        if db.sessions.remove(&hash(cookie)).is_some() {
            self.persist(&mut db).map_err(store)?;
        }
        Ok(())
    }

    /// Sign out one of an account's sessions by its `ses_…` id.
    pub fn end_session_id(&self, account: &str, id: &str) -> Result<(), AccountError> {
        let mut db = self.db.write().unwrap();
        let h = db.sessions.iter().find(|(h, s)| s.account == account && handle("ses", h) == id).map(|(h, _)| h.clone()).ok_or(AccountError::NotFound("session"))?;
        db.sessions.remove(&h);
        self.persist(&mut db).map_err(store)?;
        Ok(())
    }

    /// `[{id, created, last_seen, ua, current}]`.
    pub fn sessions_json(&self, account: &str, current_cookie: Option<&str>) -> Vec<Value> {
        let cur = current_cookie.map(hash);
        let db = self.db.read().unwrap();
        db.sessions
            .iter()
            .filter(|(_, s)| s.account == account)
            .map(|(h, s)| json!({ "id": handle("ses", h), "created": s.created, "last_seen": s.last_seen, "ua": s.ua, "current": cur.as_deref() == Some(h.as_str()) }))
            .collect()
    }

    // ------------------------------------------------------- one-time secrets

    /// An email sign-in token for `address` (already normalised).
    pub fn email_token(&self, address: &str) -> Result<String, AccountError> {
        let t = random_b64url(32).map_err(store)?;
        let mut db = self.db.write().unwrap();
        db.email_tokens.insert(hash(&t), EmailTok { address: address.into(), expires: Self::now() + self.portal.email_ttl });
        self.persist(&mut db).map_err(store)?;
        Ok(t)
    }

    /// The address of an email token, once.
    pub fn consume_email(&self, token: &str) -> Result<String, AccountError> {
        let mut db = self.db.write().unwrap();
        let t = db.email_tokens.remove(&hash(token)).ok_or(AccountError::BadToken)?;
        self.persist(&mut db).map_err(store)?;
        if t.expires <= Self::now() {
            return Err(AccountError::BadToken);
        }
        Ok(t.address)
    }

    /// An `ssh login` token, bound to the key that asked for it.
    pub fn ssh_login_token(&self, fingerprint: &str, public_key: &str) -> Result<String, AccountError> {
        let t = random_b64url(32).map_err(store)?;
        let mut db = self.db.write().unwrap();
        db.ssh_tokens.insert(hash(&t), SshTok { fingerprint: fingerprint.into(), public_key: public_key.into(), expires: Self::now() + self.portal.ssh_ttl });
        self.persist(&mut db).map_err(store)?;
        Ok(t)
    }

    /// The key behind an `ssh login` token, once: (fingerprint, openssh line).
    pub fn consume_ssh_login(&self, token: &str) -> Result<(String, String), AccountError> {
        let mut db = self.db.write().unwrap();
        let t = db.ssh_tokens.remove(&hash(token)).ok_or(AccountError::BadToken)?;
        self.persist(&mut db).map_err(store)?;
        if t.expires <= Self::now() {
            return Err(AccountError::BadToken);
        }
        Ok((t.fingerprint, t.public_key))
    }

    /// `n` fresh random bytes for a challenge.
    pub fn challenge_bytes(n: usize) -> Result<Vec<u8>, AccountError> {
        random_bytes(n).map_err(store)
    }

    /// Remember a challenge (by hash, in memory) until it is used or
    /// expires; returns when it expires.
    pub fn put_challenge(&self, bytes: &[u8], c: Challenge) -> Result<f64, AccountError> {
        let now = Self::now();
        let expires = now + self.portal.challenge_ttl;
        let mut db = self.db.write().unwrap();
        if db.challenges.len() >= self.portal.max_challenges {
            db.challenges.retain(|_, (_, e)| *e > now);
            if db.challenges.len() >= self.portal.max_challenges {
                return Err(AccountError::Busy);
            }
        }
        db.challenges.insert(hash_bytes(bytes), (c, expires));
        Ok(expires)
    }

    /// Take a challenge back, once; None if unknown, used or expired.
    pub fn take_challenge(&self, bytes: &[u8]) -> Option<Challenge> {
        let (c, e) = self.db.write().unwrap().challenges.remove(&hash_bytes(bytes))?;
        (e > Self::now()).then_some(c)
    }

    // ------------------------------------------------------------ projects

    /// Can `v` read project `project`?
    pub fn readable(&self, project: &str, v: &Viewer) -> Access {
        let db = self.db.read().unwrap();
        let Some(o) = db.owned.get(project) else { return Access::Open };
        if o.visibility == Visibility::Public {
            return Access::Open;
        }
        let token_covers = match &v.cred {
            Cred::Project(p) => p == project,
            Cred::Run(r) => project_of(r) == project,
            Cred::None => false,
        };
        if token_covers || v.account.as_deref() == Some(o.owner.as_str()) {
            Access::Allowed
        } else if v.account.is_none() && v.cred == Cred::None {
            Access::SignedOut
        } else {
            Access::Denied
        }
    }

    pub fn owned(&self, project: &str) -> Option<Owned> {
        self.db.read().unwrap().owned.get(project).cloned()
    }

    /// The account's projects, by name.
    pub fn projects_of(&self, account: &str) -> Vec<(String, Owned)> {
        self.db.read().unwrap().owned.iter().filter(|(_, o)| o.owner == account).map(|(n, o)| (n.clone(), o.clone())).collect()
    }

    /// Does the token store know `name` as a project (owned, tokens, SSH
    /// keys, claims, runs)? The run store and boards are checked by the caller.
    pub fn name_known(&self, name: &str) -> bool {
        let db = self.db.read().unwrap();
        let prefix = format!("{name}/");
        db.owned.contains_key(name)
            || db.projects.values().any(|p| p.project == name)
            || db.ssh_keys.values().any(|p| p == name)
            || db.claims.values().any(|p| p == name)
            || db.runs.keys().any(|r| r.starts_with(&prefix))
    }

    /// A new owned project. `in_use` = the caller found runs or boards under the name.
    pub fn create_project(&self, account: &str, name: &str, visibility: Visibility, in_use: bool) -> Result<(), AccountError> {
        check_project_name(name)?;
        if in_use || self.name_known(name) {
            return Err(AccountError::NameTaken);
        }
        let mut db = self.db.write().unwrap();
        if !db.accounts.contains_key(account) {
            return Err(AccountError::NotFound("account"));
        }
        if db.owned.contains_key(name) {
            return Err(AccountError::NameTaken);
        }
        db.owned.insert(name.into(), Owned { owner: account.into(), visibility, created: Self::now() });
        self.persist(&mut db).map_err(store)
    }

    pub fn set_visibility(&self, account: &str, name: &str, visibility: Visibility) -> Result<(), AccountError> {
        let mut db = self.db.write().unwrap();
        match db.owned.get_mut(name) {
            Some(o) if o.owner == account => o.visibility = visibility,
            _ => return Err(AccountError::NotFound("project")),
        }
        self.persist(&mut db).map_err(store)
    }

    /// Make an `anon-…` project the account's: `token` is a project token
    /// for it (or its `obsc_…` claim token). It becomes private and its
    /// tokens stop expiring.
    pub fn claim(&self, account: &str, project: &str, token: &str) -> Result<(), AccountError> {
        if !project.starts_with(ANON_PREFIX) {
            return Err(AccountError::BadName(format!("{project:?}: only anonymous (anon-…) projects can be claimed")));
        }
        let h = hash(token);
        let now = Self::now();
        let mut db = self.db.write().unwrap();
        let covers = db.projects.get(&h).is_some_and(|p| p.project == project && p.expires_unix.is_none_or(|e| e > now)) || db.claims.get(&h).is_some_and(|p| p == project);
        if !covers {
            return Err(AccountError::Forbidden(format!("that token is not a live token for {project}")));
        }
        if !db.accounts.contains_key(account) {
            return Err(AccountError::NotFound("account"));
        }
        match db.owned.get(project) {
            Some(o) if o.owner == account => return Ok(()),
            Some(_) => return Err(AccountError::AlreadyOwned(format!("{project} already belongs to an account (an SSH key's project is claimed with `ssh <server> login`)"))),
            None => {}
        }
        db.owned.insert(project.into(), Owned { owner: account.into(), visibility: Visibility::Private, created: now });
        for p in db.projects.values_mut().filter(|p| p.project == project) {
            p.expires_unix = None;
        }
        self.persist(&mut db).map_err(store)
    }

    /// Mint a project token for one of the account's projects: (id, token).
    pub fn mint_token(&self, account: &str, project: &str, label: Option<String>, expires_days: Option<f64>) -> Result<(String, String), AccountError> {
        let t = format!("obsp_{}", random_bytes(24).map_err(store)?.iter().map(|x| format!("{x:02x}")).collect::<String>());
        let h = hash(&t);
        let now = Self::now();
        let mut db = self.db.write().unwrap();
        if db.owned.get(project).is_none_or(|o| o.owner != account) {
            return Err(AccountError::NotFound("project"));
        }
        let label = label.map(|l| l.chars().filter(|c| !c.is_control()).take(100).collect());
        db.projects.insert(h.clone(), ProjectTok { project: project.into(), expires_unix: expires_days.filter(|d| *d > 0.0).map(|d| now + d * 86400.0), label, created: Some(now), last_used: None });
        self.persist(&mut db).map_err(store)?;
        Ok((handle("tok", &h), t))
    }

    /// Revoke a token (by `tok_…` id) of one of the account's projects.
    pub fn revoke_token(&self, account: &str, id: &str) -> Result<(), AccountError> {
        let mut db = self.db.write().unwrap();
        let owner_of = |p: &str| db.owned.get(p).map(|o| o.owner.clone());
        let h = db.projects.iter().find(|(h, p)| handle("tok", h) == id && owner_of(&p.project).as_deref() == Some(account)).map(|(h, _)| h.clone()).ok_or(AccountError::NotFound("token"))?;
        db.projects.remove(&h);
        self.persist(&mut db).map_err(store)
    }

    /// `[{id, project, label, created, expires, last_used}]` for the account's projects.
    pub fn tokens_json(&self, account: &str) -> Vec<Value> {
        let db = self.db.read().unwrap();
        db.projects
            .iter()
            .filter(|(_, p)| db.owned.get(&p.project).is_some_and(|o| o.owner == account))
            .map(|(h, p)| json!({ "id": handle("tok", h), "project": p.project, "label": p.label, "created": p.created, "expires": p.expires_unix, "last_used": p.last_used }))
            .collect()
    }

    /// Delete an account: its sessions, its projects' tokens, SSH routes
    /// and ownership. Returns the projects whose data the caller removes.
    pub fn delete_account(&self, account: &str) -> Result<Vec<String>, AccountError> {
        let mut db = self.db.write().unwrap();
        let acct = db.accounts.remove(account).ok_or(AccountError::NotFound("account"))?;
        let projects: Vec<String> = db.owned.iter().filter(|(_, o)| o.owner == account).map(|(n, _)| n.clone()).collect();
        let mine = |p: &str| projects.iter().any(|x| x == p);
        db.owned.retain(|_, o| o.owner != account);
        db.sessions.retain(|_, s| s.account != account);
        db.projects.retain(|_, p| !mine(&p.project));
        db.claims.retain(|_, p| !mine(p));
        db.ssh_keys.retain(|_, p| !mine(p));
        for c in &acct.creds {
            if let CredKind::Ssh { fingerprint, .. } = &c.kind {
                db.ssh_keys.remove(fingerprint);
            }
        }
        let gone: Vec<String> = db.runs.keys().filter(|r| mine(project_of(r))).cloned().collect();
        for r in gone {
            if let Some(Some(h)) = db.runs.remove(&r) {
                db.run_tokens.remove(&h);
            }
        }
        self.persist(&mut db).map_err(store)?;
        Ok(projects)
    }
}

/// Record a use: `last_used`, a passkey's new sign count, a migrated
/// key's missing public key line.
fn touch(a: &mut Account, kind: &CredKind, now: f64) {
    for c in a.creds.iter_mut().filter(|c| c.kind.key() == kind.key()) {
        c.last_used = Some(now);
        match (&mut c.kind, kind) {
            (CredKind::Passkey { sign_count, .. }, CredKind::Passkey { sign_count: n, .. }) => *sign_count = *n,
            (CredKind::Ssh { public_key: p @ None, .. }, CredKind::Ssh { public_key: Some(l), .. }) => *p = Some(l.clone()),
            _ => {}
        }
    }
}

/// An SSH key's project belongs to the key's account.
fn own_key_project(db: &mut crate::auth::Db, kind: &CredKind, account: &str) {
    let CredKind::Ssh { fingerprint, .. } = kind else { return };
    let Some(p) = db.ssh_keys.get(fingerprint).cloned() else { return };
    let live_owner = db.owned.get(&p).is_some_and(|o| db.accounts.contains_key(&o.owner));
    if !live_owner {
        let created = db.owned.get(&p).map_or_else(modgrad_obs::now_unix, |o| o.created);
        let visibility = db.owned.get(&p).map_or(Visibility::Public, |o| o.visibility);
        db.owned.insert(p, Owned { owner: account.into(), visibility, created });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{AuthConfig, PortalConfig};

    fn email(a: &str) -> CredKind {
        CredKind::Email { address: a.into() }
    }
    fn ssh(fp: &str) -> CredKind {
        CredKind::Ssh { fingerprint: fp.into(), public_key: Some(format!("ssh-ed25519 AAAA {fp}")), label: None, verified: true }
    }

    #[test]
    fn linking_rule() {
        let a = Auth::memory(AuthConfig::default());
        let s = a.sign_in(None, email("a@x.org")).unwrap();
        assert!(s.created && !s.linked);
        assert_eq!(a.sign_in(None, email("a@x.org")).unwrap(), SignIn { account: s.account.clone(), linked: false, created: false });
        let z = CredKind::Zafu { public_key: "ab".repeat(32) };
        assert_eq!(a.sign_in(Some(&s.account), z.clone()).unwrap(), SignIn { account: s.account.clone(), linked: true, created: false });
        let other = a.sign_in(None, email("b@x.org")).unwrap();
        a.new_session(&other.account, "ua").unwrap();
        assert_eq!(a.sign_in(Some(&other.account), z), Err(AccountError::CredTaken), "no silent merge");
        assert_eq!(a.sign_in(Some(&s.account), email("b@x.org")), Err(AccountError::CredTaken));
        assert_eq!(a.account(&s.account).unwrap().creds.len(), 2);
    }

    #[test]
    fn auto_ssh_account_folds_in_when_its_key_is_proven() {
        let a = Auth::memory(AuthConfig::default());
        let (project, new) = a.ssh_key_project_with("SHA256:k", Some("ssh-ed25519 AAAA k")).unwrap();
        assert!(new);
        let auto = a.owned(&project).unwrap().owner;
        assert!(a.account(&auto).unwrap().auto);
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        let s = a.sign_in(Some(&me), ssh("SHA256:k")).unwrap();
        assert_eq!(s, SignIn { account: me.clone(), linked: true, created: false });
        assert!(a.account(&auto).is_none());
        assert_eq!(a.owned(&project).unwrap().owner, me, "the key's project came along");
        // a key whose account signed in on the web is a real account: 409
        let (_, _) = a.ssh_key_project_with("SHA256:j", Some("ssh-ed25519 AAAA j")).unwrap();
        let j = a.sign_in(None, ssh("SHA256:j")).unwrap().account;
        a.new_session(&j, "").unwrap();
        assert_eq!(a.sign_in(Some(&me), ssh("SHA256:j")), Err(AccountError::CredTaken));
    }

    #[test]
    fn web_added_ssh_key_is_a_claim_until_proven() {
        let a = Auth::memory(AuthConfig::default());
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        let id = a.add_ssh_key(&me, "SHA256:v", "ssh-ed25519 AAAA v", Some("laptop".into())).unwrap();
        assert_eq!(id, ssh("SHA256:v").id());
        // the door does not route an unconfirmed key
        assert_eq!(a.ssh_key_project_with("SHA256:v", None), Err(crate::AuthError::Unconfirmed));
        // proven by the account that added it: confirmed
        assert_eq!(a.sign_in(Some(&me), ssh("SHA256:v")).unwrap().account, me);
        let (p, new) = a.ssh_key_project_with("SHA256:v", None).unwrap();
        assert!(new);
        assert_eq!(a.owned(&p).unwrap().owner, me);
        // someone else pastes a key they do not hold; the holder proves it and wins
        let squatter = a.sign_in(None, email("s@x.org")).unwrap().account;
        a.add_ssh_key(&squatter, "SHA256:w", "ssh-ed25519 AAAA w", None).unwrap();
        let holder = a.sign_in(None, ssh("SHA256:w")).unwrap();
        assert!(holder.created && holder.account != squatter);
        assert_eq!(a.account(&squatter).unwrap().creds.len(), 1, "the claim is dropped");
        assert_eq!(a.add_ssh_key(&squatter, "SHA256:w", "ssh-ed25519 AAAA w", None), Err(AccountError::CredTaken));
    }

    #[test]
    fn last_cred_and_removal() {
        let a = Auth::memory(AuthConfig::default());
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        assert_eq!(a.remove_cred(&me, &email("me@x.org").id()), Err(AccountError::LastCred));
        a.sign_in(Some(&me), CredKind::Zafu { public_key: "cd".repeat(32) }).unwrap();
        a.remove_cred(&me, &email("me@x.org").id()).unwrap();
        assert_eq!(a.remove_cred(&me, "cred_nope"), Err(AccountError::NotFound("credential")));
        // the address is free again
        assert!(a.sign_in(None, email("me@x.org")).unwrap().created);
    }

    #[test]
    fn sessions_slide_and_expire() {
        let a = Auth::memory(AuthConfig::default()).with_portal(PortalConfig { seen_every: 0.0, ..Default::default() });
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        let c = a.new_session(&me, "firefox\n").unwrap();
        assert_eq!(c.len(), 43);
        assert_eq!(a.session(&c), Some(me.clone()));
        let list = a.sessions_json(&me, Some(&c));
        assert_eq!((list.len(), list[0]["current"].as_bool(), list[0]["ua"].as_str()), (1, Some(true), Some("firefox")));
        a.end_session_id(&me, list[0]["id"].as_str().unwrap()).unwrap();
        assert_eq!(a.session(&c), None);
        let short = Auth::memory(AuthConfig::default()).with_portal(PortalConfig { session_days: 1e-9, ..Default::default() });
        let me = short.sign_in(None, email("me@x.org")).unwrap().account;
        let c = short.new_session(&me, "").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(short.session(&c), None);
        assert!(short.db.read().unwrap().sessions.is_empty() || { short.persist(&mut short.db.write().unwrap()).unwrap(); short.db.read().unwrap().sessions.is_empty() });
    }

    #[test]
    fn one_time_secrets() {
        let a = Auth::memory(AuthConfig::default());
        let t = a.email_token("me@x.org").unwrap();
        assert_eq!(a.consume_email(&t).unwrap(), "me@x.org");
        assert_eq!(a.consume_email(&t), Err(AccountError::BadToken), "single use");
        let s = a.ssh_login_token("SHA256:k", "ssh-ed25519 AAAA k").unwrap();
        assert_eq!(a.consume_ssh_login(&s).unwrap().0, "SHA256:k");
        assert_eq!(a.consume_ssh_login(&s), Err(AccountError::BadToken));
        a.put_challenge(b"chal", Challenge::Zafu).unwrap();
        assert_eq!(a.take_challenge(b"chal"), Some(Challenge::Zafu));
        assert_eq!(a.take_challenge(b"chal"), None);
        let x = Auth::memory(AuthConfig::default()).with_portal(PortalConfig { email_ttl: 1e-9, challenge_ttl: 1e-9, max_challenges: 1, ..Default::default() });
        let t = x.email_token("me@x.org").unwrap();
        x.put_challenge(b"c1", Challenge::Zafu).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(x.consume_email(&t), Err(AccountError::BadToken), "expired");
        assert_eq!(x.take_challenge(b"c1"), None, "expired");
        let y = Auth::memory(AuthConfig::default()).with_portal(PortalConfig { max_challenges: 1, ..Default::default() });
        y.put_challenge(b"c2", Challenge::Zafu).unwrap();
        assert_eq!(y.put_challenge(b"c3", Challenge::Zafu), Err(AccountError::Busy), "bounded");
    }

    #[test]
    fn projects_tokens_and_access() {
        let a = Auth::memory(AuthConfig::default());
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        let you = a.sign_in(None, email("you@x.org")).unwrap().account;
        for bad in ["ab", "Abc", "-ab", "anon-x", "api", "a_b", &"x".repeat(41)] {
            assert!(matches!(a.create_project(&me, bad, Visibility::Private, false), Err(AccountError::BadName(_))), "{bad}");
        }
        a.create_project(&me, "mine", Visibility::Private, false).unwrap();
        assert_eq!(a.create_project(&you, "mine", Visibility::Private, false), Err(AccountError::NameTaken));
        a.add_project_token("ops").unwrap();
        assert_eq!(a.create_project(&me, "ops", Visibility::Private, false), Err(AccountError::NameTaken), "an operator's project");
        let mine = |acct: Option<&str>, cred: Cred| a.readable("mine", &Viewer { account: acct.map(String::from), cred });
        assert_eq!(mine(None, Cred::None), Access::SignedOut);
        assert_eq!(mine(Some(&you), Cred::None), Access::Denied);
        assert_eq!(mine(Some(&me), Cred::None), Access::Allowed);
        assert_eq!(mine(None, Cred::Project("mine".into())), Access::Allowed);
        assert_eq!(mine(None, Cred::Run("mine/r".into())), Access::Allowed);
        assert_eq!(mine(None, Cred::Project("ops".into())), Access::Denied);
        assert_eq!(a.readable("ops", &Viewer::nobody()), Access::Open);
        assert_eq!(a.set_visibility(&you, "mine", Visibility::Public), Err(AccountError::NotFound("project")));
        a.set_visibility(&me, "mine", Visibility::Public).unwrap();
        assert_eq!(mine(None, Cred::None), Access::Open);
        // tokens
        assert_eq!(a.mint_token(&you, "mine", None, None), Err(AccountError::NotFound("project")));
        let (id, t) = a.mint_token(&me, "mine", Some("ci".into()), Some(7.0)).unwrap();
        assert!(id.starts_with("tok_") && t.starts_with("obsp_"));
        assert_eq!(a.cred(Some(&t)).unwrap(), Cred::Project("mine".into()));
        let list = a.tokens_json(&me);
        assert_eq!((list.len(), list[0]["label"].as_str()), (1, Some("ci")));
        assert!(!list[0].to_string().contains(&t), "listing never shows the token");
        assert_eq!(a.revoke_token(&you, &id), Err(AccountError::NotFound("token")));
        a.revoke_token(&me, &id).unwrap();
        assert!(a.cred(Some(&t)).is_err());
    }

    #[test]
    fn claim_anonymous_project() {
        let a = Auth::memory(AuthConfig::default());
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        let p = a.new_anon_project().unwrap();
        assert!(p.expires_unix.is_some());
        assert!(matches!(a.claim(&me, &p.project, "obsp_wrong"), Err(AccountError::Forbidden(_))));
        assert!(matches!(a.claim(&me, "mine", &p.token), Err(AccountError::BadName(_))));
        a.claim(&me, &p.project, &p.token).unwrap();
        let o = a.owned(&p.project).unwrap();
        assert_eq!((o.owner.as_str(), o.visibility), (me.as_str(), Visibility::Private));
        assert!(a.tokens_json(&me)[0]["expires"].is_null(), "claimed projects stop expiring");
        let you = a.sign_in(None, email("you@x.org")).unwrap().account;
        assert!(matches!(a.claim(&you, &p.project, &p.claim), Err(AccountError::AlreadyOwned(_))));
        a.claim(&me, &p.project, &p.claim).unwrap();
    }

    #[test]
    fn delete_account_takes_its_projects() {
        let a = Auth::memory(AuthConfig::default());
        let me = a.sign_in(None, email("me@x.org")).unwrap().account;
        a.create_project(&me, "mine", Visibility::Private, false).unwrap();
        let (_, t) = a.mint_token(&me, "mine", None, None).unwrap();
        let rt = a.register_run("mine/r", false).unwrap().unwrap();
        let c = a.new_session(&me, "").unwrap();
        assert_eq!(a.delete_account(&me).unwrap(), vec!["mine".to_string()]);
        assert_eq!(a.session(&c), None);
        assert!(a.cred(Some(&t)).is_err() && a.cred(Some(&rt)).is_err());
        assert!(a.owned("mine").is_none() && !a.name_known("mine"));
    }

    #[test]
    fn names_and_addresses() {
        assert_eq!(normalize_email("  Me@Example.ORG "), Some("me@example.org".into()));
        for bad in ["me", "me@x", "a@b@c.d", "me@x.org\r\nBcc: y@z.org", "<me@x.org>", "me @x.org", "@x.org"] {
            assert_eq!(normalize_email(bad), None, "{bad:?}");
        }
        assert!(check_project_name("my-proj-1").is_ok());
        assert_eq!(project_of("p/r/x"), "p");
    }
}
