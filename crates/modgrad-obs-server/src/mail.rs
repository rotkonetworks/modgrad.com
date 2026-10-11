//! Outgoing mail for email sign-in links, behind [`Mailer`]
//! (`modgrad-server --mail <spec>`, see [`from_spec`]):
//!
//! * `log` (default): nothing is sent; the message, link included, is
//!   logged at info level (`RUST_LOG=info`). For development and tests.
//! * `cmd:<path>`: the message as RFC 5322 text on the stdin of
//!   `<path> -t -i` (sendmail, msmtp and friends).
//! * `jmap` (feature `jmap`): JMAP submission (e.g. Stalwart) with Basic
//!   auth from `MODGRAD_JMAP_URL` (the session URL, usually
//!   `https://host/.well-known/jmap`), `MODGRAD_JMAP_USER`,
//!   `MODGRAD_JMAP_PASS`; From is the JMAP user.
//!
//! [`MemoryMailer`] keeps messages in memory for tests.

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub struct Mail {
    pub to: String,
    pub subject: String,
    /// Plain text.
    pub body: String,
}

pub trait Mailer: Send + Sync {
    /// Deliver (or hand off) one message. Runs on a blocking thread.
    fn send(&self, m: &Mail) -> Result<(), String>;
    /// Shown at startup.
    fn describe(&self) -> String;
}

/// The sign-in message: subject, the link, its lifetime, what to do if
/// unexpected.
pub fn sign_in_mail(to: &str, link: &str, minutes: u64) -> Mail {
    Mail {
        to: to.into(),
        subject: "Sign in to modgrad".into(),
        body: format!(
            "Open this link to sign in to modgrad:\n\n{link}\n\nIt works once and expires in {minutes} minutes.\n\nIf you didn't ask to sign in, ignore this email; nothing happens without the link.\n"
        ),
    }
}

/// `--mail log`: logs instead of sending.
pub struct LogMailer;

impl Mailer for LogMailer {
    fn send(&self, m: &Mail) -> Result<(), String> {
        tracing::info!(to = %m.to, "mail not sent (--mail log): {}\n{}", m.subject, m.body);
        Ok(())
    }
    fn describe(&self) -> String {
        "log (sign-in links are logged at info level, not sent; RUST_LOG=info)".into()
    }
}

/// Keeps every message (tests).
#[derive(Clone, Default)]
pub struct MemoryMailer(pub Arc<Mutex<Vec<Mail>>>);

impl Mailer for MemoryMailer {
    fn send(&self, m: &Mail) -> Result<(), String> {
        self.0.lock().unwrap().push(m.clone());
        Ok(())
    }
    fn describe(&self) -> String {
        "memory".into()
    }
}

/// `--mail cmd:<path>`: RFC 5322 on the stdin of `<path> -t -i`.
pub struct CmdMailer {
    pub path: std::path::PathBuf,
    pub from: String,
}

impl Mailer for CmdMailer {
    fn send(&self, m: &Mail) -> Result<(), String> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let mut child = Command::new(&self.path).args(["-t", "-i"]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().map_err(|e| format!("{}: {e}", self.path.display()))?;
        child.stdin.take().ok_or("no stdin")?.write_all(rfc5322(m, &self.from, modgrad_obs::now_unix()).as_bytes()).map_err(|e| e.to_string())?;
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(format!("{} exited with {}: {}", self.path.display(), out.status, String::from_utf8_lossy(&out.stderr).trim()));
        }
        Ok(())
    }
    fn describe(&self) -> String {
        format!("{} -t -i (from {})", self.path.display(), self.from)
    }
}

/// `--mail <spec>`; `from` is the From address for `cmd:`.
pub fn from_spec(spec: &str, from: &str) -> Result<Arc<dyn Mailer>, String> {
    match spec {
        "log" => Ok(Arc::new(LogMailer)),
        "jmap" => jmap_from_env(),
        s => match s.strip_prefix("cmd:") {
            Some(p) if !p.is_empty() => Ok(Arc::new(CmdMailer { path: p.into(), from: from.into() })),
            _ => Err(format!("--mail {spec:?}: want log, cmd:<path> or jmap")),
        },
    }
}

#[cfg(feature = "jmap")]
fn jmap_from_env() -> Result<Arc<dyn Mailer>, String> {
    Ok(Arc::new(jmap::JmapMailer::from_env()?))
}

#[cfg(not(feature = "jmap"))]
fn jmap_from_env() -> Result<Arc<dyn Mailer>, String> {
    Err("--mail jmap: this modgrad-server was built without the jmap feature (cargo build --features jmap)".into())
}

/// `Thu, 01 Jan 1970 00:00:00 +0000`.
fn rfc2822_date(unix: f64) -> String {
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let s = unix.max(0.0) as i64;
    let (days, rem) = (s.div_euclid(86400), s.rem_euclid(86400));
    // civil_from_days (H. Hinnant), as v1::auto_name
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{}, {d:02} {} {y:04} {:02}:{:02}:{:02} +0000", DAYS[days.rem_euclid(7) as usize], MONTHS[(m - 1) as usize], rem / 3600, rem / 60 % 60, rem % 60)
}

/// The message as RFC 5322 text (CRLF line ends, UTF-8 body).
pub fn rfc5322(m: &Mail, from: &str, unix: f64) -> String {
    let body = m.body.replace("\r\n", "\n").replace('\n', "\r\n");
    format!(
        "From: {from}\r\nTo: {}\r\nSubject: {}\r\nDate: {}\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: 8bit\r\nAuto-Submitted: auto-generated\r\n\r\n{body}",
        m.to,
        m.subject,
        rfc2822_date(unix)
    )
}

/// The JMAP request that creates the message as a draft and submits it
/// (RFC 8621 Email/set + EmailSubmission/set, the draft destroyed once sent).
pub fn jmap_send_request(account: &str, identity: &str, drafts: &str, from: &str, m: &Mail) -> serde_json::Value {
    serde_json::json!({
        "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail", "urn:ietf:params:jmap:submission"],
        "methodCalls": [
            ["Email/set", {
                "accountId": account,
                "create": { "m": {
                    "mailboxIds": { drafts: true },
                    "keywords": { "$draft": true },
                    "from": [{ "email": from }],
                    "to": [{ "email": m.to }],
                    "subject": m.subject,
                    "bodyValues": { "b": { "value": m.body } },
                    "textBody": [{ "partId": "b", "type": "text/plain" }],
                }},
            }, "0"],
            ["EmailSubmission/set", {
                "accountId": account,
                "create": { "s": { "identityId": identity, "emailId": "#m" } },
                "onSuccessDestroyEmail": ["#s"],
            }, "1"],
        ],
    })
}

#[cfg(feature = "jmap")]
mod jmap {
    use super::{Mail, Mailer, jmap_send_request};
    use base64::Engine;
    use serde_json::{Value, json};

    pub struct JmapMailer {
        session_url: String,
        user: String,
        pass: String,
    }

    impl JmapMailer {
        pub fn from_env() -> Result<JmapMailer, String> {
            let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty()).ok_or_else(|| format!("--mail jmap needs {k} in the environment"));
            Ok(JmapMailer { session_url: get("MODGRAD_JMAP_URL")?, user: get("MODGRAD_JMAP_USER")?, pass: get("MODGRAD_JMAP_PASS")? })
        }

        fn auth(&self) -> String {
            format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", self.user, self.pass)))
        }

        fn post(&self, api: &str, body: &Value) -> Result<Value, String> {
            let mut r = ureq::post(api).header("Authorization", &self.auth()).header("Content-Type", "application/json").send(body.to_string()).map_err(|e| format!("jmap {api}: {e}"))?;
            serde_json::from_str(&r.body_mut().read_to_string().map_err(|e| e.to_string())?).map_err(|e| format!("jmap reply: {e}"))
        }
    }

    /// The arguments of the reply to call `id`.
    fn reply<'a>(r: &'a Value, id: &str) -> Result<&'a Value, String> {
        r["methodResponses"].as_array().and_then(|a| a.iter().find(|m| m[2] == id)).map(|m| &m[1]).ok_or_else(|| format!("jmap: no reply to call {id}: {r}"))
    }

    impl Mailer for JmapMailer {
        fn send(&self, m: &Mail) -> Result<(), String> {
            let mut s = ureq::get(&self.session_url).header("Authorization", &self.auth()).call().map_err(|e| format!("jmap session: {e}"))?;
            let session: Value = serde_json::from_str(&s.body_mut().read_to_string().map_err(|e| e.to_string())?).map_err(|e| format!("jmap session: {e}"))?;
            let api = session["apiUrl"].as_str().ok_or("jmap session has no apiUrl")?;
            let account = session["primaryAccounts"]["urn:ietf:params:jmap:mail"].as_str().ok_or("jmap session has no mail account")?;
            let q = json!({
                "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail", "urn:ietf:params:jmap:submission"],
                "methodCalls": [
                    ["Identity/get", { "accountId": account }, "i"],
                    ["Mailbox/query", { "accountId": account, "filter": { "role": "drafts" } }, "d"],
                ],
            });
            let r = self.post(api, &q)?;
            let ids = reply(&r, "i")?["list"].as_array().cloned().unwrap_or_default();
            let identity = ids.iter().find(|i| i["email"].as_str().is_some_and(|e| e.eq_ignore_ascii_case(&self.user))).or(ids.first()).ok_or("jmap: no sending identity")?;
            let from = identity["email"].as_str().unwrap_or(&self.user).to_string();
            let drafts = reply(&r, "d")?["ids"][0].as_str().ok_or("jmap: no drafts mailbox")?.to_string();
            let r = self.post(api, &jmap_send_request(account, identity["id"].as_str().unwrap_or_default(), &drafts, &from, m))?;
            for (id, what) in [("0", "Email/set"), ("1", "EmailSubmission/set")] {
                let a = reply(&r, id)?;
                if a.get("notCreated").is_some_and(|n| n.as_object().is_some_and(|o| !o.is_empty())) {
                    return Err(format!("jmap {what} refused: {}", a["notCreated"]));
                }
            }
            Ok(())
        }
        fn describe(&self) -> String {
            format!("jmap {} as {}", self.session_url, self.user)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_shape() {
        let m = sign_in_mail("me@x.org", "https://modgrad.com/login#email=abc", 15);
        assert!(m.body.contains("https://modgrad.com/login#email=abc") && m.body.contains("expires in 15 minutes") && m.body.contains("didn't ask"));
        let t = rfc5322(&m, "modgrad <noreply@modgrad.com>", 0.0);
        assert!(t.starts_with("From: modgrad <noreply@modgrad.com>\r\nTo: me@x.org\r\nSubject: Sign in to modgrad\r\nDate: Thu, 01 Jan 1970 00:00:00 +0000\r\n"), "{t}");
        assert!(t.contains("\r\n\r\nOpen this link") && !t.replace("\r\n", "").contains('\n'));
        assert_eq!(rfc2822_date(1_760_140_800.0 + 3661.0), "Sat, 11 Oct 2025 01:01:01 +0000");
        let j = jmap_send_request("A", "I", "D", "f@x.org", &m);
        assert_eq!(j["methodCalls"][1][1]["create"]["s"]["emailId"], "#m");
        assert_eq!(j["methodCalls"][0][1]["create"]["m"]["mailboxIds"]["D"], true);
        assert!(from_spec("smtp", "x").is_err() && from_spec("cmd:", "x").is_err() && from_spec("log", "x").is_ok());
    }
}
