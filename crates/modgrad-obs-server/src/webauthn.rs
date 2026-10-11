//! WebAuthn (passkey) verification: the relying-party checks of WebAuthn
//! Level 2/3 §7.1 (registration) and §7.2 (authentication), and nothing
//! else. Pure functions over the bytes the browser hands back; challenges,
//! accounts and routes live elsewhere.
//!
//! What is verified:
//!
//! * `clientDataJSON`: `type`, `challenge` (base64url without padding,
//!   compared as bytes), `origin` (exact string match), `crossOrigin` must
//!   not be `true`. `tokenBinding` and unknown members are ignored. The
//!   hash covers the raw bytes as received.
//! * `authenticatorData`: rpIdHash == SHA-256(rp id), UP set, UV recorded,
//!   BS only with BE, attested credential data (registration), extensions
//!   (ED) skipped by measuring the CBOR item, no trailing bytes.
//! * COSE public keys: EdDSA (-8, OKP Ed25519), ES256 (-7, EC2 P-256) and
//!   RS256 (-257, RSA); the key's own `alg` must match. Anything else is
//!   [`WebauthnError::Unsupported`].
//! * Assertion signatures over `authData || SHA-256(clientDataJSON)` with
//!   `ring`, and the signature counter (see [`verify_assertion`]).
//!
//! What is deliberately NOT verified: **attestation**. The `fmt` must be a
//! text string and `attStmt` a map; for `"none"` the statement must be empty
//! (as the spec requires), for every other format (`packed`, `tpm`,
//! `android-key`, …) the statement is ignored and the credential is treated
//! exactly like `none`. A [`RegisteredKey`] therefore says nothing about the
//! make or model of the authenticator, only that the holder of this key
//! answered our challenge on our origin. The registration options ask for
//! `attestation: "none"` anyway.
//!
//! Parsing is defensive: inputs are size-capped, CBOR is measured with a
//! bounded, depth-limited walker before `ciborium` decodes it, and no path
//! indexes or slices without a bounds check.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ciborium::value::Value;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// COSE algorithm identifiers we accept.
pub const ALG_EDDSA: i64 = -8;
pub const ALG_ES256: i64 = -7;
pub const ALG_RS256: i64 = -257;

const MAX_CLIENT_DATA: usize = 16 * 1024;
const MAX_ATTESTATION_OBJECT: usize = 64 * 1024;
const MAX_AUTH_DATA: usize = 16 * 1024;
const MAX_COSE_KEY: usize = 2048;
const MAX_SIGNATURE: usize = 1024; // RSA-8192
const MAX_CRED_ID: usize = 1023; // L3 §7.1 step 24
const MAX_CBOR_DEPTH: u32 = 16;

const FLAG_UP: u8 = 0x01;
const FLAG_UV: u8 = 0x04;
const FLAG_BE: u8 = 0x08;
const FLAG_BS: u8 = 0x10;
const FLAG_AT: u8 = 0x40;
const FLAG_ED: u8 = 0x80;

/// A credential accepted by [`verify_registration`]; store it on the account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredKey {
    pub cred_id: Vec<u8>,
    /// The COSE_Key exactly as the authenticator encoded it.
    pub public_key_cose: Vec<u8>,
    pub alg: i64,
    pub sign_count: u32,
    pub user_verified: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum WebauthnError {
    #[error("unsupported")]
    Unsupported,
    #[error("{0}")]
    Invalid(&'static str),
}

use WebauthnError::Invalid;

/// §7.1: verify a `navigator.credentials.create()` response. `expected_challenge`
/// is the raw challenge bytes we issued; `origin` is e.g. `https://modgrad.com`.
pub fn verify_registration(client_data_json: &[u8], attestation_object: &[u8], expected_challenge: &[u8], rp_id: &str, origin: &str) -> Result<RegisteredKey, WebauthnError> {
    check_client_data(client_data_json, "webauthn.create", expected_challenge, origin)?;
    let auth_data = attestation_auth_data(attestation_object)?;
    let ad = parse_auth_data(&auth_data)?;
    check_rp_and_presence(&ad, rp_id)?;
    let att = ad.attested.ok_or(Invalid("no attested credential data"))?;
    let alg = cose_alg(att.cose_key)?;
    // Parse now so an unusable key is refused at registration, not at login.
    CoseKey::parse(att.cose_key, alg)?;
    Ok(RegisteredKey {
        cred_id: att.cred_id.to_vec(),
        public_key_cose: att.cose_key.to_vec(),
        alg,
        sign_count: ad.sign_count,
        user_verified: ad.flags & FLAG_UV != 0,
    })
}

/// §7.2: verify a `navigator.credentials.get()` response against a stored
/// key. Returns the authenticator's new signature count, to be stored.
///
/// Counter rule: if both the stored and the new count are nonzero, the new
/// one must be strictly greater, else the authenticator may be cloned and
/// the assertion is refused. A zero on either side passes: synced passkeys
/// (iCloud Keychain, Google Password Manager) always report 0, and an
/// authenticator that stops counting is not treated as a clone. (§7.2 step
/// 21 would also refuse stored > 0, new == 0; we deliberately do not.)
#[allow(clippy::too_many_arguments)]
pub fn verify_assertion(public_key_cose: &[u8], alg: i64, stored_count: u32, client_data_json: &[u8], authenticator_data: &[u8], signature: &[u8], expected_challenge: &[u8], rp_id: &str, origin: &str) -> Result<u32, WebauthnError> {
    check_client_data(client_data_json, "webauthn.get", expected_challenge, origin)?;
    if authenticator_data.len() > MAX_AUTH_DATA {
        return Err(Invalid("authenticator data too large"));
    }
    if signature.is_empty() || signature.len() > MAX_SIGNATURE {
        return Err(Invalid("bad signature length"));
    }
    let ad = parse_auth_data(authenticator_data)?;
    check_rp_and_presence(&ad, rp_id)?;
    let key = CoseKey::parse(public_key_cose, alg)?;

    let mut signed = Vec::with_capacity(authenticator_data.len() + 32);
    signed.extend_from_slice(authenticator_data);
    signed.extend_from_slice(&Sha256::digest(client_data_json));
    key.verify(&signed, signature)?;

    let new = ad.sign_count;
    if new != 0 && stored_count != 0 && new <= stored_count {
        return Err(Invalid("sign count did not increase"));
    }
    Ok(new)
}

// ---- clientDataJSON -------------------------------------------------------

#[derive(Deserialize)]
struct ClientData {
    #[serde(rename = "type")]
    ty: String,
    challenge: String,
    origin: String,
    #[serde(rename = "crossOrigin", default)]
    cross_origin: Option<bool>,
}

fn check_client_data(raw: &[u8], ty: &str, expected_challenge: &[u8], origin: &str) -> Result<(), WebauthnError> {
    if raw.len() > MAX_CLIENT_DATA {
        return Err(Invalid("clientDataJSON too large"));
    }
    let cd: ClientData = serde_json::from_slice(raw).map_err(|_| Invalid("malformed clientDataJSON"))?;
    if cd.ty != ty {
        return Err(Invalid("wrong clientData type"));
    }
    let challenge = URL_SAFE_NO_PAD.decode(cd.challenge.as_bytes()).map_err(|_| Invalid("malformed challenge"))?;
    if expected_challenge.is_empty() || challenge != expected_challenge {
        return Err(Invalid("challenge mismatch"));
    }
    if cd.origin != origin {
        return Err(Invalid("origin mismatch"));
    }
    if cd.cross_origin == Some(true) {
        return Err(Invalid("cross-origin request"));
    }
    Ok(())
}

// ---- attestationObject ----------------------------------------------------

/// The `authData` bytes of an attestation object, after checking its shape.
/// `attStmt` is not verified (see the module docs).
fn attestation_auth_data(raw: &[u8]) -> Result<Vec<u8>, WebauthnError> {
    if raw.len() > MAX_ATTESTATION_OBJECT {
        return Err(Invalid("attestation object too large"));
    }
    if cbor_item_len(raw, 0)? != raw.len() {
        return Err(Invalid("trailing bytes after attestation object"));
    }
    let Value::Map(entries) = decode(raw)? else {
        return Err(Invalid("attestation object is not a map"));
    };
    let (mut fmt, mut stmt, mut auth_data) = (None, None, None);
    for (k, v) in entries {
        let slot = match k.as_text() {
            Some("fmt") => &mut fmt,
            Some("attStmt") => &mut stmt,
            Some("authData") => &mut auth_data,
            _ => continue,
        };
        if slot.replace(v).is_some() {
            return Err(Invalid("duplicate key in attestation object"));
        }
    }
    let Some(Value::Text(fmt)) = fmt else {
        return Err(Invalid("malformed attestation format"));
    };
    let Some(Value::Map(stmt)) = stmt else {
        return Err(Invalid("malformed attestation statement"));
    };
    if fmt == "none" && !stmt.is_empty() {
        return Err(Invalid("attestation \"none\" with a statement"));
    }
    match auth_data {
        Some(Value::Bytes(b)) => Ok(b),
        _ => Err(Invalid("missing authData")),
    }
}

fn decode(raw: &[u8]) -> Result<Value, WebauthnError> {
    ciborium::de::from_reader_with_recursion_limit(raw, MAX_CBOR_DEPTH as usize).map_err(|_| Invalid("malformed CBOR"))
}

// ---- authenticatorData ----------------------------------------------------

struct AuthData<'a> {
    rp_id_hash: &'a [u8],
    flags: u8,
    sign_count: u32,
    attested: Option<Attested<'a>>,
}

struct Attested<'a> {
    cred_id: &'a [u8],
    cose_key: &'a [u8],
}

fn take<'a>(buf: &mut &'a [u8], n: usize) -> Result<&'a [u8], WebauthnError> {
    if buf.len() < n {
        return Err(Invalid("authenticator data truncated"));
    }
    let (head, rest) = buf.split_at(n);
    *buf = rest;
    Ok(head)
}

/// rpIdHash(32) flags(1) signCount(4, BE) [attestedCredentialData] [extensions].
fn parse_auth_data(raw: &[u8]) -> Result<AuthData<'_>, WebauthnError> {
    let mut rest = raw;
    let rp_id_hash = take(&mut rest, 32)?;
    let flags = take(&mut rest, 1)?[0];
    let count = take(&mut rest, 4)?;
    let sign_count = u32::from_be_bytes([count[0], count[1], count[2], count[3]]);
    if flags & FLAG_BS != 0 && flags & FLAG_BE == 0 {
        return Err(Invalid("backup state without backup eligibility"));
    }
    let attested = if flags & FLAG_AT != 0 {
        let _aaguid = take(&mut rest, 16)?;
        let len = take(&mut rest, 2)?;
        let len = usize::from(u16::from_be_bytes([len[0], len[1]]));
        if len == 0 || len > MAX_CRED_ID {
            return Err(Invalid("bad credential id length"));
        }
        let cred_id = take(&mut rest, len)?;
        let key_len = cbor_item_len(rest, 0)?;
        if key_len > MAX_COSE_KEY {
            return Err(Invalid("credential public key too large"));
        }
        let cose_key = take(&mut rest, key_len)?;
        Some(Attested { cred_id, cose_key })
    } else {
        None
    };
    if flags & FLAG_ED != 0 {
        if rest.first().map(|b| b >> 5) != Some(5) {
            return Err(Invalid("extensions are not a map"));
        }
        let n = cbor_item_len(rest, 0)?;
        take(&mut rest, n)?;
    }
    if !rest.is_empty() {
        return Err(Invalid("trailing bytes in authenticator data"));
    }
    Ok(AuthData { rp_id_hash, flags, sign_count, attested })
}

fn check_rp_and_presence(ad: &AuthData<'_>, rp_id: &str) -> Result<(), WebauthnError> {
    if ad.rp_id_hash != Sha256::digest(rp_id.as_bytes()).as_slice() {
        return Err(Invalid("rp id hash mismatch"));
    }
    if ad.flags & FLAG_UP == 0 {
        return Err(Invalid("user not present"));
    }
    Ok(())
}

// ---- bounded CBOR walker --------------------------------------------------

/// (major type, argument, header length) of the item at the start of `buf`.
/// Indefinite lengths are refused: CTAP2 canonical CBOR never uses them.
fn cbor_head(buf: &[u8]) -> Result<(u8, u64, usize), WebauthnError> {
    let &b = buf.first().ok_or(Invalid("truncated CBOR"))?;
    let (major, info) = (b >> 5, b & 0x1f);
    let n = match info {
        0..=23 => return Ok((major, u64::from(info), 1)),
        24 => 1,
        25 => 2,
        26 => 4,
        27 => 8,
        _ => return Err(Invalid("unsupported CBOR length encoding")),
    };
    let bytes = buf.get(1..1 + n).ok_or(Invalid("truncated CBOR"))?;
    let arg = bytes.iter().fold(0u64, |acc, &x| (acc << 8) | u64::from(x));
    Ok((major, arg, 1 + n))
}

/// Length in bytes of the single CBOR data item at the start of `buf`.
/// Every nested item consumes at least one byte, so the walk is bounded by
/// `buf.len()`; nesting is bounded by `MAX_CBOR_DEPTH`.
fn cbor_item_len(buf: &[u8], depth: u32) -> Result<usize, WebauthnError> {
    if depth > MAX_CBOR_DEPTH {
        return Err(Invalid("CBOR nested too deeply"));
    }
    let (major, arg, mut pos) = cbor_head(buf)?;
    match major {
        0 | 1 | 7 => {}
        2 | 3 => {
            let n = usize::try_from(arg).map_err(|_| Invalid("truncated CBOR"))?;
            pos = pos.checked_add(n).ok_or(Invalid("truncated CBOR"))?;
        }
        _ => {
            let items = match major {
                4 => arg,
                5 => arg.checked_mul(2).ok_or(Invalid("truncated CBOR"))?,
                _ => 1, // 6: tag + one item
            };
            for _ in 0..items {
                let rest = buf.get(pos..).filter(|r| !r.is_empty()).ok_or(Invalid("truncated CBOR"))?;
                pos += cbor_item_len(rest, depth + 1)?;
            }
        }
    }
    if pos > buf.len() {
        return Err(Invalid("truncated CBOR"));
    }
    Ok(pos)
}

// ---- COSE keys ------------------------------------------------------------

enum CoseKey {
    Ed25519(Vec<u8>),
    /// Uncompressed SEC1 point `04 || x || y`.
    P256(Vec<u8>),
    Rsa { n: Vec<u8>, e: Vec<u8> },
}

fn cose_map(raw: &[u8]) -> Result<Vec<(i128, Value)>, WebauthnError> {
    if raw.len() > MAX_COSE_KEY || cbor_item_len(raw, 0)? != raw.len() {
        return Err(Invalid("malformed COSE key"));
    }
    let Value::Map(entries) = decode(raw)? else {
        return Err(Invalid("COSE key is not a map"));
    };
    let mut out: Vec<(i128, Value)> = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        // Text labels are legal COSE but none we need; skip them.
        let Value::Integer(k) = k else { continue };
        let k = i128::from(k);
        if out.iter().any(|(seen, _)| *seen == k) {
            return Err(Invalid("duplicate label in COSE key"));
        }
        out.push((k, v));
    }
    Ok(out)
}

fn label(map: &[(i128, Value)], l: i128) -> Option<&Value> {
    map.iter().find(|(k, _)| *k == l).map(|(_, v)| v)
}

fn int(map: &[(i128, Value)], l: i128) -> Option<i128> {
    match label(map, l)? {
        Value::Integer(i) => Some(i128::from(*i)),
        _ => None,
    }
}

fn bytes(map: &[(i128, Value)], l: i128) -> Result<&[u8], WebauthnError> {
    match label(map, l) {
        Some(Value::Bytes(b)) => Ok(b),
        _ => Err(Invalid("malformed COSE key")),
    }
}

/// The `alg` (label 3) a COSE key declares.
fn cose_alg(raw: &[u8]) -> Result<i64, WebauthnError> {
    let map = cose_map(raw)?;
    let alg = int(&map, 3).ok_or(Invalid("COSE key without alg"))?;
    i64::try_from(alg).map_err(|_| WebauthnError::Unsupported)
}

impl CoseKey {
    /// Parse a COSE_Key, requiring its `alg` to equal `alg`.
    fn parse(raw: &[u8], alg: i64) -> Result<CoseKey, WebauthnError> {
        let map = cose_map(raw)?;
        if int(&map, 3) != Some(i128::from(alg)) {
            return Err(Invalid("COSE key alg mismatch"));
        }
        let kty = int(&map, 1).ok_or(Invalid("COSE key without kty"))?;
        match (alg, kty) {
            (ALG_EDDSA, 1) => {
                if int(&map, -1) != Some(6) {
                    return Err(WebauthnError::Unsupported); // Ed448 or unknown
                }
                let x = bytes(&map, -2)?;
                if x.len() != 32 {
                    return Err(Invalid("bad Ed25519 key"));
                }
                Ok(CoseKey::Ed25519(x.to_vec()))
            }
            (ALG_ES256, 2) => {
                if int(&map, -1) != Some(1) {
                    return Err(Invalid("ES256 key not on P-256"));
                }
                let (x, y) = (bytes(&map, -2)?, bytes(&map, -3)?); // y as bool = compressed: refused
                if x.len() != 32 || y.len() != 32 {
                    return Err(Invalid("bad P-256 key"));
                }
                let mut point = Vec::with_capacity(65);
                point.push(0x04);
                point.extend_from_slice(x);
                point.extend_from_slice(y);
                Ok(CoseKey::P256(point))
            }
            (ALG_RS256, 3) => {
                let (n, e) = (bytes(&map, -1)?, bytes(&map, -2)?);
                if n.is_empty() || e.is_empty() || n.len() > 1024 || e.len() > 8 {
                    return Err(Invalid("bad RSA key"));
                }
                Ok(CoseKey::Rsa { n: n.to_vec(), e: e.to_vec() })
            }
            (ALG_EDDSA | ALG_ES256 | ALG_RS256, _) => Err(Invalid("COSE key type does not match alg")),
            _ => Err(WebauthnError::Unsupported),
        }
    }

    fn verify(&self, msg: &[u8], sig: &[u8]) -> Result<(), WebauthnError> {
        use ring::signature::{ECDSA_P256_SHA256_ASN1, ED25519, RSA_PKCS1_2048_8192_SHA256, RsaPublicKeyComponents, UnparsedPublicKey};
        let ok = match self {
            CoseKey::Ed25519(pk) => UnparsedPublicKey::new(&ED25519, pk).verify(msg, sig),
            CoseKey::P256(pk) => UnparsedPublicKey::new(&ECDSA_P256_SHA256_ASN1, pk).verify(msg, sig),
            CoseKey::Rsa { n, e } => RsaPublicKeyComponents { n, e }.verify(&RSA_PKCS1_2048_8192_SHA256, msg, sig),
        };
        ok.map_err(|_| Invalid("bad signature"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbor_walker_measures_items() {
        // {1: 2, "a": [h'0102', -1]}
        let item = [0xa2, 0x01, 0x02, 0x61, b'a', 0x82, 0x42, 0x01, 0x02, 0x20];
        assert_eq!(cbor_item_len(&item, 0).unwrap(), item.len());
        let mut more = item.to_vec();
        more.extend_from_slice(&[0xff, 0xff]);
        assert_eq!(cbor_item_len(&more, 0).unwrap(), item.len());
        for cut in 0..item.len() {
            assert!(cbor_item_len(&item[..cut], 0).is_err(), "cut {cut}");
        }
        // huge declared lengths do not overflow or allocate
        assert!(cbor_item_len(&[0x5b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff], 0).is_err());
        assert!(cbor_item_len(&[0x9b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00], 0).is_err());
        assert!(cbor_item_len(&[0xbb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00], 0).is_err());
        // indefinite length refused; deep nesting refused
        assert!(cbor_item_len(&[0x9f, 0xff], 0).is_err());
        assert!(cbor_item_len(&[0x81; 64], 0).is_err());
    }
}
