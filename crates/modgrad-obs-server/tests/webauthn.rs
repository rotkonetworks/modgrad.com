//! Passkey verification (`modgrad_obs_server::webauthn`): a software
//! authenticator for each algorithm (register, then assert), one negative
//! test per check, a no-panic fuzz, and real browser/authenticator vectors
//! from py_webauthn so we are not only testing against ourselves.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ciborium::value::Value;
use modgrad_obs_server::webauthn::{RegisteredKey, WebauthnError, verify_assertion, verify_registration};
use proptest::prelude::*;
use ring::rand::SystemRandom;
use ring::signature::{self, EcdsaKeyPair, Ed25519KeyPair, KeyPair, RsaKeyPair};
use sha2::{Digest, Sha256};

const RP: &str = "modgrad.test";
const ORIGIN: &str = "https://modgrad.test";
const CHALLENGE: &[u8] = b"0123456789abcdef0123456789abcdef";

fn b64(s: &str) -> Vec<u8> {
    URL_SAFE_NO_PAD.decode(s).unwrap()
}

fn cbor(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(v, &mut out).unwrap();
    out
}

fn int_map(entries: Vec<(i64, Value)>) -> Value {
    Value::Map(entries.into_iter().map(|(k, v)| (Value::Integer(k.into()), v)).collect())
}

fn client_data(ty: &str, challenge: &[u8], origin: &str) -> Vec<u8> {
    format!(r#"{{"type":"{ty}","challenge":"{}","origin":"{origin}","crossOrigin":false}}"#, URL_SAFE_NO_PAD.encode(challenge)).into_bytes()
}

fn is_invalid(r: Result<impl std::fmt::Debug, WebauthnError>, why: &str) {
    match r {
        Err(WebauthnError::Invalid(m)) => assert_eq!(m, why),
        other => panic!("expected Invalid({why:?}), got {other:?}"),
    }
}

enum Signer {
    Ed(Ed25519KeyPair),
    Ec(EcdsaKeyPair),
    Rsa(RsaKeyPair),
}

/// A software authenticator holding one credential.
struct Soft {
    signer: Signer,
    cred_id: Vec<u8>,
    count: u32,
    rng: SystemRandom,
}

impl Soft {
    fn new(alg: i64) -> Soft {
        Soft::with_rsa(alg, include_bytes!("fixtures/webauthn_rsa2048.pk8.der"))
    }

    /// RSA keys are fixed test keys (ring cannot generate them).
    fn with_rsa(alg: i64, rsa_pkcs8: &[u8]) -> Soft {
        let rng = SystemRandom::new();
        let signer = match alg {
            -8 => Signer::Ed(Ed25519KeyPair::from_pkcs8(Ed25519KeyPair::generate_pkcs8(&rng).unwrap().as_ref()).unwrap()),
            -7 => {
                let a = &signature::ECDSA_P256_SHA256_ASN1_SIGNING;
                Signer::Ec(EcdsaKeyPair::from_pkcs8(a, EcdsaKeyPair::generate_pkcs8(a, &rng).unwrap().as_ref(), &rng).unwrap())
            }
            -257 => Signer::Rsa(RsaKeyPair::from_pkcs8(rsa_pkcs8).unwrap()),
            _ => unreachable!(),
        };
        Soft { signer, cred_id: vec![0xc7; 20], count: 0, rng }
    }

    fn cose(&self) -> Vec<u8> {
        let b = |x: &[u8]| Value::Bytes(x.to_vec());
        let key = match &self.signer {
            Signer::Ed(kp) => int_map(vec![(1, 1.into()), (3, (-8).into()), (-1, 6.into()), (-2, b(kp.public_key().as_ref()))]),
            Signer::Ec(kp) => {
                let p = kp.public_key().as_ref();
                int_map(vec![(1, 2.into()), (3, (-7).into()), (-1, 1.into()), (-2, b(&p[1..33])), (-3, b(&p[33..65]))])
            }
            Signer::Rsa(kp) => {
                let c: signature::RsaPublicKeyComponents<Vec<u8>> = kp.public().into();
                int_map(vec![(1, 3.into()), (3, (-257).into()), (-1, b(&c.n)), (-2, b(&c.e))])
            }
        };
        cbor(&key)
    }

    /// authenticatorData; `attested` adds the AT block, `ext` an ED map.
    fn auth_data(&self, rp: &str, flags: u8, attested: bool, ext: bool) -> Vec<u8> {
        let mut d = Sha256::digest(rp.as_bytes()).to_vec();
        d.push(flags | if attested { 0x40 } else { 0 } | if ext { 0x80 } else { 0 });
        d.extend_from_slice(&self.count.to_be_bytes());
        if attested {
            d.extend_from_slice(&[0u8; 16]);
            d.extend_from_slice(&(self.cred_id.len() as u16).to_be_bytes());
            d.extend_from_slice(&self.cred_id);
            d.extend_from_slice(&self.cose());
        }
        if ext {
            d.extend_from_slice(&cbor(&Value::Map(vec![("credProtect".into(), 2.into()), ("hmac-secret".into(), true.into())])));
        }
        d
    }

    fn attestation(&self, fmt: &str, stmt: Value, auth_data: Vec<u8>) -> Vec<u8> {
        cbor(&Value::Map(vec![("fmt".into(), fmt.into()), ("attStmt".into(), stmt), ("authData".into(), Value::Bytes(auth_data))]))
    }

    fn register(&self) -> (Vec<u8>, Vec<u8>) {
        let ad = self.auth_data(RP, 0x05, true, false);
        (client_data("webauthn.create", CHALLENGE, ORIGIN), self.attestation("none", Value::Map(vec![]), ad))
    }

    fn sign(&self, auth_data: &[u8], cdj: &[u8]) -> Vec<u8> {
        let mut msg = auth_data.to_vec();
        msg.extend_from_slice(&Sha256::digest(cdj));
        match &self.signer {
            Signer::Ed(kp) => kp.sign(&msg).as_ref().to_vec(),
            Signer::Ec(kp) => kp.sign(&self.rng, &msg).unwrap().as_ref().to_vec(),
            Signer::Rsa(kp) => {
                let mut sig = vec![0; kp.public().modulus_len()];
                kp.sign(&signature::RSA_PKCS1_SHA256, &self.rng, &msg, &mut sig).unwrap();
                sig
            }
        }
    }

    /// (clientDataJSON, authenticatorData, signature) for a login.
    fn assert(&mut self, flags: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let cdj = client_data("webauthn.get", CHALLENGE, ORIGIN);
        let ad = self.auth_data(RP, flags, false, false);
        let sig = self.sign(&ad, &cdj);
        (cdj, ad, sig)
    }
}

fn login(k: &RegisteredKey, stored: u32, (cdj, ad, sig): &(Vec<u8>, Vec<u8>, Vec<u8>)) -> Result<u32, WebauthnError> {
    verify_assertion(&k.public_key_cose, k.alg, stored, cdj, ad, sig, CHALLENGE, RP, ORIGIN)
}

fn registered(soft: &Soft) -> RegisteredKey {
    let (cdj, att) = soft.register();
    verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN).unwrap()
}

// ---- round trips ----------------------------------------------------------

#[test]
fn register_then_assert_each_alg() {
    for alg in [-8, -7, -257] {
        let mut soft = Soft::new(alg);
        soft.count = 5;
        let k = registered(&soft);
        assert_eq!((k.alg, k.sign_count, k.user_verified), (alg, 5, true));
        assert_eq!(k.cred_id, soft.cred_id);
        assert_eq!(k.public_key_cose, soft.cose());
        soft.count = 6;
        assert_eq!(login(&k, 5, &soft.assert(0x01)).unwrap(), 6, "alg {alg}");
        soft.count = 40;
        assert_eq!(login(&k, 6, &soft.assert(0x05)).unwrap(), 40, "alg {alg}");
    }
}

#[test]
fn zero_counters_are_allowed() {
    let mut soft = Soft::new(-7);
    let k = registered(&soft);
    assert_eq!(k.sign_count, 0);
    assert_eq!(login(&k, 0, &soft.assert(0x01)).unwrap(), 0);
    assert_eq!(login(&k, 0, &soft.assert(0x01)).unwrap(), 0);
    soft.count = 1;
    assert_eq!(login(&k, 0, &soft.assert(0x01)).unwrap(), 1);
}

#[test]
fn extensions_are_skipped() {
    let mut soft = Soft::new(-8);
    let ad = soft.auth_data(RP, 0x01, true, true);
    let att = soft.attestation("none", Value::Map(vec![]), ad);
    let k = verify_registration(&client_data("webauthn.create", CHALLENGE, ORIGIN), &att, CHALLENGE, RP, ORIGIN).unwrap();
    assert_eq!(k.public_key_cose, soft.cose());
    assert!(!k.user_verified);
    soft.count = 1;
    let cdj = client_data("webauthn.get", CHALLENGE, ORIGIN);
    let ad = soft.auth_data(RP, 0x01, false, true);
    let sig = soft.sign(&ad, &cdj);
    assert_eq!(login(&k, 0, &(cdj, ad, sig)).unwrap(), 1);
}

#[test]
fn other_attestation_formats_are_treated_as_none() {
    let soft = Soft::new(-7);
    let ad = soft.auth_data(RP, 0x05, true, false);
    // a self-attestation-shaped statement with a bogus signature: ignored
    let stmt = Value::Map(vec![("alg".into(), (-7).into()), ("sig".into(), Value::Bytes(vec![0; 70]))]);
    for fmt in ["packed", "tpm", "made-up"] {
        let att = soft.attestation(fmt, stmt.clone(), ad.clone());
        assert!(verify_registration(&client_data("webauthn.create", CHALLENGE, ORIGIN), &att, CHALLENGE, RP, ORIGIN).is_ok(), "{fmt}");
    }
    // multi-KB certificate chains (tpm / android-key) decode fine and are ignored
    let big = Value::Map(vec![("x5c".into(), Value::Array(vec![Value::Bytes(vec![0x30; 10 * 1024]), Value::Bytes(vec![0x30; 3000])])), ("note".into(), "x".repeat(5 * 1024).into())]);
    let att = soft.attestation("tpm", big, ad.clone());
    assert!(verify_registration(&client_data("webauthn.create", CHALLENGE, ORIGIN), &att, CHALLENGE, RP, ORIGIN).is_ok());
    // "none" must carry an empty statement; fmt must be text
    let cdj = client_data("webauthn.create", CHALLENGE, ORIGIN);
    is_invalid(verify_registration(&cdj, &soft.attestation("none", stmt, ad.clone()), CHALLENGE, RP, ORIGIN), "attestation \"none\" with a statement");
    let att = cbor(&Value::Map(vec![("fmt".into(), 0.into()), ("attStmt".into(), Value::Map(vec![])), ("authData".into(), Value::Bytes(ad))]));
    is_invalid(verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN), "malformed attestation format");
}

// ---- negative: client data ------------------------------------------------

#[test]
fn registration_rejects_bad_client_data() {
    let soft = Soft::new(-7);
    let (_, att) = soft.register();
    let reg = |cdj: Vec<u8>| verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN);
    is_invalid(reg(client_data("webauthn.create", CHALLENGE, "https://evil.test")), "origin mismatch");
    is_invalid(reg(client_data("webauthn.create", CHALLENGE, "https://modgrad.test:443")), "origin mismatch");
    is_invalid(reg(client_data("webauthn.create", b"another challenge", ORIGIN)), "challenge mismatch");
    is_invalid(reg(client_data("webauthn.get", CHALLENGE, ORIGIN)), "wrong clientData type");
    let padded = format!(r#"{{"type":"webauthn.create","challenge":"{}=","origin":"{ORIGIN}"}}"#, URL_SAFE_NO_PAD.encode(&CHALLENGE[..31]));
    is_invalid(reg(padded.into_bytes()), "malformed challenge");
    let cross = format!(r#"{{"type":"webauthn.create","challenge":"{}","origin":"{ORIGIN}","crossOrigin":true}}"#, URL_SAFE_NO_PAD.encode(CHALLENGE));
    is_invalid(reg(cross.into_bytes()), "cross-origin request");
    is_invalid(reg(b"not json".to_vec()), "malformed clientDataJSON");
    is_invalid(reg(b"{}".to_vec()), "malformed clientDataJSON");
    // crossOrigin absent (older browsers) and unknown members are fine
    let legacy = format!(r#"{{"challenge":"{}","clientExtensions":{{}},"hashAlgorithm":"SHA-256","origin":"{ORIGIN}","type":"webauthn.create","tokenBinding":{{"status":"present","id":"x"}}}}"#, URL_SAFE_NO_PAD.encode(CHALLENGE));
    assert!(reg(legacy.into_bytes()).is_ok());
}

#[test]
fn assertion_rejects_bad_client_data() {
    let mut soft = Soft::new(-8);
    let k = registered(&soft);
    soft.count = 1;
    let ad = soft.auth_data(RP, 0x01, false, false);
    let signed = |cdj: Vec<u8>| {
        let sig = soft.sign(&ad, &cdj);
        login(&k, 0, &(cdj, ad.clone(), sig))
    };
    // each one correctly signed: the client-data checks alone must refuse it
    is_invalid(signed(client_data("webauthn.get", CHALLENGE, "https://evil.test")), "origin mismatch");
    is_invalid(signed(client_data("webauthn.get", b"stale challenge", ORIGIN)), "challenge mismatch");
    is_invalid(signed(client_data("webauthn.create", CHALLENGE, ORIGIN)), "wrong clientData type");
}

// ---- negative: authenticator data -----------------------------------------

#[test]
fn registration_rejects_bad_auth_data() {
    let soft = Soft::new(-7);
    let cdj = client_data("webauthn.create", CHALLENGE, ORIGIN);
    let reg = |ad: Vec<u8>| verify_registration(&cdj, &soft.attestation("none", Value::Map(vec![]), ad), CHALLENGE, RP, ORIGIN);
    is_invalid(reg(soft.auth_data("evil.test", 0x05, true, false)), "rp id hash mismatch");
    is_invalid(reg(soft.auth_data(RP, 0x04, true, false)), "user not present");
    is_invalid(reg(soft.auth_data(RP, 0x05, false, false)), "no attested credential data");
    is_invalid(reg(soft.auth_data(RP, 0x15, true, false)), "backup state without backup eligibility");
    let mut trailing = soft.auth_data(RP, 0x05, true, false);
    trailing.push(0);
    is_invalid(reg(trailing), "trailing bytes in authenticator data");
    // a wrong rp id passed by the server, not just a wrong hash from the client
    let (cdj2, att) = soft.register();
    is_invalid(verify_registration(&cdj2, &att, CHALLENGE, "evil.test", ORIGIN), "rp id hash mismatch");
}

#[test]
fn assertion_rejects_bad_auth_data_and_signatures() {
    for alg in [-8, -7, -257] {
        let mut soft = Soft::new(alg);
        let k = registered(&soft);
        soft.count = 10;

        let (cdj, ad, sig) = soft.assert(0x01);
        is_invalid(verify_assertion(&k.public_key_cose, k.alg, 0, &cdj, &ad, &sig, CHALLENGE, "evil.test", ORIGIN), "rp id hash mismatch");

        let no_up = soft.assert(0x04);
        is_invalid(login(&k, 0, &no_up), "user not present");

        let (cdj, ad, mut sig) = soft.assert(0x01);
        let last = sig.len() - 1;
        sig[last] ^= 1;
        is_invalid(login(&k, 0, &(cdj, ad, sig)), "bad signature");

        // flip UV after signing: the signature covers authData
        let (cdj, mut ad, sig) = soft.assert(0x01);
        ad[32] |= 0x04;
        is_invalid(login(&k, 0, &(cdj, ad, sig)), "bad signature");

        // bump the counter after signing
        let (cdj, mut ad, sig) = soft.assert(0x01);
        ad[36] ^= 0x80;
        is_invalid(login(&k, 0, &(cdj, ad, sig)), "bad signature");

        // client data changed after signing (same checks pass, hash differs)
        let (_, ad, sig) = soft.assert(0x01);
        let other = format!(r#"{{"type":"webauthn.get","challenge":"{}","origin":"{ORIGIN}","crossOrigin":false,"x":1}}"#, URL_SAFE_NO_PAD.encode(CHALLENGE));
        is_invalid(login(&k, 0, &(other.into_bytes(), ad, sig)), "bad signature");

        // a different key's signature
        let mut other = Soft::with_rsa(alg, include_bytes!("fixtures/webauthn_rsa2048_b.pk8.der"));
        other.count = 10;
        let forged = other.assert(0x01);
        is_invalid(login(&k, 0, &forged), "bad signature");
    }
}

#[test]
fn sign_count_regression_is_refused() {
    let mut soft = Soft::new(-7);
    let k = registered(&soft);
    soft.count = 7;
    let a = soft.assert(0x01);
    is_invalid(login(&k, 7, &a), "sign count did not increase");
    is_invalid(login(&k, 9, &a), "sign count did not increase");
    assert_eq!(login(&k, 6, &a).unwrap(), 7);
    // a zero on either side passes (more lenient than §7.2 step 21, by design)
    soft.count = 0;
    assert_eq!(login(&k, 7, &soft.assert(0x01)).unwrap(), 0);
}

#[test]
fn key_and_alg_must_agree() {
    let mut soft = Soft::new(-7);
    let k = registered(&soft);
    soft.count = 1;
    let a = soft.assert(0x01);
    is_invalid(verify_assertion(&k.public_key_cose, -8, 0, &a.0, &a.1, &a.2, CHALLENGE, RP, ORIGIN), "COSE key alg mismatch");
    // ES384 (-35), PS256 (-37): not offered, not accepted
    for alg in [-35, -37] {
        let key = cbor(&int_map(vec![(1, 2.into()), (3, alg.into()), (-1, 2.into()), (-2, Value::Bytes(vec![1; 48])), (-3, Value::Bytes(vec![1; 48]))]));
        assert!(matches!(verify_assertion(&key, alg, 0, &a.0, &a.1, &a.2, CHALLENGE, RP, ORIGIN), Err(WebauthnError::Unsupported)), "{alg}");
    }
    // Ed448 (OKP crv 7) under EdDSA
    let ed448 = cbor(&int_map(vec![(1, 1.into()), (3, (-8).into()), (-1, 7.into()), (-2, Value::Bytes(vec![1; 57]))]));
    assert!(matches!(verify_assertion(&ed448, -8, 0, &a.0, &a.1, &a.2, CHALLENGE, RP, ORIGIN), Err(WebauthnError::Unsupported)));
    // compressed EC2 point (y as bool) and kty/alg disagreement
    let compressed = cbor(&int_map(vec![(1, 2.into()), (3, (-7).into()), (-1, 1.into()), (-2, Value::Bytes(vec![1; 32])), (-3, true.into())]));
    is_invalid(verify_assertion(&compressed, -7, 0, &a.0, &a.1, &a.2, CHALLENGE, RP, ORIGIN), "malformed COSE key");
    let okp_es256 = cbor(&int_map(vec![(1, 1.into()), (3, (-7).into()), (-1, 6.into()), (-2, Value::Bytes(vec![1; 32]))]));
    is_invalid(verify_assertion(&okp_es256, -7, 0, &a.0, &a.1, &a.2, CHALLENGE, RP, ORIGIN), "COSE key type does not match alg");
    // registration refuses a credential whose key it could never verify
    let ad = {
        let mut d = Sha256::digest(RP.as_bytes()).to_vec();
        d.push(0x41);
        d.extend_from_slice(&[0; 4 + 16]);
        d.extend_from_slice(&[0, 4, 1, 2, 3, 4]);
        d.extend_from_slice(&ed448);
        d
    };
    let att = soft.attestation("none", Value::Map(vec![]), ad);
    assert!(matches!(verify_registration(&client_data("webauthn.create", CHALLENGE, ORIGIN), &att, CHALLENGE, RP, ORIGIN), Err(WebauthnError::Unsupported)));
}

#[test]
fn truncated_and_garbage_inputs_are_refused() {
    let mut soft = Soft::new(-7);
    let (cdj, att) = soft.register();
    let k = verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN).unwrap();
    for cut in 0..att.len() {
        assert!(verify_registration(&cdj, &att[..cut], CHALLENGE, RP, ORIGIN).is_err(), "attestation cut at {cut}");
    }
    // truncate the authData inside a well-formed attestation object
    let ad = soft.auth_data(RP, 0x05, true, false);
    for cut in 0..ad.len() {
        let att = soft.attestation("none", Value::Map(vec![]), ad[..cut].to_vec());
        assert!(verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN).is_err(), "authData cut at {cut}");
    }
    soft.count = 1;
    let (gcdj, gad, gsig) = soft.assert(0x01);
    for cut in 0..gad.len() {
        assert!(verify_assertion(&k.public_key_cose, k.alg, 0, &gcdj, &gad[..cut], &gsig, CHALLENGE, RP, ORIGIN).is_err(), "assertion authData cut at {cut}");
    }
    for cut in 0..k.public_key_cose.len() {
        assert!(verify_assertion(&k.public_key_cose[..cut], k.alg, 0, &gcdj, &gad, &gsig, CHALLENGE, RP, ORIGIN).is_err(), "COSE cut at {cut}");
    }
    for junk in [&b""[..], b"\xff", b"\xa1", b"\xbf\xff", b"\x9f\x9f\x9f", b"\xa3\x63fmt", b"\x5b\xff\xff\xff\xff\xff\xff\xff\xff", &[0x81; 4096]] {
        assert!(verify_registration(&cdj, junk, CHALLENGE, RP, ORIGIN).is_err());
        assert!(verify_assertion(junk, -7, 0, &gcdj, &gad, &gsig, CHALLENGE, RP, ORIGIN).is_err());
    }
    // an empty expected challenge never matches (a server bug, not a pass)
    is_invalid(verify_registration(&client_data("webauthn.create", b"", ORIGIN), &att, b"", RP, ORIGIN), "challenge mismatch");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn fuzz_never_panics(att in proptest::collection::vec(any::<u8>(), 0..2048), ad in proptest::collection::vec(any::<u8>(), 0..512), sig in proptest::collection::vec(any::<u8>(), 0..300)) {
        let cdj = client_data("webauthn.create", CHALLENGE, ORIGIN);
        let _ = verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN);
        let _ = verify_registration(&att, &att, CHALLENGE, RP, ORIGIN);
        let gcdj = client_data("webauthn.get", CHALLENGE, ORIGIN);
        for alg in [-8, -7, -257] {
            let _ = verify_assertion(&att, alg, 0, &gcdj, &ad, &sig, CHALLENGE, RP, ORIGIN);
        }
    }

    /// Mutations of a valid attestation object: the parser sees nearly
    /// well-formed CBOR, which random bytes rarely reach.
    #[test]
    fn fuzz_mutated_attestation_never_panics(flips in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8)) {
        let soft = Soft::new(-8);
        let ad = soft.auth_data(RP, 0x05, true, true);
        let (cdj, _) = soft.register();
        let mut att = soft.attestation("none", Value::Map(vec![]), ad.clone());
        let mut ad = ad;
        for (i, b) in &flips {
            let n = i.index(att.len());
            att[n] = *b;
            let m = i.index(ad.len());
            ad[m] = *b;
        }
        let _ = verify_registration(&cdj, &att, CHALLENGE, RP, ORIGIN);
        let _ = verify_registration(&cdj, &soft.attestation("none", Value::Map(vec![]), ad), CHALLENGE, RP, ORIGIN);
    }
}

// ---- real vectors ---------------------------------------------------------
//
// From the py_webauthn test suite (duo-labs, BSD-3-Clause), captured from
// real browsers/authenticators against rp id "localhost", origin
// "http://localhost:5000":
//   https://raw.githubusercontent.com/duo-labs/py_webauthn/master/tests/test_verify_registration_response.py
//   https://raw.githubusercontent.com/duo-labs/py_webauthn/master/tests/test_verify_authentication_response.py
// The registration and the assertions are different credentials (the suite
// has no matched pair), so each is checked against its own published key.

const V_RP: &str = "localhost";
const V_ORIGIN: &str = "http://localhost:5000";

/// test_verifies_none_attestation_response: ES256, fmt "none", count 23.
#[test]
fn real_es256_none_registration() {
    let cdj = b64("eyJ0eXBlIjoid2ViYXV0aG4uY3JlYXRlIiwiY2hhbGxlbmdlIjoiVHdON240V1R5R0tMYzRaWS1xR3NGcUtuSE00bmdscXN5VjBJQ0psTjJUTzlYaVJ5RnRya2FEd1V2c3FsLWdrTEpYUDZmbkYxTWxyWjUzTW00UjdDdnciLCJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjUwMDAiLCJjcm9zc09yaWdpbiI6ZmFsc2V9");
    let att = b64("o2NmbXRkbm9uZWdhdHRTdG10oGhhdXRoRGF0YVjESZYN5YgOjGh0NBcPZHZgW4_krrmihjLHmVzzuoMdl2NFAAAAFwAAAAAAAAAAAAAAAAAAAAAAQPctcQPE5oNRRJk_nO_371mf7qE7qIodzr0eOf6ACvnMB1oQG165dqutoi1U44shGezu5_gkTjmOPeJO0N8a7P-lAQIDJiABIVggSFbUJF-42Ug3pdM8rDRFu_N5oiVEysPDB6n66r_7dZAiWCDUVnB39FlGypL-qAoIO9xWHtJygo2jfDmHl-_eKFRLDA");
    let challenge = b64("TwN7n4WTyGKLc4ZY-qGsFqKnHM4nglqsyV0ICJlN2TO9XiRyFtrkaDwUvsql-gkLJXP6fnF1MlrZ53Mm4R7Cvw");
    let k = verify_registration(&cdj, &att, &challenge, V_RP, V_ORIGIN).unwrap();
    assert_eq!(k.cred_id, b64("9y1xA8Tmg1FEmT-c7_fvWZ_uoTuoih3OvR45_oAK-cwHWhAbXrl2q62iLVTjiyEZ7O7n-CROOY494k7Q3xrs_w"));
    assert_eq!(k.public_key_cose, b64("pQECAyYgASFYIEhW1CRfuNlIN6XTPKw0RbvzeaIlRMrDwwep-uq_-3WQIlgg1FZwd_RZRsqS_qgKCDvcVh7ScoKNo3w5h5fv3ihUSww"));
    assert_eq!((k.alg, k.sign_count, k.user_verified), (-7, 23, true));
    is_invalid(verify_registration(&cdj, &att, &challenge, "example.com", V_ORIGIN), "rp id hash mismatch");
    is_invalid(verify_registration(&cdj, &att, &challenge, V_RP, "http://localhost:5001"), "origin mismatch");
}

/// test_raises_exception_on_invalid_attestation_type (its unmodified input):
/// a YubiKey "packed" full attestation with an x5c chain. We do not verify
/// attestation, so it registers like "none".
#[test]
fn real_packed_registration_ignores_statement() {
    let cdj = b64("eyJ0eXBlIjoid2ViYXV0aG4uY3JlYXRlIiwiY2hhbGxlbmdlIjoicERSbWtkZHVBaS1BVTJ4Nm8tRnFxaEkzWEsybmxWbHNDU3IwNHpXa050djg0SndyTUh0RWxSSEhVV0xFRGhrckVhUThCMWxCY0lIX1ZTUnFwX1JBQXciLCJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjUwMDAiLCJjcm9zc09yaWdpbiI6ZmFsc2V9");
    let att = b64("o2NmbXRmcGFja2VkZ2F0dFN0bXSjY2FsZyZjc2lnWEcwRQIgRpuZ6hdaLAgWgCFTIo4BGSTBAxwwqk4u3s1-JAzv_H4CIQCZnfoic34aOwlac1A09eflEtb0V1kO7yGhHOw5P5wVWmN4NWOBWQLBMIICvTCCAaWgAwIBAgIEKudiYzANBgkqhkiG9w0BAQsFADAuMSwwKgYDVQQDEyNZdWJpY28gVTJGIFJvb3QgQ0EgU2VyaWFsIDQ1NzIwMDYzMTAgFw0xNDA4MDEwMDAwMDBaGA8yMDUwMDkwNDAwMDAwMFowbjELMAkGA1UEBhMCU0UxEjAQBgNVBAoMCVl1YmljbyBBQjEiMCAGA1UECwwZQXV0aGVudGljYXRvciBBdHRlc3RhdGlvbjEnMCUGA1UEAwweWXViaWNvIFUyRiBFRSBTZXJpYWwgNzE5ODA3MDc1MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEKgOGXmBD2Z4R_xCqJVRXhL8Jr45rHjsyFykhb1USGozZENOZ3cdovf5Ke8fj2rxi5tJGn_VnW4_6iQzKdIaeP6NsMGowIgYJKwYBBAGCxAoCBBUxLjMuNi4xLjQuMS40MTQ4Mi4xLjEwEwYLKwYBBAGC5RwCAQEEBAMCBDAwIQYLKwYBBAGC5RwBAQQEEgQQbUS6m_bsLkm5MAyP6SDLczAMBgNVHRMBAf8EAjAAMA0GCSqGSIb3DQEBCwUAA4IBAQByV9A83MPhFWmEkNb4DvlbUwcjc9nmRzJjKxHc3HeK7GvVkm0H4XucVDB4jeMvTke0WHb_jFUiApvpOHh5VyMx5ydwFoKKcRs5x0_WwSWL0eTZ5WbVcHkDR9pSNcA_D_5AsUKOBcbpF5nkdVRxaQHuuIuwV4k1iK2IqtMNcU8vL6w21U261xCcWwJ6sMq4zzVO8QCKCQhsoIaWrwz828GDmPzfAjFsJiLJXuYivdHACkeJ5KHMt0mjVLpfJ2BCML7_rgbmvwL7wBW80VHfNdcKmKjkLcpEiPzwcQQhiN_qHV90t-p4iyr5xRSpurlP5zic2hlRkLKxMH2_kRjhqSn4aGF1dGhEYXRhWMRJlg3liA6MaHQ0Fw9kdmBbj-SuuaKGMseZXPO6gx2XY0UAAAAqbUS6m_bsLkm5MAyP6SDLcwBAFsWBrFcw8yRjxV8z18Egh91o1AScNRYkIuUoY6wIlIhslDpP7eydKi1q5s9g1ugDP9mqBlPDDFPRbH6YLwHbtqUBAgMmIAEhWCAq3y0RWh8nLzanBZQwTA7yAbUy9KEDAM0b3N9Elrb0VCJYIJrX7ygtpyInb5mXBE7g9YEow6xWrJ400HhL2r4q5tzV");
    let challenge = b64("pDRmkdduAi-AU2x6o-FqqhI3XK2nlVlsCSr04zWkNtv84JwrMHtElRHHUWLEDhkrEaQ8B1lBcIH_VSRqp_RAAw");
    let k = verify_registration(&cdj, &att, &challenge, V_RP, V_ORIGIN).unwrap();
    assert_eq!(k.cred_id, b64("FsWBrFcw8yRjxV8z18Egh91o1AScNRYkIuUoY6wIlIhslDpP7eydKi1q5s9g1ugDP9mqBlPDDFPRbH6YLwHbtg"));
    assert_eq!((k.alg, k.sign_count), (-7, 42));
}

/// test_verify_authentication_response_with_EC2_public_key: ES256, legacy
/// clientData (hashAlgorithm, no crossOrigin), stored 77 -> 78, UV off.
#[test]
fn real_es256_assertion() {
    let key = b64("pQECAyYgASFYIIeDTe-gN8A-zQclHoRnGFWN8ehM1b7yAsa8I8KIvmplIlgg4nFGT5px8o6gpPZZhO01wdy9crDSA_Ngtkx0vGpvPHI");
    let ad = b64("SZYN5YgOjGh0NBcPZHZgW4_krrmihjLHmVzzuoMdl2MBAAAATg");
    let cdj = b64("eyJjaGFsbGVuZ2UiOiJ4aTMwR1BHQUZZUnhWRHBZMXNNMTBEYUx6VlFHNjZudi1fN1JVYXpIMHZJMll2RzhMWWdERW52TjVmWlpOVnV2RUR1TWk5dGUzVkxxYjQyTjBma0xHQSIsImNsaWVudEV4dGVuc2lvbnMiOnt9LCJoYXNoQWxnb3JpdGhtIjoiU0hBLTI1NiIsIm9yaWdpbiI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAwMCIsInR5cGUiOiJ3ZWJhdXRobi5nZXQifQ");
    let sig = b64("MEUCIGisVZOBapCWbnJJvjelIzwpixxIwkjCCb5aCHafQu68AiEA88v-2pJNNApPFwAKFiNuf82-2hBxYW5kGwVweeoxCwo");
    let challenge = b64("xi30GPGAFYRxVDpY1sM10DaLzVQG66nv-_7RUazH0vI2YvG8LYgDEnvN5fZZNVuvEDuMi9te3VLqb42N0fkLGA");
    assert_eq!(verify_assertion(&key, -7, 77, &cdj, &ad, &sig, &challenge, V_RP, V_ORIGIN).unwrap(), 78);
    is_invalid(verify_assertion(&key, -7, 78, &cdj, &ad, &sig, &challenge, V_RP, V_ORIGIN), "sign count did not increase");
    let mut bad = sig.clone();
    bad[10] ^= 1;
    is_invalid(verify_assertion(&key, -7, 77, &cdj, &ad, &bad, &challenge, V_RP, V_ORIGIN), "bad signature");
    // the registration vector's key does not verify this assertion
    let other = b64("pQECAyYgASFYIEhW1CRfuNlIN6XTPKw0RbvzeaIlRMrDwwep-uq_-3WQIlgg1FZwd_RZRsqS_qgKCDvcVh7ScoKNo3w5h5fv3ihUSww");
    is_invalid(verify_assertion(&other, -7, 77, &cdj, &ad, &sig, &challenge, V_RP, V_ORIGIN), "bad signature");
}

/// test_verify_authentication_response_with_RSA_public_key: RS256, UV on,
/// stored 0 -> 1.
#[test]
fn real_rs256_assertion() {
    let key = b64("pAEDAzkBACBZAQDfV20epzvQP-HtcdDpX-cGzdOxy73WQEvsU7Dnr9UWJophEfpngouvgnRLXaEUn_d8HGkp_HIx8rrpkx4BVs6X_B6ZjhLlezjIdJbLbVeb92BaEsmNn1HW2N9Xj2QM8cH-yx28_vCjf82ahQ9gyAr552Bn96G22n8jqFRQKdVpO-f-bvpvaP3IQ9F5LCX7CUaxptgbog1SFO6FI6ob5SlVVB00lVXsaYg8cIDZxCkkENkGiFPgwEaZ7995SCbiyCpUJbMqToLMgojPkAhWeyktu7TlK6UBWdJMHc3FPAIs0lH_2_2hKS-mGI1uZAFVAfW1X-mzKL0czUm2P1UlUox7IUMBAAE");
    let ad = b64("SZYN5YgOjGh0NBcPZHZgW4_krrmihjLHmVzzuoMdl2MFAAAAAQ");
    let cdj = b64("eyJ0eXBlIjoid2ViYXV0aG4uZ2V0IiwiY2hhbGxlbmdlIjoiaVBtQWkxUHAxWEw2b0FncTNQV1p0WlBuWmExekZVRG9HYmFRMF9LdlZHMWxGMnMzUnRfM280dVN6Y2N5MHRtY1RJcFRUVDRCVTFULUk0bWFhdm5kalEiLCJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjUwMDAiLCJjcm9zc09yaWdpbiI6ZmFsc2V9");
    let sig = b64("iOHKX3erU5_OYP_r_9HLZ-CexCE4bQRrxM8WmuoKTDdhAnZSeTP0sjECjvjfeS8MJzN1ArmvV0H0C3yy_FdRFfcpUPZzdZ7bBcmPh1XPdxRwY747OrIzcTLTFQUPdn1U-izCZtP_78VGw9pCpdMsv4CUzZdJbEcRtQuRS03qUjqDaovoJhOqEBmxJn9Wu8tBi_Qx7A33RbYjlfyLm_EDqimzDZhyietyop6XUcpKarKqVH0M6mMrM5zTjp8xf3W7odFCadXEJg-ERZqFM0-9Uup6kJNLbr6C5J4NDYmSm3HCSA6lp2iEiMPKU8Ii7QZ61kybXLxsX4w4Dm3fOLjmDw");
    let challenge = b64("iPmAi1Pp1XL6oAgq3PWZtZPnZa1zFUDoGbaQ0_KvVG1lF2s3Rt_3o4uSzccy0tmcTIpTTT4BU1T-I4maavndjQ");
    assert_eq!(verify_assertion(&key, -257, 0, &cdj, &ad, &sig, &challenge, V_RP, V_ORIGIN).unwrap(), 1);
}

/// test_verify_authentication_response_with_OKP_public_key: EdDSA, stored 3 -> 7.
#[test]
fn real_eddsa_assertion() {
    let key = b64("pAEBAycgBiFYIMz6_SUFLiDid2Yhlq0YboyJ-CDrIrNpkPUGmJp4D3Dp");
    let ad = b64("SZYN5YgOjGh0NBcPZHZgW4_krrmihjLHmVzzuoMdl2MBAAAABw");
    let cdj = b64("eyJ0eXBlIjoid2ViYXV0aG4uZ2V0IiwiY2hhbGxlbmdlIjoiZVo0ZWVBM080ank1Rkl6cURhU0o2SkROR3UwYkJjNXpJMURqUV9rTHNvMVdOcWtHNms1bUNZZjFkdFFoVlVpQldaV2xaa3pSNU1GZWVXQ3BKUlVOWHciLCJvcmlnaW4iOiJodHRwOi8vbG9jYWxob3N0OjUwMDAiLCJjcm9zc09yaWdpbiI6ZmFsc2V9");
    let sig = b64("RRWV8mYDRvK7YdQgdtZD4pJ2dh1D_IWZ_D6jsZo6FHJBoenbj0CVT5nA20vUzlRhN4R6dOEUHmUwP1F8eRBhBg");
    let challenge = b64("eZ4eeA3O4jy5FIzqDaSJ6JDNGu0bBc5zI1DjQ_kLso1WNqkG6k5mCYf1dtQhVUiBWZWlZkzR5MFeeWCpJRUNXw");
    assert_eq!(verify_assertion(&key, -8, 3, &cdj, &ad, &sig, &challenge, V_RP, V_ORIGIN).unwrap(), 7);
}
