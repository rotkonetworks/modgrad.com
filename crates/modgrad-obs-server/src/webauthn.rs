//! WebAuthn verification boundary for passkey sign-in ([`crate::portal`]).
//!
//! This build has no verifier: both functions refuse with
//! [`WebauthnError::Unsupported`], which the portal answers with 501. The
//! real implementation (attestation "none", ES256 / EdDSA / RS256) drops
//! in with exactly these signatures.

/// A credential that passed registration.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredKey {
    pub cred_id: Vec<u8>,
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

pub fn verify_registration(_client_data_json: &[u8], _attestation_object: &[u8], _expected_challenge: &[u8], _rp_id: &str, _origin: &str) -> Result<RegisteredKey, WebauthnError> {
    Err(WebauthnError::Unsupported)
}

/// Returns the new sign count.
#[allow(clippy::too_many_arguments)]
pub fn verify_assertion(_public_key_cose: &[u8], _alg: i64, _stored_count: u32, _client_data_json: &[u8], _authenticator_data: &[u8], _signature: &[u8], _expected_challenge: &[u8], _rp_id: &str, _origin: &str) -> Result<u32, WebauthnError> {
    Err(WebauthnError::Unsupported)
}
