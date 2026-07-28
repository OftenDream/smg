//! Stand-in for [`super::gigatoken_encoder`] when the optional `gigatoken`
//! feature is off (the default).
//!
//! It mirrors the real module's API exactly so `huggingface.rs` carries no
//! `cfg` branches: `fast_path_requested()` is always `false`, so
//! `GigatokenEncoder::try_new` is never reached and the encoder is
//! uninhabited. Everything here compiles away.

use tokenizers::Tokenizer as HfTokenizer;

use crate::traits::TokenIdType;

/// Never true in this build: the fast path is compiled out entirely.
#[inline]
pub(crate) fn fast_path_requested() -> bool {
    false
}

/// Uninhabited: with the feature off there is no gigatoken vocabulary to hold,
/// and `try_new` is unreachable because `fast_path_requested()` is `false`.
pub(crate) enum GigatokenEncoder {}

impl GigatokenEncoder {
    #[inline]
    pub(crate) fn try_new(_file_path: &str, _hf: &HfTokenizer) -> Option<Self> {
        None
    }

    /// Unreachable — no value of this type can be constructed.
    #[inline]
    pub(crate) fn encode(&self, _input: &str) -> Option<Vec<TokenIdType>> {
        match *self {}
    }

    /// Unreachable — no value of this type can be constructed.
    #[inline]
    pub(crate) fn encode_batch(&self, _inputs: &[&str]) -> Option<Vec<Vec<TokenIdType>>> {
        match *self {}
    }
}
