#![expect(clippy::expect_used, reason = "test code — panics are intentional")]
#![expect(clippy::print_stderr, reason = "test skip diagnostics")]
//! Contract tests for the optional gigatoken prompt-encode fast path.
//!
//! The contract is deliberately narrow and is asserted here in full:
//!
//!   1. Tokens are IDENTICAL to the HuggingFace backend, always. Whether the
//!      fast path engaged is an implementation detail; producing different
//!      tokens is never acceptable.
//!   2. It must refuse to engage when it cannot be proven equivalent — most
//!      importantly when the tokenizer has a post-processor that adds special
//!      tokens, because the fast path does not implement `add_special_tokens`.
//!   3. It must be safe to call concurrently from many threads (the encoder
//!      keeps a pool of forked instances behind try-lock).
//!
//! These run against the TinyLlama tokenizer the rest of this crate's tests
//! already cache, so they work in CI with no model weights. Point
//! `SMG_TEST_GIGATOKEN_TOKENIZER` at a `tokenizer.json` to additionally cover a
//! vocabulary where the fast path is expected to engage (e.g. GLM-5.2).
//!
//! NOTE: nothing here asserts on wall-clock time. Timing assertions live in
//! `encode_size_sweep.rs`, which is `#[ignore]`d, because a CI runner under
//! load fails them for reasons unrelated to this code.

// The shared helper module also carries fixtures other tests use; this one
// only needs the cached-tokenizer path.
#[expect(
    dead_code,
    reason = "shared test fixtures; this test only needs the cached tokenizer"
)]
mod common;

use std::{
    sync::{Mutex, MutexGuard, OnceLock},
    thread,
};

use llm_tokenizer::{huggingface::HuggingFaceTokenizer, traits::Encoder};

/// `set_var`/`remove_var` are process-global; cargo runs tests in threads.
/// Every test that touches `SMG_TOKENIZER_BACKEND` holds this first.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Loads the same tokenizer twice: once forced onto the HuggingFace path, once
/// with the fast path requested. Returns `(reference, candidate)`.
fn load_pair(path: &str) -> (HuggingFaceTokenizer, HuggingFaceTokenizer) {
    let _guard = env_lock();
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    let reference = HuggingFaceTokenizer::from_file(path).expect("load reference tokenizer");

    std::env::set_var("SMG_TOKENIZER_BACKEND", "gigatoken");
    let candidate = HuggingFaceTokenizer::from_file(path).expect("load candidate tokenizer");
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    (reference, candidate)
}

/// Inputs chosen to hit the places byte-level BPE implementations disagree:
/// empty/whitespace, control bytes, the ASCII/non-ASCII boundary the scan hops
/// over, multi-codepoint grapheme clusters, RTL, combining marks, and
/// chat-template markers (added tokens, not ordinary merges).
fn corpus() -> Vec<String> {
    let mut v: Vec<String> = [
        "",
        " ",
        "\n",
        "\t\t\n\r\n  ",
        "a",
        "Hello, world!",
        "fn main() { let x: Vec<u32> = vec![1, 2, 3]; }",
        "  \t\n\n   trailing and   repeated   whitespace \n",
        "0123456789 1234 99999 007 3.14159 1e-9 0xDEADBEEF",
        "日本語のテキストと中文混排とハングル한국어",
        "emoji: 👨‍👩‍👧‍👦 🇯🇵 🏳️‍🌈 🚀🚀🚀",
        "combining: éé àà ñ ǅǆ ﬁﬂ",
        "עברית وعربي RTL text mixed with LTR",
        "<|system|>\nYou are helpful.<|user|>\nhi<|assistant|>\n",
        "[gMASK]<sop><|user|>\nwrite a haiku<|assistant|>\n<think>",
        "<tool_call>{\"name\": \"get_weather\"}</tool_call>",
        "control\u{0}bytes\u{1}and\u{1f}del\u{7f}here",
        "\u{feff}BOM-prefixed text",
        "replacement chars: \u{fffd}\u{fffd}\u{fffd}",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();

    // Length sweep across the 16- and 32-byte scan windows and well past them,
    // including exact multiples and one-off from multiples.
    for n in [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 255, 4096, 65_536] {
        v.push("a".repeat(n));
        v.push("あ".repeat(n));
        v.push(format!("{} tail", "word ".repeat(n)));
    }
    v
}

fn assert_parity(reference: &HuggingFaceTokenizer, candidate: &HuggingFaceTokenizer, label: &str) {
    for (i, case) in corpus().iter().enumerate() {
        for add_special in [false, true] {
            let want_enc = reference
                .encode(case, add_special)
                .expect("reference encode");
            let got_enc = candidate
                .encode(case, add_special)
                .expect("candidate encode");
            assert_eq!(
                want_enc.token_ids(),
                got_enc.token_ids(),
                "[{label}] token mismatch on case {i} (len {}, add_special_tokens={add_special}); \
                 first 80 chars: {:?}",
                case.len(),
                case.chars().take(80).collect::<String>()
            );
        }
    }
}

/// The core contract, on a tokenizer that is always available in CI.
///
/// TinyLlama's post-processor adds BOS, so the self-check is expected to REFUSE
/// the fast path here. That is exactly the interesting case: the encoder must
/// silently fall back and still be correct. If a future gigatoken learns
/// `add_special_tokens` and engages, this test keeps passing — it asserts the
/// contract, not the mechanism.
#[test]
fn tokens_match_huggingface_on_cached_tokenizer() {
    let path = common::ensure_tokenizer_cached();
    let (reference, candidate) = load_pair(path.to_str().expect("utf-8 path"));
    assert_parity(&reference, &candidate, "tinyllama");
}

/// Same contract on a vocabulary where the fast path is expected to engage.
/// Opt in with `SMG_TEST_GIGATOKEN_TOKENIZER=/path/to/tokenizer.json`.
#[test]
fn tokens_match_huggingface_on_opt_in_tokenizer() {
    let Ok(path) = std::env::var("SMG_TEST_GIGATOKEN_TOKENIZER") else {
        eprintln!("skipping: set SMG_TEST_GIGATOKEN_TOKENIZER to a tokenizer.json to run");
        return;
    };
    assert!(
        std::path::Path::new(&path).exists(),
        "SMG_TEST_GIGATOKEN_TOKENIZER points at a missing file: {path}"
    );
    let (reference, candidate) = load_pair(&path);
    assert_parity(&reference, &candidate, "opt-in");
}

/// The encoder hands out forked instances from a fixed-size pool under
/// try-lock. Hammer it from more threads than there are slots and assert every
/// thread still gets the reference answer.
#[test]
fn concurrent_encode_is_correct_under_contention() {
    let path = std::env::var("SMG_TEST_GIGATOKEN_TOKENIZER").unwrap_or_else(|_| {
        common::ensure_tokenizer_cached()
            .to_str()
            .expect("utf-8 path")
            .to_owned()
    });
    let (reference, candidate) = load_pair(&path);

    let cases = corpus();
    let expected: Vec<Vec<u32>> = cases
        .iter()
        .map(|c| {
            reference
                .encode(c, false)
                .expect("reference")
                .token_ids()
                .to_vec()
        })
        .collect();

    thread::scope(|scope| {
        for _ in 0..64 {
            scope.spawn(|| {
                for (case, want) in cases.iter().zip(&expected) {
                    let enc = candidate.encode(case, false).expect("candidate");
                    assert_eq!(
                        enc.token_ids(),
                        want.as_slice(),
                        "concurrent mismatch (len {})",
                        case.len()
                    );
                }
            });
        }
    });
}

/// Without the env var the fast path must never be selected, regardless of
/// whether the crate was built with the feature.
#[test]
fn absent_env_var_leaves_the_huggingface_path_untouched() {
    let path = common::ensure_tokenizer_cached();
    let path = path.to_str().expect("utf-8 path");
    let (reference, _) = load_pair(path);

    let guard = env_lock();
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    let plain = HuggingFaceTokenizer::from_file(path).expect("load plain");
    drop(guard);

    for case in corpus() {
        let a = reference.encode(&case, true).expect("reference");
        let b = plain.encode(&case, true).expect("plain");
        assert_eq!(a.token_ids(), b.token_ids());
    }
}

/// An unrecognised backend name must fall back rather than fail the load.
#[test]
fn unknown_backend_name_falls_back_cleanly() {
    let path = common::ensure_tokenizer_cached();
    let path = path.to_str().expect("utf-8 path");

    let guard = env_lock();
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    let reference = HuggingFaceTokenizer::from_file(path).expect("load reference");
    std::env::set_var("SMG_TOKENIZER_BACKEND", "definitely-not-a-backend");
    let candidate = HuggingFaceTokenizer::from_file(path).expect("load must not fail");
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    drop(guard);

    for case in corpus() {
        let a = reference.encode(&case, true).expect("reference");
        let b = candidate.encode(&case, true).expect("candidate");
        assert_eq!(a.token_ids(), b.token_ids());
    }
}

/// `encode_batch` must agree with `encode` element-by-element AND with the
/// HuggingFace reference. The batch path is a different implementation (a
/// parallel ragged shard-and-gather), so it needs its own parity assertion —
/// batch sizes are chosen to straddle the ragged path's 2-document threshold
/// and to mix tiny and large documents in one call.
#[test]
fn batch_encode_matches_single_encode_and_huggingface() {
    let path = std::env::var("SMG_TEST_GIGATOKEN_TOKENIZER").unwrap_or_else(|_| {
        common::ensure_tokenizer_cached()
            .to_str()
            .expect("utf-8 path")
            .to_owned()
    });
    let (reference, candidate) = load_pair(&path);
    let cases = corpus();

    for batch_len in [1usize, 2, 3, 8, 64, cases.len()] {
        let batch: Vec<&str> = cases.iter().take(batch_len).map(String::as_str).collect();
        let got = candidate
            .encode_batch(&batch, false)
            .expect("candidate batch encode");
        assert_eq!(got.len(), batch.len(), "batch length changed");

        for (i, (enc, case)) in got.iter().zip(&batch).enumerate() {
            let want_hf = reference.encode(case, false).expect("reference encode");
            assert_eq!(
                enc.token_ids(),
                want_hf.token_ids(),
                "batch[{i}] of {batch_len} disagrees with HuggingFace (len {})",
                case.len()
            );
            let want_single = candidate.encode(case, false).expect("candidate single");
            assert_eq!(
                enc.token_ids(),
                want_single.token_ids(),
                "batch[{i}] of {batch_len} disagrees with its own single-encode path"
            );
        }
    }

    // Reversed order must not leak state between rows.
    let mut rev: Vec<&str> = cases.iter().map(String::as_str).collect();
    rev.reverse();
    let got = candidate.encode_batch(&rev, false).expect("reversed batch");
    for (enc, case) in got.iter().zip(&rev) {
        let want = reference.encode(case, false).expect("reference");
        assert_eq!(
            enc.token_ids(),
            want.token_ids(),
            "order-dependent batch result"
        );
    }
}
