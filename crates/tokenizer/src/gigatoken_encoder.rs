//! Optional gigatoken fast path for prompt encode.
//!
//! Compiled only with the `gigatoken` cargo feature; see
//! `gigatoken_encoder_disabled.rs` for the default no-op stub.
//!
//! Selected at runtime by `SMG_TOKENIZER_BACKEND=gigatoken`, and even then only
//! if the load-time parity self-check reproduces HuggingFace exactly. Only
//! `Encoder::encode` is routed here — decode, chat templating and
//! special-token handling stay on the HuggingFace implementation, which is the
//! only backend with incremental decode (`decode_step`); gigatoken has no
//! streaming decode API.
//!
//! The failure mode is always "fall back to HuggingFace", never "serve
//! different tokens".

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

use gigatoken_rs::{
    encode_docs_ragged,
    load_tokenizer::hf::{load_hf_slice, HfTokenizer as GigaHf},
    Tokenizer as GigaTokenizer, WorkerPool,
};
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{info, warn};

use crate::traits::TokenIdType;

/// Probe strings the fast path must reproduce exactly before it is allowed to
/// serve traffic: ASCII prose, code punctuation, CJK, emoji (multi-codepoint),
/// combining accents, whitespace runs, digit groups, and GLM/Qwen-style chat
/// markers so the added-token path is covered too.
const PARITY_PROBES: &[&str] = &[
    "",
    " ",
    "Hello, world!",
    "fn main() { let x: Vec<u32> = vec![1, 2, 3]; }",
    "  \t\n\n   trailing and   repeated   whitespace \n",
    "0123456789 1234 99999 007",
    "日本語のテキストと中文混排",
    "emoji: 👨‍👩‍👧‍👦 🇯🇵 🏳️‍🌈",
    "combining: éé àà ñ",
    "עברית وعربي RTL text",
    "<|system|>\nYou are helpful.<|user|>\nhi<|assistant|>\n",
    "[gMASK]<sop><|user|>\nwrite a haiku<|assistant|>\n<think>",
    "<tool_call>{\"name\": \"get_weather\"}</tool_call>",
];

/// gigatoken's encoder takes `&mut self` for its pretoken cache, so a pool of
/// forked instances stands in for the shared `&self` that `Encoder` provides.
/// Slots are probed without blocking; the scan falls back to a blocking
/// acquire so encode never fails under contention.
pub(crate) struct GigatokenEncoder {
    slots: Vec<Mutex<GigaTokenizer>>,
    cursor: AtomicUsize,
    /// Prototype for the parallel ragged batch path. MUST stay unmutated:
    /// `WorkerPool` forks a worker per slot on first use and never re-compares
    /// it against the prototype, so mutating this would yield stale workers.
    /// Every encode goes through `slots` or a pool fork, never through `proto`.
    proto: GigaTokenizer,
    workers: WorkerPool,
}

impl GigatokenEncoder {
    /// Returns `None` — leaving the caller on the HuggingFace path — whenever
    /// the fast path cannot be proven equivalent: unsupported vocabulary
    /// shape, or any probe where the two backends disagree.
    pub(crate) fn try_new(file_path: &str, hf: &HfTokenizer) -> Option<Self> {
        let raw = match std::fs::read(file_path) {
            Ok(r) => r,
            Err(e) => {
                warn!(path = file_path, error = %e, "gigatoken: cannot read tokenizer file");
                return None;
            }
        };

        let base = match load_hf_slice(&raw) {
            Ok(GigaHf::Bpe(t)) => t,
            Ok(GigaHf::SentencePiece(_)) => {
                warn!("gigatoken: SentencePiece vocabularies are not wired into this fast path");
                return None;
            }
            Err(e) => {
                warn!(error = %e, "gigatoken: failed to load vocabulary");
                return None;
            }
        };

        // The parity probes are all short, so they can never trip a configured
        // `truncation.max_length` (or padding) — HF would apply either in
        // `post_process` regardless of `add_special_tokens`, and gigatoken
        // would not. No probe content can catch that, so gate on the config
        // itself: any truncation/padding means the fast path could serve
        // different tokens for long prompts. Disable it outright.
        if hf.get_truncation().is_some() || hf.get_padding().is_some() {
            warn!("gigatoken: tokenizer configures truncation/padding; fast path disabled");
            return None;
        }

        let mut probe = base.fork();
        let mut ids: Vec<TokenIdType> = Vec::new();
        for text in PARITY_PROBES {
            // The fast path ignores `add_special_tokens`, so it may only engage
            // when the flag makes no difference to HF — i.e. no token-adding
            // post-processor is configured.
            let hf_true = match hf.encode(*text, true) {
                Ok(e) => e.get_ids().to_vec(),
                Err(e) => {
                    warn!(error = %e, "gigatoken: HF reference encode failed during self-check");
                    return None;
                }
            };
            let hf_false = match hf.encode(*text, false) {
                Ok(e) => e.get_ids().to_vec(),
                Err(e) => {
                    warn!(error = %e, "gigatoken: HF reference encode failed during self-check");
                    return None;
                }
            };
            if hf_true != hf_false {
                warn!(
                    probe = text,
                    "gigatoken: tokenizer adds special tokens via post-processor; fast path disabled"
                );
                return None;
            }
            ids.clear();
            probe.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
            if ids != hf_true {
                warn!(
                    probe = text,
                    hf_len = hf_true.len(),
                    giga_len = ids.len(),
                    "gigatoken: parity self-check failed; fast path disabled"
                );
                return None;
            }
        }

        // Each fork eagerly zeroes a vocab-sized pretoken cache (~8 MiB for a
        // ~150k vocab, committed RSS, not lazy) — so the pool is priced by the
        // vocabulary, not the core count. And with encode at ~1 ms instead of
        // ~45 ms, slots are held ~45x shorter, so contention collapses: 8 is
        // plenty. Bound or widen explicitly with SMG_GIGATOKEN_SLOTS.
        let slots = std::env::var("SMG_GIGATOKEN_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8)
            .max(1);

        info!(
            slots,
            probes = PARITY_PROBES.len(),
            "gigatoken encode fast path enabled (decode stays on HuggingFace)"
        );

        Some(Self {
            slots: (0..slots).map(|_| Mutex::new(base.fork())).collect(),
            cursor: AtomicUsize::new(0),
            proto: base,
            workers: WorkerPool::new(),
        })
    }

    /// `None` means "this call could not be served safely" — the caller falls
    /// back to HuggingFace for it. A poisoned slot (a panic escaped a prior
    /// encode, leaving `pretoken_cache`/`token_arena` mid-mutation) is never
    /// reused: reusing it is the one path that could serve *wrong* ids rather
    /// than falling back. The pool self-heals by shrinking; if every slot is
    /// poisoned, every call routes back to HuggingFace.
    pub(crate) fn encode(&self, input: &str) -> Option<Vec<TokenIdType>> {
        let n = self.slots.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % n;
        let mut out = Vec::new();
        for i in 0..n {
            match self.slots[(start + i) % n].try_lock() {
                Ok(mut slot) => {
                    slot.encode_with_added_tokens_flat(input.as_bytes(), &mut out);
                    return Some(out);
                }
                // Busy or poisoned: try the next slot. (Poisoned slots stay
                // permanently out of service.)
                Err(_) => continue,
            }
        }
        // Every slot busy: block on healthy slots in ring order rather than
        // silently diverging to a different backend mid-benchmark. A poisoned
        // result here is skipped, never `into_inner()`d back into service.
        for i in 0..n {
            match self.slots[(start + i) % n].lock() {
                Ok(mut slot) => {
                    slot.encode_with_added_tokens_flat(input.as_bytes(), &mut out);
                    return Some(out);
                }
                Err(_poisoned) => continue,
            }
        }
        warn!("gigatoken: all pool slots poisoned; routing encode back to HuggingFace");
        None
    }

    /// Batch encode via gigatoken's parallel ragged path, which shards the
    /// documents across a rayon pool (splitting oversized ones at
    /// pretoken-safe boundaries) and returns one flat id buffer plus per-row
    /// lengths. Token- and order-identical to encoding each input on its own.
    ///
    /// This exists because the HuggingFace `encode_batch` it replaces is itself
    /// rayon-parallel; a serial map over `encode` would hand back part of the
    /// win on large batches.
    /// `None` falls the whole batch back to HuggingFace (same contract as
    /// [`Self::encode`]).
    pub(crate) fn encode_batch(&self, inputs: &[&str]) -> Option<Vec<Vec<TokenIdType>>> {
        // Below two documents the chunk/gather bookkeeping is pure overhead.
        if inputs.len() < 2 {
            return inputs.iter().map(|s| self.encode(s)).collect();
        }
        let docs: Vec<&[u8]> = inputs.iter().map(|s| s.as_bytes()).collect();
        let (ids, lens) = encode_docs_ragged(&self.workers, &self.proto, &docs);

        // The ragged contract is one row per input with sum(lens) == ids.len().
        // A violation must be *detected*, not clamped: clamping would hand the
        // caller silently empty/truncated rows, or a Vec whose length doesn't
        // match `inputs` — positionally misaligned tokens downstream. Validate
        // up front and fall back to per-input encode (which parity-checks per
        // slot and can itself route to HuggingFace).
        let total: usize = lens.iter().map(|&l| l.max(0) as usize).sum();
        if lens.len() != inputs.len() || lens.iter().any(|&l| l < 0) || total != ids.len() {
            warn!(
                rows = lens.len(),
                inputs = inputs.len(),
                ids = ids.len(),
                total,
                "gigatoken: ragged batch contract violated; per-input encode"
            );
            return inputs.iter().map(|s| self.encode(s)).collect();
        }

        let mut out = Vec::with_capacity(lens.len());
        let mut at = 0usize;
        for len in lens {
            let end = at + len as usize;
            out.push(ids[at..end].to_vec());
            at = end;
        }
        Some(out)
    }
}

/// Whether `SMG_TOKENIZER_BACKEND` selects the gigatoken fast path.
pub(crate) fn fast_path_requested() -> bool {
    matches!(
        std::env::var("SMG_TOKENIZER_BACKEND").as_deref(),
        Ok("gigatoken")
    )
}
