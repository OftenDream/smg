//! Backend-agnostic encode latency accounting.
//!
//! Lives outside the gigatoken module because it is useful on its own and must
//! compile whether or not the optional `gigatoken` feature is enabled — both
//! backends are timed through the same code path so an A/B is directly
//! comparable.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use tracing::info;

/// Rolling encode-latency counters, logged when `SMG_TOKENIZER_TIMING=1`.
/// Both backends are timed through the same code path so the two legs of an
/// A/B are directly comparable.
#[derive(Default)]
pub(crate) struct EncodeTiming {
    enabled: bool,
    calls: AtomicU64,
    total_ns: AtomicU64,
    total_bytes: AtomicU64,
    max_ns: AtomicU64,
}

impl EncodeTiming {
    pub(crate) fn from_env() -> Self {
        Self {
            enabled: matches!(
                std::env::var("SMG_TOKENIZER_TIMING").as_deref(),
                Ok("1") | Ok("true")
            ),
            ..Default::default()
        }
    }

    #[inline]
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Records one encode and logs a cumulative line every 256 calls. Emitting
    /// per-call would itself perturb what we are trying to measure.
    pub(crate) fn record(&self, elapsed_ns: u64, input_bytes: usize, backend: &str) {
        let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        let total = self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed) + elapsed_ns;
        let bytes = self
            .total_bytes
            .fetch_add(input_bytes as u64, Ordering::Relaxed)
            + input_bytes as u64;
        self.max_ns.fetch_max(elapsed_ns, Ordering::Relaxed);
        if n.is_multiple_of(256) {
            info!(
                backend,
                calls = n,
                mean_us = (total as f64 / n as f64) / 1000.0,
                max_us = self.max_ns.load(Ordering::Relaxed) as f64 / 1000.0,
                mean_input_kb = (bytes as f64 / n as f64) / 1024.0,
                "tokenizer encode timing"
            );
        }
    }
}

/// Timed encode shared by both backends.
#[inline]
pub(crate) fn timed<T>(
    timing: &EncodeTiming,
    backend: &str,
    input_len: usize,
    f: impl FnOnce() -> T,
) -> T {
    if !timing.enabled() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    timing.record(t.elapsed().as_nanos() as u64, input_len, backend);
    out
}
