#![expect(clippy::print_stdout, reason = "benchmark table output")]
#![expect(clippy::print_stderr, reason = "test skip diagnostics")]
//! Encode-latency sweep across the prompt sizes long-prompt gateway workloads
//! actually produce, measured through SMG's public tokenizer API and built
//! with SMG's own shipped release profile (`opt-level = "z"`, fat LTO).
//!
//! That profile matters: a size-optimised build is materially slower at
//! tokenization than a `-O3` build, so numbers taken from a standalone
//! `-O3` harness understate what the gateway actually spends.
//!
//! Run explicitly (it is `#[ignore]`d so normal test runs stay fast):
//!   SMG_TEST_GIGATOKEN_TOKENIZER=/path/to/tokenizer.json \
//!     cargo test -p llm-tokenizer --release --features gigatoken \
//!     --test encode_size_sweep -- --ignored --nocapture
//!
//! gigatoken's nightly `simd` feature is not reachable from here on purpose
//! (see GIGATOKEN.md, "Known limitations"); measuring it needs a local patch of
//! the dependency on a nightly toolchain.

use std::time::Instant;

use llm_tokenizer::{huggingface::HuggingFaceTokenizer, traits::Encoder};

/// Point `SMG_TEST_GIGATOKEN_TOKENIZER` at a `tokenizer.json` (a vocabulary the
/// fast path is expected to engage on, e.g. GLM-5.2). Skips when unset so this
/// stays runnable on any host.
fn tokenizer_path() -> Option<String> {
    let p = std::env::var("SMG_TEST_GIGATOKEN_TOKENIZER").ok()?;
    std::path::Path::new(&p).exists().then_some(p)
}

/// Both tests select the backend through the process-wide
/// `SMG_TOKENIZER_BACKEND` env var; built concurrently they race and can swap
/// which instance actually gets the fast path (inverting every table). One
/// lock around the env-mutating construction window fixes that.
static BACKEND_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[expect(
    clippy::expect_used,
    reason = "test setup helper; allow-expect-in-tests only exempts #[test] fns"
)]
fn build_hf_and_giga(path: &str) -> (HuggingFaceTokenizer, HuggingFaceTokenizer) {
    let _guard = BACKEND_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    let hf = HuggingFaceTokenizer::from_file(path).expect("hf");
    std::env::set_var("SMG_TOKENIZER_BACKEND", "gigatoken");
    let giga = HuggingFaceTokenizer::from_file(path).expect("giga");
    std::env::remove_var("SMG_TOKENIZER_BACKEND");
    (hf, giga)
}

/// Bracketing the workloads: 4 KB is the size the original assessment used;
/// 176 KB is the coding replay's mean 44k-token prompt; 860 KB is its
/// 215k-token session-retirement threshold.
const SIZES: &[usize] = &[4096, 16384, 65536, 176128, 393216, 860160];

fn corpus() -> String {
    let mut out = String::new();
    // Rooted at this repo checkout (crates/tokenizer -> repo root) so the sweep
    // is reproducible on any machine; callers assert the result is non-empty so
    // a bad root fails loudly instead of measuring nothing.
    let Some(root) = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
    else {
        return out; // empty; callers' non-empty assertions fail loudly
    };
    let mut stack = vec![
        root.join("crates"),
        root.join("model_gateway/src"),
        root.join("docs"),
    ];
    while let Some(p) = stack.pop() {
        if out.len() > 24 * 1024 * 1024 {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&p) else {
            continue;
        };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
            } else if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("rs") | Some("md") | Some("py") | Some("toml")
            ) {
                if let Ok(s) = std::fs::read_to_string(&path) {
                    out.push_str(&s);
                    out.push('\n');
                }
            }
        }
    }
    out
}

fn slices(corpus: &str, size: usize, n: usize) -> Vec<String> {
    let mut v = Vec::new();
    let span = corpus.len().saturating_sub(size + 1).max(1);
    let stride = (span / n.max(1)).max(1);
    for i in 0..n {
        let mut s = (i * stride) % span;
        while s < corpus.len() && !corpus.is_char_boundary(s) {
            s += 1;
        }
        let mut e = (s + size).min(corpus.len());
        while e > s && !corpus.is_char_boundary(e) {
            e -= 1;
        }
        if e > s {
            v.push(corpus[s..e].to_string());
        }
    }
    v
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore]
fn encode_latency_by_prompt_size() {
    let Some(path) = tokenizer_path() else {
        eprintln!("skipping: set SMG_TEST_GIGATOKEN_TOKENIZER to a tokenizer.json");
        return;
    };
    let (hf, giga) = build_hf_and_giga(&path);

    let corpus = corpus();
    assert!(
        !corpus.is_empty(),
        "corpus is empty; the sweep would measure nothing (and the parity \
         assertion would pass vacuously)"
    );
    eprintln!("corpus {} MiB", corpus.len() / 1024 / 1024);
    println!(
        "\n{:>10} {:>9} {:>5} {:>12} {:>14} {:>9} {:>12}",
        "bytes", "tokens", "n", "hf_ms", "gigatoken_ms", "speedup", "saved_ms"
    );

    let mut mismatches = 0;
    for &size in SIZES {
        let n = (corpus.len() / (size + 1)).clamp(4, 32);
        let sl = slices(&corpus, size, n);
        if sl.is_empty() {
            continue;
        }

        for s in sl.iter().take(3) {
            let a = hf.encode(s, true).expect("hf");
            let b = giga.encode(s, true).expect("giga");
            if a.token_ids() != b.token_ids() {
                mismatches += 1;
            }
        }

        for s in sl.iter().take(2) {
            let _ = hf.encode(s, true);
            let _ = giga.encode(s, true);
        }

        let mut h = Vec::new();
        let mut tokens = 0;
        for s in &sl {
            let t = Instant::now();
            let e = hf.encode(s, true).expect("hf");
            h.push(t.elapsed().as_secs_f64() * 1e3);
            tokens = e.token_ids().len();
        }
        let mut g = Vec::new();
        for s in &sl {
            let t = Instant::now();
            let _ = giga.encode(s, true).expect("giga");
            g.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (hm, gm) = (median(h), median(g));
        println!(
            "{:>10} {:>9} {:>5} {:>12.2} {:>14.2} {:>8.1}x {:>12.2}",
            size,
            tokens,
            sl.len(),
            hm,
            gm,
            hm / gm,
            hm - gm
        );
    }
    assert_eq!(mismatches, 0, "token parity broke during the size sweep");
}

/// Batch-encode sweep. The HuggingFace `encode_batch` this replaces is itself
/// rayon-parallel, so this is parallel-vs-parallel — unlike the single-encode
/// sweep above, a large win here is not guaranteed.
#[test]
#[ignore]
fn batch_encode_latency_by_batch_size() {
    let Some(path) = tokenizer_path() else {
        eprintln!("skipping: set SMG_TEST_GIGATOKEN_TOKENIZER to a tokenizer.json");
        return;
    };
    let (hf, giga) = build_hf_and_giga(&path);

    let corpus = corpus();
    assert!(
        !corpus.is_empty(),
        "corpus is empty; the sweep would measure nothing"
    );
    println!(
        "\n{:>6} {:>10} {:>12} {:>14} {:>9} {:>14}",
        "batch", "doc_bytes", "hf_ms", "gigatoken_ms", "speedup", "hf_MB/s"
    );

    for &doc_bytes in &[4_096usize, 65_536] {
        // Guard the modulus below: a corpus shorter than doc_bytes+1 would
        // underflow (release wraps -> effectively an infinite char-boundary
        // walk; debug panics). Skip sizes the corpus cannot cover.
        let Some(span) = corpus.len().checked_sub(doc_bytes + 1) else {
            eprintln!("corpus too small for doc_bytes={doc_bytes}; skipping");
            continue;
        };
        for &batch in &[1usize, 2, 8, 32, 128] {
            // Distinct slices so neither backend can benefit from a repeat.
            let docs: Vec<String> = (0..batch)
                .map(|i| {
                    let start = (i * 7_919 * doc_bytes) % span.max(1);
                    let mut a = start;
                    while !corpus.is_char_boundary(a) {
                        a += 1;
                    }
                    let mut b = a + doc_bytes;
                    while !corpus.is_char_boundary(b) {
                        b -= 1;
                    }
                    corpus[a..b].to_owned()
                })
                .collect();
            let refs: Vec<&str> = docs.iter().map(String::as_str).collect();

            // Warm both (pool forks, page faults) outside the timed region.
            for _ in 0..2 {
                let _ = hf.encode_batch(&refs, false).expect("hf warm");
                let _ = giga.encode_batch(&refs, false).expect("giga warm");
            }

            let reps = if batch * doc_bytes > 1 << 22 { 3 } else { 10 };
            let mut hf_ms = Vec::new();
            let mut g_ms = Vec::new();
            for _ in 0..reps {
                let t = Instant::now();
                let _ = hf.encode_batch(&refs, false).expect("hf");
                hf_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let _ = giga.encode_batch(&refs, false).expect("giga");
                g_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (h, g) = (median(hf_ms), median(g_ms));
            let mb = (batch * doc_bytes) as f64 / 1e6;
            println!(
                "{:>6} {:>10} {:>12.2} {:>14.2} {:>8.1}x {:>14.1}",
                batch,
                doc_bytes,
                h,
                g,
                h / g,
                mb / (h / 1e3)
            );
        }
    }
}
