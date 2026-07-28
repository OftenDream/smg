# gigatoken prompt-encode fast path

Optional, off by default. Replaces **only** `Encoder::encode` (prompt
tokenization) with [gigatoken](https://github.com/marcelroed/gigatoken)
(Marcel Rød, MIT). Decode, chat templating and special-token handling stay on
`tokenizers`.

## Why

SMG re-tokenizes the whole prompt on every request. Measured on the gateway,
that cost is linear in prompt size at **1.13–1.58 ms per 1,000 prompt tokens**.
It matters because the engine's prefix cache absorbs the *prefill* while the
gateway still pays full tokenization, so on cache-heavy traffic it is a large
share of TTFT.

## Enabling it

Two independent switches — both are required, which is deliberate:

1. **Compile it in:** `cargo build --release -p smg-python --features gigatoken`
2. **Select it at runtime:** `SMG_TOKENIZER_BACKEND=gigatoken`

Optional: `SMG_TOKENIZER_TIMING=1` logs rolling encode latency for whichever
backend is active (both go through the same timing path, so an A/B is directly
comparable).

Optional: `SMG_GIGATOKEN_SLOTS=N` sizes the encoder pool (default **8**). Each
slot is a forked gigatoken instance whose pretoken cache is sized by the
*vocabulary* and committed at load (~8 MiB of faulted RSS per slot on a ~150k
vocab) — so the pool costs memory per tokenizer, per model. With encode at
~1 ms slots are held ~45x shorter than the HuggingFace path they replace, so 8
covers heavy concurrency; raise it only if `gigatoken` encode timing shows lock
contention, and bound it on memory-tight router pods.

gigatoken's own nightly-only `simd` feature is deliberately not reachable from
this crate — see "Performance" for the measurement and "Known limitations" for
why it cannot be exposed as a cargo feature at all.

## Safety contract

The fast path does not implement `add_special_tokens`, and a tokenizer whose
post-processor injects BOS/EOS would therefore produce different tokens. So it
**proves equivalence at load time before it is allowed to serve traffic**:

- It encodes 13 probe strings (ASCII, code punctuation, CJK, multi-codepoint
  emoji, combining marks, RTL, control bytes, chat-template markers) and
  requires byte-identical output against `tokenizers`.
- For every probe it also requires `encode(text, true) == encode(text, false)`,
  i.e. that no post-processor adds tokens. If any probe disagrees it logs the
  reason and stays on the HuggingFace path.
- A tokenizer that configures **truncation or padding** disables the fast path
  outright: HuggingFace applies both in `post_process` regardless of
  `add_special_tokens`, gigatoken applies neither, and no short probe can trip
  a `max_length` — so this is gated on the tokenizer config itself, not on
  probe content.
- A **poisoned pool slot** (a panic escaped a prior encode) is permanently
  retired, never reused — its caches may be mid-mutation. The pool self-heals
  by shrinking; if every slot is poisoned, encode routes back to HuggingFace.
- The parallel **ragged batch contract** (one row per input, `sum(lens) ==
  ids.len()`) is validated per call; a violation falls back to per-input
  encode rather than slicing a misaligned buffer.

**The failure mode is always "fall back", never "serve different tokens".**
That invariant is what the test suite asserts, rather than asserting that the
fast path engaged — see `tests/gigatoken_fastpath.rs`.

## Testing

```bash
# Contract tests. Run in CI with no model weights (uses the cached TinyLlama
# tokenizer, whose post-processor adds BOS -- so this also covers the refusal
# path). Run them with the feature both off and on.
cargo test -p llm-tokenizer --test gigatoken_fastpath
cargo test -p llm-tokenizer --features gigatoken --test gigatoken_fastpath

# Additionally cover a vocabulary the fast path actually engages on:
SMG_TEST_GIGATOKEN_TOKENIZER=/path/to/glm-5.2/tokenizer.json \
  cargo test -p llm-tokenizer --features gigatoken --test gigatoken_fastpath

# Latency sweep (ignored by default; uses SMG's shipped release profile).
SMG_TEST_GIGATOKEN_TOKENIZER=/path/to/tokenizer.json \
  cargo test -p llm-tokenizer --release --features gigatoken \
  --test encode_size_sweep -- --ignored --nocapture
```

## Performance

Measured on GLM-5.2's vocabulary, in SMG's shipped release profile
(`opt-level = "z"`, fat LTO), stable toolchain, scalar scanners:

| prompt bytes | tokens | HuggingFace | gigatoken | speedup |
|---|---|---|---|---|
| 4,096 | 1,006 | 1.12 ms | 0.04 ms | 27.1x |
| 16,384 | 3,824 | 4.47 ms | 0.12 ms | 37.3x |
| 65,536 | 13,918 | 17.24 ms | 0.40 ms | 42.6x |
| 176,128 | 37,156 | 45.59 ms | 0.97 ms | 47.2x |
| 860,160 | 187,252 | 218.90 ms | 4.07 ms | 53.7x |

End-to-end on GLM-5.2-NVFP4 (TP8, B200, a long-prompt coding workload), against
the mean of two baseline runs: **TTFT p50 −14% / −21% / −21% / −8.5%** across the
four load points, throughput neutral except **−1.8% at saturation**. The TTFT
gain is largest when the GPU is not the bottleneck, which is what you would
expect from removing gateway work.

### Batch encode

`Encoder::encode_batch` is also routed. Note the HuggingFace `encode_batch` it
replaces is *itself* rayon-parallel, so this is parallel-vs-parallel; gigatoken
uses its ragged shard-and-gather path (`encode_docs_ragged`):

| batch | doc bytes | HuggingFace | gigatoken | speedup |
|---|---|---|---|---|
| 1 | 4,096 | 1.09 ms | 0.05 ms | 21.6x |
| 8 | 4,096 | 1.70 ms | 0.13 ms | 13.3x |
| 128 | 4,096 | 11.16 ms | 1.57 ms | 7.1x |
| 1 | 65,536 | 18.46 ms | 0.73 ms | 25.3x |
| 32 | 65,536 | 40.42 ms | 6.28 ms | 6.4x |
| 128 | 65,536 | 158.05 ms | 11.09 ms | 14.2x |

The speedup *narrows* as the batch grows (21.6x → 7.1x at 4 KB) because HF's
rayon parallelism amortises better at larger batches — expected, not a defect.

**However: `encode_batch` currently has no production callers.** Every request
path — chat, completion, embedding, generate, messages — calls `encode` once per
request (the embedding stage handles a single text). So routing it is
correctness/future-proofing and contributes **zero** end-to-end gain today. It
becomes real the moment a multi-input embedding or a batched prefill path lands.
Do not quote the table above as an end-to-end improvement.

gigatoken's `simd` feature measured **within noise** of the stable scalar build
on this vocabulary (4.12 vs 4.07 ms at 860 KB). Note GLM-5.2 uses a
tiktoken-style BPE while gigatoken's `portable_simd` code lives in its
*sentencepiece* path, so that comparison does not show SIMD is useless in
general — only that it is not on the hot path here, where it never executes.

## Known limitations

- **Encode only.** No streaming/incremental decode API, so `decode_step` and
  all decode stay on `tokenizers`.
- **WordPiece is unsupported**, so BERT-family embedding/classify models must
  not enable this. They will fail the load-time self-check and fall back, but
  do not enable it for them in the first place.
- **Not on crates.io.** Consumed via a build-packaging mirror of
  [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) (see the
  dependency comment in `Cargo.toml` for the exact delta). Pinned by git rev,
  never a branch, so encoder output stays reproducible across builds.
- **gigatoken's `simd` feature cannot be exposed as a cargo feature.** It turns
  on `#![feature(portable_simd)]`, which is nightly-only, and CI lints with
  `cargo clippy --all-targets --all-features`. `--all-features` switches on
  every declared feature unconditionally, so merely *declaring* a passthrough
  breaks the stable build with `error[E0554]`. Feature-gating a nightly-only
  attribute does not protect it from `--all-features`. To measure it, patch the
  dependency locally against a nightly toolchain; exposing it properly would
  need gigatoken to gate that attribute on a build-script channel probe
  (e.g. `#![cfg_attr(gigatoken_nightly, feature(portable_simd))]`) rather than
  on a cargo feature.
- The upstream project is young; treat a rev bump as a change that needs the
  parity suite re-run, not a routine dependency update.
