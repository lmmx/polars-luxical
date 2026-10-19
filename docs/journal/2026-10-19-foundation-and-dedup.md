# 2026-10-19: Foundation and Document Deduplication

This entry covers the first development pass through the roadmap in
`2026-10-17-state-of-the-library.md`: the remaining Foundation tasks, then Core Utility
up to a working deduplication API with a large-corpus run. It is stacked on PR #26,
which fixed the build (ethnum) and made the tests able to fail.

## Foundation

**Parity with the reference implementation.** Before changing anything, I installed
`luxical` 1.1.2 from PyPI in an isolated venv and embedded all 708 PEPs with it, using
the `luxical_one_rc4.npz` weights. polars-luxical matched it to a max absolute
difference of 3e-6 per dimension (minimum cosine 0.9999998). That settles the open
question from `test-suite-health`: the hand-ported FNV hashing, padding sentinel, TF-IDF
and layer normalisation are faithful. Eight reference embeddings now live in
`tests/data/upstream_parity.json`, and `tests/parity_test.py` checks them to 1e-5. They
cover short strings, unicode, a repeated sentence, and PEP excerpts and full PEPs. The
fixture guarded every later change in this pass. Upstream took 9.4s for the PEPs; this
plugin took 3.1s before the performance work below.

**The Test workflow runs again.** The file had three problems that would have broken
it:
- it triggered on pushes to `main`, but the default branch is `master`;
- it ran `maturin develop` without installing the dev group that provides maturin;
- each of its 12 matrix jobs would download the ~900 MB model.

It now builds through `uv sync`, which runs the maturin backend. A new
`POLARS_LUXICAL_CACHE_DIR` variable points the model cache at a directory that
`actions/cache` restores across OSes. All 12 jobs (Linux, macOS arm/x86 and Windows,
each on Python 3.11–3.13) pass.

**The Rust unit tests run in CI too.** `cargo test` failed to link because pyo3's
`extension-module` feature leaves libpython symbols unresolved. That feature is now a
default cargo feature, so maturin builds are unchanged and `cargo test
--no-default-features` links against libpython. On the runners the uv-managed
interpreter's `LIBDIR` has to be on `LD_LIBRARY_PATH`. This container masked the problem
because it has a system libpython3.11; the first CI run exposed it.

**The doc-half benchmark is honest now.** As suspected in `doc-half-benchmark`, the old
97.46% top-1 measured first half → whole document. With the halves embedded separately
the real numbers are:

| corpus | method | top-1 | top-1% | mean rank |
|---|---|---:|---:|---:|
| 50,000 FineWeb docs (≥100 words) | Luxical-One | 64.3% | **97.5%** | **101** |
| | word 1–2 gram TF-IDF | **76.8%** | 95.7% | 475 |
| 5,000 FineWeb docs | Luxical-One | 79.4% | **97.2%** | **11.1** |
| | TF-IDF | **85.2%** | 95.5% | 53.8 |
| 708 PEPs | Luxical-One | 68.5% | 87.9% | 10.5 |
| | TF-IDF | **87.6%** | **98.0%** | **2.9** |
| 708 PEPs, half → whole (old method) | Luxical-One | 97.5% | 99.9% | 1.05 |

This is the shape the Luxical report describes. Luxical is weaker than lexical matching
at pinpointing the exact partner, but it reliably puts the partner in the right
neighbourhood on web text. On the PEPs, a small technical corpus far from FineWeb where
TF-IDF is fitted in-domain, plain TF-IDF wins outright. The README now carries this
table instead of the inflated figure.

## Inference performance

A 1M-document run needed the work in `inference-performance`. The changes:
- **Per-document parallelism.** Each document goes through tokenisation, n-gram
  hashing, TF-IDF weighting and the first layer in a rayon task, and the dense layers
  run in parallel 256-row chunks.
- **Transposed first layer.** The first layer is stored as vocab × 96, built once at
  load straight from the safetensors mmap, so each active n-gram reads one contiguous
  384-byte row instead of 96 loads spaced 8 MB apart.
- **No per-n-gram allocations.** N-gram hashing streams FNV state with no allocations,
  and the n-gram index uses an identity hasher, since its keys are already FNV hashes.
- **Fewer dependencies.** The `sprs` CSR intermediate and `src/tfidf.rs` are gone.

Outputs are bitwise identical to the previous build on all 708 PEPs, because the
arithmetic order is preserved.

| PEPs, 4-core container | before | after |
|---|---:|---:|
| default 4-chunk frame | 2.46s | 1.67s |
| single-chunk column | 8.50s | 1.41s |

The single-chunk case went from single-threaded to parallel, which is what a large
parquet read hits. Tokenisation is now about 70% of the time: the HF tokenizers BERT
normaliser runs at about 0.9M tokens/s per thread. A faster path would mean
reimplementing BERT normalisation, which risks parity, so it was left alone.
`encode_fast` made no difference.

## Deduplication

The API lives on the DataFrame namespace:
- `mark_duplicates` adds `duplicate_of` and `duplicate_of_similarity`;
- `dedup` keeps the first row of each group;
- `near_duplicates` lists every pair at or above the threshold.

The duplicate rule is SemDeDup's: row j is a duplicate if any earlier row has cosine
≥ threshold. The reported partner is the most similar earlier row. Two search
strategies share that rule:
- **Exact:** blocked all-pairs similarity on BLAS. It covers 20k documents in about
  3.3s, and `auto` uses it up to 50k rows.
- **Clustered:** spherical k-means with one cluster per ~2,000 rows, fitted on at most
  200k sampled rows, then exact search within each cluster.

The implementation is numpy rather than Rust. The heavy work is dense matrix products,
which BLAS already does well, and nothing on the Rust side would beat it without
pulling in a BLAS binding.

**Two assignments per row.** With one cluster per row, about 4% of near-duplicate
pairs on FineWeb were split across clusters and missed. With each row searched in its
2 nearest clusters, 99.4–100% of pairs at ≥0.95 shared a cluster, measured across
k = 10, 50 and 200. The within-cluster work rises about 4x, which is still seconds.
`assignments=2` is the default.

**Calibrating the threshold** (`benchmark/dedup_eval.py`). The setup:
- 20,000 FineWeb documents, plus 1,000 copies of randomly chosen ones;
- each copy gets one of five crawl-style edits: truncation to 60–90%, dropping
  10–25% of sentences, cookie/nav boilerplate, ~5% word deletions or substitutions,
  or a paragraph shuffle;
- the TF-IDF baseline is the same word 1–2 gram cosine.

| threshold | Luxical recall | untouched docs flagged | of which TF-IDF ≥ 0.5 | TF-IDF recall | TF-IDF flagged |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 99.6% | 9,209 | 26 | 97.5% | 2 |
| 0.90 | 99.4% | 442 | 13 | 78.0% | 2 |
| 0.95 | 98.2% | 33 | 5 | 52.6% | 2 |
| 0.97 | 95.3% | 12 | 4 | 34.6% | 2 |
| 0.98 | 90.5% | 7 | 4 | 26.9% | 2 |

The two methods' scales are not comparable at the same number. TF-IDF at 0.80 is
already well separated, with 97.5% recall and 2 flagged originals. Luxical's
similarities are compressed near the top. Reading sampled pairs by band:
- **≥0.98:** true duplicates, including some TF-IDF misses: a spun "Bluehost" article
  (TF-IDF 0.68) and dumpster-rental template pages (TF-IDF 0.33).
- **0.96–0.98:** a mix of template duplicates, near-identical privacy policies and
  same-genre spam (essay mills, replica watches, casino reviews).
- **0.90–0.94:** mostly different documents on the same topic.

I set the default to **0.95**, where recall on edited copies is 98.2% and 0.17% of
untouched documents are flagged, mostly templated, spun or same-genre spam pages. For
data curation that is SemDeDup's intent: removing semantically redundant pages, not
only copies. For strict near-duplicate detection, use 0.98 or higher. The README shows
the table so users choose with the trade-off in view.

## The 1M-document run

`benchmark/dedup_fineweb.py` on FineWeb sample-10BT shard `000_00000.parquet`, on this
4-core container:

| stage | time |
|---|---:|
| read 1,048,581 documents (726M GPT-2 tokens) | 5.1s |
| load model | 1.6s |
| embed | 502.2s (2,088 docs/s, 6.1 MiB/s) |
| `mark_duplicates(method="clustered")`, 524 clusters × 2 assignments | 152.4s |
| **total** | **661.3s (11 min)** |

| threshold | marked as duplicates | share |
|---:|---:|---:|
| 0.90 | 211,911 | 20.2% |
| 0.95 | 21,441 | 2.04% |
| 0.97 | 5,755 | 0.55% |
| 0.98 | 2,714 | 0.26% |
| 0.99 | 1,224 | 0.12% |

Sampled pairs by band:
- **≥0.99:** spun articles across URLs on one site (sonagoldira.com), templated review
  pages (two mattress "updated reviews" pages), lift-ticket pages identical except for
  the resort (onthesnow.com), a LASIK article copied between two blogs.
- **0.97–0.99:** dumpster-rental template pages across sister domains, casino and SEO
  spam from unrelated sites.
- **0.95–0.97:** web-hosting adverts, payday-loan pages, GDPR statements, and two
  different news articles on Bay Area home sales.
- **0.90–0.95:** different documents on a shared topic (K-beauty products, music and
  mathematics).

The flagged share at a fixed threshold grows with corpus size: 0.17% at 0.95 on 20k
documents, 2.04% on 1M. Every document has more earlier documents to resemble, and
real cross-crawl duplicates accumulate. The calibration table above was measured at 20k
and should be read that way.

The 6.1 MiB/s is not comparable with the report's 23.1 MiB/s, which was measured on an
M4 Max. Recall of clustered search at this scale (k=524) was not measured directly; the
top-2 check covered k up to 200.

## What this means for direction

The library is now trustworthy (parity-tested, CI-tested, honest benchmarks) and does
practical deduplication at million-document scale on a laptop-class CPU. Two findings
bear on what comes next, and both are decisions for the project rather than this pass:

1. **Lexical near-duplicates.** On copies with lexical edits, a lexical method (TF-IDF
   here; MinHash-LSH is the industry standard and was not tested) separates duplicates
   from non-duplicates more cleanly than Luxical. Luxical's distinct value is
   semantic: spun and templated pages that lexical similarity misses, at a cost that
   scales. The same embeddings also serve clustering and classification. A combined
   mode, with Luxical for candidates and a lexical check to confirm, is one option.
   Positioning dedup as SemDeDup-style curation, as the README now does, is another.
2. **Short documents and short queries.** Unchanged from the previous entry: not
   addressed, and still out of scope by the roadmap.

## Current State

- `tests/parity_test.py` compares embeddings for eight texts against `luxical` 1.1.2 output stored in `tests/data/upstream_parity.json` with `atol=1e-5`
- `tests/conftest.py` provides a `model_id` fixture that registers luxical-one and skips only on `RuntimeError` from `register_model` (tests/conftest.py:13-22)
- The Test workflow triggers on pushes to `master` and on pull requests, and runs 12 jobs across Linux, macOS arm64, macOS x86_64 and Windows on Python 3.11-3.13 (.github/workflows/test.yml)
- `test-platform.yml` restores model weights with `actions/cache` keyed `luxical-one-safetensors-v1`, builds with `uv sync --frozen --group test`, and runs `cargo test --release --no-default-features` on Linux with the interpreter LIBDIR on `LD_LIBRARY_PATH` (.github/workflows/test-platform.yml:50-72)
- `cache_dir` returns `POLARS_LUXICAL_CACHE_DIR` when set, otherwise `dirs::cache_dir()/polars-luxical` (src/registry.rs:40-47)
- pyo3 `extension-module` is a default cargo feature named `extension-module` (Cargo.toml)
- `embed_batch` tokenises and projects each document through the first layer in parallel with rayon, then runs the dense layers over 256-row chunks in parallel (src/embedder/inference.rs:15-41, src/embedder/inference.rs:78-107)
- `LuxicalEmbedder::new` stores the first layer transposed as vocab_size x hidden_dim using a blocked parallel transpose (src/embedder/model.rs:57-79, src/embedder/model.rs:131-148)
- `for_each_ngram_hash` streams FNV-1a state per n-gram start without allocating, and a Rust unit test checks the hashes against the padded `hash_ngram` reference (src/ngrams.rs:47-60, src/ngrams.rs:124-138)
- The n-gram index is a `HashMap<i64, u32>` with an identity hasher (src/ngrams.rs:71-90, src/embedder/model.rs:17)
- `mark_duplicates`, `dedup` and `near_duplicates` are DataFrame namespace methods with default threshold 0.95 (python/polars_luxical/__init__.py:27, python/polars_luxical/__init__.py:232-349)
- `most_similar_earlier` and `similar_pairs` run exact blocked search up to 50,000 rows and clustered search above that, with each row assigned to its 2 nearest spherical k-means clusters (python/polars_luxical/dedup.py:24-26, python/polars_luxical/dedup.py:163-219)
- `retrieve` accepts a string, a list of strings (mean embedding) or an embedding vector (python/polars_luxical/__init__.py:142-222)
- `benchmark/doc_half_match_demo.py` embeds first and second halves separately and reports Luxical half->half, Luxical half->whole and TF-IDF half->half on PEPs or a FineWeb shard
- `benchmark/dedup_eval.py` reports recall on injected edited copies and originals flagged per threshold for Luxical and TF-IDF
- `benchmark/dedup_fineweb.py` embeds and marks duplicates across a full FineWeb shard with timings

## Missing

- No lexical (MinHash or TF-IDF) confirmation step in the dedup API (python/polars_luxical/dedup.py)
- No comparison against MinHash-LSH in `benchmark/dedup_eval.py`
- No faster tokenisation path; HF tokenizers' BERT normalisation accounts for about 70% of embedding time (src/embedder/inference.rs:28-32)
- The registry still recognises only luxical-one by Hugging Face ID (src/registry.rs:20-31)
