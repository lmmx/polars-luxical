# 2026-10-17: State of the Library

This entry reviews polars-luxical ten months after it was written. The library has had no
development since the v0.1.2 release on 2025-12-16. Every later commit on `master` is a
pre-commit-ci hook autoupdate (#7 through #24). The entry covers three things: what the
library is, what the Luxical model can and cannot do, and what has happened upstream since.
Each development task identified here has its own journal entry dated the same day (listed
at the end).

## What the library is

polars-luxical is a Rust Polars plugin (pyo3 + pyo3-polars, built with maturin). It
reimplements Luxical inference natively so `DatologyAI/luxical-one` can run inside Polars
expressions:

- `embed_text(expr, model_id=...)` is an elementwise plugin expression that returns
  `Array[f32, 192]`.
- `df.luxical.embed(columns, model_name, output_column, join_columns)` embeds one column, or
  several columns concatenated with spaces.
- `df.luxical.retrieve(query, ...)` embeds the query string, cross-joins it against every row,
  computes cosine similarity (via `polars-distance`) or a dot product, then sorts and takes
  the top-k.
- `register_model`, `clear_registry` and `list_models` manage a global model registry, held
  in a `Lazy<RwLock<HashMap<String, Arc<LuxicalEmbedder>>>>`.

The inference pipeline in `src/embedder/inference.rs` has four steps:

1. BERT-uncased tokenization through the HF `tokenizers` crate.
2. Extraction of every 1- to 5-gram of token IDs. Each n-gram is padded with a `u32::MAX`
   sentinel and FNV-1a hashed (byte-compatible with the Python implementation,
   `src/ngrams.rs`), then looked up in a hash→vocab-index map.
3. Counts → TF-IDF → L2 normalization (`src/tfidf.rs`).
4. A sparse×dense first layer, then dense ReLU layers with L2 normalization after each,
   then a final projection to 192 dims.

Weights load from safetensors (mmap) or `.npz`. Both are downloaded from Hugging Face into a
per-OS cache directory.

The speed numbers are the reason the library exists. The PEP-corpus benchmark covers 708
PEPs, about 3.6M tokens. It runs embed plus retrieve in about **1.8s, roughly 0.5ms per 1k
tokens**. polars-fastembed with `Xenova/all-MiniLM-L6-v2` takes 30s on CPU and 20s on GPU for
the same job (`benchmark/README.md`).

## What the model actually is (from the tech report)

Luxical is described in "Luxical: High-Speed Lexical-Dense Text Embeddings" (Merrick, Fang,
Carranza et al., arXiv 2512.09015, v1 9 Dec 2025, v2 11 Dec 2025 fixing typos only). The
report presents it as an *organizing* tool for web-scale text, sitting between FastText,
which is fast but only yields classifier scores, and transformer embedders, which are
flexible but expensive.

- **Features:** a TF-IDF-weighted bag over a fixed vocabulary of about 2M n-grams of up to 5
  tokens, mined from a FineWeb sample with the Space-Saving heavy-hitters algorithm. Log-scaled
  IDF, then ℓ2 normalization.
- **Network:** a small ReLU MLP on top of the sparse input, with ℓ2 normalization after each
  projection. The upstream training script's default dims are `(96, 3072, 3072, 192)`
  (`projects/luxical_one/step_3_train_luxical_one.py` in `datologyai/luxical`): the
  2M-feature sparse input projects to 96, then 3072, 3072, and finally 192. Nearly all of
  the roughly 0.2B parameters Hugging Face lists sit in the 2M × 96 first layer, about 192M
  weights or about 768 MB as f32. The dense layers add about 10M.
- **Teacher:** `Snowflake/snowflake-arctic-embed-m-v2.0`, which produces 256-dim embeddings.
- **Training data:** 50M English FineWeb documents, embedded offline by the teacher. Three
  epochs, CPU only.
- **Objective:** Gram-matrix distillation. Per batch, the student's and the teacher's pairwise
  similarity matrices are built and their diagonals removed. The loss is τ²·KL between
  temperature-scaled rows (τ = 3.0, peak LR 0.01, Adam, warmup-stable-decay schedule). The
  paper reports a batch of 3072. The upstream script defaults to `12 * 1024`, so the two
  sources disagree. The student learns to reproduce the teacher's *document–document similarity
  structure*. It is not trained to reproduce the teacher's vectors one by one.
- **Evaluations:** two, with no MTEB run.
  1. Document-half matching on 50k FineWeb documents (100k halves). Luxical-One trails
     MiniLM-L6-v2 at strict top-1. As the retrieval window widens it beats MiniLM and
     approaches the transformers.
  2. An end-to-end data-curation run. A 600B-token FineWeb pool was filtered to 60B tokens
     by each scorer:

     | Scorer | Throughput | MMLU of 3B model trained on the filtered data |
     |---|---|---|
     | Luxical-One + small MLP (2×256 hidden) | 23.1 MiB/s | 36.4% |
     | DCLM FastText | 19.0 MiB/s | 36.4% |
     | FineWeb-Edu classifier | 1.6 MiB/s | similar |

     MLP inference was under 0.25% of the Luxical pipeline's runtime.
- **Speed:** the abstract reports 3x–100x speedups over neural baselines of varying size.
  Qwen3-0.6B lags by nearly two orders of magnitude even on GPU.
- **Stated limitations:** no fine-grained positional information, which makes the model less
  suited to high-precision search or reasoning-heavy tasks. English only. The evaluation scope
  is narrow: one model, one symmetric similarity task, one curation task.
- **Stated future work:** distilled variants, multilingual models, tighter integration with
  curation systems, and connections to static embedding models.

## Why it works for dedup but not for search

The README states it plainly: "they were not trained on queries, so you cannot use them for
search!" (`README.md:15`). The mechanism behind that sentence has three parts.

1. **The training signal is purely symmetric.** Every training example is a FineWeb
   document compared against other FineWeb documents. arctic-embed-m-v2.0 is an asymmetric
   retrieval model with a dedicated query mode. The upstream teacher wrapper even defines
   `QUERY_PREFIX = "query: "` and an `is_query` flag
   (`src/luxical/teacher_embedder.py:47-48, 109-134`). But step 1 of the Luxical-One
   pipeline calls the teacher with `is_query=False` only
   (`projects/luxical_one/step_1_teacher_embed_training_docs.py:111`). The student was only
   shown the teacher's document side, so whatever the teacher knows about mapping short
   queries near their answering documents never reached the student.
2. **Short inputs are out of distribution.** A 5-word query activates a handful of n-gram
   features, where a web document activates hundreds or thousands. After TF-IDF and ℓ2
   normalization, those few features dominate the vector entirely. The MLP is then
   extrapolating from an input region it barely saw during training.
3. **Bags of n-grams cannot bridge paraphrase from almost no evidence.** "How do computers
   understand language?" and a paragraph about NLP share few n-grams. A transformer bridges
   that gap with contextual semantics. A lexical-dense model only has the learned projection
   of the few n-grams the query actually contains.

The practical rule follows from this: **the query has to look like a document.** A full
document works well, and so does a centroid of a few seed documents. Datology's own
framing is document-to-document: clustering, classification, semantic deduplication,
coarse organization such as separating teaching materials from videogame reviews.

## Use cases, ranked by payoff

1. **Semantic deduplication.** Embed, k-means into clusters, then within each cluster drop
   pairs above a cosine threshold. This is the SemDeDup recipe, which the paper cites. At
   about 0.5ms per 1k tokens a laptop can dedup a corpus that would otherwise need a GPU.
2. **Cheap classifiers on frozen embeddings.** This is the paper's headline result. A tiny MLP
   on Luxical embeddings matched FastText's downstream MMLU and outran it on throughput. The
   pattern: label a few thousand documents with an LLM judge, train the MLP to copy those
   labels, then apply it to millions of documents.
3. **Clustering and corpus mapping.** Topic maps, cluster-balanced or stratified sampling,
   diversity subset selection (k-center greedy, facility location), and mixture reweighting.
4. **"More like this" search.** The query is a seed document, or the mean of several seed
   embeddings. Optionally a transformer reranks the shortlist, with Luxical as the cheap
   first stage. The doc-half results show this mode works.
5. **Contamination and overlap checks.** Benchmark test sets embedded against a training
   corpus. Two dataset versions compared by cluster makeup.
6. **Drift monitoring inside Polars pipelines.** Batch-to-batch shift in embedding centroids
   or cluster occupancy.

There is a caveat on input length. The model is built for whole documents. Short strings
such as tweets, log lines and ticket titles fall into the same out-of-distribution regime as
queries.

## What has happened upstream since December 2025

- **No new Luxical models.** The DatologyAI Hugging Face org lists only `luxical-one`, last
  updated 2025-12-12, plus two CLIP models.
- **The upstream repo is effectively dormant.** `datologyai/luxical` carries a "No Planned
  Active Maintainance" notice: Datology has limited capacity to monitor it, accept PRs or
  maintain it. Its last release is v1.1.2 (2025-11-24), and its last commit is a README
  update on 2025-12-12. The repo does ship the full training pipeline:
  `src/luxical/training.py`, `teacher_embedder.py`, and
  `projects/luxical_one/step_1…step_3`, covering teacher embedding, n-gram mining and
  student training. That makes retraining feasible.
  It supports only macOS and Linux on Python 3.11–3.13, because of its Rust `arrow-tokenize`
  extension. polars-luxical is pure Rust, so it builds Windows wheels as well.
- **MTEB integration is unfinished.** MTEB issue #3722 ("Add model: DatologyAI/luxical-one")
  was opened 2025-12-12 and is still open with no scores. On 2026-08-08 a contributor proposed
  adding the model through the Sentence Transformers wrapper. On 2026-10-16 a maintainer told
  them to take it. A Luxical author commented that the model "was not designed to compete on
  most MTEB tasks". The MTEB maintainer asked for a retrieval comparison against model2vec,
  which nobody has published.
- **Datology's attention is on its curation product.** Curation Studio launched 2026-10-09
  (with a self-reported 6x compute reduction). A 2026 Datology blog post cites the Luxical
  paper but announces no new model.
- **The nearest fast models that handle queries are static embeddings trained for
  retrieval.** `minishlab/potion-retrieval-32M` (Model2Vec) reports 35.06 on MTEB retrieval
  against 42.92 for all-MiniLM-L6-v2, about 82% of the transformer, all self-reported.
  sentence-transformers' `static-retrieval-mrl-en-v1` reports 34.95. These occupy the
  query-capable fast niche that Luxical does not.

So the direction is unchanged from December: a lexical-dense, document-level, CPU-first
embedder. Nobody has pushed it further, and no one has released a query-capable variant.

## Critique of the repo as it stands

The README gets the headline right: document-level similarity is in, search is out. But the
repo undercuts that framing in several places, and a few claims do not hold up on close
reading. Each point has its own task entry.

1. **The examples show the use case the README rules out.** The README Usage block
   (`README.md:77-84`), `demo.py:41-52`, the `benchmark` entry point
   (`benchmark/src/benchmark/__init__.py:18`) and `tests/embed_test.py` all demonstrate
   `retrieve(query=<short string>)`. See `2026-10-17-query-retrieval-framing.md`.
2. **The doc-half benchmark measures an easier task than the one it quotes.**
   `doc_half_match_demo.py` embeds each first half concatenated with its second half, so the
   97.46% top-1 figure is for first-half → whole-document retrieval. Datology's task is
   half → half. See `2026-10-17-doc-half-benchmark.md`.
3. **The test suite cannot fail on assertions.** Every test wraps its body in
   `except Exception: pytest.skip(...)`, which also catches `AssertionError`. The dimension
   test asserts width 256 against a 192-dim model. The Test workflow is disabled. See
   `2026-10-17-test-suite-health.md`.
4. **Model loading is hardwired to luxical-one.** The registry, the cache filenames and the
   output dtype all assume a single model. That blocks any custom or future model. See
   `2026-10-17-custom-model-loading.md`.
5. **Inference is single-threaded with a strided first-layer access pattern.** `rayon` is a
   dependency but nothing uses it, so there is headroom beyond the 1.8s. See
   `2026-10-17-inference-performance.md`.
6. **The namespace offers retrieval but no dedup, clustering or classifier helpers**, even
   though those are the model's strengths. See `2026-10-17-dedup-cluster-helpers.md` and
   `2026-10-17-classifier-recipe.md`.

The rest are research threads, not repairs:

- `2026-10-17-query-aware-distillation.md`: training a Luxical variant that handles queries.
- `2026-10-17-static-embedding-comparison.md`: the MTEB comparison nobody has run.
- `2026-10-17-quantization.md`: uint8 storage and its effect on dedup thresholds.

## Sources

- DatologyAI blog, "Introducing Luxical Embeddings and the Luxical-One Model":
  https://www.datologyai.com/blog/introducing-luxical-embeddings
- Luxical technical report, arXiv 2512.09015: https://arxiv.org/abs/2512.09015
- DatologyAI Hugging Face org: https://huggingface.co/DatologyAI
- Upstream library: https://github.com/datologyai/luxical
- MTEB issue #3722: https://github.com/embeddings-benchmark/mteb/issues/3722
- potion-retrieval-32M: https://huggingface.co/minishlab/potion-retrieval-32M
- Model2Vec: https://github.com/MinishLab/model2vec
- Curation Studio coverage:
  https://fourweekmba.com/ai-datologyai-launches-curation-studio-claims-6x-less-compute/

## Current State

- `embed_text` runs Luxical inference as a Polars plugin expression and returns `Array[f32, dim]` with nulls preserved for null inputs (src/expressions.rs:26-85)
- Inference tokenizes with BERT uncased, hashes 1-to-5-grams with FNV-1a, applies TF-IDF with L2 normalization, then runs the MLP forward pass (src/embedder/inference.rs:14-41, src/ngrams.rs, src/tfidf.rs)
- `register_model`, `clear_registry` and `list_models` manage a global lowercase-keyed model registry (src/registry.rs:50-86)
- Model weights load from safetensors via mmap or from `.npz`, downloaded from Hugging Face into `dirs::cache_dir()/polars-luxical` on first use (src/registry.rs:119-191, src/embedder/model.rs:49-80)
- `df.luxical.embed` and `df.luxical.retrieve` form the DataFrame namespace, with `retrieve` computing similarity over a cross join of the query embedding against every row (python/polars_luxical/__init__.py:89-190)
- The PEP benchmark embeds and retrieves 708 PEPs (about 3.6M tokens) in about 1.8s, against 30s CPU and 20s GPU for polars-fastembed with MiniLM-L6-v2 (benchmark/README.md)
- The latest release is v0.1.2 from 2025-12-16, and all later commits on `master` are pre-commit-ci hook autoupdates #7 through #24 (git log)
- DatologyAI publishes `luxical-one` as the only Luxical model as of 2026-10-17, last updated 2025-12-12 (huggingface.co/DatologyAI)

## Missing

- The DataFrame namespace has no deduplication, clustering or classification helper (python/polars_luxical/__init__.py:88-190 defines only `embed` and `retrieve`)
- The model registry recognises no model ID besides `DatologyAI/luxical-one` and `luxical-one` (src/registry.rs:20-31)
- The repository has no MTEB or cross-model quality evaluation of polars-luxical embeddings (benchmark/ holds only the PEP timing script and the doc-half script)

## Divergence

- The README states Luxical models cannot be used for search (README.md:15) while the README Usage block calls `retrieve` with the short query "Tell me about speed" (README.md:77-84)
- The README describes `doc_half_match_demo.py` as matching document halves (README.md:92) while the script embeds first and second halves concatenated into a single vector per PEP (benchmark/doc_half_match_demo.py:43-47)
- The README says the benchmarks demonstrate fast but not useful search results (README.md:16) while benchmark/README.md reports only timings and no retrieval output
