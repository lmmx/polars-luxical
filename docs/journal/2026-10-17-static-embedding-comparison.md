# 2026-10-17: Static Embedding Comparison

No published evaluation places Luxical-One alongside the other CPU-fast embedders on the
same tasks. MTEB issue #3722 ("Add model: DatologyAI/luxical-one") has been open since
2025-12-12 with no scores. On 2026-08-08 a contributor proposed integrating the model through
the Sentence Transformers wrapper, and on 2026-10-16 a maintainer told them to go ahead. A
Luxical author commented that the model "was not designed to compete on most MTEB tasks". The
MTEB maintainer said they would like to see how it compares with model2vec on retrieval.

That comparison is exactly the experiment that would turn this repo's README claims into
numbers. polars-luxical already runs Luxical inference without the upstream Python package,
which supports only macOS and Linux on Python 3.11–3.13. polars-fastembed already runs MiniLM
and other ONNX models. A Polars-native harness could run every candidate over the same frames.

## The comparison set

| Model | Type | Retrieval-trained | Notes |
|---|---|---|---|
| `DatologyAI/luxical-one` | TF-IDF 5-gram bag → MLP, 192d | No | distilled from arctic-embed-m-v2.0 on FineWeb docs |
| `minishlab/potion-base-32M` | static (Model2Vec) | No | 32.67 MTEB retrieval, self-reported |
| `minishlab/potion-retrieval-32M` | static (Model2Vec) | Yes | 35.06 MTEB retrieval, self-reported |
| `sentence-transformers/static-retrieval-mrl-en-v1` | static | Yes | 34.95 MTEB retrieval, as listed by MinishLab |
| `all-MiniLM-L6-v2` | 6-layer transformer | Yes | 42.92 MTEB retrieval; the speed baseline in benchmark/README.md |
| plain TF-IDF cosine | sparse | n/a | lower bound for lexical-overlap tasks |

## The hypothesis to test

- **Retrieval tasks** with short queries: Luxical-One scores below both retrieval-trained
  static models and well below MiniLM, consistent with `2026-10-17-query-retrieval-framing.md`.
- **Clustering and classification tasks** on full documents: Luxical-One scores at or above
  the static models and close to MiniLM, consistent with the paper's doc-half and curation
  results.
- **The size of the gap between those two task families** is the actual finding. A large
  gap says the training objective, not the architecture, limits retrieval. That would be the
  evidence `2026-10-17-query-aware-distillation.md` needs before anyone spends effort on it.

Throughput should be reported for every model on the same hardware, in MiB/s and ms per 1k
tokens. That way quality can be read against speed, which is the whole reason these
models exist.

## Caveats

- MTEB's clustering and classification tasks are mostly short texts: titles, abstracts,
  sentences. Luxical-One was built for whole web documents, so MTEB may understate it on
  its home turf. A long-document track (PEPs, FineWeb samples, arXiv full texts) would be
  needed to measure the target use case.
- The static-model retrieval numbers are self-reported by MinishLab. Rerunning them in
  the same harness removes that caveat.

## Current State

- `polars-luxical` computes luxical-one embeddings in Rust with no dependency on the upstream `luxical` Python package (src/embedder/, Cargo.toml)
- The PEP benchmark compares wall-clock time against polars-fastembed with `Xenova/all-MiniLM-L6-v2` and reports no quality metric for MiniLM (benchmark/README.md)
- The README links polars-fastembed as the package for query retrieval models (README.md:177-179)

## Missing

- No MTEB task runner or MTEB-compatible model wrapper (repository-wide)
- No quality comparison against Model2Vec potion models, `static-retrieval-mrl-en-v1`, MiniLM or TF-IDF cosine (benchmark/)
- No long-document clustering or classification benchmark (benchmark/)
