# Polars Luxical

A high-performance Polars plugin for Luxical text embeddings, implemented in Rust.

## Overview

This plugin provides [Luxical](https://github.com/datologyai/luxical) embeddings directly within Polars expressions. Luxical combines:

- Subword tokenization (BERT uncased)
- N-gram feature extraction with TF-IDF weighting
- Sparse-to-dense neural network projection via knowledge distillation

Luxical models embed whole documents at a fraction of the cost of transformer models,
for document-level tasks: **near-duplicate detection, clustering, classification and
"more like this" retrieval** over large corpora.

They were trained only on document-to-document similarity, never on queries, so they
are **not suitable for short-query search** ("tell me about speed"). Use a document,
passage or a few seed documents as the query instead, or use
[polars-fastembed](https://github.com/lmmx/polars-fastembed) for query retrieval.

The Rust implementation matches the reference `luxical` Python package to within 3e-6
per dimension (`tests/parity_test.py`).

## Installation
```bash
pip install polars-luxical
```

Or build from source:
```bash
maturin develop --release
```

## Model Download

Models are automatically downloaded from HuggingFace Hub and cached locally on first use.

**Cache locations:**
- **Linux:** `~/.cache/polars-luxical/`
- **macOS:** `~/Library/Caches/polars-luxical/`
- **Windows:** `C:\Users\<User>\AppData\Local\polars-luxical\`

Set `POLARS_LUXICAL_CACHE_DIR` to use a different cache directory.

To use a local model file instead:
```python
register_model("/path/to/your/model")
```

Both `.safetensors` and `.npz` formats are supported.

## Usage

### Embed documents

```python
import polars as pl
from polars_luxical import register_model, embed_text

model_id = "DatologyAI/luxical-one"
register_model(model_id)  # downloads and caches automatically

df = pl.DataFrame({"id": [1, 2, 3], "text": [...]})

# As an expression
df = df.with_columns(embed_text("text", model_id=model_id).alias("embedding"))

# Or with the namespace API
df = df.luxical.embed(columns="text", model_name=model_id, output_column="embedding")
```

### Find and remove near-duplicates

```python
# Add duplicate_of (row position of the most similar earlier row) and
# duplicate_of_similarity; both are null for rows that are not duplicates
marked = df.luxical.mark_duplicates(threshold=0.95)

# Keep the first of each group of near-duplicates
deduped = df.luxical.dedup(threshold=0.95)

# Every pair at or above the threshold: left, right (row positions), similarity
pairs = df.luxical.near_duplicates(threshold=0.95)
```

A row is a near-duplicate when an earlier row has cosine similarity at or above
`threshold`. Up to 50,000 rows every pair is compared exactly. Above that, rows are
grouped with spherical k-means and compared within groups (as in
[SemDeDup](https://arxiv.org/abs/2303.09540)); each row is searched in its 2 nearest
groups, which on FineWeb keeps over 99% of near-duplicate pairs together. Pass
`method="exact"` or `method="clustered"` to choose.

**Choosing a threshold.** Luxical similarities are compressed near the top: different
pages of the same genre (two privacy policies, two casino reviews) can score 0.94-0.97.
Measured on 20,000 FineWeb documents with 1,000 injected edited copies
(`benchmark/dedup_eval.py`):

| threshold | edited copies found | untouched documents flagged | what gets flagged |
|-----------|--------------------:|----------------------------:|-------------------|
| 0.90 | 99.4% | 2.2% | mostly same-topic pages |
| **0.95** (default) | 98.2% | 0.17% | templated, spun and same-genre spam pages |
| 0.98 | 90.5% | 0.035% | copies and template pages |

Lower thresholds remove more semantically redundant content (the SemDeDup use case);
higher thresholds approach strict near-duplicate detection. The share flagged grows
with corpus size, since every document has more earlier documents to resemble: on a
full 1M-document FineWeb shard, 0.95 marks 2.0% and 0.98 marks 0.26%.

### Retrieve similar documents

```python
# Query by example: a document, a list of seed documents (their mean embedding is
# used), or a precomputed embedding vector
results = df.luxical.retrieve(
    query=["Seed document one ...", "Seed document two ..."],
    model_name=model_id,
    k=10,
)
```

See `demo.py` for a runnable example.

## Benchmarks

### Document-half matching

Each document is split in half; each first half queries all second halves, and we
record whether its own second half ranks first (top-1) or within the top 1%
(`benchmark/doc_half_match_demo.py`, the task from the
[Luxical report](https://arxiv.org/abs/2512.09015)). Baseline: word 1-2 gram TF-IDF
cosine fitted on the same corpus.

| corpus | method | top-1 | top-1% | mean rank |
|--------|--------|------:|-------:|----------:|
| 50,000 FineWeb docs | Luxical-One | 64.3% | **97.5%** | **101** |
| | TF-IDF | **76.8%** | 95.7% | 475 |
| 708 PEPs | Luxical-One | 68.5% | 87.9% | 10.5 |
| | TF-IDF | **87.6%** | **98.0%** | **2.9** |

Luxical is weaker than TF-IDF at pinpointing the exact match but better at placing it
in the right neighbourhood on web text, which is the coarse-grained similarity it was
trained for. On a small technical corpus far from its training data (the PEPs), plain
TF-IDF wins.

(Earlier versions of this README reported 97.5% top-1 on the PEPs. That figure matched
first halves against embeddings of the *whole* document, which contains the query.)

### Throughput

On a 4-core cloud container, no GPU:

| workload | time |
|----------|-----:|
| Embed 708 PEPs (3.6M tokens) | 1.4s |
| Embed 1,048,581 FineWeb documents (726M GPT-2 tokens, 3 GB of text) | 502s (2,088 docs/s, 6.1 MiB/s) |
| Mark near-duplicates across those 1,048,581 documents (clustered) | 152s |

End to end, deduplicating a full FineWeb sample-10BT shard takes 11 minutes
(`benchmark/dedup_fineweb.py`). Embedding is parallel across cores; about 70% of it is
spent in BERT tokenisation. For comparison, polars-fastembed with
`Xenova/all-MiniLM-L6-v2` takes about 30s on CPU for the 708 PEPs.

## Available Models

| Model ID | Description | Embedding Dim |
|----------|-------------|---------------|
| `DatologyAI/luxical-one` | English web documents, distilled from snowflake-arctic-embed-m-v2.0 | 192 |

## API Reference

### Functions

**`register_model(model_name: str, providers: list[str] | None = None) -> None`**

Register/load a Luxical model into the global registry. If already loaded, this is a no-op.

- `model_name`: HuggingFace model ID (e.g., `"DatologyAI/luxical-one"`) or local path.
- `providers`: Ignored (kept for API compatibility).

**`embed_text(expr, *, model_id: str | None = None) -> pl.Expr`**

Embed text using a Luxical model.

- `expr`: Column expression containing text to embed.
- `model_id`: Model name/ID. If `None`, uses the default model.

**`clear_registry() -> None`**

Clear all loaded models from the registry (frees memory).

**`list_models() -> list[str]`**

Return a list of currently loaded model names.

### DataFrame Namespace

**`df.luxical.embed(columns, model_name, output_column="embedding", join_columns=True)`**

Embed text from specified columns.

**`df.luxical.retrieve(query, model_name, embedding_column="embedding", k=None, threshold=None, similarity_metric="cosine", add_similarity_column=True)`**

Retrieve rows most similar to a query: a text, a list of texts (mean embedding), or an
embedding vector.

**`df.luxical.mark_duplicates(embedding_column="embedding", threshold=0.95, *, method="auto", n_clusters=None, assignments=2, seed=0, output_column="duplicate_of")`**

Add `duplicate_of` and `duplicate_of_similarity` columns marking near-duplicates of
earlier rows.

**`df.luxical.dedup(embedding_column="embedding", threshold=0.95, *, method="auto", n_clusters=None, assignments=2, seed=0)`**

Drop near-duplicates of earlier rows. Like SemDeDup, a row is dropped if it is close
to any earlier row, even one that was itself dropped.

**`df.luxical.near_duplicates(embedding_column="embedding", threshold=0.95, *, method="auto", n_clusters=None, assignments=2, seed=0)`**

All pairs of rows with cosine similarity at or above `threshold`.

## See also

- [polars-fastembed](https://github.com/lmmx/polars-fastembed) - a similar package with more embedding models,
  including ones suitable for search retrieval with a query

## License

Apache 2.0
