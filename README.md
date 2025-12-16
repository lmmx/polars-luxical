# Polars Luxical

A high-performance Polars plugin for Luxical text embeddings, implemented in Rust.

## Overview

This plugin provides [Luxical](https://github.com/datologyai/luxical) embeddings directly within Polars expressions. Luxical combines:

- Subword tokenization (BERT uncased)
- N-gram feature extraction with TF-IDF weighting  
- Sparse-to-dense neural network projection via knowledge distillation

Luxical models achieve dramatically higher throughput than transformer-based embedding models while maintaining competitive quality for document-level similarity tasks like clustering, classification, and semantic deduplication.

It should be noted that they were not trained on queries, so you cannot use them for search!
A demonstration of this is given in the benchmarks, where the results are fast but not useful.

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

To use a local model file instead:
```python
register_model("/path/to/your/model")
```

Both `.safetensors` and `.npz` formats are supported.

## Usage
```python
import polars as pl
from polars_luxical import register_model, embed_text

# Register a Luxical model (downloads and caches automatically)
register_model("DatologyAI/luxical-one")

# Create a DataFrame
df = pl.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "Hello world",
        "Machine learning is fascinating",
        "Polars and Rust are fast",
    ],
})

# Embed text
df_emb = df.with_columns(
    embed_text("text", model_id="DatologyAI/luxical-one").alias("embedding")
)
print(df_emb)

# Or use the namespace API
df_emb = df.luxical.embed(
    columns="text",
    model_name="DatologyAI/luxical-one",
    output_column="embedding",
)

# Retrieve similar documents
results = df_emb.luxical.retrieve(
    query="Tell me about speed",
    model_name="DatologyAI/luxical-one",
    embedding_column="embedding",
    k=3,
)
print(results)
```

## Available Models

| Model ID | Description | Embedding Dim |
|----------|-------------|---------------|
| `DatologyAI/luxical-one` | English web documents, distilled from snowflake-arctic-embed-m-v2.0 | 192 |

## Performance

Luxical embeddings avoid transformer inference entirely, achieving throughput up to ~100x faster than large transformer embedding models (e.g., Qwen3-0.6B) and significantly faster than smaller models like MiniLM-L6-v2, particularly on CPU.

For benchmarks and methodology, see the [Luxical technical report](https://arxiv.org/abs/2512.09015).

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

Retrieve rows most similar to a query.

## License

Apache 2.0
