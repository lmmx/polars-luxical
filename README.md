# Polars Luxical

A high-performance Polars plugin for fast lexical text embeddings, implemented entirely in Rust.

## Overview

This plugin provides Luxical embeddings directly within Polars expressions. Luxical uses:
- Subword tokenization
- N-gram feature extraction with TF-IDF weighting
- Sparse-to-dense neural network projection

The result is embeddings that are **10-100x faster** than transformer-based models while maintaining good quality for many retrieval tasks.

## Installation
```bash
pip install polars-luxical
```

Or build from source:
```bash
maturin develop --release
```

## Model Download

When you first use a model, it will be automatically downloaded from HuggingFace Hub and cached locally.

**Cache location:**
- **Linux:** `~/.cache/polars-luxical/`
- **macOS:** `~/Library/Caches/polars-luxical/`
- **Windows:** `C:\Users\<User>\AppData\Local\polars-luxical\`

The default model `DatologyAI/luxical-one` is approximately 50MB.

To use a local model file instead, pass the path directly:
```python
register_model("/path/to/your/model.npz")
```

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
| `DatologyAI/luxical-one` | Default model, good general-purpose embeddings | 256 |

## Performance

Luxical embeddings are extremely fast because they avoid transformer inference entirely:

| Operation | Time (1000 docs) |
|-----------|------------------|
| Tokenization | ~2ms |
| N-gram + TF-IDF | ~5ms |
| MLP projection | ~3ms |
| **Total** | **~10ms** |

Compare to ~500ms+ for MiniLM-L6 on the same hardware.

## API Reference

### Functions

`register_model(model_name: str, providers: list[str] | None = None) -> None`

> Register/load a Luxical model into the global registry. If already loaded, this is a no-op.
>
> - `model_name`: HuggingFace model ID (e.g., `"DatologyAI/luxical-one"`) or local path to a `.npz` file.
> - `providers`: Ignored (kept for API compatibility with polars-fastembed).

`embed_text(expr, *, model_id: str | None = None) -> pl.Expr`

> Embed text using a Luxical model.
>
> - `expr`: Column expression containing text to embed.
> - `model_id`: Model name/ID. If `None`, uses the default model.

`clear_registry() -> None`

> Clear all loaded models from the registry (frees memory).

`list_models() -> list[str]`

> Return a list of currently loaded model names.

### DataFrame Namespace

`df.luxical.embed(columns, model_name, output_column="embedding", join_columns=True)`

> Embed text from specified columns.

`df.luxical.retrieve(query, model_name, embedding_column="embedding", k=None, threshold=None, similarity_metric="cosine", add_similarity_column=True)`

> Retrieve rows most similar to a query.

## License

Apache 2.0
