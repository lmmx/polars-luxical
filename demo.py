"""Demo script for polars-luxical."""

import polars as pl
from polars_luxical import register_model

# Create sample data
df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "text": [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning models process text data",
            "Polars is a fast DataFrame library written in Rust",
            "Python is a popular programming language",
            "Natural language processing enables text understanding",
        ],
    },
)

print("Input DataFrame:")
print(df)
print()

# Register the model (downloads if needed)
model_id = "datologyai/luxical-one"
print(f"Loading model: {model_id}")
register_model(model_id)

# Embed the text
print("\nEmbedding text...")
df_emb = df.luxical.embed(
    columns="text",
    model_name=model_id,
    output_column="embedding",
)

print("\nEmbedded DataFrame:")
print(df_emb)
print()

# Retrieve similar documents
query = "How do computers understand language?"
print(f"Query: {query}")
print("\nTop 3 similar documents:")

results = df_emb.luxical.retrieve(
    query=query,
    model_name=model_id,
    embedding_column="embedding",
    k=3,
)
print(results.select(["id", "text", "similarity"]))
