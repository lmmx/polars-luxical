"""Tests for polars-luxical embedding functionality."""

import polars as pl
from conftest import EMBEDDING_DIM


def test_embed_text_basic(model_id):
    """Test basic embedding functionality."""
    from polars_luxical import embed_text

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [
                "Hello world",
                "Machine learning is great",
                "Polars is fast",
            ],
        },
    )

    result = df.with_columns(
        embed_text("text", model_id=model_id).alias("embedding"),
    )

    assert "embedding" in result.columns
    assert result["embedding"].dtype == pl.Array(pl.Float32, shape=EMBEDDING_DIM)
    assert len(result) == 3


def test_embed_handles_nulls(model_id):
    """Test that null values are handled correctly."""
    from polars_luxical import embed_text

    df = pl.DataFrame(
        {
            "text": ["Hello", None, "World"],
        },
    )

    result = df.with_columns(
        embed_text("text", model_id=model_id).alias("embedding"),
    )

    # Check that null input produces null output
    assert result["embedding"][1] is None
    assert result["embedding"][0] is not None
    assert result["embedding"][2] is not None


def test_namespace_api(model_id):
    """Test the DataFrame namespace API."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "text": ["Hello", "World"],
        },
    )

    # Test embed
    result = df.luxical.embed(
        columns="text",
        model_name=model_id,
        output_column="emb",
    )
    assert "emb" in result.columns

    # Test retrieve
    retrieved = result.luxical.retrieve(
        query="Greeting",
        model_name=model_id,
        embedding_column="emb",
        k=1,
    )
    assert len(retrieved) == 1
    assert "similarity" in retrieved.columns


def test_registry_operations(model_id):
    """Test model registry functions."""
    from polars_luxical import clear_registry, list_models, register_model

    clear_registry()
    assert list_models() == []

    register_model(model_id)
    assert model_id in list_models()

    # Registering again should be a no-op
    register_model(model_id)
    assert len(list_models()) == 1

    clear_registry()
    assert list_models() == []
