"""Tests for polars-luxical embedding functionality."""

import polars as pl
import pytest

MODEL_ID = "datologyai/luxical-one"
EMBEDDING_DIM = 192


def load_model_or_skip():
    """Register the model, skipping the test if it cannot be downloaded or loaded."""
    from polars_luxical import register_model

    try:
        register_model(MODEL_ID)
    except RuntimeError as e:
        pytest.skip(f"Model not available: {e}")


def test_embed_text_basic():
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

    load_model_or_skip()
    result = df.with_columns(
        embed_text("text", model_id=MODEL_ID).alias("embedding"),
    )

    assert "embedding" in result.columns
    assert result["embedding"].dtype == pl.Array(pl.Float32, shape=EMBEDDING_DIM)
    assert len(result) == 3


def test_embed_handles_nulls():
    """Test that null values are handled correctly."""
    from polars_luxical import embed_text

    df = pl.DataFrame(
        {
            "text": ["Hello", None, "World"],
        },
    )

    load_model_or_skip()
    result = df.with_columns(
        embed_text("text", model_id=MODEL_ID).alias("embedding"),
    )

    # Check that null input produces null output
    assert result["embedding"][1] is None
    assert result["embedding"][0] is not None
    assert result["embedding"][2] is not None


def test_namespace_api():
    """Test the DataFrame namespace API."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "text": ["Hello", "World"],
        },
    )

    load_model_or_skip()

    # Test embed
    result = df.luxical.embed(
        columns="text",
        model_name=MODEL_ID,
        output_column="emb",
    )
    assert "emb" in result.columns

    # Test retrieve
    retrieved = result.luxical.retrieve(
        query="Greeting",
        model_name=MODEL_ID,
        embedding_column="emb",
        k=1,
    )
    assert len(retrieved) == 1
    assert "similarity" in retrieved.columns


def test_registry_operations():
    """Test model registry functions."""
    from polars_luxical import clear_registry, list_models, register_model

    clear_registry()
    assert list_models() == []

    load_model_or_skip()
    assert MODEL_ID in list_models()

    # Registering again should be a no-op
    register_model(MODEL_ID)
    assert len(list_models()) == 1

    clear_registry()
    assert list_models() == []
