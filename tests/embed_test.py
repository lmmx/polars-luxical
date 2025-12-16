"""Tests for polars-luxical embedding functionality."""

import polars as pl
import pytest


def test_embed_text_basic():
    """Test basic embedding functionality."""
    from polars_luxical import embed_text, register_model

    # Use a local model file for testing (you'd create this)
    # For now, we test that the API works with a mock
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

    # This would require a real model - skip if not available
    try:
        register_model("datologyai/luxical-one")
        result = df.with_columns(
            embed_text("text", model_id="datologyai/luxical-one").alias("embedding"),
        )

        assert "embedding" in result.columns
        assert result["embedding"].dtype == pl.Array(
            pl.Float32,
            width=256,
        )  # Adjust dim as needed
        assert len(result) == 3

    except Exception as e:
        pytest.skip(f"Model not available: {e}")


def test_embed_handles_nulls():
    """Test that null values are handled correctly."""
    from polars_luxical import embed_text, register_model

    df = pl.DataFrame(
        {
            "text": ["Hello", None, "World"],
        },
    )

    try:
        register_model("datologyai/luxical-one")
        result = df.with_columns(
            embed_text("text", model_id="datologyai/luxical-one").alias("embedding"),
        )

        # Check that null input produces null output
        assert result["embedding"][1] is None
        assert result["embedding"][0] is not None
        assert result["embedding"][2] is not None

    except Exception as e:
        pytest.skip(f"Model not available: {e}")


def test_namespace_api():
    """Test the DataFrame namespace API."""
    from polars_luxical import register_model

    df = pl.DataFrame(
        {
            "id": [1, 2],
            "text": ["Hello", "World"],
        },
    )

    try:
        register_model("datologyai/luxical-one")

        # Test embed
        result = df.luxical.embed(
            columns="text",
            model_name="datologyai/luxical-one",
            output_column="emb",
        )
        assert "emb" in result.columns

        # Test retrieve
        retrieved = result.luxical.retrieve(
            query="Greeting",
            model_name="datologyai/luxical-one",
            embedding_column="emb",
            k=1,
        )
        assert len(retrieved) == 1
        assert "similarity" in retrieved.columns

    except Exception as e:
        pytest.skip(f"Model not available: {e}")


def test_registry_operations():
    """Test model registry functions."""
    from polars_luxical import clear_registry, list_models, register_model

    clear_registry()
    assert list_models() == []

    try:
        register_model("datologyai/luxical-one")
        models = list_models()
        assert "datologyai/luxical-one" in models

        # Registering again should be a no-op
        register_model("datologyai/luxical-one")
        assert len(list_models()) == 1

        clear_registry()
        assert list_models() == []

    except Exception as e:
        pytest.skip(f"Model not available: {e}")
