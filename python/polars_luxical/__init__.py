from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import polars_distance as pld
from polars.api import register_dataframe_namespace
from polars.plugins import register_plugin_function

from polars_luxical._polars_luxical import clear_registry as _clear_registry
from polars_luxical._polars_luxical import list_models as _list_models
from polars_luxical._polars_luxical import register_model as _register_model

from .utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# Determine the correct plugin path
if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

__all__ = [
    "embed_text",
    "register_model",
    "clear_registry",
    "list_models",
]


def register_model(model_name: str, providers: list[str] | None = None) -> None:
    """
    Register/load a Luxical model into the global registry.
    If already loaded, this is a no-op.

    Args:
        model_name: HuggingFace model ID (e.g., "datologyai/luxical-one")
                    or local path to a .npz model file.
        providers: Ignored (kept for API compatibility with polars-fastembed).
    """
    _register_model(model_name)


def clear_registry() -> None:
    """Clear all loaded models from the registry."""
    _clear_registry()


def list_models() -> list[str]:
    """Return a list of currently loaded model names."""
    return _list_models()


def plug(expr: IntoExpr, **kwargs) -> pl.Expr:
    """Helper to register a plugin function."""
    func_name = inspect.stack()[1].function
    into_expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=lib,
        function_name=func_name,
        args=into_expr,
        is_elementwise=True,
        kwargs=kwargs,
    )


def embed_text(expr: IntoExpr, *, model_id: str | None = None) -> pl.Expr:
    """
    Embed text using a Luxical model.

    Args:
        expr: Column expression containing text to embed.
        model_id: Model name/ID. If None, uses the default model.

    Returns:
        Expression producing fixed-size float32 arrays.
    """
    return plug(expr, model_id=model_id)


@register_dataframe_namespace("luxical")
class LuxicalPlugin:
    """Polars DataFrame namespace for Luxical embeddings."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def embed(
        self,
        columns: str | list[str],
        model_name: str,
        output_column: str = "embedding",
        join_columns: bool = True,
    ) -> pl.DataFrame:
        """
        Embed text from specified columns.

        Args:
            columns: Column name(s) containing text to embed.
            model_name: Luxical model name/ID.
            output_column: Name for the embedding column.
            join_columns: If True and multiple columns given, concatenate them.

        Returns:
            DataFrame with embedding column added.
        """
        if isinstance(columns, str):
            columns = [columns]

        if join_columns and len(columns) > 1:
            self._df = self._df.with_columns(
                pl.concat_str(columns, separator=" ").alias("_text_to_embed"),
            )
            text_col = "_text_to_embed"
        else:
            text_col = columns[0]

        new_df = self._df.with_columns(
            embed_text(text_col, model_id=model_name).alias(output_column),
        )

        if join_columns and len(columns) > 1:
            new_df = new_df.drop("_text_to_embed")

        return new_df

    def retrieve(
        self,
        query: str,
        model_name: str | None = None,
        embedding_column: str = "embedding",
        k: int | None = None,
        threshold: float | None = None,
        similarity_metric: str = "cosine",
        add_similarity_column: bool = True,
    ) -> pl.DataFrame:
        """
        Retrieve rows most similar to a query.

        Args:
            query: Query text to embed and compare against.
            model_name: Model to use for embedding the query.
            embedding_column: Column containing document embeddings.
            k: Return top-k results (None for all).
            threshold: Minimum similarity threshold.
            similarity_metric: "cosine" or "dot".
            add_similarity_column: Add a "similarity" column to results.

        Returns:
            DataFrame sorted by similarity (descending).
        """
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

        # Embed the query
        q_df = pl.DataFrame({"_q": [query]}).with_columns(
            embed_text("_q", model_id=model_name).alias("_q_emb"),
        )

        # Cross join query embedding with all rows
        result_df = self._df.join(q_df.select("_q_emb"), how="cross")

        # Compute similarity
        if similarity_metric == "cosine":
            similarity_expr = 1 - pld.col(embedding_column).dist_arr.cosine("_q_emb")
        elif similarity_metric == "dot":
            similarity_expr = pl.col(embedding_column).dot(pl.col("_q_emb"))
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        if add_similarity_column:
            result_df = result_df.with_columns(similarity_expr.alias("similarity"))

        result_df = result_df.drop("_q_emb")

        if threshold is not None:
            result_df = result_df.filter(pl.col("similarity") >= threshold)

        result_df = result_df.sort("similarity", descending=True)

        if k is not None:
            result_df = result_df.head(k)

        return result_df
