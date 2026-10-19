from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars_distance as pld
from polars.api import register_dataframe_namespace
from polars.plugins import register_plugin_function

from polars_luxical._polars_luxical import clear_registry as _clear_registry
from polars_luxical._polars_luxical import list_models as _list_models
from polars_luxical._polars_luxical import register_model as _register_model

from .dedup import Method, most_similar_earlier, similar_pairs
from .utils import parse_into_expr, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

#: Default cosine similarity at or above which two documents count as near-duplicates.
#: On 20k FineWeb documents, 0.95 recalls 98% of lightly edited copies while flagging
#: about 1 in 600 untouched documents, mostly templated or spun pages; the share grows
#: with corpus size (2% of a 1M-document shard). See benchmark/dedup_eval.py.
DEFAULT_DUPLICATE_THRESHOLD = 0.95

# Determine the correct plugin path
if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

__all__ = [
    "DEFAULT_DUPLICATE_THRESHOLD",
    "clear_registry",
    "embed_text",
    "list_models",
    "register_model",
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
        query: str | Sequence[str] | Sequence[float] | np.ndarray,
        model_name: str | None = None,
        embedding_column: str = "embedding",
        k: int | None = None,
        threshold: float | None = None,
        similarity_metric: str = "cosine",
        add_similarity_column: bool = True,
    ) -> pl.DataFrame:
        """
        Retrieve rows most similar to a query.

        Luxical models compare documents with documents: the query works best as a
        document (or passage) rather than a short search phrase.

        Args:
            query: Query text to embed; a list of texts, whose mean embedding is used
                (e.g. a few seed documents); or a precomputed embedding vector.
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

        q_vec = self._query_vector(query, model_name)
        q_df = pl.DataFrame(
            [pl.Series("_q_emb", [q_vec], dtype=pl.Array(pl.Float32, len(q_vec)))],
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

    @staticmethod
    def _query_vector(
        query: str | Sequence[str] | Sequence[float] | np.ndarray,
        model_name: str | None,
    ) -> np.ndarray:
        """Resolve a retrieve() query to one embedding vector."""
        if isinstance(query, str):
            query = [query]
        if isinstance(query, np.ndarray) or not isinstance(query[0], str):
            return np.asarray(query, dtype=np.float32).ravel()
        embeddings = (
            pl.DataFrame({"_q": list(query)})
            .select(embed_text("_q", model_id=model_name))
            .to_series()
            .to_numpy()
        )
        centroid = embeddings.mean(axis=0)
        return (centroid / np.linalg.norm(centroid)).astype(np.float32)

    def _embedding_matrix(self, embedding_column: str) -> tuple[np.ndarray, np.ndarray]:
        """Non-null embeddings as an (n, dim) array, with their row positions."""
        if embedding_column not in self._df.columns:
            raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")
        col = self._df.get_column(embedding_column)
        valid = np.flatnonzero(col.is_not_null().to_numpy())
        return col.drop_nulls().to_numpy(), valid

    def mark_duplicates(
        self,
        embedding_column: str = "embedding",
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        *,
        method: Method = "auto",
        n_clusters: int | None = None,
        assignments: int = 2,
        seed: int = 0,
        output_column: str = "duplicate_of",
    ) -> pl.DataFrame:
        """
        Mark rows that are near-duplicates of an earlier row.

        A row is a duplicate when some earlier row has cosine similarity
        ``>= threshold`` with it. Adds ``output_column`` (row position of the most
        similar earlier row, null when the row is not a duplicate) and
        ``{output_column}_similarity`` (its cosine similarity).

        Args:
            embedding_column: Column of embeddings (e.g. from ``embed``).
            threshold: Cosine similarity at or above which rows are near-duplicates.
            method: ``"exact"`` compares all pairs; ``"clustered"`` groups rows with
                spherical k-means first and compares within groups (SemDeDup), which
                scales to millions of rows but can miss pairs split across groups;
                ``"auto"`` uses exact up to 50,000 rows.
            n_clusters: Number of clusters for ``"clustered"`` (default: one per
                ~2,000 rows).
            assignments: For ``"clustered"``, how many nearest clusters each row is
                searched in. Two keeps over 99% of near-duplicate pairs together.
            seed: Random seed for clustering.
            output_column: Name of the duplicate-of column.

        Returns:
            DataFrame with the two columns added.
        """
        embeddings, positions = self._embedding_matrix(embedding_column)
        best_idx, best_sim = most_similar_earlier(
            embeddings,
            method=method,
            n_clusters=n_clusters,
            assignments=assignments,
            seed=seed,
        )
        is_dup = best_sim >= threshold
        dup_of = np.full(self._df.height, -1, dtype=np.int64)
        dup_sim = np.full(self._df.height, np.nan, dtype=np.float32)
        dup_of[positions[is_dup]] = positions[best_idx[is_dup]]
        dup_sim[positions[is_dup]] = best_sim[is_dup]
        return self._df.with_columns(
            pl.Series(output_column, dup_of).replace(-1, None).cast(pl.UInt32),
            pl.Series(f"{output_column}_similarity", dup_sim).fill_nan(None),
        )

    def dedup(
        self,
        embedding_column: str = "embedding",
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        *,
        method: Method = "auto",
        n_clusters: int | None = None,
        assignments: int = 2,
        seed: int = 0,
    ) -> pl.DataFrame:
        """
        Drop rows that are near-duplicates of an earlier row, keeping the first.

        Same arguments as ``mark_duplicates``. Rows with a null embedding are kept.
        Like SemDeDup, a row is dropped if it is close to any earlier row, even one
        that was itself dropped.
        """
        marked = self.mark_duplicates(
            embedding_column,
            threshold,
            method=method,
            n_clusters=n_clusters,
            assignments=assignments,
            seed=seed,
            output_column="__duplicate_of",
        )
        return marked.filter(pl.col("__duplicate_of").is_null()).drop(
            "__duplicate_of",
            "__duplicate_of_similarity",
        )

    def near_duplicates(
        self,
        embedding_column: str = "embedding",
        threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        *,
        method: Method = "auto",
        n_clusters: int | None = None,
        assignments: int = 2,
        seed: int = 0,
    ) -> pl.DataFrame:
        """
        List all pairs of rows with cosine similarity ``>= threshold``.

        Same arguments as ``mark_duplicates``. Returns columns ``left`` and ``right``
        (row positions, ``left < right``) and ``similarity``, sorted by position.
        Join back on ``with_row_index()`` to see the rows.
        """
        embeddings, positions = self._embedding_matrix(embedding_column)
        left, right, sim = similar_pairs(
            embeddings,
            threshold,
            method=method,
            n_clusters=n_clusters,
            assignments=assignments,
            seed=seed,
        )
        return pl.DataFrame(
            {
                "left": pl.Series(positions[left], dtype=pl.UInt32),
                "right": pl.Series(positions[right], dtype=pl.UInt32),
                "similarity": pl.Series(sim, dtype=pl.Float32),
            },
        )
