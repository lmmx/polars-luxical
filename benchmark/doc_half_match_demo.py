"""Half-document matching benchmark on the PEP corpus (or a FineWeb shard).

Each document is split in half by words. Every first half is used as a query against all
second halves, and we record the rank of its own second half. This is the task from
the Luxical report ("document halves"), with both halves embedded separately.

Three settings are reported side by side:

- luxical half->half: the real task
- luxical half->whole: the second-half target embedded together with the first half,
  i.e. the target contains the query text (what earlier versions of this script did)
- tf-idf half->half: word 1-2 gram TF-IDF cosine baseline (scikit-learn)

Usage: uv run python doc_half_match_demo.py [--parquet FINEWEB.parquet --docs 5000]
"""

import argparse
import time

import numpy as np
import polars as pl
from benchmark.dataset import load_peps
from benchmark.schema import TEXT_COL
from polars_luxical import embed_text, register_model
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODEL_ID = "datologyai/luxical-one"


def split_half(text: str) -> tuple[str, str]:
    """Split a document roughly in half by words."""
    words = text.split()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def ranks_of_matches(queries, targets, block: int = 2048) -> np.ndarray:
    """1-based rank of target i among all targets for query i, in row blocks.

    Works for dense arrays and scipy sparse matrices; ties count against the match.
    """
    ranks = []
    for i0 in range(0, queries.shape[0], block):
        i1 = min(i0 + block, queries.shape[0])
        sim = queries[i0:i1] @ targets.T
        sim = sim.toarray() if hasattr(sim, "toarray") else sim
        diag = sim[np.arange(i1 - i0), np.arange(i0, i1)][:, None]
        ranks.append(1 + (sim > diag).sum(axis=1))
    return np.concatenate(ranks)


def summarise(name: str, ranks: np.ndarray) -> dict:
    n = len(ranks)
    top1pct = max(1, n // 100)
    return {
        "setting": name,
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        f"top{top1pct}(1%)": float((ranks <= top1pct).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
    }


def embed(df: pl.DataFrame, col: str) -> np.ndarray:
    return df.select(embed_text(col, model_id=DEFAULT_MODEL_ID)).to_series().to_numpy()


def load_docs(parquet: str | None, n_docs: int) -> pl.DataFrame:
    """PEPs, or the first `n_docs` FineWeb documents of at least 100 words."""
    if parquet is None:
        return load_peps().select(pl.col("pep").alias("doc"), TEXT_COL)
    return (
        pl.scan_parquet(parquet)
        .select(pl.col("id").alias("doc"), TEXT_COL)
        .filter(pl.col(TEXT_COL).str.split(" ").list.len() >= 100)
        .head(n_docs)
        .collect()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", help="FineWeb parquet shard (default: PEPs)")
    parser.add_argument("--docs", type=int, default=5000)
    args = parser.parse_args()

    docs_df = load_docs(args.parquet, args.docs)
    halves = [split_half(t) for t in docs_df[TEXT_COL]]
    df = docs_df.select("doc").with_columns(
        pl.Series("first_half", [a for a, _ in halves]),
        pl.Series("second_half", [b for _, b in halves]),
        pl.Series("whole", [f"{a} {b}" for a, b in halves]),
    )
    print(f"Loaded {len(df)} documents")

    register_model(DEFAULT_MODEL_ID)
    start = time.perf_counter()
    first = embed(df, "first_half")
    second = embed(df, "second_half")
    whole = embed(df, "whole")
    print(f"Embedded 3 x {len(df)} texts in {time.perf_counter() - start:.2f}s")

    lux_ranks = ranks_of_matches(first, second)
    results = [
        summarise("luxical half->half", lux_ranks),
        summarise("luxical half->whole", ranks_of_matches(first, whole)),
    ]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    vectorizer.fit(df["first_half"].to_list() + df["second_half"].to_list())
    tf_first = vectorizer.transform(df["first_half"].to_list())
    tf_second = vectorizer.transform(df["second_half"].to_list())
    results.append(
        summarise(
            "tf-idf half->half",
            ranks_of_matches(tf_first, tf_second),
        ),
    )

    with pl.Config(tbl_cols=-1, float_precision=4, tbl_width_chars=200):
        print(pl.DataFrame(results))

        top_other = np.concatenate(
            [
                np.argmax(first[i : i + 2048] @ second.T, axis=1)
                for i in range(0, len(first), 2048)
            ],
        )
        top_doc = pl.Series("top_retrieved_doc", df["doc"].to_numpy()[top_other])
        misses = (
            df.with_columns(rank=pl.Series(lux_ranks), top_retrieved_doc=top_doc)
            .filter(pl.col("rank") > 1)
            .select("doc", "top_retrieved_doc", "rank")
            .sort("rank", descending=True)
        )
        print(f"\nluxical half->half: {len(misses)} documents not matched at rank 1")
        print(misses.head(20))


if __name__ == "__main__":
    main()
