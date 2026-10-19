"""Half-document matching benchmark on the PEP corpus.

Each PEP is split in half by words. Every first half is used as a query against all
second halves, and we record the rank of its own second half. This is the task from
the Luxical report ("document halves"), with both halves embedded separately.

Three settings are reported side by side:

- luxical half->half: the real task
- luxical half->whole: the second-half target embedded together with the first half,
  i.e. the target contains the query text (what earlier versions of this script did)
- tf-idf half->half: word 1-2 gram TF-IDF cosine baseline (scikit-learn)
"""

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


def ranks_of_matches(sim: np.ndarray) -> np.ndarray:
    """1-based rank of the diagonal entry within each row (ties count against it)."""
    diag = np.diag(sim)[:, None]
    return 1 + (sim > diag).sum(axis=1)


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


def main():
    docs_df = load_peps()
    halves = [split_half(t) for t in docs_df[TEXT_COL]]
    df = docs_df.select("pep").with_columns(
        pl.Series("first_half", [a for a, _ in halves]),
        pl.Series("second_half", [b for _, b in halves]),
        pl.Series("whole", [f"{a} {b}" for a, b in halves]),
    )
    print(f"Loaded {len(df)} PEPs")

    register_model(DEFAULT_MODEL_ID)
    start = time.perf_counter()
    first = embed(df, "first_half")
    second = embed(df, "second_half")
    whole = embed(df, "whole")
    print(f"Embedded 3 x {len(df)} texts in {time.perf_counter() - start:.2f}s")

    lux_ranks = ranks_of_matches(first @ second.T)
    results = [
        summarise("luxical half->half", lux_ranks),
        summarise("luxical half->whole", ranks_of_matches(first @ whole.T)),
    ]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    vectorizer.fit(df["first_half"].to_list() + df["second_half"].to_list())
    tf_first = vectorizer.transform(df["first_half"].to_list())
    tf_second = vectorizer.transform(df["second_half"].to_list())
    results.append(
        summarise(
            "tf-idf half->half",
            ranks_of_matches((tf_first @ tf_second.T).toarray()),
        ),
    )

    with pl.Config(tbl_cols=-1, float_precision=4, tbl_width_chars=200):
        print(pl.DataFrame(results))

        top_other = np.argmax(first @ second.T, axis=1)
        top_pep = pl.Series("top_retrieved_pep", df["pep"].to_numpy()[top_other])
        misses = (
            df.with_columns(rank=pl.Series(lux_ranks), top_retrieved_pep=top_pep)
            .filter(pl.col("rank") > 1)
            .select("pep", "top_retrieved_pep", "rank")
            .sort("rank", descending=True)
        )
        print(f"\nluxical half->half: {len(misses)} PEPs not matched at rank 1")
        print(misses.head(20))


if __name__ == "__main__":
    main()
