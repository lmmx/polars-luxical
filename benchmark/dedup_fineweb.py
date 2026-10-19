"""Semantic deduplication of a full FineWeb shard (~1M documents), timed end to end.

Reads one FineWeb sample-10BT parquet shard, embeds every document with Luxical-One,
marks near-duplicates with clustered search, and reports timings, how many documents
each threshold would remove, and example duplicate pairs.

Data: https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet

Usage: uv run python dedup_fineweb.py PATH/TO/000_00000.parquet [--rows N] [--out DIR]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from polars_luxical import embed_text, register_model

MODEL_ID = "datologyai/luxical-one"
THRESHOLDS = [0.90, 0.95, 0.97, 0.98, 0.99]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="limit rows (default all)",
    )
    parser.add_argument("--out", default=None, help="write embeddings + marks here")
    args = parser.parse_args()

    print(f"CPU cores: {os.cpu_count()}")
    timings = {}

    start = time.perf_counter()
    lf = pl.scan_parquet(args.parquet).select("id", "url", "text", "token_count")
    if args.rows:
        lf = lf.head(args.rows)
    df = lf.collect()
    timings["read parquet"] = time.perf_counter() - start
    n_tokens = df["token_count"].sum()
    print(f"{df.height:,} documents, {n_tokens:,} GPT-2 tokens")

    start = time.perf_counter()
    register_model(MODEL_ID)
    timings["load model"] = time.perf_counter() - start

    start = time.perf_counter()
    df = df.with_columns(embedding=embed_text("text", model_id=MODEL_ID))
    timings["embed"] = time.perf_counter() - start

    start = time.perf_counter()
    marked = df.luxical.mark_duplicates(threshold=min(THRESHOLDS), method="clustered")
    timings["mark duplicates (clustered)"] = time.perf_counter() - start

    total = sum(timings.values())
    print("\nTimings")
    for name, secs in timings.items():
        print(f"  {name:<30} {secs:8.1f}s")
    print(f"  {'total':<30} {total:8.1f}s")
    print(
        f"  embedding throughput: {df.height / timings['embed']:,.0f} docs/s, "
        f"{df['text'].str.len_bytes().sum() / timings['embed'] / 2**20:.1f} MiB/s",
    )

    sim = marked["duplicate_of_similarity"]
    summary = pl.DataFrame(
        {
            "threshold": THRESHOLDS,
            "duplicates": [int((sim >= t).sum()) for t in THRESHOLDS],
        },
    ).with_columns(share=pl.col("duplicates") / df.height)
    with pl.Config(float_precision=4):
        print("\nDocuments marked as near-duplicates of an earlier document")
        print(summary)

    rng = np.random.default_rng(0)
    pairs = (
        marked.with_row_index()
        .filter(pl.col("duplicate_of").is_not_null())
        .select("index", "duplicate_of", "duplicate_of_similarity")
    )
    for lo, hi in [(0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.90, 0.95)]:
        band = pairs.filter(
            pl.col("duplicate_of_similarity").is_between(lo, hi, "left"),
        )
        print(f"\n--- pairs with similarity in [{lo}, {min(hi, 1.0)}): {band.height:,}")
        sample = band.sample(min(4, band.height), seed=int(rng.integers(1 << 31)))
        for j, i, s in sample.iter_rows():
            print(f"sim={s:.3f}")
            print("  A:", df["url"][i], "|", df["text"][i][:140].replace("\n", " "))
            print("  B:", df["url"][j], "|", df["text"][j][:140].replace("\n", " "))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        marked.select(
            "id",
            "embedding",
            "duplicate_of",
            "duplicate_of_similarity",
        ).write_parquet(
            out / "fineweb_luxical_dedup.parquet",
        )


if __name__ == "__main__":
    main()
