"""Near-duplicate detection on FineWeb with injected near-duplicates.

Takes N FineWeb documents, makes an edited copy of a random subset with one of five
web-crawl-style edits, and measures, per similarity threshold:

- recall: fraction of (source, copy) pairs at or above the threshold
- originals flagged: fraction of the N untouched documents that `mark_duplicates`
  flags as a duplicate of another document. FineWeb is MinHash-deduplicated within
  each crawl but not across crawls, so some of these are genuine duplicates
  (inspect the printed examples) rather than false positives.

The same is reported for a word 1-2 gram TF-IDF cosine baseline.

Data: one shard of FineWeb sample-10BT, e.g.
https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet

Usage: uv run python dedup_eval.py PATH/TO/000_00000.parquet [--docs 20000] [--copies 1000]
"""

import argparse
import re
import time

import numpy as np
import polars as pl
from polars_luxical import embed_text, register_model
from polars_luxical.dedup import _scan, assign_clusters, normalize_rows
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_ID = "datologyai/luxical-one"
THRESHOLDS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

BOILERPLATE = [
    (
        "We use cookies to improve your experience. By continuing to browse you agree "
        "to our use of cookies. Accept all cookies or manage your preferences."
    ),
    "Home | About Us | Contact | Privacy Policy | Terms of Service | Sitemap",
    (
        "Share this article: Facebook Twitter LinkedIn Email Print. Subscribe to our "
        "newsletter for the latest updates delivered straight to your inbox."
    ),
    "Copyright 2024 All rights reserved. Powered by WordPress. Theme by Example.",
]


def truncate(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    return " ".join(words[: max(1, int(len(words) * rng.uniform(0.6, 0.9)))])


def drop_sentences(text: str, rng: np.random.Generator) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keep = rng.random(len(sentences)) > rng.uniform(0.1, 0.25)
    kept = [s for s, k in zip(sentences, keep) if k] or sentences[:1]
    return " ".join(kept)


def add_boilerplate(text: str, rng: np.random.Generator) -> str:
    header, footer = rng.choice(BOILERPLATE, 2, replace=False)
    return f"{header}\n{text}\n{footer}"


def word_edits(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    out = []
    for w in words:
        r = rng.random()
        if r < 0.03:
            continue  # delete
        if r < 0.05:
            out.append(words[rng.integers(len(words))])  # substitute
        else:
            out.append(w)
    return " ".join(out)


def shuffle_paragraphs(text: str, rng: np.random.Generator) -> str:
    paragraphs = [p for p in text.split("\n") if p.strip()]
    rng.shuffle(paragraphs)
    return "\n".join(paragraphs)


EDITS = {
    "truncate": truncate,
    "drop_sentences": drop_sentences,
    "boilerplate": add_boilerplate,
    "word_edits": word_edits,
    "shuffle_paragraphs": shuffle_paragraphs,
}


def tfidf_best_earlier(texts: list[str]):
    """Exact most-similar-earlier cosine under TF-IDF, blocked over sparse rows."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    m = vectorizer.fit_transform(texts).astype(np.float32).tocsr()
    n = m.shape[0]
    best = np.full(n, -np.inf, dtype=np.float32)
    block = 2048
    for j0 in range(0, n, block):
        j1 = min(j0 + block, n)
        sim = (m[:j1] @ m[j0:j1].T).toarray()
        ii = np.arange(j1)[:, None]
        jj = np.arange(j0, j1)[None, :]
        sim = np.where(ii < jj, sim, -np.inf)
        best[j0:j1] = sim.max(axis=0)
    return best, vectorizer, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--docs", type=int, default=20_000)
    parser.add_argument("--copies", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    docs = pl.scan_parquet(args.parquet).select("id", "text").head(args.docs).collect()
    sources = rng.choice(len(docs), args.copies, replace=False)
    edit_names = rng.choice(list(EDITS), args.copies)
    copies = [
        EDITS[name](docs["text"][int(i)], rng) for i, name in zip(sources, edit_names)
    ]
    print(f"{len(docs)} FineWeb documents, {len(copies)} edited copies")

    register_model(MODEL_ID)
    all_text = pl.DataFrame({"text": docs["text"].to_list() + copies})
    start = time.perf_counter()
    emb = normalize_rows(
        all_text.select(embed_text("text", model_id=MODEL_ID)).to_series().to_numpy(),
    )
    print(f"Embedded {len(all_text)} documents in {time.perf_counter() - start:.1f}s")

    orig, copy = emb[: len(docs)], emb[len(docs) :]
    pair_sim = (orig[sources] * copy).sum(axis=1)

    start = time.perf_counter()
    orig_best_idx, orig_best, _ = _scan(orig, np.inf, collect_pairs=False)
    print(
        f"Exact search over {len(docs)} originals in {time.perf_counter() - start:.1f}s",
    )

    # Clustered search: a pair is only found if both rows share one of their clusters
    labels = assign_clusters(emb, seed=args.seed)
    src_labels, copy_labels = labels[sources], labels[len(docs) :]
    same_cluster = (src_labels[:, :, None] == copy_labels[:, None, :]).any(axis=(1, 2))

    start = time.perf_counter()
    tf_orig_best, vectorizer, tf_matrix = tfidf_best_earlier(docs["text"].to_list())
    tf_orig = vectorizer.transform(docs["text"][sources].to_list())
    tf_copy = vectorizer.transform(copies)
    tf_pair_sim = np.asarray(tf_orig.multiply(tf_copy).sum(axis=1)).ravel()
    print(f"TF-IDF baseline in {time.perf_counter() - start:.1f}s")

    # Lexical overlap (TF-IDF cosine) of each original's most Luxical-similar earlier doc
    has_prev = orig_best_idx >= 0
    lexical = np.zeros(len(orig_best), dtype=np.float32)
    j = np.flatnonzero(has_prev)
    lexical[j] = np.asarray(
        tf_matrix[orig_best_idx[j]].multiply(tf_matrix[j]).sum(axis=1),
    ).ravel()

    rows = []
    for t in THRESHOLDS:
        found = pair_sim >= t
        flagged = orig_best >= t
        row = {
            "threshold": t,
            "luxical_recall": found.mean(),
            "luxical_recall_clustered": (found & same_cluster).mean(),
            "luxical_originals_flagged": int(flagged.sum()),
            "of_which_tfidf>=0.5": int((flagged & (lexical >= 0.5)).sum()),
            "tfidf_recall": (tf_pair_sim >= t).mean(),
            "tfidf_originals_flagged": int((tf_orig_best >= t).sum()),
        }
        rows.append(row)

    by_edit = (
        pl.DataFrame({"edit": edit_names, "luxical": pair_sim, "tfidf": tf_pair_sim})
        .group_by("edit")
        .agg(
            pl.len().alias("n"),
            pl.col("luxical").median().alias("luxical_median_sim"),
            (pl.col("luxical") >= 0.9).mean().alias("luxical_recall@0.90"),
            pl.col("tfidf").median().alias("tfidf_median_sim"),
            (pl.col("tfidf") >= 0.9).mean().alias("tfidf_recall@0.90"),
        )
        .sort("edit")
    )

    with pl.Config(tbl_cols=-1, float_precision=3, tbl_width_chars=220, tbl_rows=-1):
        print("\nRecall on injected pairs vs fraction of originals flagged")
        print(pl.DataFrame(rows))
        print("\nBy edit type")
        print(by_edit)
        print(
            "\nClustered search keeps both sides of",
            f"{same_cluster.mean():.1%}",
            "of pairs together",
        )

    # Examples of originals flagged as near-duplicates of each other
    best_idx, best_sim = orig_best_idx, orig_best
    for lo, hi in [
        (0.99, 1.01),
        (0.98, 0.99),
        (0.97, 0.98),
        (0.96, 0.97),
        (0.94, 0.96),
        (0.90, 0.94),
    ]:
        band = np.flatnonzero((best_sim >= lo) & (best_sim < hi))
        print(
            f"\n--- original pairs with similarity in [{lo}, {min(hi, 1.0)}): {len(band)}",
        )
        for j in rng.choice(band, min(3, len(band)), replace=False):
            i = best_idx[j]
            print(f"sim={best_sim[j]:.3f} tfidf={lexical[j]:.3f}")
            print("  A:", docs["text"][int(i)][:160].replace("\n", " "))
            print("  B:", docs["text"][int(j)][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
