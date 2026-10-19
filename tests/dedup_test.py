"""Tests for near-duplicate detection."""

import numpy as np
import polars as pl
import pytest
from polars_luxical import dedup

DIM = 192


def unit_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    return dedup.normalize_rows(rng.standard_normal((n, DIM)))


def with_noisy_copies(
    rng: np.random.Generator,
    base: np.ndarray,
    sources: list[int],
    noise: float = 0.01,
) -> np.ndarray:
    """Append a noisy copy of each source row (noise 0.01 per dim: cosine ~0.99)."""
    copies = base[sources] + noise * rng.standard_normal((len(sources), DIM))
    return np.vstack([base, dedup.normalize_rows(copies)])


def brute_force(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sim = x @ x.T
    sim[np.triu_indices(len(x))] = -np.inf  # keep only i < j, as sim[j, i]
    idx = sim.argmax(axis=1)
    best = sim[np.arange(len(x)), idx]
    return np.where(np.isfinite(best), idx, -1), best


def test_blocked_scan_matches_brute_force():
    rng = np.random.default_rng(0)
    x = with_noisy_copies(rng, unit_rows(rng, 300), [3, 10, 10, 299])
    idx, sim, _ = dedup._scan(x, np.inf, collect_pairs=False, block=37)
    exp_idx, exp_sim = brute_force(x)
    np.testing.assert_array_equal(idx, exp_idx)
    np.testing.assert_allclose(sim, exp_sim, rtol=1e-5)


def test_blocked_pairs_match_brute_force():
    rng = np.random.default_rng(1)
    x = with_noisy_copies(rng, unit_rows(rng, 200), [0, 5, 5, 150], noise=0.1)
    pairs = dedup._scan(x, 0.5, collect_pairs=True, block=16)[2]
    found = {(int(i), int(j)) for p in pairs for i, j in zip(p[0], p[1])}
    sim = x @ x.T
    expected = {(i, j) for i, j in zip(*np.nonzero(sim >= 0.5)) if i < j}
    assert found == expected
    assert {(0, 200), (5, 201), (5, 202), (150, 203)} <= found


def test_mark_duplicates_points_to_source():
    rng = np.random.default_rng(2)
    base = unit_rows(rng, 100)
    x = with_noisy_copies(rng, base, [7, 42])
    df = pl.DataFrame({"embedding": x.tolist()}).cast(
        {"embedding": pl.Array(pl.Float32, DIM)},
    )
    marked = df.luxical.mark_duplicates(threshold=0.95)
    dups = marked.with_row_index().filter(pl.col("duplicate_of").is_not_null())
    assert dups["index"].to_list() == [100, 101]
    assert dups["duplicate_of"].to_list() == [7, 42]
    assert (dups["duplicate_of_similarity"] >= 0.95).all()

    kept = df.luxical.dedup(threshold=0.95)
    assert kept.height == 100
    assert kept.columns == ["embedding"]


def test_null_embeddings_are_kept_and_skipped():
    rng = np.random.default_rng(3)
    x = with_noisy_copies(rng, unit_rows(rng, 5), [1])
    rows = x.tolist()
    rows.insert(2, None)  # null at position 2 shifts the copy of row 1 to position 6
    df = pl.DataFrame(
        {"embedding": pl.Series(rows, dtype=pl.Array(pl.Float32, DIM))},
    )
    marked = df.luxical.mark_duplicates(threshold=0.95)
    assert marked["duplicate_of"].to_list() == [None] * 6 + [1]
    assert df.luxical.dedup(threshold=0.95).height == 6
    pairs = df.luxical.near_duplicates(threshold=0.95)
    assert pairs.select("left", "right").rows() == [(1, 6)]


def test_clustered_finds_duplicates_within_clusters():
    rng = np.random.default_rng(4)
    centers = unit_rows(rng, 20)
    labels = rng.integers(0, 20, 5000)
    base = dedup.normalize_rows(
        centers[labels] + 0.03 * rng.standard_normal((5000, DIM)),
    )
    sources = rng.choice(5000, 50, replace=False).tolist()
    x = with_noisy_copies(rng, base, sources, noise=0.002)

    exact_idx, exact_sim = dedup.most_similar_earlier(x, method="exact")
    clus_idx, clus_sim = dedup.most_similar_earlier(
        x,
        method="clustered",
        n_clusters=20,
    )
    exact_dups = set(np.flatnonzero(exact_sim >= 0.95))
    clus_dups = set(np.flatnonzero(clus_sim >= 0.95))
    assert exact_dups == set(range(5000, 5050))
    assert clus_dups == exact_dups
    np.testing.assert_array_equal(clus_idx[5000:], exact_idx[5000:])


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        dedup.most_similar_earlier(np.eye(3), method="fast")


def test_text_dedup_end_to_end(model_id):
    """Lightly edited copies of a document are flagged; unrelated documents are not."""
    doc = (
        "Polars is a DataFrame library written in Rust. It uses Apache Arrow as its "
        "memory model and supports lazy evaluation, query optimisation and parallel "
        "execution. Expressions compose into pipelines that run on all cores. "
    ) * 4
    texts = [
        doc,
        "A recipe for sourdough bread: mix flour, water and starter, then rest.",
        doc.replace("Rust", "the Rust language") + "Read the user guide for more.",
        "The 1966 World Cup final was played at Wembley Stadium in London.",
    ]
    df = pl.DataFrame({"text": texts}).luxical.embed("text", model_name=model_id)
    marked = df.luxical.mark_duplicates()
    assert marked["duplicate_of"].to_list() == [None, None, 0, None]


def test_retrieve_accepts_vectors_and_seed_documents(model_id):
    df = pl.DataFrame(
        {"text": ["Rust and Polars are fast", "Bread baking at home", "Cake recipes"]},
    ).luxical.embed("text", model_name=model_id)
    vec = df["embedding"][1].to_numpy()
    by_vector = df.luxical.retrieve(vec, model_name=model_id, k=1)
    assert by_vector["text"].to_list() == ["Bread baking at home"]
    by_seeds = df.luxical.retrieve(
        ["Sourdough loaf baking", "Baking bread with yeast"],
        model_name=model_id,
        k=1,
    )
    assert by_seeds["text"].to_list() == ["Bread baking at home"]


def test_clustered_pairs_match_exact_without_repeats():
    rng = np.random.default_rng(5)
    centers = unit_rows(rng, 10)
    base = dedup.normalize_rows(
        centers[rng.integers(0, 10, 2000)] + 0.03 * rng.standard_normal((2000, DIM)),
    )
    x = with_noisy_copies(rng, base, rng.choice(2000, 30, replace=False).tolist())
    exact = dedup.similar_pairs(x, 0.95, method="exact")
    clustered = dedup.similar_pairs(x, 0.95, method="clustered", n_clusters=10)
    for e, c in zip(exact, clustered):
        np.testing.assert_allclose(c, e, rtol=1e-5)
    assert len(set(zip(clustered[0], clustered[1]))) == len(clustered[0])
