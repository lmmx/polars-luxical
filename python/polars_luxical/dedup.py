"""Near-duplicate detection over embedding matrices.

Two search strategies share one definition of a duplicate: row ``j`` is a duplicate when
some earlier row ``i < j`` has cosine similarity ``>= threshold`` with it (the SemDeDup
rule). Its reported partner is the most similar such earlier row.

- ``exact``: blocked all-pairs similarity, ``O(n^2)`` but BLAS-bound; finds every pair.
- ``clustered``: spherical k-means, then exact search within each cluster (SemDeDup).
  Each row joins its ``assignments`` most similar clusters (default 2), so a pair is
  found when the rows share any cluster. With one assignment per row, about 4% of
  near-duplicate pairs on FineWeb are split across clusters and missed; with two,
  under 1%.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Method = Literal["auto", "exact", "clustered"]

#: Above this many rows, ``method="auto"`` switches from exact to clustered search.
AUTO_EXACT_MAX_ROWS = 50_000
#: Target mean cluster size when ``n_clusters`` is not given.
TARGET_CLUSTER_SIZE = 2_000

_BLOCK = 4096


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """Return float32 rows scaled to unit L2 norm (zero rows stay zero)."""
    x = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1)


def _scan(
    x: np.ndarray,
    threshold: float,
    *,
    collect_pairs: bool,
    block: int = _BLOCK,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, ...]]]:
    """Exact blocked search over normalised rows ``x``.

    Returns, per row, the index of and similarity to its most similar earlier row (-1
    and -inf for the first row), plus the (i, j, sim) pairs at or above ``threshold``
    when ``collect_pairs`` is set.
    """
    n = len(x)
    best_idx = np.full(n, -1, dtype=np.int64)
    best_sim = np.full(n, -np.inf, dtype=np.float32)
    pairs: list[tuple[np.ndarray, ...]] = []
    for j0 in range(0, n, block):
        j1 = min(j0 + block, n)
        cols = x[j0:j1]
        for i0 in range(0, j1, block):
            i1 = min(i0 + block, n)
            sim = x[i0:i1] @ cols.T
            if i1 > j0:
                # Only earlier rows count: mask i >= j where the blocks overlap
                ii = np.arange(i0, i1)[:, None]
                jj = np.arange(j0, j1)[None, :]
                sim = np.where(ii < jj, sim, -np.inf)
            col_best = sim.argmax(axis=0)
            col_sim = sim[col_best, np.arange(j1 - j0)]
            better = col_sim > best_sim[j0:j1]
            best_sim[j0:j1] = np.where(better, col_sim, best_sim[j0:j1])
            best_idx[j0:j1] = np.where(better, col_best + i0, best_idx[j0:j1])
            if collect_pairs:
                ri, rj = np.nonzero(sim >= threshold)
                if len(ri):
                    pairs.append((ri + i0, rj + j0, sim[ri, rj]))
    return best_idx, best_sim, pairs


def spherical_kmeans(
    x: np.ndarray,
    n_clusters: int,
    *,
    n_iter: int = 10,
    sample_size: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Fit unit-norm centroids to normalised rows by cosine similarity.

    Centroids are fitted on a random sample of at most ``sample_size`` rows.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    n_clusters = max(1, min(n_clusters, n))
    sample = x if n <= sample_size else x[rng.choice(n, sample_size, replace=False)]
    centroids = sample[rng.choice(len(sample), n_clusters, replace=False)].copy()
    for _ in range(n_iter):
        labels = _assign(sample, centroids)[:, 0]
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, sample)
        counts = np.bincount(labels, minlength=n_clusters)
        empty = counts == 0
        if empty.any():
            sums[empty] = sample[rng.choice(len(sample), empty.sum(), replace=False)]
        centroids = normalize_rows(sums)
    return centroids


def _assign(
    x: np.ndarray,
    centroids: np.ndarray,
    top: int = 1,
    chunk: int = 65_536,
) -> np.ndarray:
    """Labels of each row's ``top`` most similar centroids, shape (n, top)."""
    top = min(top, len(centroids))
    parts = []
    for i in range(0, len(x), chunk):
        sim = x[i : i + chunk] @ centroids.T
        if top == 1:
            parts.append(sim.argmax(axis=1)[:, None])
        else:
            parts.append(np.argpartition(-sim, top - 1, axis=1)[:, :top])
    return np.concatenate(parts)


def assign_clusters(
    x: np.ndarray,
    n_clusters: int | None = None,
    *,
    assignments: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Spherical k-means labels of each row's ``assignments`` nearest clusters, (n, a).

    ``n_clusters`` defaults to one cluster per ~2,000 rows.
    """
    k = n_clusters or max(1, round(len(x) / TARGET_CLUSTER_SIZE))
    return _assign(x, spherical_kmeans(x, k, seed=seed), top=assignments)


def _resolve_method(method: Method, n: int) -> Literal["exact", "clustered"]:
    if method == "auto":
        return "exact" if n <= AUTO_EXACT_MAX_ROWS else "clustered"
    if method not in ("exact", "clustered"):
        raise ValueError(f"Unknown method: {method!r}")
    return method


def _cluster_members(
    x: np.ndarray,
    n_clusters: int | None,
    assignments: int,
    seed: int,
) -> list[np.ndarray]:
    """Row positions in each cluster, ascending (so "earlier" is preserved)."""
    labels = assign_clusters(x, n_clusters, assignments=assignments, seed=seed)
    flat_labels = labels.ravel()
    flat_rows = np.repeat(np.arange(len(x)), labels.shape[1])
    order = np.argsort(flat_labels, kind="stable")
    bounds = np.flatnonzero(np.diff(flat_labels[order])) + 1
    return np.split(flat_rows[order], bounds)


def most_similar_earlier(
    embeddings: np.ndarray,
    *,
    method: Method = "auto",
    n_clusters: int | None = None,
    assignments: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row, the index of (and cosine similarity to) its most similar earlier row.

    Rows with no earlier row (or, for ``clustered``, none in their cluster) get -1 and
    -inf.
    """
    x = normalize_rows(embeddings)
    if _resolve_method(method, len(x)) == "exact":
        best_idx, best_sim, _ = _scan(x, np.inf, collect_pairs=False)
        return best_idx, best_sim
    best_idx = np.full(len(x), -1, dtype=np.int64)
    best_sim = np.full(len(x), -np.inf, dtype=np.float32)
    for members in _cluster_members(x, n_clusters, assignments, seed):
        idx, sim, _ = _scan(x[members], np.inf, collect_pairs=False)
        better = sim > best_sim[members]
        best_idx[members[better]] = members[idx[better]]
        best_sim[members[better]] = sim[better]
    return best_idx, best_sim


def similar_pairs(
    embeddings: np.ndarray,
    threshold: float,
    *,
    method: Method = "auto",
    n_clusters: int | None = None,
    assignments: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs ``i < j`` with cosine similarity ``>= threshold``, as (i, j, sim)."""
    x = normalize_rows(embeddings)
    if _resolve_method(method, len(x)) == "exact":
        groups = [np.arange(len(x))]
    else:
        groups = _cluster_members(x, n_clusters, assignments, seed)
    found: list[tuple[np.ndarray, ...]] = []
    for members in groups:
        _, _, pairs = _scan(x[members], threshold, collect_pairs=True)
        found.extend((members[i], members[j], s) for i, j, s in pairs)
    if not found:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float32)
    left, right, sim = (np.concatenate(parts) for parts in zip(*found))
    # Within a cluster, members are in original order, so left < right already holds.
    # A pair sharing two clusters is found twice: keep one.
    order = np.lexsort((right, left))
    left, right, sim = left[order], right[order], sim[order]
    first = np.ones(len(left), dtype=bool)
    first[1:] = (left[1:] != left[:-1]) | (right[1:] != right[:-1])
    return left[first], right[first], sim[first]
