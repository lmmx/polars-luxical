# 2026-10-17: Dedup and Cluster Helpers

Both the README and Datology name clustering, classification and semantic deduplication as
the workloads where Luxical earns its keep. The DataFrame namespace offers only `embed` and
`retrieve`, and `retrieve` targets the one workload the model is weakest at. A user who
wants to dedup a corpus has to write the all-pairs or clustered similarity logic
themselves, outside the plugin.

## What semantic dedup looks like in practice

The SemDeDup recipe (Abbas et al., 2023, cited by the Luxical report) is the standard
approach at web scale:

1. Embed every document.
2. k-means into K clusters. K is chosen so clusters hold a few thousand documents, which
   makes within-cluster all-pairs cheap.
3. Within each cluster, compute the pairwise cosine matrix and mark every document whose
   maximum similarity to an earlier kept document exceeds 1 − ε.
4. Keep one representative per near-duplicate group. Common choices are the one closest to
   the cluster centroid, or the longest.

ε is the one knob that matters. It has to be calibrated per embedder and per corpus, by
eyeballing pairs at a few thresholds. A threshold tuned on transformer embeddings does not
transfer to Luxical's similarity scale, because Gram-matrix distillation with τ = 3.0 matches
*relative* similarity structure, not absolute cosine values.

Polars handles most of this without help:

- the embeddings are already `Array[f32, 192]`;
- cluster assignment is a `group_by`;
- within-cluster pairs are a self-join on the cluster id.

The missing pieces are an efficient k-means and a blocked pairwise similarity, which
avoids materialising a long-format cross join.

## Evidence from this repo

The doc-half run's misses are near-duplicate families. The Steering Council election PEPs
(8102/8103, 8105–8107), release-schedule PEPs (361/373, 398/392) and packaging-metadata PEPs
(241/314) all rank a sibling above the document's own other half. That is the signal a dedup
pass on the PEP corpus would surface, and it makes a ready-made worked example for docs:
"here are the near-duplicate clusters in the PEPs at ε = …".

## Helpers the namespace could expose

- `df.luxical.near_duplicates(embedding_column, threshold, by=None)`: pairs (or groups)
  above a cosine threshold, optionally restricted within a `by` column such as a cluster id.
- `df.luxical.cluster(embedding_column, k, ...)`: k-means assignment as a new column. This
  could live in Rust, alongside the embedder, for speed.
- `df.luxical.dedup(...)`: the composed SemDeDup pipeline, returning the kept rows.
- Seed-centroid retrieval and query-by-vector, described in
  `2026-10-17-query-retrieval-framing.md`.

## Current State

- The `luxical` DataFrame namespace defines `embed` and `retrieve` only (python/polars_luxical/__init__.py:88-190)
- `retrieve` computes similarity between one embedded query string and every row through a cross join (python/polars_luxical/__init__.py:162-176)
- The README names clustering, classification and semantic deduplication as the model's target tasks (README.md:13)
- The doc-half output lists PEPs 8102, 8106 and 8107 ranked below sibling Steering Council election PEPs 8103 and 8105 (README.md:122-124)

## Missing

- No namespace method or expression computes pairwise similarity within a frame (python/polars_luxical/__init__.py)
- No clustering or k-means routine (src/, python/polars_luxical/)
- No deduplication routine or near-duplicate threshold calibration example (README.md, benchmark/)
