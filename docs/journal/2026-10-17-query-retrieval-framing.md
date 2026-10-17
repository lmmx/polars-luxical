# 2026-10-17: Query Retrieval Framing

The README makes the right call in its Overview: Luxical models "were not trained on queries,
so you cannot use them for search!" Every runnable example in the repo then contradicts it.
The Usage block, `demo.py`, the `benchmark` entry point and the namespace test all show
`retrieve(query=<short natural-language string>)`. That is exactly the asymmetric
query→document pattern the model never learned.

Anyone skimming the README sees "Retrieve similar documents" with `query="Tell me about
speed"`. They copy it, get plausible-looking but semantically weak rankings, and conclude
the embeddings are bad. In fact the embeddings are good at a different job.

## Why short queries fail

The full mechanism is in `2026-10-17-state-of-the-library.md`. In short:

1. Luxical-One was distilled only on document–document similarity, using FineWeb Gram
   matrices with the teacher's document side.
2. A short query activates a handful of n-gram features. After TF-IDF and ℓ2 normalization,
   those few features dominate the input completely. The MLP is extrapolating there.
3. A bag of n-grams has no way to bridge paraphrase from so little evidence.

## What does work

The retrieval API itself is fine. The framing is the problem. `retrieve` is just a cosine
top-k against an embedded string, and it works well when that string is
document-shaped:

- **Query by example:** pass a whole document, or a substantial passage, as `query`. The
  doc-half results show this works, once the measurement caveat in
  `2026-10-17-doc-half-benchmark.md` is accounted for.
- **Seed centroid:** embed several seed documents, take the mean vector and renormalize, then
  rank against that. This is the standard pattern for targeted crawling and topic expansion
  ("find more like these 20 examples"). The current `retrieve` signature cannot express it,
  because the method only accepts a `str` query and always embeds it
  (python/polars_luxical/__init__.py:162-164). Passing a precomputed vector, or an
  expression, is not supported.
- **Two-stage:** Luxical produces a cheap shortlist from a document-shaped query, and a
  transformer (for example via polars-fastembed) reranks it.

The README already points to polars-fastembed for "search retrieval with a query"
(`README.md:177-179`). That pointer belongs next to the warning, where readers will see it.

## A smaller issue in `retrieve`

`retrieve` cross-joins the single query embedding onto every row, computes similarity, then
sorts the entire frame before `head(k)` (python/polars_luxical/__init__.py:166-190). For a
single query over a few thousand rows this costs nothing. When `retrieve` is called once per
row, as `doc_half_match_demo.py` does, the cost is O(N²) materialised rows plus N full sorts.
The demo also re-embeds every query text, even though the frame already holds embeddings for
those texts. A top-k without a full sort, or one matrix product over all queries, would avoid
both costs. This only affects benchmarks and dedup-style all-pairs uses, not the single-query
path.

## Current State

- `retrieve` accepts a single `str` query, embeds the query with `embed_text`, cross-joins the query embedding against every row, then sorts all rows by similarity before applying `k` (python/polars_luxical/__init__.py:133-190)
- `retrieve` computes cosine similarity through `polars_distance` `dist_arr.cosine` or a dot product through `pl.col(...).dot` (python/polars_luxical/__init__.py:170-175)
- The README Overview states Luxical models were not trained on queries and cannot be used for search (README.md:15)
- The README "See also" section points to polars-fastembed for retrieval with a query (README.md:177-179)

## Missing

- `retrieve` has no parameter for a precomputed query vector or a centroid of several seed embeddings (python/polars_luxical/__init__.py:133-142)
- The README has no example of query-by-document or seed-centroid retrieval (README.md:43-85)

## Divergence

- The README Usage block calls `retrieve` with the short query "Tell me about speed" (README.md:77-84) despite README.md:15 stating the models cannot be used for search
- `demo.py` calls `retrieve` with the short query "How do computers understand language?" (demo.py:41-52)
- The `benchmark` entry point retrieves against the short query "Typed dictionaries and mappings" (benchmark/src/benchmark/__init__.py:18, benchmark/src/benchmark/__init__.py:46-51)
- `test_namespace_api` exercises `retrieve` with the one-word query "Greeting" (tests/embed_test.py:90-95)
- README.md:16 states the benchmarks demonstrate fast but not useful search results, while benchmark/README.md contains only timing figures and no retrieval output
