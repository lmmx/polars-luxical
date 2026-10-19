# Benchmarks

Run from this directory with `uv run --frozen python <script>`.

| script | what it measures |
|--------|------------------|
| `src/benchmark/__init__.py` (`just bench`) | Time to embed all PEPs and run one short query (the query result is not meaningful: Luxical is not trained on queries) |
| `doc_half_match_demo.py [--parquet SHARD --docs N]` | Document-half matching on PEPs or a FineWeb shard: Luxical half->half, Luxical half->whole, and a TF-IDF baseline |
| `dedup_eval.py SHARD` | Near-duplicate recall on injected edited copies vs untouched documents flagged, per threshold, against TF-IDF |
| `dedup_fineweb.py SHARD [--rows N]` | End-to-end semantic deduplication of a FineWeb shard, with timings and sampled pairs |

`SHARD` is a FineWeb sample-10BT parquet file, e.g.
https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet
(1,048,581 documents, 2.1 GB). Results from these scripts are recorded in the top-level
README and in `docs/journal/2026-10-19-foundation-and-dedup.md`.

## PEP embedding speed

The benchmark to beat is polars-fastembed with `Xenova/all-MiniLM-L6-v2`, which took 30s
on CPU and 20s on GPU to embed and retrieve over all PEPs (3.6M tokens, about 8ms per
1k tokens on CPU).

polars-luxical embeds the same 708 PEPs in about 1.4s on a 4-core container, about
0.4ms per 1k tokens.
