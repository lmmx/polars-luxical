# 2026-10-17: Doc-Half Benchmark

The README presents the document-half matching demo as the main evidence that the model is
useful. It quotes Datology's framing, that a good embedder should place most document halves
within the top 1% of their match, and reports 97.46% top-1 on 708 PEPs. Reading the script
closely shows it measures something easier than Datology's task.

## What the script actually does

`benchmark/doc_half_match_demo.py`:

1. Splits each PEP in half by whitespace-separated words (`split_half`).
2. Builds a frame with `pep`, `first_half` and `second_half` columns.
3. Calls `df.luxical.embed(columns=["first_half", "second_half"], ...)`. `join_columns`
   defaults to `True`, so `embed` concatenates the two columns with a space and embeds the
   result (python/polars_luxical/__init__.py:116-125). **The `embedding` column therefore
   holds the embedding of the whole PEP**, not of its second half.
4. For each row, calls `retrieve(query=first_half_text)` against that column. It then finds
   where the row's own `second_half` string lands in the ranking.

So the task is: given the first half of a document, find the full document. The full
document contains the query verbatim. In a TF-IDF bag-of-n-grams representation, the target
vector is the query's own features plus the second half's features. Any lexical method,
including plain TF-IDF cosine with no learned projection, should score very highly here.
The 97.46% top-1 / 99.86% top-5 / mean rank 1.05 is real for this task. It is not evidence
of half-to-half semantic matching.

Datology's evaluation is half→half. The 50k FineWeb documents are split into 100k halves,
each half queries all the others, and the error rate is plotted against window size k. On
that task Luxical-One trails MiniLM-L6-v2 at strict top-1 and only overtakes it at wider
windows. A like-for-like PEP number needs the two halves embedded separately.

## What the misses show

The 18 misses are informative even under the easier task. They cluster in templated or
near-duplicate document families:

- the Steering Council election PEPs 8102/8103 and 8105/8106/8107;
- release-schedule PEPs 361/373 and 398/392;
- packaging metadata PEPs 241/314.

Those are exactly the cases where lexical overlap between *different* documents is high. In
other words, they are the near-duplicates a dedup pass would flag. The misses are a free
demonstration of the dedup use case.

## Secondary issues

- **Cost.** The evaluation loop calls `retrieve` once per row. Each call re-embeds the query
  text and cross-joins and fully sorts all 708 rows (benchmark/doc_half_match_demo.py:54-62).
  That is O(N²) and wastes time re-embedding text already in the frame. A single N×N
  similarity matrix from two embedding columns would compute every rank in one pass.
- **Rank lookup by string.** The rank comes from `retrieved_texts.index(second_half_text)`.
  When two PEPs have identical second halves, it returns the first match, which may be the
  wrong PEP. That is unlikely with PEPs but possible with templated documents.
- **Copy-paste docstring.** The benchmark package docstring says "Benchmark Polars
  FastEmbed" and "Deterministic, no network" (benchmark/src/benchmark/__init__.py:1-8). The
  package benchmarks polars-luxical, and `register_model` downloads weights on first use.
- **Comparison baselines.** No MiniLM, TF-IDF-only or potion numbers sit alongside the
  Luxical figure. Without them, 97% cannot be read as good or bad for the PEP corpus.

## Current State

- `doc_half_match_demo.py` splits each PEP at the midpoint word into `first_half` and `second_half` columns (benchmark/doc_half_match_demo.py:11-36)
- `doc_half_match_demo.py` embeds `first_half` and `second_half` concatenated into one `embedding` per PEP — `df.luxical.embed` joins multiple columns with a space when `join_columns` is `True`, the default (benchmark/doc_half_match_demo.py:43-47, python/polars_luxical/__init__.py:116-125)
- `doc_half_match_demo.py` calls `retrieve` once per row with the row's `first_half` as the query and records the position of the row's `second_half` string in the ranking (benchmark/doc_half_match_demo.py:54-70)
- The README reports 97.46% top-1, 99.86% top-5 and mean rank 1.05 over 708 PEPs, with 18 non-top-1 cases including PEPs 8102, 8106 and 8107 ranked below sibling Steering Council election PEPs (README.md:96-124)

## Missing

- No benchmark script embeds first and second halves separately for half-to-half matching (benchmark/doc_half_match_demo.py is the only doc-half script)
- No benchmark script reports a TF-IDF-only, MiniLM or static-embedding baseline on the PEP doc-half task (benchmark/)

## Divergence

- README.md:92 describes the demo as Datology's "matching document halves" task, while benchmark/doc_half_match_demo.py:43-47 matches first halves against whole-document embeddings that contain the query text
- The benchmark package docstring describes "Benchmark Polars FastEmbed" and "Deterministic, no network" (benchmark/src/benchmark/__init__.py:1-8), while the package imports `polars_luxical` and `register_model` downloads weights from Hugging Face when no cached copy is present (src/registry.rs:151-166)
