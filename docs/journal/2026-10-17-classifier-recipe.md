# 2026-10-17: Classifier Recipe

The Luxical paper's strongest result is a classifier result, not a retrieval one. A small
MLP (two hidden layers of 256) on frozen Luxical-One embeddings was used as a quality filter
over a 600B-token FineWeb pool, filtering it to 60B tokens:

| Filter | Throughput | MMLU (3B model trained on filtered data) |
|---|---|---|
| Luxical-One + MLP | 23.1 MiB/s | 36.4% |
| DCLM FastText | 19.0 MiB/s | 36.4% |
| FineWeb-Edu classifier | 1.6 MiB/s, more than 10x slower even on GPU | similar |

MLP inference was under 0.25% of the Luxical pipeline's runtime. Embedding dominates; the
classifier head costs almost nothing. The blog adds that the quality differences between
filters were small next to the gain from filtering at all.

That makes "Luxical + tiny head" a drop-in replacement for FastText, with one advantage
FastText lacks. The *same embeddings* serve dedup, clustering and any number of classifier
heads, so a curation pipeline embeds once and reuses the vectors.

## The practitioner recipe

1. **Label a seed set with a strong judge.** A few thousand documents labelled by an LLM
   against a rubric, for example educational value 0–5 in FineWeb-Edu style, topic,
   toxicity, language register, or "is this code".
2. **Embed with Luxical.**
3. **Train a small head on the embeddings.** Logistic regression or a 2×256 MLP, trained in
   seconds on CPU.
4. **Score the full corpus.** Throughput is bounded by embedding, which is already the fast
   path.
5. **Threshold or bucket.** Keep the top X%, or stratify by score.

In Polars terms, the head is a 192→256→256→C matrix chain. It could run as a second plugin
expression over the embedding column, or with numpy over `to_numpy()`. Either way the scores
come back as a column that feeds `filter` or `group_by`.

## Open questions

- **How small can the labelled set be** before the Luxical+head falls behind a FastText model
  trained on the same labels? FastText learns its own features, while the head relies on
  frozen distilled ones. The crossover point is not published.
- **Head capacity.** Is a linear probe enough at 192 dims, or does the 2×256 MLP matter? The
  paper reports only the MLP.
- **Domain shift.** Luxical-One's n-gram vocabulary and IDF were mined from FineWeb. On text
  far from web English, such as code, LaTeX-heavy papers or chat logs, the active-feature
  count per document falls and embeddings degrade. FastText trained in-domain would not
  suffer the same way.

## Current State

- `embed_text` produces `Array[f32, 192]` embeddings for luxical-one as a Polars column (src/expressions.rs:26-85)
- The README lists classification among the model's target tasks (README.md:13)

## Missing

- No classifier head training or inference routine (src/, python/polars_luxical/)
- No example of training a probe or MLP on luxical embeddings (README.md, benchmark/)
- No comparison of a luxical-embedding classifier against FastText on a labelled set (benchmark/)
