# 2026-10-17: Query-Aware Distillation

This is the deepest open thread. Luxical-One fails at short-query search because of how it
was trained, not because of what it is: a TF-IDF n-gram bag feeding a small ReLU MLP. The
full training recipe is open source in `datologyai/luxical`, so someone could distil a
variant that also handles queries. Nobody has published one. The Luxical paper's
future-work list mentions distilled variants and connections to static embedding models,
but not query handling specifically.

## Where the current recipe leaves queries out

- **Pipeline:** upstream `datologyai/luxical` ships it as three scripts under
  `projects/luxical_one/`: step 1 teacher-embeds the training docs, step 2 mines n-grams
  with Space-Saving, and step 3 trains the student. These sit on top of
  `src/luxical/training.py` and `src/luxical/teacher_embedder.py`.
- **Teacher:** `Snowflake/snowflake-arctic-embed-m-v2.0`, an *asymmetric* retrieval model.
  Queries get a `query: ` prefix and documents are embedded without one. Arctic's
  query→document geometry is exactly what search needs. The upstream teacher wrapper already
  supports both sides (`QUERY_PREFIX = "query: "`, `DOC_PREFIX = ""`, and an `is_query`
  argument in `teacher_embedder.py:47-48, 109-134`). Step 1 calls it with `is_query=False`
  only (`step_1_teacher_embed_training_docs.py:111`). The query path is written but unused.
- **Data:** 50M English FineWeb documents, with the teacher's document-side embeddings
  only.
- **Loss:** per-batch Gram-matrix KL (τ = 3.0). The paper reports batch 3072, while the
  step-3 script defaults to `12 * 1024`. The student matches the teacher's
  document×document similarity distribution row by row.
- **Student dims:** `(96, 3072, 3072, 192)` by the step-3 default. A second tower for
  queries could share the 2M × 96 first layer, which holds nearly all the parameters, and
  differ only in the about 10M dense weights.
- **Result:** the student never sees a short input, never sees a query-side teacher vector,
  and is never asked to place a query near its answering documents.

## Candidate recipes

1. **Mixed Gram batches.** Add query strings, such as MS MARCO, Natural Questions or
   synthetic LLM-generated queries for FineWeb documents, to each batch. Their teacher targets
   are query-mode embeddings. The Gram matrix then contains query×document blocks, and the KL
   loss teaches query→document geometry directly. This needs the smallest code change: new
   data plus a step-1 pass with `is_query=True` for the query rows.
2. **Separate query tower.** Keep the document student frozen and train a second small MLP
   (shared n-gram vocabulary, IDF and possibly the 2M × 96 first layer) only on queries, with a query×document Gram loss against
   frozen document embeddings. This is analogous to asymmetric static models. It costs a
   second set of weights and a `mode` argument at inference.
3. **Contrastive fine-tune on top of the distilled student.** This is the route Model2Vec
   took from `potion-base-32M` to `potion-retrieval-32M`: distil first, then fine-tune on
   retrieval pairs. The potion numbers give a reference point. 32.67 → 35.06 MTEB retrieval
   from the retrieval fine-tune, against 42.92 for MiniLM-L6-v2 (MinishLab, self-reported).

## The capacity question

Can a bag of n-grams align queries with documents at all? The input has no word order
beyond the 5-gram window and no context, and short queries activate few features. Static
models face the same constraint and reach about 82% of MiniLM on retrieval. Luxical has two
structural differences from static models:

- **(a)** a vocabulary of up to 5-grams, which captures multi-word phrases a unigram static
  model averages away;
- **(b)** a nonlinear MLP after the bag, where static models use a linear mean.

Both should help in principle. Whether they do on short queries is the experiment.

A secondary question is **IDF on queries**. The IDF weights come from FineWeb documents.
In a 5-word query, every term has a count of about 1, so TF-IDF reduces to IDF alone. Whether
document-mined IDF is the right weighting for query terms is untested.

## Prerequisites in this repo

- Loading arbitrary models (`2026-10-17-custom-model-loading.md`). A new student is
  unloadable today unless given as a local path, and its output width is ignored by schema
  resolution.
- A retrieval evaluation (`2026-10-17-static-embedding-comparison.md`). Without one there is
  no way to tell whether the variant improved.
- A query/document `mode` on `embed_text`, if recipe 2 is chosen.

## Current State

- `embed_text` applies the same tokenization, TF-IDF and MLP path to every input string with no query or document mode (src/embedder/inference.rs:14-41, src/expressions.rs:10-14)
- The registry loads models from a local path when the path exists, otherwise only luxical-one by Hugging Face ID (src/registry.rs:119-133)
- The README attributes luxical-one to distillation from snowflake-arctic-embed-m-v2.0 (README.md:132)

## Missing

- No query-mode or second-tower weights in the model loaders (src/embedder/safetensors_loader.rs, src/embedder/npz_loader.rs)
- No training or distillation code in polars-luxical (repository contains inference only; training lives in upstream `datologyai/luxical` `projects/luxical_one/`)
- No query-to-document retrieval evaluation (benchmark/)
