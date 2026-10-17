# 2026-10-17: Inference Performance

The 1.8s PEP benchmark (about 3.6M tokens, about 0.5ms per 1k tokens) already beats
polars-fastembed/MiniLM by roughly 16x on CPU, and this entry is not a complaint about speed.
But reading `src/embedder/inference.rs` shows the current number comes from a naive
implementation. Several structural choices leave throughput on the table. The paper's
Python+Numba pipeline reports 23.1 MiB/s end to end on an M4 Max for the curation run. No
like-for-like MiB/s figure for polars-luxical exists to compare against.

## What the hot path does

`embed_batch` runs four sequential stages over the whole batch Polars hands it:

1. **Tokenize** each text with `tokenizer.encode(text, false)` in a plain `.iter().map()`,
   one text at a time on one thread. There is no truncation, which matches Luxical's
   whole-document intent.
2. **Build the BoW.** For each document, `extract_ngrams_hashed` allocates a `Vec<i64>` of
   every 1- to 5-gram hash. `hash_ngram` allocates a padded `Vec<u32>` and then a `Vec<u8>`
   per n-gram. So for a 5,000-token document there are about 25,000 n-grams, each costing
   two heap allocations plus an FNV pass, followed by a `HashMap<i64, u32>` lookup into a
   map of about 2M entries. Counts go into a per-document `HashMap<u32, u32>`, then into a
   `TriMat`, then into CSR.
3. **TF-IDF + normalize** builds a second CSR matrix.
4. **Forward.** `sparse_dense_matmul` covers the first layer, then come dense `dot`s.

## Specific inefficiencies

- **`rayon` is a dependency that nothing uses.** It is declared in `Cargo.toml`, but no
  `par_iter` appears anywhere in `src/`. Tokenization and BoW construction are both
  per-document and trivially parallel. Polars may parallelise across chunks, but a single
  contiguous String column usually arrives as one chunk.
- **The first-layer access pattern is strided.** Layers are stored row-major as
  `output_dim × input_dim`, using `Array2::from_shape_vec((shape[0], shape[1]), ...)` in
  the loaders. `sparse_dense_matmul` iterates the non-zeros of each sparse row and reads
  `dense.column(col_idx)`. Luxical-One's first layer is 96 × about 2M (upstream default dims
  `(96, 3072, 3072, 192)`). In row-major order each column's 96 elements are about 2M floats
  (about 8 MB) apart, so every active n-gram costs 96 scattered loads, each a likely cache
  and TLB miss. Storing the first layer transposed (`input_dim × 96`) makes each active
  n-gram's weights one contiguous 384-byte row, about six cache lines. That is an
  embedding-bag gather-and-sum, which is what FastText and Model2Vec do. The upstream repo
  even carries `projects/miscellaneous/sparse_matmul_benchmarking.py` for exactly this
  kernel.
- **mmap is defeated.** The safetensors loader slices the mmap'd tensor with
  `bytemuck::cast_slice` and then calls `.to_vec()`, copying every layer, including the
  96 × 2M first layer (about 768 MB as f32), into an owned heap array. Startup pays a full copy and resident memory
  doubles while it runs. The README says safetensors is preferred for "faster loading via
  mmap", but the copy undoes that benefit.
- **Allocation per n-gram.** `hash_ngram` builds a padded `Vec<u32>` and a `Vec<u8>` for
  every n-gram. A rolling FNV over a fixed-size stack buffer would allocate nothing.
- **Double CSR construction.** `TriMat → CSR → TF-IDF CSR` could be one pass. Each document's
  counts and IDF weights could be computed directly into the first-layer accumulator, with no
  materialised sparse matrix at all.

## Critique of the measurement itself

`benchmark/README.md` reports the 1.8s figure for "the same operation" as polars-fastembed:
embedding plus retrieval with a short query. It does not separate embedding time from
retrieval time, nor model-load time from inference time. It also states no hardware. The
MiniLM figure comes from a different library with different overheads. A per-stage timing
(tokenize, BoW, forward) on a stated CPU, in MiB/s so it is comparable with the paper's
23.1 MiB/s, would show where the remaining time goes.

## Current State

- `embed_batch` tokenizes, builds the BoW matrix, applies TF-IDF and runs the forward pass sequentially over one batch (src/embedder/inference.rs:14-41)
- Tokenization calls `tokenizer.encode(text, false)` per text through a sequential iterator (src/embedder/inference.rs:20-29)
- `build_bow_matrix` collects n-gram hashes per document into a `HashMap<u32, u32>` of counts, then into a `TriMat` converted to CSR (src/embedder/inference.rs:44-63)
- `hash_ngram` allocates a padded `Vec<u32>` and a `Vec<u8>` for every n-gram before hashing (src/ngrams.rs:19-28)
- `sparse_dense_matmul` reads `dense.column(col_idx)` from layers stored row-major as `output_dim × input_dim` (src/embedder/inference.rs:94-110, src/embedder/safetensors_loader.rs:103)
- The safetensors loader copies each mmap'd layer into an owned `Array2` with `layer_data.to_vec()` (src/embedder/safetensors_loader.rs:102-105)
- `rayon` appears in `Cargo.toml` dependencies with no use in `src/` (Cargo.toml:8)
- benchmark/README.md reports 1.8s for PEP embedding plus retrieval with no hardware description and no per-stage breakdown

## Missing

- No per-stage timing or MiB/s throughput measurement (benchmark/)

## Divergence

- The registry comment describes safetensors as "faster loading via mmap" (src/registry.rs:136) while the safetensors loader copies every layer out of the mmap into heap memory (src/embedder/safetensors_loader.rs:102-105)
