# 2026-10-17: Quantization

The upstream `datologyai/luxical` README says "Luxical models typically experience no
quality degradation from uint8 quantization". It adds that dictionary-encoding formats such
as Parquet "may automatically compress roundtrip-quantized data by 4x".

The upstream scheme is `misc_utils.fast_8bit_uniform_scalar_quantize(emb, limit=0.5)`, a
single global uniform scalar quantizer:

- each component is clipped to [−0.5, 0.5];
- the range is split into 255 bins of width 1/255 ≈ 0.0039;
- the result is stored as `round((x + 0.5) / bin_width)`;
- the worst-case per-component error is about 0.002.

The training pipeline already uses the same idea on the teacher side. The step-1 output
dataset is named `…arctic2m_mrl_8bit_quantized_scale_0dot3`, which indicates MRL-truncated,
8-bit quantized teacher embeddings with a 0.3 scale.
polars-luxical emits `Array[f32, 192]`: 768 bytes per document before compression. uint8
would be 192 bytes, a 4x saving. For web-scale corpora this is the difference between
embeddings that fit in RAM and embeddings that do not.

## Why "no quality loss" needs checking per use case

The upstream claim is most plausibly about ranking: top-k neighbours and classifier inputs.
Ranking is robust to small uniform perturbations because ordering is preserved. Dedup is
different. It applies an *absolute* threshold (cosine ≥ 1 − ε) near the top of the
similarity range, where quantization error is largest relative to the gap being measured.
At ε = 0.05, a few thousandths of cosine error moves pairs across the boundary. The
questions:

- **Scheme.** The upstream global `limit=0.5` scheme is the reference. The final layer has
  no ReLU and outputs signed, unit-norm 192-dim vectors (src/embedder/inference.rs:84-88),
  so typical components are around 1/√192 ≈ 0.072 in magnitude. That spends most of the
  255 bins on a range few components reach. The open question is whether per-dimension
  scaling calibrated on a sample, or a tighter global limit, buys meaningful precision.
- **Threshold stability.** At ε ∈ {0.02, 0.05, 0.1}, how many pairs flip between f32 and uint8?
  This metric matters more for dedup than recall@k.
- **Where to compute.** Quantize after embedding, compute similarity in int arithmetic
  (int8 dot products are fast and SIMD-friendly), and dequantize only for display.
- **Storage.** Polars `Array[u8, 192]` in Parquet with zstd, checked against the upstream
  compression claim on the PEP corpus.

A related question is **weight quantization**. The first layer is 96 × about 2M f32, about
768 MB. That dominates model size and memory, and it is copied out of the mmap at
load (see `2026-10-17-inference-performance.md`). f16 or int8 weights with f32 accumulation
would shrink it 2–4x and improve cache behaviour in the gather.

## Current State

- `embed_text` returns `Array[f32, dim]` and casts the output builder to `Float32` arrays (src/expressions.rs:66-82)
- The final MLP layer applies L2 normalization without ReLU, so output components take signed values (src/embedder/inference.rs:84-88)
- Model layers load as `f32` arrays from safetensors or npz (src/embedder/safetensors_loader.rs:102-105)

## Missing

- No quantized embedding output type or quantization option on `embed_text` (src/expressions.rs:10-14)
- No measurement of near-duplicate threshold stability between f32 and quantized embeddings (benchmark/)
- No quantized weight format support in the loaders (src/embedder/safetensors_loader.rs, src/embedder/npz_loader.rs)
