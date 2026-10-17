# 2026-10-17: Custom Model Loading

polars-luxical was written for exactly one model, and the code assumes that in several
places. That was fine in December 2025, when `luxical-one` was the only Luxical model.
Upstream has released nothing new since. Still, any of the research threads (a query-aware
distillation, a domain-specific retrain, a multilingual variant, a quantized export) produces
a *new* model that the plugin cannot load cleanly. This entry records each single-model
assumption.

## The assumptions

1. **The registry is a hardcoded match.** `get_model_info` recognises only
   `"datologyai/luxical-one"` and `"luxical-one"`, with literal Hugging Face URLs for the
   safetensors file (`artifacts/luxical-one/model.safetensors`) and the npz file
   (`luxical_one_rc4.npz`). Any other ID fails with
   `Unknown model '…'. Known models: DatologyAI/luxical-one`. There is no generic
   `{repo}/resolve/main/...` fallback and no `hf-hub` resolution.
2. **The cache is flat and keyed by filename alone.** The cache directory is
   `dirs::cache_dir()/polars-luxical/`, and the files are named `model.safetensors` and
   `luxical_one_rc4.npz` with no per-model subdirectory. A second model whose `ModelInfo` also
   names `model.safetensors` would read luxical-one's cached weights instead of downloading
   its own. There is no revision pinning either. If DatologyAI re-uploads weights to `main`,
   existing caches keep the old file indefinitely and new installs get the new one.
3. **The output dtype comes from the default model, whichever model is requested.**
   `output_type_func` calls `get_or_load_model(&None)`, which resolves to
   `DEFAULT_MODEL_ID`, and builds the schema `Array[f32, dim]` from that model's
   `output_dim`. `embed_text` ignores `kwargs.model_id` during schema resolution. A model with a
   different output width would produce a Series whose width disagrees with the declared
   schema. Schema resolution also triggers a download or load of luxical-one even when the
   user only registered a local model.
4. **Local paths work, with a quirk.** `load_model_impl` checks `PathBuf::from(model_name).exists()`
   first, so `register_model("/path/to/model.safetensors")` loads. The registry key is the
   lowercased path string, so mixed-case paths on a case-sensitive filesystem register under a
   different key from the one the user typed. `get_or_load_model` lowercases consistently, so
   lookups still match.
5. **`providers` is accepted and dropped.** The Python `register_model(model_name, providers=None)`
   signature mirrors polars-fastembed, and the README documents `providers` as ignored.

## Why it matters now

The mechanics of loading custom models gate every research task in the other entries:

- `2026-10-17-query-aware-distillation.md` produces new weights that must load by path or
  by Hugging Face ID, possibly with a different output dim.
- `2026-10-17-quantization.md` may want a uint8 or f16 weight export.
- A domain retrain (for example on code or scientific text, which FineWeb under-represents)
  changes the n-gram vocabulary and IDF.

The safetensors format the loader reads already carries everything the model needs:
tokenizer JSON, recognised n-grams, IDF, `num_layers`, and `nn_layer_{i}`. Generic loading
only needs a URL template and a per-model cache directory. The format itself needs no
changes.

## Current State

- `get_model_info` maps only `datologyai/luxical-one` and `luxical-one` to hardcoded Hugging Face safetensors and npz URLs (src/registry.rs:20-31)
- `load_model_impl` loads from a local path when `PathBuf::from(model_name)` exists, then from cached `model.safetensors` or `luxical_one_rc4.npz`, then by download (src/registry.rs:119-166)
- The model cache stores files directly under `dirs::cache_dir()/polars-luxical` with no per-model or per-revision subdirectory (src/registry.rs:40-44, src/registry.rs:136-149)
- `output_type_func` resolves the output width from `get_or_load_model(&None)`, which loads `DEFAULT_MODEL_ID` `DatologyAI/luxical-one` (src/expressions.rs:16-24, src/registry.rs:17, src/registry.rs:89-93)
- The safetensors loader reads `embedder.tokenizer`, recognised n-grams, IDF values, `embedder.num_layers` and `embedder.nn_layer_{i}` tensors (src/embedder/safetensors_loader.rs:38-118)
- The Python `register_model` accepts a `providers` argument and passes only `model_name` to the Rust registry (python/polars_luxical/__init__.py:37-47)

## Missing

- No generic Hugging Face repository resolution for model IDs outside the hardcoded match (src/registry.rs:20-31)
- No revision or hash pinning for downloaded weights (src/registry.rs:151-166)

## Divergence

- The README states models download from the Hugging Face Hub (README.md:30-31) while `load_model_impl` rejects every Hugging Face ID other than luxical-one with "Unknown model" (src/registry.rs:127-133)
