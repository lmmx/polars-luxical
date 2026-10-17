# 2026-10-17: Test Suite Health

The Python test suite appears to cover the core API: basic embedding, null handling, the
namespace methods and the registry. In practice it cannot fail on a wrong result, and it does
not run in CI anyway.

## The swallowed-assertion pattern

Every test in `tests/embed_test.py` wraps its whole body in:

```python
try:
    ...
    assert ...
except Exception as e:
    pytest.skip(f"Model not available: {e}")
```

The intent was to skip when the model cannot be downloaded. But `AssertionError` is a
subclass of `Exception`, so **every failed assertion becomes a skip**. A test run where
every assertion fails reports as all-skipped, which looks the same as having no network.

The suite already contains a concrete case. `test_embed_text_basic` asserts the output dtype
is `pl.Array(pl.Float32, width=256)`, with the comment "Adjust dim as needed". Luxical-One
outputs 192 dims (the README's Available Models table, and `output_dim` taken from the last
layer's row count). With the model available, that assertion fails, the `except` catches it,
and the test is reported as skipped with the message "Model not available: …". This is the
one test that checks the model's output shape, and it has been silently skipping since it
was written.

256 is the *teacher's* (arctic-embed-m-v2.0) output dimension, which suggests where the
number came from.

The narrow fix is to restrict the skip to the download/load step: catch the error from
`register_model` only, or use `pytest.importorskip`-style gating on a cached-model
fixture, and let assertions propagate.

## CI does not run the tests

`.github/workflows/test.yml.disabled` holds the Test workflow. It fans out over a Python
version matrix into the reusable `test-platform.yml`, which runs `maturin develop --release`
and then `pytest -s -v`. The `.disabled` suffix means GitHub never loads it. The only active
workflows are `CI.yml`, which builds wheels for Linux, musllinux, Windows and macOS plus an
sdist and publishes to PyPI, and `test-platform.yml`, which is only reachable through the
disabled caller. So releases ship without a single pytest run.

The Rust side has unit tests (`src/ngrams.rs`, `src/tfidf.rs`) for FNV hashing determinism,
n-gram padding and counts, and TF-IDF row normalization. No workflow runs `cargo test`
either.

## Parity is untested

There is no test comparing polars-luxical output against the reference Python `luxical`
package, even approximately (cosine ≥ 0.999 on a fixed sample). The Rust reimplementation
has hand-matched details that could drift silently: the FNV-1a byte layout, the `u32::MAX`
padding sentinel, the TF-IDF formula, and the placement of normalization between layers. The
doc-half score is too coarse to catch a subtle mismatch.

## Current State

- `tests/embed_test.py` defines `test_embed_text_basic`, `test_embed_handles_nulls`, `test_namespace_api` and `test_registry_operations`, each downloading or loading `datologyai/luxical-one` inside a `try` block (tests/embed_test.py:7, 42, 67, 104)
- Each test in `tests/embed_test.py` catches `Exception` and calls `pytest.skip` with the message "Model not available" — `AssertionError` raised inside the `try` block is reported as a skip rather than a failure (tests/embed_test.py:38-39, 63-64, 99-100, 122-123)
- `test_embed_text_basic` asserts an output dtype of `pl.Array(pl.Float32, width=256)` (tests/embed_test.py:32-35)
- `output_dim` takes the row count of the final MLP layer, and the README lists 192 as the luxical-one embedding dimension (src/embedder/safetensors_loader.rs:108, README.md:132)
- Rust unit tests cover FNV-1a determinism, n-gram padding, n-gram counts and TF-IDF row normalization (src/ngrams.rs:47-77, src/tfidf.rs:41-65)
- The Test workflow sits at `.github/workflows/test.yml.disabled` and calls the reusable `test-platform.yml`, which runs `maturin develop --release` then `pytest -s -v` (.github/workflows/test-platform.yml:54-61)
- `CI.yml` builds wheels for Linux, musllinux, Windows and macOS plus an sdist and publishes to PyPI without a test job (.github/workflows/CI.yml:79-233)

## Missing

- No test compares polars-luxical embeddings with the reference `luxical` Python package output (tests/)
- No active workflow runs `pytest` or `cargo test` (.github/workflows/CI.yml, .github/workflows/test.yml.disabled)

## Divergence

- `test_embed_text_basic` expects width 256 (tests/embed_test.py:32-35) while the README documents luxical-one as 192-dimensional (README.md:132)
