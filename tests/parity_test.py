"""Parity of the Rust implementation against the reference Python `luxical` package."""

import json
from pathlib import Path

import numpy as np
import polars as pl

FIXTURE = Path(__file__).parent / "data" / "upstream_parity.json"


def test_matches_upstream_luxical(model_id):
    """Embeddings agree with `luxical` 1.1.2 output stored in the fixture."""
    from polars_luxical import embed_text

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = np.array(fixture["embeddings"], dtype=np.float32)

    actual = (
        pl.DataFrame({"text": fixture["texts"]})
        .select(embed_text("text", model_id=model_id))
        .to_series()
        .to_numpy()
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5)
