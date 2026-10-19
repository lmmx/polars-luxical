import io
import sys

import pytest

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MODEL_ID = "datologyai/luxical-one"
EMBEDDING_DIM = 192


@pytest.fixture
def model_id() -> str:
    """Register the model, skipping the test if it cannot be downloaded or loaded."""
    from polars_luxical import register_model

    try:
        register_model(MODEL_ID)
    except RuntimeError as e:
        pytest.skip(f"Model not available: {e}")
    return MODEL_ID
