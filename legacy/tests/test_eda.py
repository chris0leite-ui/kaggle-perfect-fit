import pandas as pd
import numpy as np
from src.eda import detect_sentinels, zero_variance_cols, correlations


def test_detect_sentinels_finds_999():
    df = pd.DataFrame({"a": [1.0, 999.0, 3.0], "b": [999.0, 999.0, 2.0], "c": [1.0, 2.0, 3.0]})
    result = detect_sentinels(df)
    assert result == {"a": 1, "b": 2}


def test_detect_sentinels_empty():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert detect_sentinels(df) == {}


def test_zero_variance_cols():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": ["x", "x", "x"]})
    assert set(zero_variance_cols(df)) == {"a", "c"}


def test_correlations_sorted():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    df = pd.DataFrame({
        "strong": x + rng.standard_normal(100) * 0.1,
        "weak": x + rng.standard_normal(100) * 5,
        "target": x,
    })
    result = correlations(df, "target")
    assert result.index[0] == "strong"
    assert "target" not in result.index
