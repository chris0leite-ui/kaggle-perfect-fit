import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.formula import predict, NOISE_FEATURES, SENTINEL_VALUE


DATA_PATH = Path("data/dataset.csv")


@pytest.fixture
def dataset():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def clean_dataset(dataset):
    """Rows where x5 is not the sentinel value."""
    return dataset[dataset["x5"] != SENTINEL_VALUE].copy()


def test_predict_returns_array(clean_dataset):
    pred = predict(clean_dataset)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(clean_dataset)


def test_exact_rows_high_proportion(clean_dataset):
    """At least 93% of non-sentinel rows should be predicted exactly."""
    pred = predict(clean_dataset)
    residuals = clean_dataset["target"].values - pred
    exact = np.abs(residuals) < 0.01
    assert exact.sum() / len(clean_dataset) > 0.93


def test_r2_above_threshold(clean_dataset):
    """R² should be above 0.996 on non-sentinel rows."""
    pred = predict(clean_dataset)
    y = clean_dataset["target"].values
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.996


def test_x9_ge5_perfect(clean_dataset):
    """Rows with x9 >= 5 (equivalently x4 > 0) should be predicted perfectly."""
    subset = clean_dataset[clean_dataset["x9"] >= 5]
    pred = predict(subset)
    residuals = subset["target"].values - pred
    assert np.abs(residuals).max() < 1e-8


def test_noise_features_excluded():
    """x6 and x7 should be identified as noise."""
    assert "x6" in NOISE_FEATURES
    assert "x7" in NOISE_FEATURES


def test_sentinel_value():
    assert SENTINEL_VALUE == 999


def test_formula_coefficients_are_clean(clean_dataset):
    """Verify the formula uses exact integer/simple coefficients by checking
    the prediction matches a manually computed value for a known row."""
    row = clean_dataset[clean_dataset["x4"] > 0].iloc[0]
    expected = (
        -100 * row["x1"] ** 2
        + 10 * np.cos(5 * np.pi * row["x2"])
        + 15 * row["x4"]
        - 8 * row["x5"]
        + 15 * row["x8"]
        - 4 * row["x9"]
        + row["x10"] * row["x11"]
        - 25 * (1 if row["City"] == "Zaragoza" else 0)
        + 20
        + 92.5
    )
    pred = predict(pd.DataFrame([row]))[0]
    assert abs(pred - expected) < 1e-10
