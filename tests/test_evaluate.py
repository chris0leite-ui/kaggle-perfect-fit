"""Tests for src/evaluate.py — evaluation framework."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.evaluate import (
    compare_models,
    cross_validate_model,
    evaluate_on_holdout,
    score_mae,
    split_val_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _holdout_df(n=300):
    """Minimal holdout-like DataFrame with City for stratification."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "City": np.where(rng.rand(n) < 0.5, "Zaragoza", "Albacete"),
        "x1": rng.normal(0, 1, n),
        "target": rng.normal(0, 10, n),
    })


def _simple_regression_data(n=100):
    """Simple y = 2*x + noise for pipeline smoke tests."""
    rng = np.random.RandomState(42)
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x, "target": y})
    return df


# ---------------------------------------------------------------------------
# split_val_test tests
# ---------------------------------------------------------------------------

class TestSplitValTest:

    def test_sizes(self):
        df = _holdout_df(300)
        val, test = split_val_test(df)
        assert len(val) == 150
        assert len(test) == 150

    def test_no_overlap(self):
        df = _holdout_df(300)
        val, test = split_val_test(df)
        combined = pd.concat([val, test], ignore_index=True)
        assert len(combined) == len(df)
        # No duplicate rows (check via target column which is float, effectively unique)
        assert combined["target"].nunique() == len(combined)

    def test_reproducible(self):
        df = _holdout_df(300)
        val1, _ = split_val_test(df, seed=42)
        val2, _ = split_val_test(df, seed=42)
        pd.testing.assert_frame_equal(val1.reset_index(drop=True),
                                      val2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# score_mae tests
# ---------------------------------------------------------------------------

class TestScoreMae:

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert score_mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert score_mae(y_true, y_pred) == 1.0


# ---------------------------------------------------------------------------
# cross_validate_model tests
# ---------------------------------------------------------------------------

class TestCrossValidate:

    def _make_pipeline(self):
        """Minimal pipeline: just a linear regression on raw 'x' column."""
        from sklearn.preprocessing import FunctionTransformer
        return Pipeline([
            ("select", FunctionTransformer(
                lambda df: df[["x"]].values, validate=False)),
            ("model", LinearRegression()),
        ])

    def test_returns_correct_keys(self):
        df = _simple_regression_data()
        pipe = self._make_pipeline()
        result = cross_validate_model(pipe, df, n_splits=3)
        assert "cv_scores" in result
        assert "cv_mean" in result
        assert "cv_std" in result

    def test_score_count_equals_n_splits(self):
        df = _simple_regression_data()
        pipe = self._make_pipeline()
        result = cross_validate_model(pipe, df, n_splits=5)
        assert len(result["cv_scores"]) == 5

    def test_dummy_model_sanity(self):
        """DummyRegressor should have higher MAE than a proper linear model."""
        df = _simple_regression_data(200)
        good_pipe = self._make_pipeline()
        dummy_pipe = Pipeline([
            ("select", __import__("sklearn.preprocessing", fromlist=["FunctionTransformer"]).FunctionTransformer(
                lambda df: df[["x"]].values, validate=False)),
            ("model", DummyRegressor(strategy="mean")),
        ])
        good_result = cross_validate_model(good_pipe, df, n_splits=5)
        dummy_result = cross_validate_model(dummy_pipe, df, n_splits=5)
        assert good_result["cv_mean"] < dummy_result["cv_mean"]


# ---------------------------------------------------------------------------
# evaluate_on_holdout tests
# ---------------------------------------------------------------------------

class TestEvaluateOnHoldout:

    def test_returns_mae_and_predictions(self):
        df = _simple_regression_data(100)
        from sklearn.preprocessing import FunctionTransformer
        pipe = Pipeline([
            ("select", FunctionTransformer(
                lambda df: df[["x"]].values, validate=False)),
            ("model", LinearRegression()),
        ])
        train = df.iloc[:70].copy()
        val = df.iloc[70:].copy()
        result = evaluate_on_holdout(pipe, train, val)
        assert "mae" in result
        assert "predictions" in result
        assert isinstance(result["mae"], float)
        assert len(result["predictions"]) == 30


# ---------------------------------------------------------------------------
# compare_models tests
# ---------------------------------------------------------------------------

class TestCompareModels:

    def test_sorted_by_val_mae(self):
        results = {
            "model_a": {"cv_mean": 12.0, "cv_std": 1.0, "val_mae": 15.0},
            "model_b": {"cv_mean": 10.0, "cv_std": 0.5, "val_mae": 9.0},
            "model_c": {"cv_mean": 11.0, "cv_std": 0.8, "val_mae": 11.0},
        }
        table = compare_models(results)
        assert isinstance(table, pd.DataFrame)
        assert list(table["val_mae"]) == [9.0, 11.0, 15.0]
