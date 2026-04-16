"""Tests for src/diagnostics.py — model diagnostics and interpretability."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.diagnostics import (
    compute_ks_tests,
    compute_residuals,
    compute_cluster_mae,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df(n=120):
    """Synthetic DataFrame matching the real data schema."""
    rng = np.random.RandomState(42)
    city = rng.choice(["Zaragoza", "Albacete"], n)
    city_num = np.where(city == "Zaragoza", 1.0, 0.0)
    x4 = np.concatenate([rng.uniform(-2, -0.5, n // 2),
                         rng.uniform(0.5, 2, n // 2)])
    x9 = 0.8 * x4 + rng.normal(0, 0.3, n)
    noise = rng.normal(0, 2, n)
    target = 20 * city_num + 30 * x4 + rng.normal(0, 1, n) * 12 + noise

    return pd.DataFrame({
        "id": range(n),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "Country": "Spain",
        "City": city,
        "x4": x4,
        "x5": np.where(rng.rand(n) < 0.15, 999.0, rng.normal(5, 1, n)),
        "x6": rng.normal(0, 1, n),
        "x7": rng.normal(0, 1, n),
        "x8": rng.normal(0, 1, n),
        "x9": x9,
        "x10": rng.normal(0, 1, n),
        "x11": rng.normal(0, 1, n),
        "target": target,
    })


def _simple_pipeline():
    """Minimal pipeline for testing residuals."""
    from src.features import build_preprocessor
    prep = build_preprocessor("linear")
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(
            lambda X: X.values if hasattr(X, "values") else X, validate=False)),
        ("model", LinearRegression()),
    ])


# ---------------------------------------------------------------------------
# KS test tests
# ---------------------------------------------------------------------------

class TestComputeKsTests:

    def test_returns_dataframe_with_correct_columns(self):
        df = _sample_df(100)
        train = df.iloc[:60]
        val = df.iloc[60:80]
        test = df.iloc[80:]
        features = ["x1", "x4", "x8"]
        result = compute_ks_tests(train, val, test, features)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"feature", "ks_stat_val", "p_val", "ks_stat_test", "p_test"}
        assert set(result.columns) == expected_cols
        assert len(result) == 3

    def test_identical_distributions_high_p(self):
        """Same data split identically should give high p-values."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 300)
        df = pd.DataFrame({"x": data, "target": data})
        train = df.iloc[:100]
        val = df.iloc[100:200]
        test = df.iloc[200:]
        result = compute_ks_tests(train, val, test, ["x"])
        assert result.iloc[0]["p_val"] > 0.05
        assert result.iloc[0]["p_test"] > 0.05


# ---------------------------------------------------------------------------
# Residual analysis tests
# ---------------------------------------------------------------------------

class TestComputeResiduals:

    def test_output_columns(self):
        df = _sample_df(100)
        train = df.iloc[:70].copy()
        val = df.iloc[70:].copy()
        pipe = _simple_pipeline()
        result = compute_residuals(pipe, train, val)
        assert "predicted" in result.columns
        assert "residual" in result.columns
        assert "abs_residual" in result.columns
        assert "cluster" in result.columns

    def test_residual_definition(self):
        """residual = actual - predicted."""
        df = _sample_df(100)
        train = df.iloc[:70].copy()
        val = df.iloc[70:].copy()
        pipe = _simple_pipeline()
        result = compute_residuals(pipe, train, val)
        expected_resid = result["target"] - result["predicted"]
        np.testing.assert_allclose(result["residual"].values,
                                   expected_resid.values, atol=1e-10)

    def test_row_count(self):
        df = _sample_df(100)
        train = df.iloc[:70].copy()
        val = df.iloc[70:].copy()
        pipe = _simple_pipeline()
        result = compute_residuals(pipe, train, val)
        assert len(result) == 30


class TestComputeClusterMae:

    def test_all_clusters_present(self):
        df = _sample_df(200)
        train = df.iloc[:140].copy()
        val = df.iloc[140:].copy()
        pipe = _simple_pipeline()
        resid_df = compute_residuals(pipe, train, val)
        cluster_mae = compute_cluster_mae(resid_df)
        # Should have entries for clusters present in val
        assert len(cluster_mae) > 0
        assert "cluster" in cluster_mae.columns
        assert "mae" in cluster_mae.columns
        assert "n" in cluster_mae.columns
