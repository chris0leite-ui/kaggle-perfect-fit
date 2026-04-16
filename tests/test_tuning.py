"""Tests for src/tuning.py — hyperparameter grid search utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.tuning import grid_search_cv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _regression_df(n=120):
    """Synthetic DataFrame matching the real data schema."""
    rng = np.random.RandomState(42)
    city = rng.choice(["Zaragoza", "Albacete"], n)
    city_num = np.where(city == "Zaragoza", 1.0, 0.0)
    x4 = np.concatenate([rng.uniform(-2, -0.5, n // 2),
                         rng.uniform(0.5, 2, n // 2)])
    x8 = rng.normal(0, 1, n)
    noise = rng.normal(0, 2, n)
    target = 20 * city_num + 30 * x4 + x8 + noise

    df = pd.DataFrame({
        "id": range(n),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "Country": "Spain",
        "City": city,
        "x4": x4,
        "x5": np.where(rng.rand(n) < 0.15, 999.0, rng.normal(5, 1, n)),
        "x6": rng.normal(0, 1, n),
        "x7": rng.normal(0, 1, n),
        "x8": x8,
        "x9": 0.8 * x4 + rng.normal(0, 0.3, n),
        "x10": rng.normal(0, 1, n),
        "x11": rng.normal(0, 1, n),
        "target": target,
    })
    return df


# ---------------------------------------------------------------------------
# grid_search_cv tests
# ---------------------------------------------------------------------------

class TestGridSearchCV:

    def test_returns_expected_keys(self):
        df = _regression_df()

        def builder(alpha=1.0):
            from sklearn.linear_model import Ridge
            from src.features import build_preprocessor
            prep = build_preprocessor("linear")
            return Pipeline([
                ("preprocessor", prep),
                ("to_array", FunctionTransformer(
                    lambda X: X.values if hasattr(X, "values") else X,
                    validate=False)),
                ("model", Ridge(alpha=alpha)),
            ])

        param_grid = {"alpha": [0.1, 1.0, 10.0]}
        result = grid_search_cv(builder, param_grid, df, n_splits=3)

        assert "best_params" in result
        assert "best_score" in result
        assert "all_results" in result
        assert isinstance(result["best_score"], float)
        assert isinstance(result["best_params"], dict)

    def test_all_results_length(self):
        df = _regression_df()

        def builder(alpha=1.0):
            from sklearn.linear_model import Ridge
            from src.features import build_preprocessor
            prep = build_preprocessor("linear")
            return Pipeline([
                ("preprocessor", prep),
                ("to_array", FunctionTransformer(
                    lambda X: X.values if hasattr(X, "values") else X,
                    validate=False)),
                ("model", Ridge(alpha=alpha)),
            ])

        param_grid = {"alpha": [0.1, 1.0, 10.0]}
        result = grid_search_cv(builder, param_grid, df, n_splits=3)
        assert len(result["all_results"]) == 3  # 3 param combos

    def test_best_score_is_minimum(self):
        """best_score should be the lowest MAE across all combos."""
        df = _regression_df()

        def builder(alpha=1.0):
            from sklearn.linear_model import Ridge
            from src.features import build_preprocessor
            prep = build_preprocessor("linear")
            return Pipeline([
                ("preprocessor", prep),
                ("to_array", FunctionTransformer(
                    lambda X: X.values if hasattr(X, "values") else X,
                    validate=False)),
                ("model", Ridge(alpha=alpha)),
            ])

        param_grid = {"alpha": [0.01, 1.0, 100.0]}
        result = grid_search_cv(builder, param_grid, df, n_splits=3)
        all_scores = [r["cv_mean"] for r in result["all_results"]]
        assert result["best_score"] == min(all_scores)

    def test_multi_param_grid(self):
        """Grid with 2 parameters produces correct number of combos."""
        df = _regression_df()

        def builder(alpha=1.0, fit_intercept=True):
            from sklearn.linear_model import Ridge
            from src.features import build_preprocessor
            prep = build_preprocessor("linear")
            return Pipeline([
                ("preprocessor", prep),
                ("to_array", FunctionTransformer(
                    lambda X: X.values if hasattr(X, "values") else X,
                    validate=False)),
                ("model", Ridge(alpha=alpha, fit_intercept=fit_intercept)),
            ])

        param_grid = {"alpha": [0.1, 1.0], "fit_intercept": [True, False]}
        result = grid_search_cv(builder, param_grid, df, n_splits=3)
        assert len(result["all_results"]) == 4  # 2x2
