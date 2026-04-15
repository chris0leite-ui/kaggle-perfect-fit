"""Tests for src/features.py — feature engineering transformers and pipeline builders."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import (
    CityEncoder,
    SentinelHandler,
    SplineBasisExpander,
    X9Residualizer,
    build_preprocessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df(n=60):
    """Synthetic DataFrame matching the real data schema."""
    rng = np.random.RandomState(42)
    x4 = np.concatenate([rng.uniform(-2, -0.5, n // 2),
                         rng.uniform(0.5, 2, n // 2)])
    x9 = 0.8 * x4 + rng.normal(0, 0.3, n)  # correlated with x4
    df = pd.DataFrame({
        "id": range(n),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "Country": "Spain",
        "City": rng.choice(["Zaragoza", "Albacete"], n),
        "x4": x4,
        "x5": np.where(rng.rand(n) < 0.2, 999.0, rng.normal(5, 1, n)),
        "x6": rng.normal(0, 1, n),
        "x7": rng.normal(0, 1, n),
        "x8": rng.normal(0, 1, n),
        "x9": x9,
        "x10": rng.normal(0, 1, n),
        "x11": rng.normal(0, 1, n),
        "target": rng.normal(0, 10, n),
    })
    return df


# ---------------------------------------------------------------------------
# CityEncoder tests
# ---------------------------------------------------------------------------

class TestCityEncoder:

    def test_maps_zaragoza_to_1(self):
        df = _sample_df()
        enc = CityEncoder()
        out = enc.fit_transform(df)
        mask = df["City"] == "Zaragoza"
        assert (out.loc[mask, "City"] == 1.0).all()

    def test_maps_albacete_to_0(self):
        df = _sample_df()
        enc = CityEncoder()
        out = enc.fit_transform(df)
        mask = df["City"] == "Albacete"
        assert (out.loc[mask, "City"] == 0.0).all()

    def test_drops_id_country_x6_x7_when_curated(self):
        df = _sample_df()
        enc = CityEncoder(drop_noise=True)
        out = enc.fit_transform(df)
        for col in ["id", "Country", "x6", "x7"]:
            assert col not in out.columns

    def test_keeps_x6_x7_when_all_vars(self):
        df = _sample_df()
        enc = CityEncoder(drop_noise=False)
        out = enc.fit_transform(df)
        assert "x6" in out.columns
        assert "x7" in out.columns
        # id and Country still dropped
        assert "id" not in out.columns
        assert "Country" not in out.columns

    def test_preserves_other_columns(self):
        df = _sample_df()
        enc = CityEncoder(drop_noise=True)
        out = enc.fit_transform(df)
        expected = {"City", "x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11", "target"}
        assert set(out.columns) == expected


# ---------------------------------------------------------------------------
# SentinelHandler tests
# ---------------------------------------------------------------------------

class TestSentinelHandler:

    def test_nan_strategy(self):
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)
        handler = SentinelHandler(strategy="nan")
        out = handler.fit_transform(enc)
        sentinel_mask = df["x5"] == 999.0
        assert out.loc[sentinel_mask, "x5"].isna().all()

    def test_median_strategy(self):
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)
        handler = SentinelHandler(strategy="median")
        out = handler.fit_transform(enc)
        # No 999.0 values remain
        assert (out["x5"] != 999.0).all()
        # No NaN either
        assert out["x5"].notna().all()

    def test_adds_sentinel_indicator(self):
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)
        handler = SentinelHandler(strategy="nan")
        out = handler.fit_transform(enc)
        assert "x5_is_sentinel" in out.columns
        sentinel_mask = df["x5"] == 999.0
        assert (out.loc[sentinel_mask, "x5_is_sentinel"] == 1).all()
        assert (out.loc[~sentinel_mask, "x5_is_sentinel"] == 0).all()

    def test_median_learned_at_fit_time(self):
        """Median from fit data is used at transform time, not recomputed."""
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)

        handler = SentinelHandler(strategy="median")
        handler.fit(enc)
        stored_median = handler.median_

        # Transform a different dataset — should use stored median
        df2 = _sample_df(40)
        enc2 = CityEncoder().fit_transform(df2)
        out = handler.transform(enc2)
        sentinel_mask = df2["x5"] == 999.0
        if sentinel_mask.any():
            replaced_vals = out.loc[sentinel_mask, "x5"]
            assert (replaced_vals == stored_median).all()

    def test_no_sentinels_present(self):
        df = _sample_df()
        df["x5"] = np.random.default_rng(0).normal(5, 1, len(df))  # no 999s
        enc = CityEncoder().fit_transform(df)
        handler = SentinelHandler(strategy="nan")
        out = handler.fit_transform(enc)
        assert out["x5"].notna().all()
        assert (out["x5_is_sentinel"] == 0).all()


# ---------------------------------------------------------------------------
# X9Residualizer tests
# ---------------------------------------------------------------------------

class TestX9Residualizer:

    def test_adds_x9_resid_column(self):
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)
        enc = SentinelHandler(strategy="median").fit_transform(enc)
        resid = X9Residualizer()
        out = resid.fit_transform(enc)
        assert "x9_resid" in out.columns

    def test_residuals_uncorrelated_with_x4(self):
        df = _sample_df(200)
        enc = CityEncoder().fit_transform(df)
        enc = SentinelHandler(strategy="median").fit_transform(enc)
        resid = X9Residualizer()
        out = resid.fit_transform(enc)
        corr = np.corrcoef(out["x9_resid"].values, out["x4"].values)[0, 1]
        assert abs(corr) < 0.05, f"x9_resid should be ~uncorrelated with x4, got r={corr:.3f}"

    def test_fit_transform_consistency(self):
        """Coefficients from fit are used at transform time."""
        df = _sample_df(200)
        enc = CityEncoder().fit_transform(df)
        enc = SentinelHandler(strategy="median").fit_transform(enc)

        resid = X9Residualizer()
        resid.fit(enc)

        # Transform new data with stored coefficients
        df2 = _sample_df(50)
        enc2 = CityEncoder().fit_transform(df2)
        enc2 = SentinelHandler(strategy="median").fit_transform(enc2)
        out = resid.transform(enc2)

        expected = enc2["x9"] - (resid.coef_ * enc2["x4"] + resid.intercept_)
        np.testing.assert_allclose(out["x9_resid"].values, expected.values, atol=1e-10)


# ---------------------------------------------------------------------------
# SplineBasisExpander tests
# ---------------------------------------------------------------------------

class TestSplineBasisExpander:

    def test_replaces_x1_x2_columns(self):
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)
        enc = SentinelHandler(strategy="median").fit_transform(enc)
        spline = SplineBasisExpander()
        out = spline.fit_transform(enc)
        assert "x1" not in out.columns
        assert "x2" not in out.columns
        # Spline columns should exist
        spline_cols = [c for c in out.columns if c.startswith("x1_sp") or c.startswith("x2_sp")]
        assert len(spline_cols) > 0

    def test_correct_output_dim(self):
        df = _sample_df()
        enc = CityEncoder().fit_transform(df)
        enc = SentinelHandler(strategy="median").fit_transform(enc)
        spline = SplineBasisExpander(n_knots=6, degree=3)
        out = spline.fit_transform(enc)
        # SplineTransformer with n_knots=6, degree=3 produces n_knots + degree - 1 = 8 features per column
        n_spline_per_col = 6 + 3 - 1
        original_cols = len(enc.columns)  # before expansion
        expected = original_cols - 2 + 2 * n_spline_per_col  # remove x1,x2, add spline features
        assert len(out.columns) == expected

    def test_works_on_unseen_data(self):
        df = _sample_df(100)
        enc = CityEncoder().fit_transform(df)
        enc = SentinelHandler(strategy="median").fit_transform(enc)
        spline = SplineBasisExpander()
        spline.fit(enc)

        df2 = _sample_df(30)
        enc2 = CityEncoder().fit_transform(df2)
        enc2 = SentinelHandler(strategy="median").fit_transform(enc2)
        out = spline.transform(enc2)
        assert len(out) == 30
        assert out.notna().all().all()


# ---------------------------------------------------------------------------
# build_preprocessor tests
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:

    def test_tree_flavor_returns_pipeline(self):
        pipe = build_preprocessor("tree")
        assert isinstance(pipe, Pipeline)

    def test_tree_flavor_output(self):
        df = _sample_df()
        pipe = build_preprocessor("tree")
        out = pipe.fit_transform(df)
        assert isinstance(out, pd.DataFrame)
        # Sentinels become NaN
        sentinel_mask = df["x5"] == 999.0
        assert out.loc[sentinel_mask, "x5"].isna().all()
        # No x9_resid column (tree flavor skips residualization)
        assert "x9_resid" not in out.columns

    def test_linear_flavor_output(self):
        df = _sample_df()
        pipe = build_preprocessor("linear")
        out = pipe.fit_transform(df)
        assert isinstance(out, pd.DataFrame)
        # No NaN (median imputed)
        assert out["x5"].notna().all()
        # Has x9_resid
        assert "x9_resid" in out.columns

    def test_drop_noise_false_keeps_x6_x7(self):
        df = _sample_df()
        pipe = build_preprocessor("tree", drop_noise=False)
        out = pipe.fit_transform(df)
        assert "x6" in out.columns
        assert "x7" in out.columns

    def test_no_leakage_across_splits(self):
        """Fit on train subset, transform on holdout — uses train's median."""
        df = _sample_df(100)
        train = df.iloc[:70].copy()
        holdout = df.iloc[70:].copy()

        pipe = build_preprocessor("linear")
        pipe.fit(train)
        out = pipe.transform(holdout)

        # All x5 values should be non-NaN
        assert out["x5"].notna().all()
        # No 999.0 values
        assert (out["x5"] != 999.0).all()
