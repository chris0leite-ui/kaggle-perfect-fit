"""Tests for Round 2 model variants and features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.features import InteractionAdder, build_preprocessor
from src.models import (
    build_ebm_tuned,
    build_gam_tuned,
    build_huber_nonlinear,
    build_quantile_nonlinear,
    build_linear_nonlinear_interact,
    build_gam_interact,
    build_lgbm_tuned,
    build_histgbr_tuned,
    build_rf,
)


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
        "x10": rng.uniform(0, 6, n),
        "x11": rng.uniform(0, 6, n),
        "target": target,
    })
    return df


def _split(df, frac=0.7):
    n = int(len(df) * frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


# ---------------------------------------------------------------------------
# InteractionAdder tests
# ---------------------------------------------------------------------------

class TestInteractionAdder:

    def test_adds_interaction_column(self):
        df = _regression_df()
        enc = build_preprocessor("linear").fit_transform(df)
        adder = InteractionAdder(pairs=[("x10", "x11")])
        out = adder.fit_transform(enc)
        assert "x10_x_x11" in out.columns

    def test_interaction_values_correct(self):
        df = _regression_df()
        enc = build_preprocessor("linear").fit_transform(df)
        adder = InteractionAdder(pairs=[("x10", "x11")])
        out = adder.fit_transform(enc)
        expected = enc["x10"] * enc["x11"]
        np.testing.assert_allclose(out["x10_x_x11"].values, expected.values)

    def test_multiple_pairs(self):
        df = _regression_df()
        enc = build_preprocessor("linear").fit_transform(df)
        adder = InteractionAdder(pairs=[("x10", "x11"), ("x4", "x8")])
        out = adder.fit_transform(enc)
        assert "x10_x_x11" in out.columns
        assert "x4_x_x8" in out.columns

    def test_preserves_other_columns(self):
        df = _regression_df()
        enc = build_preprocessor("linear").fit_transform(df)
        original_cols = set(enc.columns)
        adder = InteractionAdder(pairs=[("x10", "x11")])
        out = adder.fit_transform(enc)
        assert original_cols.issubset(set(out.columns))


# ---------------------------------------------------------------------------
# Model builder tests — fit/predict shape + returns Pipeline
# ---------------------------------------------------------------------------

class TestEBMTuned:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_ebm_tuned()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_ebm_tuned(), Pipeline)

    def test_custom_params(self):
        pipe = build_ebm_tuned(min_samples_leaf=20, max_bins=128, max_rounds=2000)
        ebm = pipe.named_steps["model"]
        assert ebm.min_samples_leaf == 20
        assert ebm.max_bins == 128
        assert ebm.max_rounds == 2000


class TestGAMTuned:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_gam_tuned()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_gam_tuned(), Pipeline)

    def test_custom_params(self):
        pipe = build_gam_tuned(lam=5.0, n_splines=15)
        gam = pipe.named_steps["model"]
        assert gam.lam == 5.0
        assert gam.n_splines == 15


class TestHuberNonlinear:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_huber_nonlinear()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_huber_nonlinear(), Pipeline)


class TestQuantileNonlinear:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_quantile_nonlinear()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_quantile_nonlinear(), Pipeline)


class TestLinearNonlinearInteract:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_linear_nonlinear_interact()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_linear_nonlinear_interact(), Pipeline)


class TestGAMInteract:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_gam_interact()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_gam_interact(), Pipeline)


class TestLGBMTuned:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_lgbm_tuned()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_lgbm_tuned(), Pipeline)

    def test_custom_params(self):
        pipe = build_lgbm_tuned(n_estimators=200, learning_rate=0.1,
                                max_depth=4, min_child_samples=20)
        model = pipe.named_steps["model"]
        assert model.n_estimators == 200


class TestHistGBRTuned:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_histgbr_tuned()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_histgbr_tuned(), Pipeline)

    def test_custom_params(self):
        pipe = build_histgbr_tuned(max_iter=200, learning_rate=0.1,
                                   max_depth=4, min_samples_leaf=20)
        model = pipe.named_steps["model"]
        assert model.max_iter == 200


class TestRandomForest:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_rf()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_rf(), Pipeline)

    def test_custom_params(self):
        pipe = build_rf(n_estimators=100, max_depth=6)
        model = pipe.named_steps["model"]
        assert model.n_estimators == 100
        assert model.max_depth == 6
