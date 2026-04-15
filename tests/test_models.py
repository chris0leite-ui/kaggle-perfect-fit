"""Tests for src/models.py — model definitions and wrappers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.models import (
    AveragingEnsemble,
    EBMRegressor,
    GAMRegressor,
    build_all_models,
    build_ebm,
    build_ensemble,
    build_gam,
    build_histgbr,
    build_lgbm,
    build_linear_baseline,
    build_linear_nonlinear,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _regression_df(n=120):
    """Synthetic DataFrame matching the real data schema.

    target = 20*City + 30*x4 + x8 + noise
    """
    rng = np.random.RandomState(42)
    city = rng.choice(["Zaragoza", "Albacete"], n)
    city_num = np.where(city == "Zaragoza", 1.0, 0.0)
    x4 = np.concatenate([rng.uniform(-2, -0.5, n // 2),
                         rng.uniform(0.5, 2, n // 2)])
    x8 = rng.normal(0, 1, n)
    x9 = 0.8 * x4 + rng.normal(0, 0.3, n)
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
        "x9": x9,
        "x10": rng.normal(0, 1, n),
        "x11": rng.normal(0, 1, n),
        "target": target,
    })
    return df


def _split(df, frac=0.7):
    n = int(len(df) * frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


# ---------------------------------------------------------------------------
# Curated model tests
# ---------------------------------------------------------------------------

class TestLinearBaseline:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_linear_baseline()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_linear_baseline(), Pipeline)

    def test_beats_dummy(self):
        df = _regression_df(200)
        train, test = _split(df)
        X_tr, y_tr = train.drop(columns=["target"]), train["target"]
        X_te, y_te = test.drop(columns=["target"]), test["target"]

        pipe = build_linear_baseline()
        pipe.fit(X_tr, y_tr)
        mae_model = np.abs(pipe.predict(X_te) - y_te.values).mean()

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(np.zeros((len(y_tr), 1)), y_tr)
        mae_dummy = np.abs(dummy.predict(np.zeros((len(y_te), 1))) - y_te.values).mean()

        assert mae_model < mae_dummy


class TestLinearNonlinear:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_linear_nonlinear()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_linear_nonlinear(), Pipeline)


class TestGAM:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_gam()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_gam(), Pipeline)


class TestEBM:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_ebm()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_ebm(), Pipeline)


class TestHistGBR:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_histgbr()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_histgbr(), Pipeline)

    def test_beats_dummy(self):
        df = _regression_df(200)
        train, test = _split(df)
        X_tr, y_tr = train.drop(columns=["target"]), train["target"]
        X_te, y_te = test.drop(columns=["target"]), test["target"]

        pipe = build_histgbr()
        pipe.fit(X_tr, y_tr)
        mae_model = np.abs(pipe.predict(X_te) - y_te.values).mean()

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(np.zeros((len(y_tr), 1)), y_tr)
        mae_dummy = np.abs(dummy.predict(np.zeros((len(y_te), 1))) - y_te.values).mean()

        assert mae_model < mae_dummy


class TestLGBM:

    def test_fit_predict_shape(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_lgbm()
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_returns_pipeline(self):
        assert isinstance(build_lgbm(), Pipeline)


# ---------------------------------------------------------------------------
# All-variables variant tests
# ---------------------------------------------------------------------------

class TestAllVarsVariants:

    def test_linear_baseline_all_vars_has_more_features(self):
        df = _regression_df()
        pipe_curated = build_linear_baseline(all_vars=False)
        pipe_all = build_linear_baseline(all_vars=True)

        X = df.drop(columns=["target"])
        y = df["target"]
        pipe_curated.fit(X, y)
        pipe_all.fit(X, y)

        # All-vars model should have more coefficients (x6, x7 included)
        n_curated = len(pipe_curated.named_steps["model"].coef_)
        n_all = len(pipe_all.named_steps["model"].coef_)
        assert n_all > n_curated

    def test_histgbr_all_vars_fit_predict(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_histgbr(all_vars=True)
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)

    def test_lgbm_all_vars_fit_predict(self):
        df = _regression_df()
        train, test = _split(df)
        pipe = build_lgbm(all_vars=True)
        pipe.fit(train.drop(columns=["target"]), train["target"])
        preds = pipe.predict(test.drop(columns=["target"]))
        assert len(preds) == len(test)


# ---------------------------------------------------------------------------
# Wrapper tests
# ---------------------------------------------------------------------------

class TestGAMRegressor:

    def test_sklearn_compatible(self):
        """GAMRegressor supports fit/predict/get_params/set_params."""
        gam = GAMRegressor()
        params = gam.get_params()
        assert "n_splines" in params
        gam.set_params(n_splines=10)
        assert gam.n_splines == 10


class TestEBMRegressor:

    def test_sklearn_compatible(self):
        ebm = EBMRegressor()
        params = ebm.get_params()
        assert "interactions" in params


class TestAveragingEnsemble:

    def test_averages_predictions(self):
        """Two constant-prediction models should average to their mean."""
        from sklearn.dummy import DummyRegressor
        from sklearn.preprocessing import FunctionTransformer

        pipe_10 = Pipeline([
            ("select", FunctionTransformer(lambda X: X[["x4"]].values if hasattr(X, "columns") else X, validate=False)),
            ("model", DummyRegressor(strategy="constant", constant=10.0)),
        ])
        pipe_20 = Pipeline([
            ("select", FunctionTransformer(lambda X: X[["x4"]].values if hasattr(X, "columns") else X, validate=False)),
            ("model", DummyRegressor(strategy="constant", constant=20.0)),
        ])

        ens = AveragingEnsemble(models=[("a", pipe_10), ("b", pipe_20)])
        df = _regression_df(20)
        X = df.drop(columns=["target"])
        y = df["target"]
        ens.fit(X, y)
        preds = ens.predict(X)
        np.testing.assert_allclose(preds, 15.0)


# ---------------------------------------------------------------------------
# build_all_models test
# ---------------------------------------------------------------------------

class TestBuildAllModels:

    def test_returns_all_entries(self):
        models = build_all_models()
        assert isinstance(models, dict)
        assert len(models) >= 12
        # Check some expected keys
        assert "linear_baseline" in models
        assert "lgbm" in models
        assert "linear_baseline_all" in models
        assert "lgbm_all" in models
        assert "ensemble" in models
