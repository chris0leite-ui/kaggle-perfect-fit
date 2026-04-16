"""Model definitions — each builder returns a full sklearn Pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features import build_preprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_array(X):
    """Convert DataFrame to numpy array, passthrough if already array."""
    if hasattr(X, "values"):
        return X.values
    return X


# ---------------------------------------------------------------------------
# Wrapper classes
# ---------------------------------------------------------------------------

class GAMRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around pygam LinearGAM."""

    def __init__(self, n_splines=20, lam=0.6, spline_columns=None):
        self.n_splines = n_splines
        self.lam = lam
        self.spline_columns = spline_columns  # columns to get spline terms (by name)

    def fit(self, X, y):
        from pygam import LinearGAM, l, s

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
            X_arr = X

        # Determine which columns get spline terms
        spline_cols = self.spline_columns or ["x1", "x2"]
        spline_idx = set()
        for name in spline_cols:
            if name in self.feature_names_:
                spline_idx.add(self.feature_names_.index(name))

        # Build term formula
        terms = None
        for i in range(X_arr.shape[1]):
            if i in spline_idx:
                term = s(i, n_splines=self.n_splines)
            else:
                term = l(i)
            terms = term if terms is None else terms + term

        self.gam_ = LinearGAM(terms, lam=self.lam)
        self.gam_.fit(X_arr, np.asarray(y))
        return self

    def predict(self, X):
        X_arr = _df_to_array(X)
        return self.gam_.predict(X_arr)


class EBMRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around ExplainableBoostingRegressor."""

    def __init__(self, interactions=10, max_rounds=5000, min_samples_leaf=5,
                 max_bins=256, smoothing_rounds=200):
        self.interactions = interactions
        self.max_rounds = max_rounds
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.smoothing_rounds = smoothing_rounds

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingRegressor

        # Preserve feature names for interpretability
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            feature_names = self.feature_names_
        else:
            feature_names = None

        self.ebm_ = ExplainableBoostingRegressor(
            interactions=self.interactions,
            max_rounds=self.max_rounds,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            smoothing_rounds=self.smoothing_rounds,
            feature_names=feature_names,
        )
        X_arr = _df_to_array(X)
        self.ebm_.fit(X_arr, np.asarray(y))
        return self

    def predict(self, X):
        X_arr = _df_to_array(X)
        return self.ebm_.predict(X_arr)


class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """Average predictions from multiple pipelines."""

    def __init__(self, models=None):
        self.models = models or []

    def fit(self, X, y):
        self.fitted_models_ = []
        for name, pipe in self.models:
            from sklearn.base import clone
            cloned = clone(pipe)
            cloned.fit(X, y)
            self.fitted_models_.append((name, cloned))
        return self

    def predict(self, X):
        preds = [pipe.predict(X) for _, pipe in self.fitted_models_]
        return np.mean(preds, axis=0)


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Weighted average predictions from multiple pipelines.

    Weights can be provided manually or optimized via CV.
    """

    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights

    def fit(self, X, y):
        from sklearn.base import clone
        self.fitted_models_ = []
        for name, pipe in self.models:
            cloned = clone(pipe)
            cloned.fit(X, y)
            self.fitted_models_.append((name, cloned))

        if self.weights is None:
            self.weights_ = np.ones(len(self.models)) / len(self.models)
        else:
            w = np.array(self.weights, dtype=float)
            self.weights_ = w / w.sum()
        return self

    def predict(self, X):
        preds = np.array([pipe.predict(X) for _, pipe in self.fitted_models_])
        return np.average(preds, axis=0, weights=self.weights_)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_linear_baseline(all_vars=False):
    drop_noise = not all_vars
    prep = build_preprocessor("linear", drop_noise=drop_noise,
                              residualize_x9=not all_vars)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", LinearRegression()),
    ])


def build_linear_nonlinear(all_vars=False):
    drop_noise = not all_vars
    prep = build_preprocessor("linear_spline", drop_noise=drop_noise,
                              residualize_x9=not all_vars)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", LinearRegression()),
    ])


def build_gam(all_vars=False):
    drop_noise = not all_vars
    prep = build_preprocessor("linear", drop_noise=drop_noise,
                              residualize_x9=not all_vars)
    return Pipeline([
        ("preprocessor", prep),
        ("model", GAMRegressor(n_splines=20, lam=0.6)),
    ])


def build_ebm(all_vars=False):
    drop_noise = not all_vars
    prep = build_preprocessor("tree", drop_noise=drop_noise)
    return Pipeline([
        ("preprocessor", prep),
        ("model", EBMRegressor(interactions=10, max_rounds=5000, min_samples_leaf=5)),
    ])


def build_histgbr(all_vars=False):
    drop_noise = not all_vars
    prep = build_preprocessor("tree", drop_noise=drop_noise)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])


def build_lgbm(all_vars=False):
    from lightgbm import LGBMRegressor

    drop_noise = not all_vars
    prep = build_preprocessor("tree", drop_noise=drop_noise)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", LGBMRegressor(
            objective="mae",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )),
    ])


def build_ensemble(models=None):
    if models is None:
        models = [
            ("histgbr", build_histgbr()),
            ("lgbm", build_lgbm()),
            ("ebm", build_ebm()),
        ]
    return AveragingEnsemble(models=models)


# ---------------------------------------------------------------------------
# Round 2 model builders
# ---------------------------------------------------------------------------

def build_ebm_tuned(min_samples_leaf=15, max_bins=256, max_rounds=3000,
                    smoothing_rounds=200, interactions=10):
    prep = build_preprocessor("tree", drop_noise=True)
    return Pipeline([
        ("preprocessor", prep),
        ("model", EBMRegressor(
            interactions=interactions,
            max_rounds=max_rounds,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            smoothing_rounds=smoothing_rounds,
        )),
    ])


def build_gam_tuned(lam=2.0, n_splines=15):
    prep = build_preprocessor("linear", drop_noise=True, residualize_x9=True)
    return Pipeline([
        ("preprocessor", prep),
        ("model", GAMRegressor(n_splines=n_splines, lam=lam)),
    ])


def build_huber_nonlinear():
    from sklearn.linear_model import HuberRegressor
    prep = build_preprocessor("linear_spline", drop_noise=True,
                              residualize_x9=True)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", HuberRegressor(epsilon=1.35, max_iter=200)),
    ])


def build_quantile_nonlinear():
    from sklearn.linear_model import QuantileRegressor
    prep = build_preprocessor("linear_spline", drop_noise=True,
                              residualize_x9=True)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", QuantileRegressor(quantile=0.5, alpha=0.01,
                                    solver="highs")),
    ])


def build_linear_nonlinear_interact():
    prep = build_preprocessor("linear_spline_interact", drop_noise=True,
                              residualize_x9=True)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", LinearRegression()),
    ])


def build_gam_interact():
    prep = build_preprocessor("linear_interact", drop_noise=True,
                              residualize_x9=True)
    return Pipeline([
        ("preprocessor", prep),
        ("model", GAMRegressor(n_splines=20, lam=0.6)),
    ])


def build_lgbm_tuned(n_estimators=1000, learning_rate=0.03, max_depth=6,
                     min_child_samples=20, subsample=0.8, colsample_bytree=0.8):
    from lightgbm import LGBMRegressor
    prep = build_preprocessor("tree", drop_noise=True)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", LGBMRegressor(
            objective="mae",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            verbose=-1,
        )),
    ])


def build_histgbr_tuned(max_iter=1000, learning_rate=0.03, max_depth=6,
                        min_samples_leaf=20):
    prep = build_preprocessor("tree", drop_noise=True)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])


def build_rf(n_estimators=500, max_depth=10, min_samples_leaf=5):
    from sklearn.ensemble import RandomForestRegressor
    # RF can't handle NaN — use median imputation (linear flavor drops x9_resid)
    prep = build_preprocessor("linear", drop_noise=True, residualize_x9=False)
    return Pipeline([
        ("preprocessor", prep),
        ("to_array", FunctionTransformer(_df_to_array, validate=False)),
        ("model", RandomForestRegressor(
            criterion="absolute_error",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_ensemble_ebm_gam():
    """EBM (tight core) + GAM+interact (lighter tails) — complementary profiles."""
    return AveragingEnsemble(models=[
        ("ebm_tuned", build_ebm_tuned(min_samples_leaf=10, max_bins=128,
                                       max_rounds=2000)),
        ("gam_interact", build_gam_interact()),
    ])


def build_ensemble_ebm_gam_weighted(ebm_weight=0.7, gam_weight=0.3):
    """Weighted EBM + GAM+interact ensemble."""
    return WeightedEnsemble(
        models=[
            ("ebm_tuned", build_ebm_tuned(min_samples_leaf=10, max_bins=128,
                                           max_rounds=2000)),
            ("gam_interact", build_gam_interact()),
        ],
        weights=[ebm_weight, gam_weight],
    )


def build_all_models():
    """Return dict of all 12 models (7 curated + 5 all-vars)."""
    return {
        # Curated (hypothesis-driven feature selection)
        "linear_baseline": build_linear_baseline(),
        "linear_nonlinear": build_linear_nonlinear(),
        "gam": build_gam(),
        "ebm": build_ebm(),
        "histgbr": build_histgbr(),
        "lgbm": build_lgbm(),
        "ensemble": build_ensemble(),
        # All-variables variants (hypothesis test)
        "linear_baseline_all": build_linear_baseline(all_vars=True),
        "linear_nonlinear_all": build_linear_nonlinear(all_vars=True),
        "gam_all": build_gam(all_vars=True),
        "histgbr_all": build_histgbr(all_vars=True),
        "lgbm_all": build_lgbm(all_vars=True),
    }
