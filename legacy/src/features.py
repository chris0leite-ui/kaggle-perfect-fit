"""Feature engineering transformers and pipeline builders."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "target"
DROP_ALWAYS = ["id", "Country"]
DROP_NOISE = ["x6", "x7"]


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

class CityEncoder(BaseEstimator, TransformerMixin):
    """Encode City as binary, drop id/Country, optionally drop noise features."""

    def __init__(self, drop_noise=True):
        self.drop_noise = drop_noise

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = X.copy()
        out["City"] = out["City"].map({"Zaragoza": 1.0, "Albacete": 0.0})
        drop = list(DROP_ALWAYS)
        if self.drop_noise:
            drop.extend(DROP_NOISE)
        out = out.drop(columns=[c for c in drop if c in out.columns])
        return out


class SentinelHandler(BaseEstimator, TransformerMixin):
    """Handle x5 sentinel values (999.0) — replace with NaN or median."""

    def __init__(self, strategy="nan", sentinel=999.0, column="x5"):
        self.strategy = strategy
        self.sentinel = sentinel
        self.column = column

    def fit(self, X, y=None):
        if self.strategy == "median":
            col = X[self.column]
            mask = col == self.sentinel
            self.median_ = float(col[~mask].median())
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = X.copy()
        mask = out[self.column] == self.sentinel
        out["x5_is_sentinel"] = mask.astype(float)

        if self.strategy == "nan":
            out.loc[mask, self.column] = np.nan
        elif self.strategy == "median":
            out.loc[mask, self.column] = self.median_
        return out


class X9Residualizer(BaseEstimator, TransformerMixin):
    """Add x9_resid = x9 - (coef*x4 + intercept), learned via OLS."""

    def fit(self, X, y=None):
        x4 = X["x4"].values
        x9 = X["x9"].values
        valid = np.isfinite(x4) & np.isfinite(x9)
        x4v, x9v = x4[valid], x9[valid]
        # Simple OLS: x9 = coef * x4 + intercept
        A = np.column_stack([x4v, np.ones(len(x4v))])
        result = np.linalg.lstsq(A, x9v, rcond=None)
        self.coef_ = float(result[0][0])
        self.intercept_ = float(result[0][1])
        return self

    def transform(self, X):
        out = X.copy()
        out["x9_resid"] = out["x9"] - (self.coef_ * out["x4"] + self.intercept_)
        out = out.drop(columns=["x9"])
        return out


class InteractionAdder(BaseEstimator, TransformerMixin):
    """Add pairwise product interaction columns."""

    def __init__(self, pairs=None):
        self.pairs = pairs or [("x10", "x11")]

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        out = X.copy()
        for col_a, col_b in self.pairs:
            name = f"{col_a}_x_{col_b}"
            out[name] = out[col_a] * out[col_b]
        return out


class SplineBasisExpander(BaseEstimator, TransformerMixin):
    """Replace x1, x2 with spline basis columns."""

    def __init__(self, columns=None, n_knots=6, degree=3):
        self.columns = columns or ["x1", "x2"]
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y=None):
        self.transformers_ = {}
        for col in self.columns:
            st = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                extrapolation="linear",
            )
            st.fit(X[[col]].values)
            self.transformers_[col] = st
        return self

    def transform(self, X):
        out = X.copy()
        for col, st in self.transformers_.items():
            basis = st.transform(out[[col]].values)
            n_features = basis.shape[1]
            names = [f"{col}_sp{i}" for i in range(n_features)]
            basis_df = pd.DataFrame(basis, columns=names, index=out.index)
            out = out.drop(columns=[col])
            out = pd.concat([out, basis_df], axis=1)
        return out


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_preprocessor(flavor: str, drop_noise: bool = True,
                       residualize_x9: bool = True) -> Pipeline:
    """Build a preprocessing pipeline.

    Flavors:
        "tree":                 CityEncoder -> SentinelHandler(nan)
        "linear":               CityEncoder -> SentinelHandler(median) [-> X9Residualizer]
        "linear_spline":        CityEncoder -> SentinelHandler(median) [-> X9Residualizer] -> SplineBasisExpander
        "linear_interact":      CityEncoder -> SentinelHandler(median) [-> X9Residualizer] -> InteractionAdder
        "linear_spline_interact": CityEncoder -> SentinelHandler(median) [-> X9Residualizer] -> SplineBasisExpander -> InteractionAdder

    residualize_x9: if True (default for linear flavors), replace x9 with
        x9_resid. Set False for all-vars comparisons that want raw x9.
    """
    steps = [("encode", CityEncoder(drop_noise=drop_noise))]

    if flavor == "tree":
        steps.append(("sentinel", SentinelHandler(strategy="nan")))
    else:
        steps.append(("sentinel", SentinelHandler(strategy="median")))
        if residualize_x9:
            steps.append(("x9_resid", X9Residualizer()))
        if flavor in ("linear_spline", "linear_spline_interact"):
            steps.append(("spline", SplineBasisExpander()))
        if flavor in ("linear_interact", "linear_spline_interact"):
            steps.append(("interact", InteractionAdder()))

    return Pipeline(steps)
