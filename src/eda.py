from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pygam import LinearGAM, s
from statsmodels.graphics.regressionplots import plot_ccpr, plot_partregress


# ---------------------------------------------------------------------------
# Data-check utilities (testable)
# ---------------------------------------------------------------------------

def detect_sentinels(df: pd.DataFrame, sentinel: float = 999.0) -> dict:
    """Return {col: count} for columns containing the sentinel value."""
    return {
        col: int((df[col] == sentinel).sum())
        for col in df.select_dtypes("number").columns
        if (df[col] == sentinel).any()
    }


def zero_variance_cols(df: pd.DataFrame) -> list:
    """Return column names that have only one unique value."""
    return [col for col in df.columns if df[col].nunique() == 1]


def correlations(df: pd.DataFrame, target: str) -> pd.Series:
    """Pearson correlations of numeric columns with target, sorted by |r| descending."""
    numeric = df.select_dtypes("number").drop(columns=[target], errors="ignore")
    corr = numeric.corrwith(df[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Internal: GAM smooth with natural-spline constraints
# ---------------------------------------------------------------------------

def _gam_smooth(x: np.ndarray, y: np.ndarray, n: int = 200):
    """Fit a GAM spline and return (x_grid, y_pred). Drops non-finite rows."""
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    gam = LinearGAM(s(0, spline_order=3)).fit(xc.reshape(-1, 1), yc)
    x_grid = np.linspace(xc.min(), xc.max(), n)
    return x_grid, gam.predict(x_grid.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Plotting utilities (visual output, not unit-tested)
# ---------------------------------------------------------------------------

def scatter_plots(df: pd.DataFrame, features: list, target: str, out_dir: Path) -> None:
    """Scatter plot per feature with GAM smooth fit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in features:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df[feat], df[target], alpha=0.4, s=10, color="steelblue")
        try:
            xg, yg = _gam_smooth(df[feat].values, df[target].values)
            ax.plot(xg, yg, color="crimson", linewidth=2, label="GAM smooth")
            ax.legend(fontsize=8)
        except Exception:
            pass
        ax.set_xlabel(feat)
        ax.set_ylabel(target)
        ax.set_title(f"{feat} vs {target}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{feat}_scatter.png", dpi=100)
        plt.close(fig)


def pairwise_scatter_plot(df: pd.DataFrame, features: list, out_dir: Path) -> None:
    """Scatter matrix of all feature pairs with GAM smooth fits on off-diagonal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n))
    for i, fy in enumerate(features):
        for j, fx in enumerate(features):
            ax = axes[i, j]
            ax.tick_params(labelsize=6)
            if i == j:
                ax.hist(df[fx].dropna(), bins=20, color="steelblue", edgecolor="white")
            else:
                ax.scatter(df[fx], df[fy], alpha=0.25, s=4, color="steelblue")
                try:
                    xg, yg = _gam_smooth(df[fx].values, df[fy].values)
                    ax.plot(xg, yg, color="crimson", linewidth=1.2)
                except Exception:
                    pass
            if j == 0:
                ax.set_ylabel(fy, fontsize=8)
            if i == n - 1:
                ax.set_xlabel(fx, fontsize=8)
    fig.suptitle("Pairwise Scatter Matrix (GAM smooth)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def partial_residual_plots(df: pd.DataFrame, features: list, target: str, out_dir: Path) -> None:
    """CCPR (partial residual) plots for each feature."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X = sm.add_constant(df[features])
    model = sm.OLS(df[target], X).fit()
    for feat in features:
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_ccpr(model, feat, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{feat}_ccpr.png", dpi=100)
        plt.close(fig)


def added_variable_plots(df: pd.DataFrame, features: list, target: str, out_dir: Path) -> None:
    """Added variable (partial regression) plots for each feature."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X = sm.add_constant(df[features])
    model = sm.OLS(df[target], X).fit()
    for feat in features:
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_partregress(target, feat, [f for f in features if f != feat],
                         data=df, ax=ax, obs_labels=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{feat}_avp.png", dpi=100)
        plt.close(fig)
