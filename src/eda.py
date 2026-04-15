from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
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
# Plotting utilities (visual output, not unit-tested)
# ---------------------------------------------------------------------------

def scatter_plots(df: pd.DataFrame, features: list, target: str, out_dir: Path) -> None:
    """Save one scatter plot per feature to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in features:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df[feat], df[target], alpha=0.4, s=10)
        ax.set_xlabel(feat)
        ax.set_ylabel(target)
        ax.set_title(f"{feat} vs {target}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{feat}_scatter.png", dpi=100)
        plt.close(fig)


def partial_residual_plots(df: pd.DataFrame, features: list, target: str, out_dir: Path) -> None:
    """CCPR (partial residual) plots for each feature. Requires all features to be numeric."""
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
