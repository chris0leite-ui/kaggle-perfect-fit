"""Cluster analysis: City × x4-group interaction."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Testable functions
# ---------------------------------------------------------------------------

def find_x4_gap(x4_values: pd.Series, n_bins: int = 300):
    """Find the largest gap in a bimodal distribution.

    Returns (gap_start, gap_end, midpoint).
    """
    vals = np.sort(x4_values.dropna().values)
    diffs = np.diff(vals)
    idx = int(np.argmax(diffs))
    gap_start = float(vals[idx])
    gap_end = float(vals[idx + 1])
    midpoint = (gap_start + gap_end) / 2
    return gap_start, gap_end, midpoint


def assign_clusters(df: pd.DataFrame, x4_threshold: float | None = None):
    """Assign cluster labels based on City × x4 split.

    Returns a Series with labels like 'Albacete_high', 'Zaragoza_low'.
    """
    if x4_threshold is None:
        _, _, x4_threshold = find_x4_gap(df["x4"])
    x4_group = np.where(df["x4"] >= x4_threshold, "high", "low")
    labels = df["City"].astype(str) + "_" + x4_group
    return pd.Series(labels, index=df.index)


def replace_sentinels(series: pd.Series, sentinel: float = 999.0):
    """Replace sentinel values with the median of non-sentinel values."""
    mask = series == sentinel
    if not mask.any():
        return series.copy()
    median = series[~mask].median()
    result = series.copy()
    result[mask] = median
    return result


def cluster_stats(df: pd.DataFrame, cluster_col: str, variables: list,
                  sentinel_cols: dict | None = None):
    """Per-cluster descriptive statistics for specified variables.

    Returns a DataFrame with MultiIndex columns: (variable, statistic).
    """
    work = df.copy()
    if sentinel_cols:
        for col, sentinel in sentinel_cols.items():
            if col in work.columns:
                work[col] = replace_sentinels(work[col], sentinel)

    agg_funcs = {
        "count": "count",
        "mean": "mean",
        "median": "median",
        "std": "std",
        "min": "min",
        "max": "max",
    }

    frames = {}
    for var in variables:
        grouped = work.groupby(cluster_col)[var].agg(list(agg_funcs.values()))
        grouped.columns = pd.MultiIndex.from_tuples(
            [(var, stat) for stat in agg_funcs.keys()]
        )
        frames[var] = grouped

    return pd.concat(frames.values(), axis=1)


# ---------------------------------------------------------------------------
# Plotting functions (visual output, not unit-tested)
# ---------------------------------------------------------------------------

CLUSTER_COLORS = {
    "Albacete_high": "#e41a1c",
    "Albacete_low": "#377eb8",
    "Zaragoza_high": "#ff7f00",
    "Zaragoza_low": "#4daf4a",
}

CLUSTER_ORDER = ["Albacete_high", "Albacete_low", "Zaragoza_high", "Zaragoza_low"]


def _clean_for_plot(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Return a copy with x5 sentinels replaced."""
    work = df.copy()
    if "x5" in work.columns and "x5" in variables:
        work["x5"] = replace_sentinels(work["x5"])
    return work


def plot_boxplots(df: pd.DataFrame, cluster_col: str, variables: list,
                  out_dir: Path) -> None:
    """One box plot per variable, grouped by the 4 clusters."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = _clean_for_plot(df, variables)
    for var in variables:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [work.loc[work[cluster_col] == c, var].dropna() for c in CLUSTER_ORDER]
        bp = ax.boxplot(data, labels=CLUSTER_ORDER, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLUSTER_ORDER):
            patch.set_facecolor(CLUSTER_COLORS[c])
            patch.set_alpha(0.6)
        ax.set_ylabel(var)
        ax.set_title(f"{var} by cluster")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / f"{var}_boxplot.png", dpi=100)
        plt.close(fig)


def plot_scatter_x4_target(df: pd.DataFrame, cluster_col: str,
                           out_dir: Path) -> None:
    """Scatter of x4 vs target, colored by cluster, with gap region shaded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gap_start, gap_end, _ = find_x4_gap(df["x4"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(gap_start, gap_end, alpha=0.15, color="gray", label="gap")
    for c in CLUSTER_ORDER:
        mask = df[cluster_col] == c
        ax.scatter(df.loc[mask, "x4"], df.loc[mask, "target"],
                   alpha=0.4, s=12, color=CLUSTER_COLORS[c], label=c)
    ax.set_xlabel("x4")
    ax.set_ylabel("target")
    ax.set_title("x4 vs target by cluster")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "x4_target_scatter.png", dpi=100)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, cluster_col: str, variables: list,
                       out_dir: Path) -> None:
    """Overlaid density histograms per cluster for each variable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = _clean_for_plot(df, variables)
    for var in variables:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in CLUSTER_ORDER:
            vals = work.loc[work[cluster_col] == c, var].dropna()
            ax.hist(vals, bins=25, alpha=0.35, color=CLUSTER_COLORS[c],
                    label=c, density=True, edgecolor="white")
        ax.set_xlabel(var)
        ax.set_ylabel("density")
        ax.set_title(f"{var} distribution by cluster")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{var}_density.png", dpi=100)
        plt.close(fig)
