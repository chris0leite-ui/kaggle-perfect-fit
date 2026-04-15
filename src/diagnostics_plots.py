"""Plotting functions for diagnostic analyses."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.clusters import CLUSTER_COLORS, CLUSTER_ORDER


# ---------------------------------------------------------------------------
# TODO 1: Sentinel distribution
# ---------------------------------------------------------------------------

def plot_sentinel_distribution(crosstab: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: sentinel vs non-sentinel count per cluster."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clusters = [c for c in CLUSTER_ORDER if c in crosstab.index]
    x = np.arange(len(clusters))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    sentinel_vals = [crosstab.loc[c, "sentinel"] for c in clusters]
    non_sentinel_vals = [crosstab.loc[c, "non_sentinel"] for c in clusters]

    bars1 = ax.bar(x - width / 2, non_sentinel_vals, width, label="Non-sentinel",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, sentinel_vals, width, label="Sentinel (999)",
                   color="crimson", alpha=0.8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, str(int(h)),
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=15, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("x5 Sentinel Distribution by Cluster")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TODO 2: Residual analysis
# ---------------------------------------------------------------------------

def plot_residual_analysis(ols_result, cluster_labels: pd.Series,
                           out_path: Path) -> None:
    """3-panel figure: residuals vs fitted, QQ plot, residuals by cluster."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resid = ols_result.resid
    fitted = ols_result.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Residuals vs Fitted
    ax = axes[0]
    ax.scatter(fitted, resid, alpha=0.3, s=10, color="steelblue")
    ax.axhline(0, color="crimson", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # Panel 2: QQ plot
    ax = axes[1]
    sp_stats.probplot(resid, plot=ax)
    ax.set_title("Normal Q-Q Plot")

    # Panel 3: Residuals by cluster
    ax = axes[2]
    clusters_present = [c for c in CLUSTER_ORDER if c in cluster_labels.values]
    data = [resid[cluster_labels == c] for c in clusters_present]
    bp = ax.boxplot(data, labels=clusters_present, patch_artist=True)
    for patch, c in zip(bp["boxes"], clusters_present):
        patch.set_facecolor(CLUSTER_COLORS[c])
        patch.set_alpha(0.6)
    ax.axhline(0, color="crimson", linewidth=1, linestyle="--")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals by Cluster")
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle("Residual Analysis: target ~ City + x4", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# TODO 3: Per-cluster GAM curves
# ---------------------------------------------------------------------------

def plot_gam_per_cluster(cluster_gams: dict, feature: str,
                         out_path: Path) -> None:
    """Overlay per-cluster GAM curves for a feature."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for label in CLUSTER_ORDER:
        if label not in cluster_gams:
            continue
        info = cluster_gams[label]
        ax.plot(info["x_grid"], info["y_pred"],
                color=CLUSTER_COLORS[label], linewidth=2, alpha=0.8,
                label=f"{label} (R\u00b2={info['r_squared']:.3f}, n={info['n']})")

    ax.set_xlabel(feature)
    ax.set_ylabel("GAM prediction (target)")
    ax.set_title(f"Per-Cluster GAM: {feature} \u2192 target")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TODO 4: R-squared breakdown
# ---------------------------------------------------------------------------

def plot_r2_breakdown(breakdown_df: pd.DataFrame, out_path: Path) -> None:
    """Waterfall-style bar chart showing cumulative and marginal R-squared."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = breakdown_df["group"].tolist()
    cumulative = breakdown_df["cumulative_r2"].tolist()
    marginal = breakdown_df["marginal_r2"].tolist()

    x = np.arange(len(groups))
    bottoms = [c - m for c, m in zip(cumulative, marginal)]

    colors = ["#377eb8", "#4daf4a", "#ff7f00", "#e41a1c"]
    bars = ax.bar(x, marginal, bottom=bottoms, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5)

    for i, (bar, cum, marg) in enumerate(zip(bars, cumulative, marginal)):
        ax.text(bar.get_x() + bar.get_width() / 2, cum + 0.01,
                f"R\u00b2={cum:.3f}\n(\u0394={marg:.3f})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("R\u00b2")
    ax.set_title("Feature Group R\u00b2 Breakdown (Incremental)")
    ax.set_ylim(0, max(cumulative) * 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_ceiling_residuals(residuals: np.ndarray, out_path: Path) -> None:
    """Histogram of residuals from the ceiling model."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40, color="steelblue", edgecolor="white",
            alpha=0.8, density=True)

    # Overlay normal curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), color="crimson",
            linewidth=2, label=f"N({mu:.1f}, {sigma:.1f}\u00b2)")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Ceiling Model Residual Distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TODO 5: x4 bimodality
# ---------------------------------------------------------------------------

def plot_x4_bimodality(df: pd.DataFrame, gap_info: dict,
                       out_path: Path) -> None:
    """x4 histogram with gap annotated + per-city overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Overall histogram with gap
    ax = axes[0]
    ax.hist(df["x4"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvspan(gap_info["gap_start"], gap_info["gap_end"],
               alpha=0.3, color="crimson", label="Gap region")
    ax.set_xlabel("x4")
    ax.set_ylabel("Count")
    ax.set_title("x4 Distribution (Overall)")
    ax.annotate(f"n_below={gap_info['n_below']}",
                xy=(gap_info["gap_start"], 0), fontsize=8,
                xytext=(-0.1, ax.get_ylim()[1] * 0.8),
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate(f"n_above={gap_info['n_above']}",
                xy=(gap_info["gap_end"], 0), fontsize=8,
                xytext=(0.1, ax.get_ylim()[1] * 0.8),
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.legend(fontsize=9)

    # Panel 2: Per-city overlay
    ax = axes[1]
    for city, color in [("Albacete", "#377eb8"), ("Zaragoza", "#e41a1c")]:
        vals = df.loc[df["City"] == city, "x4"]
        ax.hist(vals, bins=40, alpha=0.5, color=color, edgecolor="white",
                label=city, density=True)
    ax.axvspan(gap_info["gap_start"], gap_info["gap_end"],
               alpha=0.2, color="gray")
    ax.set_xlabel("x4")
    ax.set_ylabel("Density")
    ax.set_title("x4 Distribution by City")
    ax.legend(fontsize=9)

    fig.suptitle("x4 Bimodality Analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
