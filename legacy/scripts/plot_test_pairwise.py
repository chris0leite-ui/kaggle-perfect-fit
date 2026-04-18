"""Pairwise scatter matrix of all numeric features in the competition test set.

Uses data/test.csv (1500 rows, the submission-data features). Plots a
10x10 grid: histograms on the diagonal, scatter on the off-diagonal,
coloured by City. Rows with x5=999 are highlighted.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "formulas"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


def main() -> None:
    df = pd.read_csv(DATA / "test.csv")
    features = ["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]
    n = len(features)
    print(f"test.csv: {df.shape}")
    print(f"  plotting {n}x{n} pairwise matrix (sentinels visible)")

    sent = (df["x5"] == SENTINEL).values
    city = df["City"].values

    colors = {"Albacete": "#1f77b4", "Zaragoza": "#d62728"}

    # Axis ranges: keep full data range for x5 so the 999 sentinels are visible;
    # other features use the tight percentile range.
    ranges = {}
    for f in features:
        vals = df[f].values
        if f == "x5":
            ranges[f] = (vals.min() - 10, vals.max() + 10)
        else:
            ranges[f] = (np.percentile(vals, 0.5), np.percentile(vals, 99.5))

    fig, axes = plt.subplots(n, n, figsize=(24, 24))

    for i, yf in enumerate(features):
        for j, xf in enumerate(features):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram including sentinels
                vals = df[xf].values
                ax.hist(vals, bins=60, color="gray", alpha=0.7)
                ax.set_xlim(ranges[xf])
                ax.set_yticks([])
                if xf == "x5":
                    ax.set_yscale("log")
            else:
                # Off-diagonal: scatter, sentinel rows in red
                x = df[xf].values
                y = df[yf].values
                # Non-sentinel rows by city
                for c, col in colors.items():
                    m = (~sent) & (city == c)
                    ax.scatter(x[m], y[m], s=3, alpha=0.4, color=col, linewidths=0)
                # Sentinel rows over the top
                ax.scatter(x[sent], y[sent], s=6, alpha=0.7, color="black",
                           linewidths=0, label=None)
                ax.set_xlim(ranges[xf])
                ax.set_ylim(ranges[yf])

            if i == n - 1:
                ax.set_xlabel(xf, fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yf, fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    handles = [plt.Line2D([], [], marker="o", color="w",
                          markerfacecolor=col, markersize=7, label=c)
               for c, col in colors.items()]
    handles.append(plt.Line2D([], [], marker="o", color="w",
                              markerfacecolor="black", markersize=7,
                              label="x5=999 sentinel"))
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.99), fontsize=11, frameon=True)
    fig.suptitle("Pairwise scatter matrix of test.csv — sentinels (x5=999) in black",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(PLOTS / "test_pairwise_scatter.png", dpi=110)
    plt.close(fig)
    print(f"wrote {PLOTS / 'test_pairwise_scatter.png'}")

    # Correlation heatmap (same as before — excludes sentinels from Pearson)
    corr_df = df[features].replace(SENTINEL, np.nan).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(features, rotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(features)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr_df.values[i, j]:+.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr_df.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pairwise Pearson correlation — test.csv (x5 sentinels excluded)")
    fig.tight_layout()
    fig.savefig(PLOTS / "test_correlation_heatmap.png", dpi=130)
    plt.close(fig)
    print(f"wrote {PLOTS / 'test_correlation_heatmap.png'}")


if __name__ == "__main__":
    main()
