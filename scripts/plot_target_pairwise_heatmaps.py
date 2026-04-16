"""Per-pair target heatmaps to surface interactions we might have missed.

For every pair of features (xi, xj) we bin the training data on a 2D grid
and colour each cell by:

  1. raw mean(target)            -> target_pairwise_raw.png
  2. mean(target - additive fit) -> target_pairwise_residual.png

The residual version removes additive main effects so interaction
structure is what remains. The additive fit uses:

  - cubic splines on x1, x2, x4, x8 (known nonlinear / strong linear)
  - linear terms on x5 (with sentinel indicator), x9, x10, x11
  - one-hot City

x6, x7 are included in the pairwise plots but excluded from the main-effect
fit because sqrt(x6^2+x7^2) is a constant — only angle carries info and
earlier experiments showed it's pure noise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "interactions"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0

FEATURES = ["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "City"]
SPLINE_FEATURES = ["x1", "x2", "x4", "x8"]
LINEAR_FEATURES = ["x5", "x9", "x10", "x11"]
N_BINS = 12
MIN_CELL_COUNT = 4


def additive_residuals(df: pd.DataFrame) -> np.ndarray:
    """Return target minus a simple additive fit (splines + linears + City)."""
    blocks: list[np.ndarray] = []
    for f in SPLINE_FEATURES:
        vals = df[f].to_numpy().reshape(-1, 1)
        st = SplineTransformer(n_knots=8, degree=3, include_bias=False)
        blocks.append(st.fit_transform(vals))
    x5 = df["x5"].to_numpy()
    is_sent = (x5 == SENTINEL).astype(float)
    x5_imp = np.where(is_sent == 1, np.median(x5[is_sent == 0]), x5)
    blocks.append(x5_imp.reshape(-1, 1))
    blocks.append(is_sent.reshape(-1, 1))
    for f in ["x9", "x10", "x11"]:
        blocks.append(df[f].to_numpy().reshape(-1, 1))
    city_z = (df["City"].to_numpy() == "Zaragoza").astype(float).reshape(-1, 1)
    blocks.append(city_z)

    X = np.hstack(blocks)
    y = df["target"].to_numpy()
    model = Ridge(alpha=1.0).fit(X, y)
    return y - model.predict(X)


def feature_values(df: pd.DataFrame, f: str) -> np.ndarray:
    """Return numeric values for a feature (City -> 0/1, x5 sentinel -> NaN)."""
    if f == "City":
        return (df["City"].to_numpy() == "Zaragoza").astype(float)
    vals = df[f].to_numpy().astype(float).copy()
    if f == "x5":
        vals[vals == SENTINEL] = np.nan
    return vals


def axis_edges(values: np.ndarray, f: str) -> np.ndarray:
    """Bin edges for a feature; binary City -> two bins centred at 0 and 1."""
    if f == "City":
        return np.array([-0.5, 0.5, 1.5])
    v = values[~np.isnan(values)]
    lo, hi = np.percentile(v, [1, 99])
    return np.linspace(lo, hi, N_BINS + 1)


def binned_mean(xv: np.ndarray, yv: np.ndarray, zv: np.ndarray,
                xedges: np.ndarray, yedges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of zv per (xedges, yedges) cell. Cells with < MIN_CELL_COUNT are NaN."""
    mask = ~(np.isnan(xv) | np.isnan(yv) | np.isnan(zv))
    sum_z, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[xedges, yedges],
                                 weights=zv[mask])
    count, _, _ = np.histogram2d(xv[mask], yv[mask], bins=[xedges, yedges])
    mean = np.full_like(sum_z, np.nan)
    ok = count >= MIN_CELL_COUNT
    mean[ok] = sum_z[ok] / count[ok]
    return mean, count


def plot_pairwise(df: pd.DataFrame, z_col: str, title: str, fname: str,
                  vmax: float | None = None) -> None:
    n = len(FEATURES)
    vals = {f: feature_values(df, f) for f in FEATURES}
    edges = {f: axis_edges(vals[f], f) for f in FEATURES}
    z = df[z_col].to_numpy()

    # Colour scale: symmetric around 0, common across all cells.
    all_means: list[float] = []
    grid: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for yf in FEATURES:
        for xf in FEATURES:
            if xf == yf:
                continue
            m, c = binned_mean(vals[xf], vals[yf], z, edges[xf], edges[yf])
            grid[(yf, xf)] = (m, c)
            all_means.extend(m[~np.isnan(m)].tolist())

    if vmax is None:
        vmax = float(np.percentile(np.abs(all_means), 99)) if all_means else 1.0

    fig, axes = plt.subplots(n, n, figsize=(24, 24))

    for i, yf in enumerate(FEATURES):
        for j, xf in enumerate(FEATURES):
            ax = axes[i, j]
            if xf == yf:
                v = vals[xf]
                good = ~(np.isnan(v) | np.isnan(z))
                edges_x = edges[xf]
                centres = 0.5 * (edges_x[:-1] + edges_x[1:])
                means = []
                for k in range(len(edges_x) - 1):
                    sel = good & (v >= edges_x[k]) & (v < edges_x[k + 1])
                    means.append(z[sel].mean() if sel.sum() >= MIN_CELL_COUNT else np.nan)
                ax.plot(centres, means, marker="o", color="#333", ms=3)
                ax.axhline(0, color="gray", lw=0.5)
                ax.set_facecolor("#f7f7f7")
            else:
                m, _ = grid[(yf, xf)]
                im = ax.imshow(m.T, origin="lower", aspect="auto",
                               extent=[edges[xf][0], edges[xf][-1],
                                       edges[yf][0], edges[yf][-1]],
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               interpolation="nearest")

            if i == n - 1:
                ax.set_xlabel(xf, fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yf, fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    cbar_ax = fig.add_axes((0.93, 0.15, 0.015, 0.7))
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label=f"mean({z_col}) per bin")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.92, 0.99))
    out = PLOTS / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}  (vmax={vmax:.2f})")


def interaction_score(df: pd.DataFrame, z_col: str) -> pd.DataFrame:
    """Rank pairs by double-centred RMS of the binned mean residual.

    For each (xi, xj), compute the grid of mean(z) per cell, subtract the
    row and column means (removing any remaining main effects), and take
    the RMS of the result. Large values = pure interaction signal.
    """
    vals = {f: feature_values(df, f) for f in FEATURES}
    edges = {f: axis_edges(vals[f], f) for f in FEATURES}
    z = df[z_col].to_numpy()

    rows = []
    for i, yf in enumerate(FEATURES):
        for j, xf in enumerate(FEATURES):
            if j <= i:
                continue
            m, c = binned_mean(vals[xf], vals[yf], z, edges[xf], edges[yf])
            m_fill = np.where(np.isnan(m), 0.0, m)
            # Double-centre: remove row and column means weighted by counts.
            w = np.where(np.isnan(m), 0.0, c)
            if w.sum() == 0:
                continue
            row_w = w.sum(axis=1, keepdims=True)
            col_w = w.sum(axis=0, keepdims=True)
            row_mean = np.where(row_w > 0,
                                (m_fill * w).sum(axis=1, keepdims=True) / np.maximum(row_w, 1),
                                0.0)
            col_mean = np.where(col_w > 0,
                                (m_fill * w).sum(axis=0, keepdims=True) / np.maximum(col_w, 1),
                                0.0)
            grand = (m_fill * w).sum() / w.sum()
            inter = m_fill - row_mean - col_mean + grand
            inter_masked = np.where(np.isnan(m), 0.0, inter)
            rms = float(np.sqrt((inter_masked ** 2 * w).sum() / w.sum()))
            n_cells = int((~np.isnan(m)).sum())
            rows.append({"xi": xf, "xj": yf, "rms": rms, "n_cells": n_cells})

    return pd.DataFrame(rows).sort_values("rms", ascending=False).reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv")
    print(f"dataset.csv: {df.shape}")

    df["target_raw"] = df["target"].astype(float)
    df["target_residual"] = additive_residuals(df)
    resid_std = df["target_residual"].std()
    print(f"additive residual std: {resid_std:.3f} "
          f"(raw target std: {df['target_raw'].std():.3f})")

    plot_pairwise(df, "target_raw",
                  "Mean(target) per (xi, xj) bin — raw",
                  "target_pairwise_raw.png")
    plot_pairwise(df, "target_residual",
                  "Mean(target - additive fit) per (xi, xj) bin — "
                  "non-zero cells indicate interactions",
                  "target_pairwise_residual.png")

    rank = interaction_score(df, "target_residual")
    print("\nTop 15 pairs by pure-interaction RMS (residual, double-centred):")
    print(rank.head(15).to_string(index=False))
    rank.to_csv(PLOTS / "interaction_ranking.csv", index=False)
    print(f"\nwrote {PLOTS / 'interaction_ranking.csv'}")


if __name__ == "__main__":
    main()
