"""Pooled-feature rediscovery (idea #1 from the DGP-archaeology plan).

Motivation
----------
PC and LiNGAM were earlier run on training features only. That pool is
selection-biased — most visibly, r(x4, x9) = +0.83 in train but +0.001
in test. Any causal / association edge that exists in train but not in
the pooled 3000-row set is a selection artifact, not DGP structure.

This script:

1. Loads dataset.csv (1500 labelled) + test.csv (1500 unlabelled),
   pools their 10 numeric features + City, handles x5=999 as NaN.
2. Computes pairwise Pearson + Spearman correlations on
   (train only) vs (test only) vs (pooled). Ranks every pair by
   |r_train - r_pooled| and |r_train - r_test| — large gaps are
   selection-bias suspects.
3. Computes partial correlations controlling for (x4, City) to expose
   edges that survive after accounting for the two strongest
   confounders; train-only edges that *appear* after partialling are
   interesting too.
4. Runs PC + DirectLiNGAM on both the train-only and pooled matrices
   and diffs the resulting adjacency sets.
5. Writes a whitelist of pooled-stable edges and a CSV of every pair's
   train/test/pooled correlation for downstream modelling scripts.

Outputs
-------
plots/pooled_rediscovery/correlation_table.csv     per-pair r_train, r_test, r_pooled
plots/pooled_rediscovery/whitelist.csv             edges stable in pool
plots/pooled_rediscovery/suspect_edges.csv         edges present only in train
plots/pooled_rediscovery/heatmaps.png              three correlation heatmaps
plots/pooled_rediscovery/partial_heatmaps.png      partial correlations
plots/pooled_rediscovery/shift_bars.png            |r_train - r_pooled| ranked
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "pooled_rediscovery"
OUT.mkdir(parents=True, exist_ok=True)

NUMERIC = ["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]
SENTINEL = 999.0
SHIFT_THRESHOLD = 0.10   # flag pairs with |r_train - r_pooled| > 0.10


def load_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["x5"] = df["x5"].replace(SENTINEL, np.nan)
        df["city_code"] = (df["City"] == "Zaragoza").astype(int)
    train["split"] = "train"
    test["split"] = "test"
    pool = pd.concat([train, test], ignore_index=True, sort=False)
    return train, test, pool


def pearson_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method="pearson")


def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method="spearman")


def partial_correlation(df: pd.DataFrame, cols: list[str], controls: list[str]) -> pd.DataFrame:
    """Partial correlation between every (i, j) controlling for `controls`.

    Implemented as: regress i and j on controls, correlate residuals.
    """
    from sklearn.linear_model import LinearRegression

    n = len(cols)
    out = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    sub = df[cols + controls].dropna()
    C = sub[controls].to_numpy()
    residuals = {}
    for c in cols:
        y = sub[c].to_numpy()
        if c in controls:
            residuals[c] = y - y.mean()
            continue
        model = LinearRegression().fit(C, y)
        residuals[c] = y - model.predict(C)
    for i, j in combinations(cols, 2):
        r, _ = pearsonr(residuals[i], residuals[j])
        out.loc[i, j] = r
        out.loc[j, i] = r
    return out


def build_pair_table(train: pd.DataFrame, test: pd.DataFrame, pool: pd.DataFrame,
                     cols: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(cols, 2):
        def r(df: pd.DataFrame, method: str) -> float:
            sub = df[[a, b]].dropna()
            if len(sub) < 20:
                return np.nan
            if method == "pearson":
                return pearsonr(sub[a], sub[b]).statistic
            return spearmanr(sub[a], sub[b]).correlation
        r_tr_p = r(train, "pearson")
        r_te_p = r(test, "pearson")
        r_po_p = r(pool, "pearson")
        r_tr_s = r(train, "spearman")
        r_po_s = r(pool, "spearman")
        rows.append({
            "a": a, "b": b,
            "r_train": r_tr_p,
            "r_test": r_te_p,
            "r_pool": r_po_p,
            "rs_train": r_tr_s,
            "rs_pool": r_po_s,
            "shift": r_tr_p - r_po_p,
            "train_test_gap": r_tr_p - r_te_p,
        })
    tab = pd.DataFrame(rows)
    tab["abs_shift"] = tab["shift"].abs()
    tab = tab.sort_values("abs_shift", ascending=False).reset_index(drop=True)
    return tab


def plot_heatmaps(mats: dict[str, pd.DataFrame], cols: list[str], title: str,
                  fname: Path, vmin: float = -1.0, vmax: float = 1.0) -> None:
    fig, axes = plt.subplots(1, len(mats), figsize=(5 * len(mats), 4.5))
    if len(mats) == 1:
        axes = [axes]
    for ax, (name, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat.values, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        ax.set_title(name, fontsize=11)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > 0.55 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_shift_bars(tab: pd.DataFrame, fname: Path, top_n: int = 20) -> None:
    top = tab.head(top_n)
    labels = [f"{r.a} · {r.b}" for r in top.itertuples()]
    x = np.arange(len(top))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(top))))
    ax.barh(x - width / 2, top["r_train"], width, label="r_train", color="#d95f02")
    ax.barh(x + width / 2, top["r_pool"], width, label="r_pool",  color="#1b9e77")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Pearson r")
    ax.set_title(f"Top {top_n} pairs by |r_train - r_pool|  "
                 "(pooled is the unbiased estimate)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def try_causal_discovery(train: pd.DataFrame, pool: pd.DataFrame,
                         cols: list[str]) -> dict[str, set[tuple[str, str]]]:
    """Run PC + DirectLiNGAM on train-only and pooled data.

    Returns {name -> set of (a, b) directed edges}. Skips quietly if
    the optional packages aren't installed.
    """
    results: dict[str, set[tuple[str, str]]] = {}

    train_arr = train[cols].dropna().to_numpy()
    pool_arr = pool[cols].dropna().to_numpy()

    try:
        from causallearn.search.ConstraintBased.PC import pc

        def pc_edges(arr: np.ndarray) -> set[tuple[str, str]]:
            cg = pc(arr, alpha=0.05, indep_test="fisherz", verbose=False)
            edges = set()
            g = cg.G.graph
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if g[i, j] == 1 and g[j, i] == -1:
                        edges.add((cols[j], cols[i]))
                    elif g[i, j] == -1 and g[j, i] == -1:
                        edges.add(tuple(sorted([cols[i], cols[j]])))
            return edges

        results["pc_train"] = pc_edges(train_arr)
        results["pc_pool"] = pc_edges(pool_arr)
    except Exception as exc:
        print(f"[causal] PC skipped: {exc}")

    try:
        import lingam

        def lingam_edges(arr: np.ndarray, thr: float = 0.1) -> set[tuple[str, str]]:
            model = lingam.DirectLiNGAM()
            model.fit(arr)
            B = model.adjacency_matrix_
            edges = set()
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j and abs(B[i, j]) >= thr:
                        edges.add((cols[j], cols[i]))
            return edges

        results["lingam_train"] = lingam_edges(train_arr)
        results["lingam_pool"] = lingam_edges(pool_arr)
    except Exception as exc:
        print(f"[causal] DirectLiNGAM skipped: {exc}")

    return results


def main() -> None:
    print("=" * 72)
    print("Pooled-feature rediscovery")
    print("=" * 72)

    train, test, pool = load_pool()
    cols_core = NUMERIC + ["city_code"]
    print(f"train={len(train)}  test={len(test)}  pool={len(pool)}")
    print(f"x5 sentinels: train={train['x5'].isna().sum()}  "
          f"test={test['x5'].isna().sum()}")

    # Pairwise table -------------------------------------------------------
    tab = build_pair_table(train, test, pool, cols_core)
    tab.to_csv(OUT / "correlation_table.csv", index=False)

    print("\nTop 15 pairs by |r_train - r_pool|:")
    cols_show = ["a", "b", "r_train", "r_test", "r_pool", "shift", "train_test_gap"]
    print(tab[cols_show].head(15).to_string(index=False,
          formatters={c: "{:+.3f}".format for c in cols_show[2:]}))

    # Heatmaps -------------------------------------------------------------
    p_train = pearson_matrix(train, cols_core)
    p_test = pearson_matrix(test, cols_core)
    p_pool = pearson_matrix(pool, cols_core)
    plot_heatmaps({"train (1500)": p_train, "test (1500)": p_test,
                   "pool (3000)": p_pool}, cols_core,
                  "Pearson correlation — feature pairs",
                  OUT / "heatmaps.png")

    # Partial correlations controlling for x4 + city_code ------------------
    controls = ["x4", "city_code"]
    free_cols = [c for c in cols_core if c not in controls]
    pc_train = partial_correlation(train, free_cols, controls)
    pc_pool = partial_correlation(pool, free_cols, controls)
    plot_heatmaps({"train partial (| x4, city)": pc_train,
                   "pool partial (| x4, city)": pc_pool}, free_cols,
                  "Partial correlations controlling for x4 and City",
                  OUT / "partial_heatmaps.png", vmin=-0.6, vmax=0.6)

    # Shift bars -----------------------------------------------------------
    plot_shift_bars(tab, OUT / "shift_bars.png", top_n=20)

    # Whitelist / suspect CSVs --------------------------------------------
    suspects = tab[tab["abs_shift"] > SHIFT_THRESHOLD].copy()
    whitelist = tab[tab["abs_shift"] <= SHIFT_THRESHOLD].copy()
    # Within the whitelist, highlight "real" pooled edges ( |r_pool| > 0.05 )
    whitelist["pooled_edge"] = whitelist["r_pool"].abs() > 0.05
    suspects.to_csv(OUT / "suspect_edges.csv", index=False)
    whitelist.to_csv(OUT / "whitelist.csv", index=False)

    print(f"\nSuspect edges (|shift| > {SHIFT_THRESHOLD}): {len(suspects)}")
    if len(suspects):
        print(suspects[cols_show].to_string(index=False,
              formatters={c: "{:+.3f}".format for c in cols_show[2:]}))
    print(f"\nPooled-stable edges with |r_pool| > 0.05:")
    real = whitelist[whitelist["pooled_edge"]]
    if len(real):
        print(real[["a", "b", "r_train", "r_test", "r_pool"]].to_string(index=False,
              formatters={c: "{:+.3f}".format for c in ["r_train", "r_test", "r_pool"]}))
    else:
        print("  (none — every pool-stable pair has |r| < 0.05)")

    # Causal discovery -----------------------------------------------------
    edges = try_causal_discovery(train, pool, cols_core)
    if edges:
        print("\nCausal-discovery edge sets:")
        for name, es in edges.items():
            print(f"  {name:14s}  |edges|={len(es)}")
        for tr_key, po_key in [("pc_train", "pc_pool"), ("lingam_train", "lingam_pool")]:
            if tr_key in edges and po_key in edges:
                only_train = edges[tr_key] - edges[po_key]
                only_pool = edges[po_key] - edges[tr_key]
                print(f"\n  {tr_key} \\ {po_key}  (train-only, likely bias): "
                      f"{sorted(only_train)}")
                print(f"  {po_key} \\ {tr_key}  (pool-only, potential DGP): "
                      f"{sorted(only_pool)}")
        # Serialise
        with open(OUT / "causal_edges.txt", "w") as fh:
            for name, es in edges.items():
                fh.write(f"[{name}]\n")
                for e in sorted(es):
                    fh.write(f"  {e}\n")

    print(f"\nAll artefacts written to {OUT}")


if __name__ == "__main__":
    main()
