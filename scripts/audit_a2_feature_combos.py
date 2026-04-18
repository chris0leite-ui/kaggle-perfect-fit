"""A2: brute-force search for 2- and 3-way feature combinations that
correlate with x5.

Signal threshold: any combo with |Pearson r| > 0.15, |Spearman r| > 0.15,
or univariate-spline R^2 > 0.03 is flagged. All prior tests were linear
on individual features (max r=0.13); this pass expands to products,
ratios, sin/cos, and triple products.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SENTINEL = 999.0
BASE = ["x1", "x2", "x4", "x6", "x7", "x8", "x9", "x10", "x11"]


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    pool = pd.concat([train[train["x5"] != SENTINEL],
                      test [test["x5"] != SENTINEL]], ignore_index=True)
    y = pool["x5"].to_numpy()
    print(f"pool: {len(pool)} non-sentinel rows")

    # Build the feature bank
    feats: dict[str, np.ndarray] = {}
    for c in BASE:
        feats[c] = pool[c].to_numpy()
    feats["city"]      = (pool["City"] == "Zaragoza").astype(float).to_numpy()
    feats["theta"]     = np.arctan2(pool["x7"].values, pool["x6"].values)
    feats["sin_theta"] = np.sin(feats["theta"])
    feats["cos_theta"] = np.cos(feats["theta"])
    feats["abs_x4"]    = np.abs(pool["x4"].values)
    feats["sign_x4"]   = np.sign(pool["x4"].values)
    feats["sign_x8"]   = np.sign(pool["x8"].values)
    for c in BASE:
        feats[f"{c}_sq"] = pool[c].values ** 2

    print(f"bank size: {len(feats)} base features\n")

    def score(name: str, v: np.ndarray) -> tuple[float, float, float]:
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.std() < 1e-10:
            return 0.0, 0.0, 0.0
        r_p = pearsonr(v, y).statistic
        r_s = spearmanr(v, y).correlation
        # Spline univariate R^2
        try:
            st = SplineTransformer(n_knots=8, degree=3,
                                    extrapolation="constant").fit(v.reshape(-1, 1))
            Z = st.transform(v.reshape(-1, 1))
            pred = Ridge(alpha=1.0).fit(Z, y).predict(Z)
            r2 = 1.0 - np.var(y - pred) / np.var(y)
        except Exception:
            r2 = 0.0
        return r_p, r_s, r2

    rows = []

    # Single features (sanity)
    for n, v in feats.items():
        p, s, r2 = score(n, v)
        rows.append((n, "single", p, s, r2))

    # 2-way products and ratios
    names = list(feats.keys())
    for a, b in combinations(names, 2):
        va, vb = feats[a], feats[b]
        rows.append((f"{a}*{b}",     "prod2", *score("", va * vb)))
        if (np.abs(vb) > 1e-6).all():
            rows.append((f"{a}/{b}", "div2",  *score("", va / np.where(np.abs(vb)>1e-6, vb, 1))))

    # 3-way products (limit to avoid combinatorial blow-up — use BASE only)
    for a, b, c in combinations(BASE + ["sin_theta", "cos_theta"], 3):
        va, vb, vc = feats[a], feats[b], feats[c]
        rows.append((f"{a}*{b}*{c}", "prod3", *score("", va * vb * vc)))

    df = pd.DataFrame(rows, columns=["combo", "kind", "pearson", "spearman", "spline_r2"])
    df["abs_p"] = df["pearson"].abs()
    df["abs_s"] = df["spearman"].abs()
    df["max_abs_r"] = df[["abs_p", "abs_s"]].max(axis=1)

    print(f"total combos scored: {len(df)}\n")

    print("TOP 20 by |Pearson|:")
    top = df.sort_values("abs_p", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  {r['combo']:<30s} [{r['kind']:>6s}]  "
              f"p={r['pearson']:+.4f}  s={r['spearman']:+.4f}  "
              f"sp_r2={r['spline_r2']:+.4f}")

    print("\nTOP 20 by spline R^2:")
    top = df.sort_values("spline_r2", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  {r['combo']:<30s} [{r['kind']:>6s}]  "
              f"p={r['pearson']:+.4f}  s={r['spearman']:+.4f}  "
              f"sp_r2={r['spline_r2']:+.4f}")

    # Threshold flag
    flagged = df[(df["max_abs_r"] > 0.15) | (df["spline_r2"] > 0.03)]
    print(f"\nflagged combos (|r|>0.15 OR spline_R^2>0.03): {len(flagged)}")
    if len(flagged):
        for _, r in flagged.sort_values("max_abs_r", ascending=False).iterrows():
            print(f"  {r['combo']:<30s} [{r['kind']:>6s}]  "
                  f"p={r['pearson']:+.4f}  s={r['spearman']:+.4f}  "
                  f"sp_r2={r['spline_r2']:+.4f}")
    else:
        print("  (none — x5 remains independent of all tested combos)")

    out = REPO / "plots" / "sentinel_audit" / "a2_feature_combos.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nfull table written to {out}")


if __name__ == "__main__":
    main()
