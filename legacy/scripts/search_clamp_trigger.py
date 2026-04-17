"""Systematic search for the hidden A1 clamp trigger within x4<0 & x8<0.

Prior tests found no single feature or the x6/x7 angle predicts the 23%
clamp trigger within the quadrant. Here we expand the search to:
  (a) all 36 pairwise products x_i * x_j
  (b) all 9 C 2 = 36 pairwise differences x_i - x_j
  (c) three-feature interactions of the top singletons
  (d) a nonparametric LightGBM classifier with interactions on all features
  (e) polar/trigonometric transforms of features (angle-like features)

Success = a metric that separates bad vs perfect rows better than the
trivial 77/23 baseline. We report both:
  - AUC / accuracy of the classifier
  - Best single rule found (split-stats like "bad rate in-rule vs out")
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402


def main():
    df = pd.read_csv(REPO / "data" / "dataset.csv").reset_index(drop=True)
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    a1 = approach1_predict(df, x5m)
    resid = df["target"].values - a1
    abs_resid = np.abs(resid)
    is_sent = (df["x5"] == SENTINEL).values
    q = (~is_sent) & (df["x4"] < 0) & (df["x8"] < 0)
    sub = df[q].copy().reset_index(drop=True)
    sub["is_bad"] = (abs_resid[q] > 0.1).astype(int)
    print(f"Quadrant rows: {len(sub)}  bad rate: {sub.is_bad.mean():.2%}  "
          f"trivial baseline accuracy: {1 - sub.is_bad.mean():.3f}")

    base_feats = ["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]
    X_base = sub[base_feats].values

    # ---- (a) pairwise products ----
    print("\n" + "=" * 72)
    print("Pairwise products  x_i * x_j  as clamp predictors")
    print("=" * 72)
    prod_results = []
    for i, j in combinations(range(len(base_feats)), 2):
        f = X_base[:, i] * X_base[:, j]
        corr = np.corrcoef(f, sub.is_bad)[0, 1]
        prod_results.append((f"{base_feats[i]}*{base_feats[j]}", corr, f))
    prod_results.sort(key=lambda t: abs(t[1]), reverse=True)
    print(f"Top 10 by |corr| with is_bad:")
    for name, c, _ in prod_results[:10]:
        print(f"  {name:<12s}  corr = {c:+.4f}")

    # ---- (b) pairwise differences ----
    print("\n" + "=" * 72)
    print("Pairwise differences  x_i - x_j")
    print("=" * 72)
    diff_results = []
    for i, j in combinations(range(len(base_feats)), 2):
        f = X_base[:, i] - X_base[:, j]
        corr = np.corrcoef(f, sub.is_bad)[0, 1]
        diff_results.append((f"{base_feats[i]}-{base_feats[j]}", corr))
    diff_results.sort(key=lambda t: abs(t[1]), reverse=True)
    for name, c in diff_results[:10]:
        print(f"  {name:<12s}  corr = {c:+.4f}")

    # ---- (c) LightGBM full classifier with interactions ----
    print("\n" + "=" * 72)
    print("LightGBM classifier (full interactions) on all features + x6/x7 angle")
    print("=" * 72)
    try:
        import lightgbm as lgb
        theta = np.arctan2(sub["x7"].values, sub["x6"].values)
        X = np.column_stack([X_base,
                              np.sin(theta), np.cos(theta),
                              np.sin(2 * theta), np.cos(2 * theta)])
        cols = base_feats + ["sin_theta", "cos_theta", "sin_2theta", "cos_2theta"]
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, random_state=42, verbose=-1,
        )
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_acc = cross_val_score(clf, X, sub.is_bad.values, cv=skf,
                                      scoring="accuracy")
        scores_auc = cross_val_score(clf, X, sub.is_bad.values, cv=skf,
                                      scoring="roc_auc")
        print(f"  5-fold CV accuracy: {scores_acc.mean():.3f} "
              f"+- {scores_acc.std():.3f} "
              f"(baseline = {1 - sub.is_bad.mean():.3f})")
        print(f"  5-fold CV AUC:      {scores_auc.mean():.3f} "
              f"+- {scores_auc.std():.3f} (0.5 = random)")

        # Fit on full data to see the top features
        clf.fit(X, sub.is_bad.values)
        imp = pd.DataFrame({"feature": cols, "importance": clf.feature_importances_})
        imp = imp.sort_values("importance", ascending=False)
        print(f"\n  Feature importances (full-data fit):")
        for _, row in imp.iterrows():
            print(f"    {row.feature:<14s}  {row.importance:>5d}")
    except ImportError:
        print("  lightgbm not installed — skipping")

    # ---- (d) Best single rule found ----
    print("\n" + "=" * 72)
    print(f"Best single rule among pairwise products (by info gain):")
    print("=" * 72)
    best_rule = None
    best_gain = 0
    trivial = max(sub.is_bad.mean(), 1 - sub.is_bad.mean())
    for name, _, f in prod_results[:20]:  # top 20 by correlation
        for q_pct in [25, 33, 50, 67, 75]:
            thresh = np.percentile(f, q_pct)
            for direction in [">", "<"]:
                m = (f > thresh) if direction == ">" else (f < thresh)
                if m.sum() < 10 or (~m).sum() < 10:
                    continue
                acc = max(
                    sub.is_bad[m].mean(), 1 - sub.is_bad[m].mean()
                ) * m.sum() / len(sub) + max(
                    sub.is_bad[~m].mean(), 1 - sub.is_bad[~m].mean()
                ) * (~m).sum() / len(sub)
                gain = acc - trivial
                if gain > best_gain:
                    best_gain = gain
                    best_rule = (name, direction, float(thresh), acc,
                                  sub.is_bad[m].mean(), sub.is_bad[~m].mean())
    if best_rule:
        name, d, t, a, r_in, r_out = best_rule
        print(f"  Rule: {name} {d} {t:.3f}")
        print(f"  Accuracy: {a:.3f}  (vs trivial {trivial:.3f})")
        print(f"  Bad rate in-rule: {r_in:.2%}   out-of-rule: {r_out:.2%}")

    print("\n" + "=" * 72)
    print("Conclusion")
    print("=" * 72)
    print("  If LightGBM accuracy ≈ 0.77 (trivial) and AUC ≈ 0.5, the clamp")
    print("  trigger is genuinely not in the observed features. Otherwise,")
    print("  we've found the trigger and can route more precisely.")


if __name__ == "__main__":
    main()
