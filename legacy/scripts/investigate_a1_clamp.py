"""Investigate the A1 'clamp' subset within x4<0 & x8<0 non-sentinel rows.

Findings (documented in CLAUDE.md):
  - A1 is exactly perfect (|resid| < 0.01) on 93.27% of non-sent rows.
  - Imperfect 6.7% are ENTIRELY in x4<0 AND x8<0 quadrant.
  - Within that quadrant, 23% of rows trigger a 'clamp' shifting the x8
    coefficient by -18.4 (residual/x8 ~= -18.4, std=0.76).

This script tests whether any observed feature or the x6/x7 angle
theta = atan2(x7, x6) predicts the clamp trigger.

Result: NO OBSERVED VARIABLE PREDICTS IT. Decision trees hit the trivial
80% baseline. Threshold rules all give ~23% bad rate regardless of cutoff.
KS / Mann-Whitney / correlation tests against theta all return null
(p > 0.6). The clamp appears to be a hidden Bernoulli(0.23) in the DGP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr
from sklearn.tree import DecisionTreeClassifier, export_text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402


def main():
    df = pd.read_csv(REPO / "data" / "dataset.csv").reset_index(drop=True)
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    a1_pred = approach1_predict(df, x5m)
    resid = df["target"].values - a1_pred
    abs_resid = np.abs(resid)
    is_sent = (df["x5"] == SENTINEL).values
    ns = ~is_sent

    print("=" * 70)
    print("Perfect-fit classification by (sign(x4), sign(x8)) quadrant")
    print("=" * 70)
    for label, mask in [("x4>0, x8>0", (df.x4>0)&(df.x8>0)&ns),
                        ("x4>0, x8<0", (df.x4>0)&(df.x8<0)&ns),
                        ("x4<0, x8>0", (df.x4<0)&(df.x8>0)&ns),
                        ("x4<0, x8<0", (df.x4<0)&(df.x8<0)&ns)]:
        frac_bad = (abs_resid[mask] > 0.1).mean()
        print(f"  {label:<14s}  n={mask.sum():<4d}  "
              f"mean|resid|={abs_resid[mask].mean():.3f}  "
              f"max|resid|={abs_resid[mask].max():.3f}  "
              f"frac>0.1={frac_bad:.2%}")

    # Focus on the mixed quadrant
    q = ns & (df.x4 < 0) & (df.x8 < 0)
    sub = df[q].copy()
    sub["is_bad"] = (abs_resid[q] > 0.1).astype(int)

    print("\n" + "=" * 70)
    print(f"Within x4<0 & x8<0 (n={q.sum()}): residual shape")
    print("=" * 70)
    bad_res = resid[q][sub.is_bad.values == 1]
    bad_x8 = sub.loc[sub.is_bad == 1, "x8"].values
    ratio = bad_res / bad_x8
    print(f"  residual / x8 on bad rows:  "
          f"mean={ratio.mean():.3f}  std={ratio.std():.3f}  "
          f"min={ratio.min():.3f}  max={ratio.max():.3f}")
    print(f"  (consistent -18.4 shift means the DGP replaces +15*x8 with "
          f"-3.4*x8 on triggered rows)")

    # Decision tree: can features classify is_bad?
    print("\n" + "=" * 70)
    print("Decision tree (max_depth=4) on features to predict clamp")
    print("=" * 70)
    feats = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                   random_state=42)
    tree.fit(sub[feats].values, sub.is_bad.values)
    print(f"Training accuracy: {tree.score(sub[feats], sub.is_bad):.3f}  "
          f"(trivial baseline = {1 - sub.is_bad.mean():.3f})")
    print(export_text(tree, feature_names=feats, show_weights=True))

    # Simple threshold rules
    print("\n" + "=" * 70)
    print("Simple threshold rules (bad rate should differ from ~23%)")
    print("=" * 70)
    for rule_name, mask in [
        ("x10 < 1.5", sub.x10 < 1.5),
        ("x10+x11 < 4", sub.x10 + sub.x11 < 4),
        ("x10*x11 < 6", sub.x10 * sub.x11 < 6),
        ("x10*x11 < 5", sub.x10 * sub.x11 < 5),
        ("x8 + x4 < -0.7", sub.x8 + sub.x4 < -0.7),
        ("|x8| > |x4|", sub.x8.abs() > sub.x4.abs()),
        ("x1 > 0.3", sub.x1 > 0.3),
        ("x9 < 4", sub.x9 < 4),
    ]:
        m_in, m_out = sub[mask], sub[~mask]
        if len(m_in) == 0 or len(m_out) == 0:
            continue
        print(f"  {rule_name:<22s}  in-rule: {m_in.is_bad.mean():.2%}  "
              f"out-of-rule: {m_out.is_bad.mean():.2%}")

    # x6/x7 angle check
    print("\n" + "=" * 70)
    print("x6/x7 angle as the clamp trigger? (radius is constant 18)")
    print("=" * 70)
    theta = np.arctan2(df["x7"].values, df["x6"].values)
    print(f"radius mean={np.hypot(df.x6, df.x7).mean():.4f}  "
          f"std={np.hypot(df.x6, df.x7).std():.6f}")

    theta_bad = theta[q][sub.is_bad.values == 1]
    theta_ok = theta[q][sub.is_bad.values == 0]
    print(f"\n  theta bad (n={len(theta_bad)}):      "
          f"mean={theta_bad.mean():+.3f}  std={theta_bad.std():.3f}")
    print(f"  theta perfect (n={len(theta_ok)}):   "
          f"mean={theta_ok.mean():+.3f}  std={theta_ok.std():.3f}")

    ks = ks_2samp(theta_bad, theta_ok)
    mw = mannwhitneyu(theta_bad, theta_ok)
    print(f"  KS test:           D={ks.statistic:.3f}  p={ks.pvalue:.4f}")
    print(f"  Mann-Whitney:      U={mw.statistic:.0f}  p={mw.pvalue:.4f}")
    for name, xs in [("theta", theta[q]), ("sin(theta)", np.sin(theta[q])),
                     ("cos(theta)", np.cos(theta[q]))]:
        r, p = pearsonr(xs, abs_resid[q])
        print(f"  corr({name:<10s}, |resid|) = {r:+.4f}  p={p:.4f}")

    print("\n  8 angular region rules:")
    for rule_name, mask in [
        ("theta > 0", theta > 0),
        ("theta < 0", theta < 0),
        ("theta in (0, pi/2)", (theta > 0) & (theta < np.pi/2)),
        ("theta in (-pi/2, 0)", (theta > -np.pi/2) & (theta < 0)),
        ("theta in (pi/2, pi)", theta > np.pi/2),
        ("theta in (-pi, -pi/2)", theta < -np.pi/2),
        ("|theta| < pi/2 (x6>0)", np.abs(theta) < np.pi/2),
        ("|theta| > pi/2 (x6<0)", np.abs(theta) > np.pi/2),
    ]:
        m = q & mask
        if m.sum() == 0:
            continue
        n_bad = ((abs_resid > 0.1) & m).sum()
        br = n_bad / m.sum()
        print(f"    {rule_name:<28s}  {n_bad}/{m.sum()} = {br:.2%}")

    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print("  No observed feature and no x6/x7 angular rule predicts the")
    print("  clamp trigger. The 23% clamp rate is uniform across every")
    print("  subset we tested. The trigger is effectively a hidden")
    print("  Bernoulli variable in the DGP that we cannot observe.")


if __name__ == "__main__":
    main()
