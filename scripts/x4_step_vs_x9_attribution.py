"""Does the 'step at x4=0' really come from x4, or from x9?

Training has r(x4, x9)=+0.83 — the two are almost synonymous:
  x4>0 rows   x9 ~ N(5.97, 0.57)
  x4<0 rows   x9 ~ N(4.02, 0.57)

So the +20 "step" A1 attributes to 1{x4>0} could equally be
(2.0 units of x9 cluster gap) × (β_x9 in the true DGP).

Protocol:

1. Fit a linear_step model with x9 included (baseline).
2. Fit the same model WITHOUT x9 in the design.
3. Fit the same model WITH x9 but where we *decorrelate* x9 from
   sign(x4) by subtracting the cluster mean of x9 (x9_wc). (This is
   the Simpson-corrected approach; here we use it as a diagnostic.)
4. Fit with x9 but sign(x4) indicator PINNED to zero (i.e. drop the
   basis step component): forces x9 and/or the x4 slope to carry the
   entire cluster contrast.

Comparing the four step coefficients tells us:

- If β_step = +20 in every variant ⇒ step is real and x9-independent.
- If β_step shrinks when x9_wc is used / x9 dropped / step forced off
  ⇒ step is an x9 artefact.

We also report the learned β_x9 in each variant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SENTINEL = 999.0
RNG = 42


def load() -> pd.DataFrame:
    train = pd.read_csv(DATA / "dataset.csv")
    train["x5_is_sent"] = (train["x5"] == SENTINEL).astype(float)
    x5 = train["x5"].where(train["x5"] != SENTINEL, np.nan)
    train["x5_imp"] = x5.fillna(x5.median())
    train["city_code"] = (train["City"] == "Zaragoza").astype(float)
    train["x1sq"] = train["x1"] ** 2
    train["cos5pi_x2"] = np.cos(5 * np.pi * train["x2"])
    train["x10x11"] = train["x10"] * train["x11"]
    train["sign_x4"] = (train["x4"] > 0).astype(float)
    # Simpson-corrected x9: within-cluster centered
    means = train.groupby("sign_x4")["x9"].transform("mean")
    train["x9_wc"] = train["x9"] - means
    return train


def cv(X: np.ndarray, y: np.ndarray) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    errs = []
    for tr, va in kf.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        errs.append(float(np.mean(np.abs(m.predict(X[va]) - y[va]))))
    return float(np.mean(errs))


def fit_report(train: pd.DataFrame, feats: list[str], label: str,
               include_step: bool, x9_feat: str | None) -> dict:
    cols = list(feats)
    x4_basis_cols = ["x4"]
    if include_step:
        x4_basis_cols.append("sign_x4")
    if x9_feat is not None:
        cols.append(x9_feat)
    full_cols = cols + x4_basis_cols
    X = train[full_cols].to_numpy()
    y = train["target"].to_numpy()
    m = LinearRegression().fit(X, y)
    coefs = dict(zip(full_cols, m.coef_))
    cv_mae = cv(X, y)
    return {"label": label, "cv_mae": cv_mae, "coefs": coefs,
            "intercept": float(m.intercept_)}


def main() -> None:
    train = load()
    FIXED = ["x1sq", "cos5pi_x2", "x5_imp", "x5_is_sent",
             "x8", "x10", "x11", "x10x11", "city_code"]

    print("Cluster stats: x9 mean by sign(x4)")
    print(train.groupby("sign_x4")["x9"].agg(["mean", "std", "count"]).to_string())
    print(f"Cluster-contrast in x9 = "
          f"{train[train.sign_x4==1]['x9'].mean() - train[train.sign_x4==0]['x9'].mean():+.3f}")

    variants = [
        fit_report(train, FIXED, "A. linear_step + raw x9 (A1-ish)",
                   include_step=True, x9_feat="x9"),
        fit_report(train, FIXED, "B. linear_step, NO x9",
                   include_step=True, x9_feat=None),
        fit_report(train, FIXED, "C. linear_step + x9_wc (Simpson-corrected)",
                   include_step=True, x9_feat="x9_wc"),
        fit_report(train, FIXED, "D. NO step + raw x9",
                   include_step=False, x9_feat="x9"),
        fit_report(train, FIXED, "E. NO step + x9_wc",
                   include_step=False, x9_feat="x9_wc"),
    ]

    print("\n" + "=" * 76)
    print(f"{'variant':<48s} {'CV MAE':>7s} {'β_x4':>7s} {'β_step':>8s} {'β_x9':>8s}")
    print("=" * 76)
    for v in variants:
        c = v["coefs"]
        print(f"{v['label']:<48s} {v['cv_mae']:>7.3f} "
              f"{c['x4']:>+7.2f} "
              f"{c.get('sign_x4', float('nan')):>+8.2f} "
              f"{(c.get('x9') or c.get('x9_wc') or float('nan')):>+8.2f}")

    print("\nInterpretation:")
    print("- If β_step drops sharply when x9 is dropped or replaced with")
    print("  x9_wc, the step is mostly x9's cluster effect being")
    print("  attributed to the x4 step.")
    print("- If β_step stays ~+20, the step is a real DGP feature that")
    print("  test rows in [-0.167, +0.167] WILL genuinely exhibit.")


if __name__ == "__main__":
    main()
