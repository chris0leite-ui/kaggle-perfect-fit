"""Rebuild locked-integer linear submissions WITHOUT the x9_wc term.

The original locked_b/locked_c submissions scored ~10.75 on Kaggle LB
because β_x9_wc = -4 extrapolates into the off-diagonal test quadrants
(48.9% of test rows) that don't exist in training. The within-cluster
Simpson slope of -4 turned out to be a training-specific artifact.

This script rebuilds the same integer-locked linear model but drops x9
entirely. The model is essentially A2's closed form with integer coefs.

Variants:
  - no_x9_b: x1^2=-100, cos(5pi*x2)=+10, x4=+30, x5=-8, x8=+14,
             x10*x11=+1, city=-25   (A1/A2 declared integers)
  - no_x9_c: same but x1^2=-102 (learned-rounded)
  - no_x9_f: same as b but x4=+31

5-fold CV + submissions written to submissions/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_gam_enhanced import make_X, cluster_means_x9, SENTINEL  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
SEED = 42
N_SPLITS = 5

# Column order with x9_mode='none':
COLS_NO_X9 = ["x1^2", "cos(5pi*x2)", "x4", "x5_imp", "x5_is_sent",
              "x8", "x10", "x11", "x10*x11", "city"]


def cv_locked_no_x9(df: pd.DataFrame, locks: dict[str, float]) -> tuple[float, float]:
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        mean_hi, mean_lo = cluster_means_x9(sub_tr)
        X_tr, names, _, _ = make_X(sub_tr, x5_med, mean_hi, mean_lo,
                                   "square", "cos", True, "none")
        X_va, _, _, _ = make_X(sub_va, x5_med, mean_hi, mean_lo,
                               "square", "cos", True, "none")
        assert names == COLS_NO_X9, names

        locked_idx = [i for i, n in enumerate(names) if n in locks]
        free_idx = [i for i in range(len(names)) if i not in locked_idx]
        locked_coefs = np.array([locks[names[i]] for i in locked_idx])

        y_tr = sub_tr["target"].values
        resid_y = y_tr - X_tr[:, locked_idx] @ locked_coefs
        if free_idx:
            lr = LinearRegression().fit(X_tr[:, free_idx], resid_y)
            intercept = lr.intercept_
            free_coefs = lr.coef_
        else:
            intercept = resid_y.mean()
            free_coefs = np.array([])

        pred = intercept + X_va[:, locked_idx] @ locked_coefs
        if free_idx:
            pred += X_va[:, free_idx] @ free_coefs
        oof[va] = pred

    return float(np.mean(np.abs(oof - y))), float(np.mean(np.abs(oof[~is_sent] - y[~is_sent])))


def build(df: pd.DataFrame, test: pd.DataFrame, locks: dict[str, float], name: str) -> None:
    x5_med = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    mean_hi, mean_lo = cluster_means_x9(df)
    X_tr, names, _, _ = make_X(df, x5_med, mean_hi, mean_lo,
                               "square", "cos", True, "none")
    X_te, _, _, _ = make_X(test, x5_med, mean_hi, mean_lo,
                           "square", "cos", True, "none")

    locked_idx = [i for i, n in enumerate(names) if n in locks]
    free_idx = [i for i in range(len(names)) if i not in locked_idx]
    locked_coefs = np.array([locks[names[i]] for i in locked_idx])

    resid_y = df["target"].values - X_tr[:, locked_idx] @ locked_coefs
    if free_idx:
        lr = LinearRegression().fit(X_tr[:, free_idx], resid_y)
        intercept = lr.intercept_
        free_coefs = lr.coef_
    else:
        intercept = resid_y.mean()
        free_coefs = np.array([])

    pred = intercept + X_te[:, locked_idx] @ locked_coefs
    if free_idx:
        pred += X_te[:, free_idx] @ free_coefs
    out = pd.DataFrame({"id": test["id"], "target": pred})
    p = SUBS / f"submission_{name}.csv"
    out.to_csv(p, index=False)
    print(f"  wrote {p.name}  mean={pred.mean():+.3f}  "
          f"range=[{pred.min():+.2f}, {pred.max():+.2f}]")


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    configs = [
        ("b_no_x9", {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
                     "x5_is_sent": 0, "x8": +14, "x10": 0, "x11": 0,
                     "x10*x11": +1, "city": -25}),
        ("c_no_x9", {"x1^2": -102, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
                     "x5_is_sent": -1, "x8": +14, "x10": 0, "x11": 0,
                     "x10*x11": +1, "city": -25}),
        ("f_no_x9", {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +31, "x5_imp": -8,
                     "x5_is_sent": 0, "x8": +14, "x10": 0, "x11": 0,
                     "x10*x11": +1, "city": -25}),
        # also a "partial-lock without x9" so we let more coefs float if useful
        ("partial_no_x9", {"cos(5pi*x2)": +10, "x5_imp": -8, "x8": +14,
                           "x10*x11": +1, "city": -25}),
    ]

    print("=" * 70)
    print(f"{'variant':<35s}  {'CV MAE':>8s}  {'non-sent':>9s}")
    print("=" * 70)
    for name, locks in configs:
        m, m_ns = cv_locked_no_x9(df, locks)
        print(f"{name:<35s}  {m:8.3f}  {m_ns:9.3f}")

    print("\n" + "=" * 70)
    print("Building submissions (refit on full dataset.csv)")
    print("=" * 70)
    for name, locks in configs:
        build(df, test, locks, f"linear_enh_locked_{name}")


if __name__ == "__main__":
    main()
