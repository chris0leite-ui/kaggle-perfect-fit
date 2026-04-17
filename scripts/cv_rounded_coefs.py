"""Test integer-rounded coefficients against the free-fit baseline.

The enhanced linear model learned coefficients suspiciously close to
A1/A2's integers. If the DGP is integer, rounding should give
equivalent-or-better CV and may generalise better to the leaderboard.

Five configurations, 5-fold CV on dataset.csv:

  A. free          — LinearRegression fits all 11 coefs + intercept  (CV 2.93 baseline)
  B. A1/A2 integers — every coef fixed at A1/A2's declared value; only
                      intercept + x5_is_sent offset are fit (2 params)
  C. learned-rounded — coefs fixed at the round() of the free fit
                      (−102, +10, +30, −8, +14, 0, 0, +1, −25, −4);
                      only intercept + x5_is_sent offset fit
  D. partial-lock   — lock only the "obviously integer" ones
                      (cos(5π·x2)=+10, x5=−8, x8=+14, x10·x11=+1, city=−25,
                      x9_wc=−4); let x1², x4, x10, x11, x5_is_sent, intercept float
  E. partial-lock +  — like D but also round x1² and x4 (−100, +30) and
                      let intercept + x10 + x11 + x5_is_sent absorb the drift

Coefficient lock is implemented by subtracting the fixed contribution
from the target, then fitting OLS on the remaining features.
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

# Column order produced by make_X(x1_basis='square', x2_basis='cos',
# include_x10x11=True, x9_mode='wc') — matches cv_gam_enhanced.py:
COLS = ["x1^2", "cos(5pi*x2)", "x4", "x5_imp", "x5_is_sent",
        "x8", "x10", "x11", "x10*x11", "city", "x9_wc"]


def cv_with_locks(df: pd.DataFrame, locks: dict[str, float]) -> tuple[float, float]:
    """Run 5-fold CV. Features named in `locks` have their coefs fixed;
    remaining features + intercept are fit by OLS on the residualised target.
    Returns (overall MAE, non-sentinel MAE).
    """
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
                                   "square", "cos", True, "wc")
        X_va, _, _, _ = make_X(sub_va, x5_med, mean_hi, mean_lo,
                               "square", "cos", True, "wc")
        assert names == COLS, f"unexpected columns: {names}"

        locked_idx = [i for i, n in enumerate(names) if n in locks]
        free_idx = [i for i in range(len(names)) if i not in locked_idx]
        locked_coefs = np.array([locks[names[i]] for i in locked_idx])

        # Subtract the locked contribution from the training target.
        y_tr = sub_tr["target"].values
        locked_contrib_tr = X_tr[:, locked_idx] @ locked_coefs
        resid_y = y_tr - locked_contrib_tr

        if free_idx:
            lr = LinearRegression().fit(X_tr[:, free_idx], resid_y)
            intercept = lr.intercept_
            free_coefs = lr.coef_
        else:
            intercept = resid_y.mean()
            free_coefs = np.array([])

        # Predict.
        pred_va = intercept + X_va[:, locked_idx] @ locked_coefs
        if free_idx:
            pred_va += X_va[:, free_idx] @ free_coefs
        oof[va] = pred_va

    m = float(np.mean(np.abs(oof - y)))
    m_ns = float(np.mean(np.abs(oof[~is_sent] - y[~is_sent])))
    return m, m_ns


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    configs = [
        ("A. free (baseline)",
         {}),

        ("B. A1/A2 integers (all locked)",
         {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
          "x5_is_sent": 0, "x8": +14, "x10": 0, "x11": 0,
          "x10*x11": +1, "city": -25, "x9_wc": -4}),

        ("C. learned-rounded (all locked)",
         {"x1^2": -102, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
          "x5_is_sent": -1, "x8": +14, "x10": 0, "x11": 0,
          "x10*x11": +1, "city": -25, "x9_wc": -4}),

        ("D. partial-lock (obvious integers)",
         {"cos(5pi*x2)": +10, "x5_imp": -8, "x8": +14,
          "x10*x11": +1, "city": -25, "x9_wc": -4}),

        ("E. partial-lock (obvious + x4, x1²)",
         {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
          "x8": +14, "x10*x11": +1, "city": -25, "x9_wc": -4}),

        ("F. partial-lock (+x4=+31 variant)",
         {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +31, "x5_imp": -8,
          "x8": +14, "x10*x11": +1, "city": -25, "x9_wc": -4}),

        ("G. learned-rounded but x1²=-100 (A1 value)",
         {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
          "x5_is_sent": 0, "x8": +14, "x10": 0, "x11": 0,
          "x10*x11": +1, "city": -25, "x9_wc": -4}),
    ]

    print("=" * 80)
    print(f"{'variant':<50s}  {'CV MAE':>8s}  {'non-sent':>9s}")
    print("=" * 80)
    rows = []
    for label, locks in configs:
        m, m_ns = cv_with_locks(df, locks)
        print(f"{label:<50s}  {m:8.3f}  {m_ns:9.3f}")
        rows.append({"variant": label, "CV_MAE": m, "non_sentinel": m_ns})

    out = REPO / "plots" / "gam_enhanced" / "cv_rounded_coefs.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    # Build submission for the best locked variant — refit the free
    # coefs on the full dataset.csv then score test.csv.
    print("\n" + "=" * 80)
    print("Building 'locked-integer' submissions on full dataset")
    print("=" * 80)

    def build(name: str, locks: dict[str, float]) -> None:
        x5_med = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
        mean_hi, mean_lo = cluster_means_x9(df)
        X_tr, names, _, _ = make_X(df, x5_med, mean_hi, mean_lo,
                                   "square", "cos", True, "wc")
        X_te, _, _, _ = make_X(test, x5_med, mean_hi, mean_lo,
                               "square", "cos", True, "wc")

        locked_idx = [i for i, n in enumerate(names) if n in locks]
        free_idx = [i for i in range(len(names)) if i not in locked_idx]
        locked_coefs = np.array([locks[names[i]] for i in locked_idx])

        y = df["target"].values
        resid_y = y - X_tr[:, locked_idx] @ locked_coefs
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
        print(f"    intercept={intercept:+.3f}  free_coefs=" +
              ", ".join(f"{names[i]}:{c:+.3f}" for i, c in zip(free_idx, free_coefs)))

    # Build all locked variants that scored reasonably — we'll figure out
    # which to keep after seeing the CV ranking.
    for label, locks in configs[1:]:  # skip baseline
        safe = (label.split(".", 1)[0].strip()).lower().replace(" ", "_")
        build(f"linear_enh_locked_{safe}", locks)


if __name__ == "__main__":
    main()
