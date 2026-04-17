"""Tune the cross_LE ensemble — LB confirmed 2.94 on our first attempt.

cross_LE = 0.5 * (LIN_x4 + EBM_x9).  Explore three axes:

  1. Blend weight w ∈ {0.3, 0.4, 0.5, 0.6, 0.7} for  w*LIN_x4 + (1-w)*EBM_x9
  2. Locked-integer LIN_x4 (A1/A2 coefs, intercept-only fit)
     vs free-fit LIN_x4.
  3. Triple ensemble adding full EBM in varying weights:
        0.5*(LIN_x4 + EBM_x9) + λ * EBM_full   (mixing parameter λ)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import FEATURES_ALL, SENTINEL, SEED, preprocess  # noqa: E402
from cv_x4_x9_swap_ensemble import ebm_features, design_matrix, fit_ebm  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5

# Locked-integer coefs for LIN_x4 (no x9 variant)
# Column order: x1^2, cos(5pi*x2), x4, x5_imp, x5_is_sent, x8, x10, x11, x10*x11, city
LOCKED_COEFS_B = {  # A1/A2 declared integers (x1^2 = -100)
    "x1^2": -100, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
    "x5_is_sent": 0, "x8": +14, "x10": 0, "x11": 0,
    "x10*x11": +1, "city": -25,
}
LOCKED_COEFS_C = {  # learned-rounded (x1^2 = -102)
    "x1^2": -102, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
    "x5_is_sent": -1, "x8": +14, "x10": 0, "x11": 0,
    "x10*x11": +1, "city": -25,
}

LIN_COL_ORDER = ["x1^2", "cos(5pi*x2)", "x4", "x5_imp", "x5_is_sent",
                 "x8", "x10", "x11", "x10*x11", "city"]


def lin_x4_locked(df_tr, df_va, df_full, locks: dict[str, float]):
    """Fit LIN_x4 with integer-locked coefs (only intercept floats)."""
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())

    def X(df):
        return design_matrix(df, x5_med, include_x4=True, include_x9=False)

    X_tr = X(df_tr); X_full = X(df_full)
    locked_vec = np.array([locks[c] for c in LIN_COL_ORDER])
    resid = df_tr["target"].values - X_tr @ locked_vec
    intercept = resid.mean()
    return X(df_va) @ locked_vec + intercept


def lin_x4_free(df_tr, df_va):
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5_med, include_x4=True, include_x9=False)
    X_va = design_matrix(df_va, x5_med, include_x4=True, include_x9=False)
    return LinearRegression().fit(X_tr, df_tr["target"].values).predict(X_va)


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def cv_oofs(df: pd.DataFrame):
    """Return OOF predictions for the base models we need."""
    y = df["target"].values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(df)) for k in
           ["lin_x4_free", "lin_x4_b", "lin_x4_c",
            "ebm_x9", "ebm_full"]}
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # LIN_x4 variants
        oof["lin_x4_free"][va] = lin_x4_free(sub_tr, sub_va)
        oof["lin_x4_b"][va]    = lin_x4_locked(sub_tr, sub_va, df, LOCKED_COEFS_B)
        oof["lin_x4_c"][va]    = lin_x4_locked(sub_tr, sub_va, df, LOCKED_COEFS_C)

        # EBM_x9 (no x4)
        feats = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(sub_tr, feats, x5m)
        X_va = preprocess(sub_va, feats, x5m)
        oof["ebm_x9"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        # EBM_full
        feats = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats, x5m)
        X_va = preprocess(sub_va, feats, x5m)
        oof["ebm_full"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        print(f"  fold {fold+1}/{N_SPLITS} in {time.time()-t0:.0f}s")
    return oof


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values

    print("=" * 70)
    print("5-fold CV OOF for base models")
    print("=" * 70)
    oof = cv_oofs(df)

    print("\n" + "=" * 70)
    print("Base models")
    print("=" * 70)
    for name in ["lin_x4_free", "lin_x4_b", "lin_x4_c", "ebm_x9", "ebm_full"]:
        m = mae(oof[name], y); mn = mae(oof[name][~is_sent], y[~is_sent])
        print(f"  {name:<18s}  overall={m:.3f}  non-sent={mn:.3f}")

    # ------------------------------------------------------------------
    # Weight sweep for cross_LE (LIN_x4_free + EBM_x9)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Weight sweep: w * LIN_x4 + (1-w) * EBM_x9  [free LIN_x4]")
    print("=" * 70)
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p = w * oof["lin_x4_free"] + (1-w) * oof["ebm_x9"]
        m = mae(p, y); mn = mae(p[~is_sent], y[~is_sent])
        print(f"  w={w:.1f}    overall={m:.3f}  non-sent={mn:.3f}")

    # Same but with locked LIN_x4
    print("\nWeight sweep with LOCKED LIN_x4 (variant B, A1/A2 integers):")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p = w * oof["lin_x4_b"] + (1-w) * oof["ebm_x9"]
        m = mae(p, y); mn = mae(p[~is_sent], y[~is_sent])
        print(f"  w={w:.1f}    overall={m:.3f}  non-sent={mn:.3f}")

    print("\nWeight sweep with LOCKED LIN_x4 (variant C, x1^2 = -102):")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p = w * oof["lin_x4_c"] + (1-w) * oof["ebm_x9"]
        m = mae(p, y); mn = mae(p[~is_sent], y[~is_sent])
        print(f"  w={w:.1f}    overall={m:.3f}  non-sent={mn:.3f}")

    # ------------------------------------------------------------------
    # Triple ensemble: cross_LE + EBM_full
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Triple: (1-λ) * cross_LE + λ * EBM_full  [free LIN_x4, 0.5 blend]")
    print("=" * 70)
    cross_LE = 0.5 * (oof["lin_x4_free"] + oof["ebm_x9"])
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        p = (1-lam) * cross_LE + lam * oof["ebm_full"]
        m = mae(p, y); mn = mae(p[~is_sent], y[~is_sent])
        print(f"  λ={lam:.1f}    overall={m:.3f}  non-sent={mn:.3f}")

    print("\nTriple with LOCKED LIN_x4 (variant B) + EBM_x9 + EBM_full:")
    cross_LE_b = 0.5 * (oof["lin_x4_b"] + oof["ebm_x9"])
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5]:
        p = (1-lam) * cross_LE_b + lam * oof["ebm_full"]
        m = mae(p, y); mn = mae(p[~is_sent], y[~is_sent])
        print(f"  λ={lam:.1f}    overall={m:.3f}  non-sent={mn:.3f}")

    # ------------------------------------------------------------------
    # Build submissions on full data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Building submissions for top variants")
    print("=" * 70)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # Build LIN_x4 free + locked, EBM_x9, EBM_full on FULL data
    X_tr = design_matrix(df, x5m_full, True, False)
    X_te = design_matrix(test, x5m_full, True, False)
    lin_x4_free_pred = LinearRegression().fit(X_tr, df["target"].values).predict(X_te)

    def _lock(locks):
        v = np.array([locks[c] for c in LIN_COL_ORDER])
        resid = df["target"].values - X_tr @ v
        intercept = resid.mean()
        return X_te @ v + intercept

    lin_x4_b_pred = _lock(LOCKED_COEFS_B)
    lin_x4_c_pred = _lock(LOCKED_COEFS_C)

    feats_x9 = ebm_features(with_x4=False, with_x9=True)
    X_tr = preprocess(df, feats_x9, x5m_full)
    X_te = preprocess(test, feats_x9, x5m_full)
    ebm_x9_pred = fit_ebm(X_tr, df["target"].values).predict(X_te)

    feats_full = ebm_features(with_x4=True, with_x9=True)
    X_tr = preprocess(df, feats_full, x5m_full)
    X_te = preprocess(test, feats_full, x5m_full)
    ebm_full_pred = fit_ebm(X_tr, df["target"].values).predict(X_te)

    submissions = {
        "cross_LE_free_50":    0.5 * (lin_x4_free_pred + ebm_x9_pred),  # LB 2.94 baseline
        "cross_LE_free_60":    0.6 * lin_x4_free_pred + 0.4 * ebm_x9_pred,
        "cross_LE_free_40":    0.4 * lin_x4_free_pred + 0.6 * ebm_x9_pred,
        "cross_LE_locked_b_50":0.5 * (lin_x4_b_pred + ebm_x9_pred),
        "cross_LE_locked_c_50":0.5 * (lin_x4_c_pred + ebm_x9_pred),
        "cross_LE_triple_full20": 0.4 * lin_x4_free_pred + 0.4 * ebm_x9_pred + 0.2 * ebm_full_pred,
    }
    for name, p in submissions.items():
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(
            SUBS / f"submission_ensemble_{name}.csv", index=False)
        print(f"  wrote submission_ensemble_{name}.csv  "
              f"mean={p.mean():+.3f}  range=[{p.min():+.2f}, {p.max():+.2f}]")


if __name__ == "__main__":
    main()
