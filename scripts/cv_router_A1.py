"""Routing ensemble using A1 (not locked_b_full) as the memoriser.

A1 closed form from scripts/compare_formulas.py::approach1_predict:
   -100*x1^2 + 10*cos(5π*x2) + 15*x4 - 8*x5_imp + 15*x8
   - 4*x9 + x10*x11 - 25*zaragoza + 20*1(x4>0) + 92.5

A1 has ZERO free parameters. CV 1.80 overall, non-sent 0.38 — near-
perfect fit on training. LB 10.80 because of catastrophic failure on
the 34% test rows in the x4 gap + the off-diagonal x4-x9 rows.

Routing rule (defaults):
  SAFE iff  x5 != 999  AND  |x4| > 0.167  AND  sign(x4)·x9-cluster match

SAFE → A1;  everything else → triple ensemble (CV 2.824).
Sentinels always route to triple (irreducible noise).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import FEATURES_ALL, SENTINEL, SEED, preprocess  # noqa: E402
from cv_x4_x9_swap_ensemble import ebm_features, design_matrix, fit_ebm  # noqa: E402
from cv_cross_LE_tune import LOCKED_COEFS_B, LIN_COL_ORDER  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5
X4_GAP_THRESH = 0.167


def safe_mask(df: pd.DataFrame) -> np.ndarray:
    x4 = df["x4"].to_numpy()
    x9 = df["x9"].to_numpy()
    is_sent = (df["x5"].to_numpy() == SENTINEL)
    x4_clear = np.abs(x4) > X4_GAP_THRESH
    cluster_match = ((x4 > 0) & (x9 > 5)) | ((x4 < 0) & (x9 < 5))
    return (~is_sent) & x4_clear & cluster_match


def lin_x4_locked_nox9_predict(df_tr, df_va):
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5_med, include_x4=True, include_x9=False)
    X_va = design_matrix(df_va, x5_med, include_x4=True, include_x9=False)
    locked_vec = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
    resid = df_tr["target"].values - X_tr @ locked_vec
    return X_va @ locked_vec + resid.mean()


def fit_ebm_heavy(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10, max_rounds=4000, min_samples_leaf=10,
        max_bins=128, smoothing_rounds=4000,
        interaction_smoothing_rounds=1000, random_state=SEED,
    ).fit(X, y)


def cv_all(df):
    y = df["target"].values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(df)) for k in
           ["a1", "lin_x4_nox9", "ebm_x9", "ebm_full"]}
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # A1 has no parameters, just applies the formula.
        oof["a1"][va] = approach1_predict(sub_va, x5m)

        oof["lin_x4_nox9"][va] = lin_x4_locked_nox9_predict(sub_tr, sub_va)

        feats9 = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(sub_tr, feats9, x5m); X_va = preprocess(sub_va, feats9, x5m)
        oof["ebm_x9"][va] = fit_ebm_heavy(X_tr, sub_tr["target"].values).predict(X_va)

        feats_f = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats_f, x5m); X_va = preprocess(sub_va, feats_f, x5m)
        oof["ebm_full"][va] = fit_ebm_heavy(X_tr, sub_tr["target"].values).predict(X_va)

        print(f"  fold {fold+1}/{N_SPLITS} in {time.time()-t0:.0f}s")
    return oof


def mae(p, y):
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - y)))


def report(name, pred, y, is_sent, safe):
    print(f"  {name:<44s}  overall={mae(pred, y):.3f}  "
          f"non-sent={mae(pred[~is_sent], y[~is_sent]):.3f}")
    safe_ns = safe & ~is_sent
    unsafe_ns = ~safe & ~is_sent
    print(f"    {'safe non-sent':<38s}  n={safe_ns.sum():<4d}  "
          f"mae={mae(pred[safe_ns], y[safe_ns]):.3f}")
    print(f"    {'unsafe non-sent':<38s}  n={unsafe_ns.sum():<4d}  "
          f"mae={mae(pred[unsafe_ns], y[unsafe_ns]):.3f}")
    print(f"    {'sentinel':<38s}  n={is_sent.sum():<4d}  "
          f"mae={mae(pred[is_sent], y[is_sent]):.3f}")


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()
    safe = safe_mask(df)

    print(f"Train: safe={safe.sum()}  unsafe non-sent={(~safe & ~is_sent).sum()}  "
          f"sentinel={is_sent.sum()}")
    test_safe = safe_mask(test)
    test_sent = (test["x5"] == SENTINEL).to_numpy()
    print(f"Test:  safe={test_safe.sum()}  unsafe non-sent={(~test_safe & ~test_sent).sum()}  "
          f"sentinel={test_sent.sum()}")

    print("\n" + "=" * 80)
    print("5-fold CV on base models (A1 has no parameters)")
    print("=" * 80)
    oof = cv_all(df)

    triple = 0.25 * oof["lin_x4_nox9"] + 0.25 * oof["ebm_x9"] + 0.50 * oof["ebm_full"]

    print("\n" + "=" * 80)
    print("Individual models")
    print("=" * 80)
    report("A1 (LB 10.80, near-perfect fit)", oof["a1"], y, is_sent, safe)
    print()
    report("Triple (CV 2.82)", triple, y, is_sent, safe)

    # Routing variants
    routed_default      = np.where(safe, oof["a1"], triple)
    routed_sent_to_a1   = np.where(safe | (is_sent & (np.abs(df["x4"].values) > X4_GAP_THRESH)), oof["a1"], triple)

    print("\n" + "=" * 80)
    print("Routing: safe rows → A1, rest → triple")
    print("=" * 80)
    report("ROUTED default", routed_default, y, is_sent, safe)
    print()
    report("ROUTED + send safe sentinels to A1", routed_sent_to_a1, y, is_sent, safe)

    # ---------- Blend: partial routing ----------
    print("\n" + "=" * 80)
    print("Partial routing: safe → α·A1 + (1-α)·triple")
    print("=" * 80)
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        p = np.where(safe, alpha * oof["a1"] + (1 - alpha) * triple, triple)
        m = mae(p, y); mn = mae(p[~is_sent], y[~is_sent])
        print(f"  α={alpha:.1f}  overall={m:.3f}  non-sent={mn:.3f}  "
              f"safe-ns mae={mae(p[safe & ~is_sent], y[safe & ~is_sent]):.3f}")

    # ---------- Build submission on full data ----------
    print("\n" + "=" * 80)
    print("Building ROUTED submission on full dataset")
    print("=" * 80)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    a1_pred = approach1_predict(test, x5m_full)

    lin_x4_pred = lin_x4_locked_nox9_predict(df, test)
    feats9 = ebm_features(with_x4=False, with_x9=True)
    X_tr = preprocess(df, feats9, x5m_full); X_te = preprocess(test, feats9, x5m_full)
    ebm_x9_pred = fit_ebm_heavy(X_tr, df["target"].values).predict(X_te)
    feats_f = ebm_features(with_x4=True, with_x9=True)
    X_tr = preprocess(df, feats_f, x5m_full); X_te = preprocess(test, feats_f, x5m_full)
    ebm_full_pred = fit_ebm_heavy(X_tr, df["target"].values).predict(X_te)
    triple_pred = 0.25 * lin_x4_pred + 0.25 * ebm_x9_pred + 0.50 * ebm_full_pred

    routed_pred = np.where(test_safe, a1_pred, triple_pred)
    pd.DataFrame({"id": test["id"], "target": routed_pred}).to_csv(
        SUBS / "submission_router_A1_triple.csv", index=False)
    print(f"  wrote submission_router_A1_triple.csv  "
          f"mean={routed_pred.mean():+.3f}  range=[{routed_pred.min():+.2f}, "
          f"{routed_pred.max():+.2f}]  safe_n={test_safe.sum()}/{len(test)}")


if __name__ == "__main__":
    main()
