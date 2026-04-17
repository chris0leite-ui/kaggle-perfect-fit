"""Routing ensemble: use a training-memorising model on 'safe' rows and
the triple ensemble on the rest.

Routing rule (defaults):
  SAFE iff  x5 != 999  AND  |x4| > 0.167  AND  sign(x4) matches cluster
       i.e. (x4 > 0.167 AND x9 > 5)  OR  (x4 < -0.167 AND x9 < 5)

  Sentinel rows (x5 == 999) ALWAYS route to triple -- both models share
  the same irreducible sentinel noise floor and triple is more robust.

Memoriser model:
  locked_b_full = A2 skeleton with ALL A1/A2 integer coefficients locked,
  INCLUDING β_x9_wc = -4 (the 'x9_wc' feature, which hit CV 2.90 but
  LB 10.75 -- the failure was purely on off-diagonal test rows, exactly
  the rows the router now diverts away from it).
  Features: x1^2, cos(5π·x2), x4, x5_imp, x5_is_sent, x8, x10, x11,
            x10*x11, city, x9_wc. Integer coefs; only intercept is fit.

Triple ensemble (unsafe-row model):
  0.25 * LIN_x4_locked_b   (no x9)
  + 0.25 * EBM_x9           (no x4)
  + 0.50 * EBM_full         (all features, heavy-smoothed)
  This is our CV 2.824 ensemble.

5-fold CV on dataset.csv. Reports:
  - routed ensemble CV MAE (overall / non-sent / sent)
  - per-region breakdown: safe rows, gap rows, off-diagonal rows
  - comparison to pure locked_b_full and pure triple
  - also tries sentinel routing ON/OFF and a no-sentinel-exception variant
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
from cv_gam_enhanced import make_X, cluster_means_x9  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5
X4_GAP_THRESH = 0.167

LIN_COL_ORDER = ["x1^2", "cos(5pi*x2)", "x4", "x5_imp", "x5_is_sent",
                 "x8", "x10", "x11", "x10*x11", "city"]
LOCKED_B_NOX9 = {"x1^2": -100, "cos(5pi*x2)": +10, "x4": +30, "x5_imp": -8,
                 "x5_is_sent": 0, "x8": +14, "x10": 0, "x11": 0,
                 "x10*x11": +1, "city": -25}
# Full memoriser: adds x9_wc column at index 10 (after city)
FULL_COL_ORDER = LIN_COL_ORDER + ["x9_wc"]
LOCKED_B_FULL = {**LOCKED_B_NOX9, "x9_wc": -4}


def safe_mask(df: pd.DataFrame) -> np.ndarray:
    x4 = df["x4"].to_numpy()
    x9 = df["x9"].to_numpy()
    is_sent = (df["x5"].to_numpy() == SENTINEL)
    x4_clear = np.abs(x4) > X4_GAP_THRESH
    cluster_match = ((x4 > 0) & (x9 > 5)) | ((x4 < 0) & (x9 < 5))
    return (~is_sent) & x4_clear & cluster_match


def locked_predict_full(df_tr, df_test, locks: dict[str, float]):
    """Full locked model INCLUDING x9_wc (uses training-fold cluster means)."""
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    mean_hi, mean_lo = cluster_means_x9(df_tr)
    X_tr, names, _, _ = make_X(df_tr, x5_med, mean_hi, mean_lo,
                               "square", "cos", True, "wc")
    X_te, _, _, _ = make_X(df_test, x5_med, mean_hi, mean_lo,
                           "square", "cos", True, "wc")
    assert names == FULL_COL_ORDER, f"column mismatch: {names}"
    locked_vec = np.array([locks[c] for c in FULL_COL_ORDER])
    resid = df_tr["target"].values - X_tr @ locked_vec
    return X_te @ locked_vec + resid.mean()


def locked_predict_nox9(df_tr, df_test, locks: dict[str, float]):
    """No-x9 locked model (the 'LIN_x4' component of the triple)."""
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5_med, include_x4=True, include_x9=False)
    X_te = design_matrix(df_test, x5_med, include_x4=True, include_x9=False)
    locked_vec = np.array([locks[c] for c in LIN_COL_ORDER])
    resid = df_tr["target"].values - X_tr @ locked_vec
    return X_te @ locked_vec + resid.mean()


def fit_ebm_heavy(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10, max_rounds=4000, min_samples_leaf=10,
        max_bins=128, smoothing_rounds=4000,
        interaction_smoothing_rounds=1000, random_state=SEED,
    ).fit(X, y)


def cv_all(df: pd.DataFrame):
    y = df["target"].values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(df)) for k in
           ["locked_b_full", "lin_x4_nox9", "ebm_x9", "ebm_full"]}
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        oof["locked_b_full"][va] = locked_predict_full(sub_tr, sub_va, LOCKED_B_FULL)
        oof["lin_x4_nox9"][va]   = locked_predict_nox9(sub_tr, sub_va, LOCKED_B_NOX9)

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
    print(f"  {name:<38s}  overall={mae(pred, y):.3f}  "
          f"non-sent={mae(pred[~is_sent], y[~is_sent]):.3f}")
    print(f"    {'  on safe rows':<36s}  "
          f"n={safe.sum():<4d}  mae={mae(pred[safe], y[safe]):.3f}")
    print(f"    {'  on unsafe non-sent rows':<36s}  "
          f"n={(~safe & ~is_sent).sum():<4d}  "
          f"mae={mae(pred[~safe & ~is_sent], y[~safe & ~is_sent]):.3f}")
    print(f"    {'  on sentinel rows':<36s}  "
          f"n={is_sent.sum():<4d}  mae={mae(pred[is_sent], y[is_sent]):.3f}")


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()
    safe = safe_mask(df)

    print(f"Train row counts:")
    print(f"  safe (use memoriser)           n={safe.sum()}")
    print(f"  unsafe non-sent (use triple)   n={(~safe & ~is_sent).sum()}")
    print(f"  sentinel (use triple)          n={is_sent.sum()}")
    print(f"Test row counts (preview):")
    test_safe = safe_mask(test)
    test_sent = (test["x5"] == SENTINEL).to_numpy()
    print(f"  safe                           n={test_safe.sum()}")
    print(f"  unsafe non-sent                n={(~test_safe & ~test_sent).sum()}")
    print(f"  sentinel                       n={test_sent.sum()}")

    print("\n" + "=" * 80)
    print("5-fold CV on 4 base models")
    print("=" * 80)
    oof = cv_all(df)

    triple = 0.25 * oof["lin_x4_nox9"] + 0.25 * oof["ebm_x9"] + 0.50 * oof["ebm_full"]
    routed_default = np.where(safe, oof["locked_b_full"], triple)
    # Alt: route safe rows AND safe sentinel rows (i.e. ignore sentinel rule)
    safe_including_sent = (np.abs(df["x4"]) > X4_GAP_THRESH).to_numpy() & (
        ((df["x4"] > 0) & (df["x9"] > 5)) | ((df["x4"] < 0) & (df["x9"] < 5))
    ).to_numpy()
    routed_no_sent_rule = np.where(safe_including_sent, oof["locked_b_full"], triple)

    print("\n" + "=" * 80)
    print("Individual models — check memoriser does well on safe rows")
    print("=" * 80)
    report("locked_b_full (LB 10.75)", oof["locked_b_full"], y, is_sent, safe)
    print()
    report("triple locked_b λ=0.5 (CV 2.82)", triple, y, is_sent, safe)

    print("\n" + "=" * 80)
    print("Routed ensembles")
    print("=" * 80)
    report("ROUTED default (safe→locked, rest→triple, sentinels→triple)",
           routed_default, y, is_sent, safe)
    print()
    report("ROUTED ignore-sentinel-rule (safe→locked even on sent)",
           routed_no_sent_rule, y, is_sent, safe)

    # ---------- Build submission on full data ----------
    print("\n" + "=" * 80)
    print("Building ROUTED ensemble submission on full dataset")
    print("=" * 80)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # Memoriser on full data
    locked_full_pred = locked_predict_full(df, test, LOCKED_B_FULL)

    # Triple components on full data
    lin_x4_pred = locked_predict_nox9(df, test, LOCKED_B_NOX9)
    feats9 = ebm_features(with_x4=False, with_x9=True)
    X_tr = preprocess(df, feats9, x5m_full); X_te = preprocess(test, feats9, x5m_full)
    ebm_x9_pred = fit_ebm_heavy(X_tr, df["target"].values).predict(X_te)
    feats_f = ebm_features(with_x4=True, with_x9=True)
    X_tr = preprocess(df, feats_f, x5m_full); X_te = preprocess(test, feats_f, x5m_full)
    ebm_full_pred = fit_ebm_heavy(X_tr, df["target"].values).predict(X_te)
    triple_pred = 0.25 * lin_x4_pred + 0.25 * ebm_x9_pred + 0.50 * ebm_full_pred

    test_safe = safe_mask(test)
    routed_pred = np.where(test_safe, locked_full_pred, triple_pred)
    pd.DataFrame({"id": test["id"], "target": routed_pred}).to_csv(
        SUBS / "submission_router_locked_b_triple.csv", index=False)
    print(f"  wrote submission_router_locked_b_triple.csv  "
          f"mean={routed_pred.mean():+.3f}  range=[{routed_pred.min():+.2f}, "
          f"{routed_pred.max():+.2f}]  safe_n={test_safe.sum()}")


if __name__ == "__main__":
    main()
