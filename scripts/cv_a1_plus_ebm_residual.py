"""A1 + EBM residual corrector.

Instead of using EBM to learn the whole DGP (which trains on the selection-
biased joint and inherits the x4-x9 shift), use EBM to model only the
RESIDUALS after A1. A1 is exactly the DGP on 93% of non-sent rows, so
the EBM only has to learn:
  - the clamp deviation (~-18.4*x8 on some x4<0 & x8<0 rows)
  - the sentinel correction (-8*x5_imputed bias)
  - any small drift

This is an easier task and — because A1's features drop out where A1 is
right — the EBM doesn't see the spurious x4-x9 joint structure there.

Variants:
  1. EBM on residuals, all features
  2. EBM on residuals, with A1_prediction as an extra feature (helps
     the model condition on the A1 baseline)
  3. Same but trained only on non-sentinel training rows (sentinels
     dominate the residual signal otherwise)

Final prediction = A1(x) + EBM_residual(x).

Compare to:
  - Triple (CV 2.824)
  - Router (CV 1.839)
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
from cv_x4_x9_swap_ensemble import ebm_features, fit_ebm  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5


def fit_ebm_heavy(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10, max_rounds=4000, min_samples_leaf=10,
        max_bins=128, smoothing_rounds=4000,
        interaction_smoothing_rounds=1000, random_state=SEED,
    ).fit(X, y)


def make_features(df, features, x5_median, include_a1_pred=False):
    X = preprocess(df, features, x5_median)
    if include_a1_pred:
        X = X.copy()
        X["a1_pred"] = approach1_predict(df, x5_median)
    return X


def cv_variant(df, variant):
    """variant in {'full', 'with_a1', 'nonsent_only', 'with_a1_nonsent'}."""
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))

    include_a1 = variant in ("with_a1", "with_a1_nonsent")
    nonsent_only = variant in ("nonsent_only", "with_a1_nonsent")

    feats = ebm_features(with_x4=True, with_x9=True)

    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # A1 predictions (same on train and test since no parameters)
        a1_tr = approach1_predict(sub_tr, x5m)
        a1_va = approach1_predict(sub_va, x5m)
        residual_tr = sub_tr["target"].values - a1_tr

        X_tr = make_features(sub_tr, feats, x5m, include_a1_pred=include_a1)
        X_va = make_features(sub_va, feats, x5m, include_a1_pred=include_a1)

        if nonsent_only:
            mask = (sub_tr["x5"] != SENTINEL).values
            X_tr_fit = X_tr[mask].reset_index(drop=True)
            residual_fit = residual_tr[mask]
        else:
            X_tr_fit = X_tr
            residual_fit = residual_tr

        ebm = fit_ebm_heavy(X_tr_fit, residual_fit)
        residual_va_pred = ebm.predict(X_va)
        oof[va] = a1_va + residual_va_pred

        print(f"  fold {fold+1}/{N_SPLITS} in {time.time()-t0:.0f}s")
    m = float(np.mean(np.abs(oof - y)))
    mn = float(np.mean(np.abs(oof[~is_sent] - y[~is_sent])))
    ms = float(np.mean(np.abs(oof[is_sent] - y[is_sent])))
    return m, mn, ms, oof


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()

    # First get A1 predictions (baseline reference)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    a1_full = approach1_predict(df, x5m_full)
    print(f"A1 alone:          overall={mae(a1_full, y):.3f}  "
          f"non-sent={mae(a1_full[~is_sent], y[~is_sent]):.3f}  "
          f"sent={mae(a1_full[is_sent], y[is_sent]):.3f}")

    print("\n" + "=" * 78)
    print("A1 + EBM residual corrector")
    print("=" * 78)
    print(f"{'variant':<32s}  {'overall':>8s}  {'non-sent':>8s}  {'sent':>7s}")
    print("-" * 78)
    results = {}
    for variant in ["full", "with_a1", "nonsent_only", "with_a1_nonsent"]:
        t0 = time.time()
        m, mn, ms, oof = cv_variant(df, variant)
        results[variant] = (m, mn, ms, oof)
        print(f"{variant:<32s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  [{time.time()-t0:.0f}s]")

    print("\n" + "=" * 78)
    print("Comparison")
    print("=" * 78)
    print(f"  Triple (current CV-best):             CV 2.824")
    print(f"  Router_A1_triple (safe->A1):          CV 1.839")
    print(f"  A1 alone:                              CV {mae(a1_full, y):.3f}")
    best = min(results.items(), key=lambda kv: kv[1][0])
    print(f"  Best A1+EBM_residual ({best[0]}):       CV {best[1][0]:.3f}")

    # Build submission for the best variant if competitive
    if best[1][0] < 2.5:
        print("\n" + "=" * 78)
        print(f"Building submission for best variant: {best[0]}")
        print("=" * 78)
        variant = best[0]
        include_a1 = variant in ("with_a1", "with_a1_nonsent")
        nonsent_only = variant in ("nonsent_only", "with_a1_nonsent")
        feats = ebm_features(with_x4=True, with_x9=True)

        a1_train_full = approach1_predict(df, x5m_full)
        a1_test_full = approach1_predict(test, x5m_full)
        residual_full = df["target"].values - a1_train_full
        X_tr = make_features(df, feats, x5m_full, include_a1_pred=include_a1)
        X_te = make_features(test, feats, x5m_full, include_a1_pred=include_a1)
        if nonsent_only:
            mask = (df["x5"] != SENTINEL).values
            X_tr_fit = X_tr[mask].reset_index(drop=True)
            resid_fit = residual_full[mask]
        else:
            X_tr_fit = X_tr
            resid_fit = residual_full
        ebm = fit_ebm_heavy(X_tr_fit, resid_fit)
        preds = a1_test_full + ebm.predict(X_te)
        pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
            SUBS / f"submission_a1_plus_ebm_resid_{variant}.csv", index=False)
        print(f"  wrote submission_a1_plus_ebm_resid_{variant}.csv  "
              f"mean={preds.mean():+.3f}  range=[{preds.min():+.2f}, {preds.max():+.2f}]")


if __name__ == "__main__":
    main()
