"""Synthetic off-diagonal training data via A1 as DGP oracle.

Training has r(x4, x9) = 0.83 due to selection bias, but test has
r = 0.001 (true DGP). Models trained on the biased joint inherit the
shift. Idea: generate synthetic training rows with randomly-permuted
x9 and targets recomputed via A1 (which is exact DGP on 93% of rows).

The synthetic set has:
  - x4 distribution matching training (unchanged)
  - x9 distribution matching training marginal (independent of x4)
  - target = old_target - 4*(new_x9 - old_x9)  (A1's x9 contribution)

Training becomes original + synthetic → 2.7× size, but now the joint
(x4, x9) in training matches the test joint.

Only use synthetic rows where A1 is exactly perfect on the original
(|resid| < 0.01, the 1192 non-sent rows outside the clamp quadrant
problem zone).

CV strategy: within each fold, generate synthetic ONLY from the training
portion to avoid leakage. Validate on original held-out rows.

Compare:
  - Triple on original training (CV 2.824, LB ~2.7-2.85 projected)
  - Triple on augmented training
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
from cv_cross_LE_tune import LOCKED_COEFS_B, LIN_COL_ORDER  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5


def generate_synthetic(df_tr: pd.DataFrame, seed: int, multiplier: int = 1) -> pd.DataFrame:
    """Generate `multiplier` synthetic rows per A1-perfect non-sent training row.

    Each synthetic row has a new x9 sampled from the training-fold's x9
    marginal; target is adjusted by -4 * (new_x9 - old_x9) to stay
    consistent with A1's DGP.
    """
    x5_median = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    a1_orig = approach1_predict(df_tr, x5_median)
    is_sent = (df_tr["x5"] == SENTINEL).values
    is_perfect = np.abs(df_tr["target"].values - a1_orig) < 0.01
    eligible = ~is_sent & is_perfect

    x9_pool = df_tr.loc[~is_sent, "x9"].values
    rng = np.random.default_rng(seed)

    synthetics = []
    base = df_tr[eligible].copy().reset_index(drop=True)
    for k in range(multiplier):
        new_x9 = rng.choice(x9_pool, size=len(base), replace=True)
        s = base.copy()
        old_x9 = s["x9"].values
        s["x9"] = new_x9
        s["target"] = s["target"].values - 4.0 * (new_x9 - old_x9)
        # Mark with fresh ids so they don't collide
        s["id"] = -1  # synthetic marker
        synthetics.append(s)
    return pd.concat(synthetics, ignore_index=True)


def fit_ebm_heavy(X, y, sample_weight=None):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10, max_rounds=4000, min_samples_leaf=10,
        max_bins=128, smoothing_rounds=4000,
        interaction_smoothing_rounds=1000, random_state=SEED,
    ).fit(X, y, sample_weight=sample_weight)


def cv_augmented(df, multiplier: int):
    """Train on original + synthetic per fold, validate on original."""
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {"lin_x4": np.zeros(len(df)),
           "ebm_x9": np.zeros(len(df)),
           "ebm_full": np.zeros(len(df))}

    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        if multiplier > 0:
            synth = generate_synthetic(sub_tr, seed=SEED + fold,
                                        multiplier=multiplier)
            combined = pd.concat([sub_tr, synth], ignore_index=True)
        else:
            combined = sub_tr

        # LIN_x4 (locked_b) — intercept fit on combined data
        X_tr_lin = design_matrix(combined, x5m, True, False)
        X_va_lin = design_matrix(sub_va, x5m, True, False)
        locked = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
        resid = combined["target"].values - X_tr_lin @ locked
        oof["lin_x4"][va] = X_va_lin @ locked + resid.mean()

        # EBM_x9 on combined
        feats9 = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(combined, feats9, x5m); X_va = preprocess(sub_va, feats9, x5m)
        oof["ebm_x9"][va] = fit_ebm_heavy(X_tr, combined["target"].values).predict(X_va)

        # EBM_full on combined
        feats_f = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(combined, feats_f, x5m); X_va = preprocess(sub_va, feats_f, x5m)
        oof["ebm_full"][va] = fit_ebm_heavy(X_tr, combined["target"].values).predict(X_va)

        print(f"  fold {fold+1}/{N_SPLITS} in {time.time()-t0:.0f}s  "
              f"combined n={len(combined)}")

    triple = 0.25 * oof["lin_x4"] + 0.25 * oof["ebm_x9"] + 0.50 * oof["ebm_full"]
    m = float(np.mean(np.abs(triple - y)))
    mn = float(np.mean(np.abs(triple[~is_sent] - y[~is_sent])))
    return m, mn, oof, triple


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values

    # Sanity: verify the synthetic data actually decorrelates x4, x9
    print("Sanity check (one-time, full-data synthetic sample):")
    synth = generate_synthetic(df, seed=SEED, multiplier=1)
    print(f"  original n={len(df)}, synthetic n={len(synth)}")
    print(f"  orig corr(x4, x9):      {np.corrcoef(df.x4, df.x9)[0,1]:+.3f}")
    print(f"  synth corr(x4, x9):     {np.corrcoef(synth.x4, synth.x9)[0,1]:+.3f}")
    combined = pd.concat([df, synth], ignore_index=True)
    print(f"  combined corr(x4, x9):  {np.corrcoef(combined.x4, combined.x9)[0,1]:+.3f}")

    # Run CV for multipliers 0 (baseline) and 1, 2
    print("\n" + "=" * 78)
    print("5-fold CV — triple ensemble on original + synthetic")
    print("=" * 78)
    print(f"{'multiplier':<12s}  {'overall':>8s}  {'non-sent':>9s}   notes")
    for mult in [0, 1, 2]:
        m, mn, oof, triple = cv_augmented(df, multiplier=mult)
        label = f"mult={mult}"
        if mult == 0:
            label += " (baseline)"
        print(f"  {label:<12s}  {m:8.3f}  {mn:9.3f}")

    # Build submission for multiplier=1 (or 2 if better)
    print("\n" + "=" * 78)
    print("Building submission (mult=1)")
    print("=" * 78)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    synth_full = generate_synthetic(df, seed=SEED, multiplier=1)
    combined_full = pd.concat([df, synth_full], ignore_index=True)

    X_tr_lin = design_matrix(combined_full, x5m_full, True, False)
    X_te_lin = design_matrix(test, x5m_full, True, False)
    locked = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
    resid = combined_full["target"].values - X_tr_lin @ locked
    lin_x4_pred = X_te_lin @ locked + resid.mean()

    feats9 = ebm_features(with_x4=False, with_x9=True)
    X_tr = preprocess(combined_full, feats9, x5m_full)
    X_te = preprocess(test, feats9, x5m_full)
    ebm_x9_pred = fit_ebm_heavy(X_tr, combined_full["target"].values).predict(X_te)

    feats_f = ebm_features(with_x4=True, with_x9=True)
    X_tr = preprocess(combined_full, feats_f, x5m_full)
    X_te = preprocess(test, feats_f, x5m_full)
    ebm_full_pred = fit_ebm_heavy(X_tr, combined_full["target"].values).predict(X_te)

    triple_aug = 0.25 * lin_x4_pred + 0.25 * ebm_x9_pred + 0.50 * ebm_full_pred
    pd.DataFrame({"id": test["id"], "target": triple_aug}).to_csv(
        SUBS / "submission_triple_augmented.csv", index=False)
    print(f"  wrote submission_triple_augmented.csv  "
          f"mean={triple_aug.mean():+.3f}  "
          f"range=[{triple_aug.min():+.2f}, {triple_aug.max():+.2f}]")

    # Also build a router variant using the augmented triple for unsafe rows
    # Safe rows still use A1
    a1_test = approach1_predict(test, x5m_full)
    from cv_router_A1 import safe_mask
    test_safe = safe_mask(test)
    routed_aug = np.where(test_safe, a1_test, triple_aug)
    pd.DataFrame({"id": test["id"], "target": routed_aug}).to_csv(
        SUBS / "submission_router_A1_triple_augmented.csv", index=False)
    print(f"  wrote submission_router_A1_triple_augmented.csv  "
          f"mean={routed_aug.mean():+.3f}")


if __name__ == "__main__":
    main()
