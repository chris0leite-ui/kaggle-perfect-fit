"""Router enhanced with a LightGBM clamp classifier.

Prior search found LightGBM AUC ~0.76 for predicting the A1 clamp trigger
within the x4<0 & x8<0 quadrant -- meaningful predictive signal in
feature interactions even though no single feature works.

Architecture:
  1. Safe rows (3 clean quadrants): route to A1 (always correct)
  2. Clamp-risk rows (x4<0, x8<0): use the classifier probability:
       prediction = (1-p) * A1  +  p * triple
     where p is the OOF clamp probability. Hard-route if p > threshold.
  3. Unsafe non-sent rows: route to triple (as before)
  4. Sentinels: route to triple

Test both soft and hard routing variants.
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
X4_GAP = 0.167


def safe_mask(df):
    x4 = df["x4"].to_numpy(); x9 = df["x9"].to_numpy()
    is_sent = (df["x5"].to_numpy() == SENTINEL)
    x4_clear = np.abs(x4) > X4_GAP
    cluster_match = ((x4 > 0) & (x9 > 5)) | ((x4 < 0) & (x9 < 5))
    return (~is_sent) & x4_clear & cluster_match


def clamp_candidate(df):
    """Rows in the x4<0 & x8<0 quadrant — where A1 may or may not be perfect."""
    return (df["x5"].to_numpy() != SENTINEL) & (df["x4"].to_numpy() < 0) & \
           (df["x8"].to_numpy() < 0)


def fit_clamp_classifier(df_tr, y_bad_tr):
    """Train LightGBM classifier on all features + x6/x7 angle."""
    import lightgbm as lgb
    theta = np.arctan2(df_tr["x7"].values, df_tr["x6"].values)
    X = np.column_stack([
        df_tr[["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]].values,
        np.sin(theta), np.cos(theta), np.sin(2*theta), np.cos(2*theta),
    ])
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=5, random_state=SEED, verbose=-1,
    )
    clf.fit(X, y_bad_tr)
    return clf


def predict_clamp_prob(clf, df):
    theta = np.arctan2(df["x7"].values, df["x6"].values)
    X = np.column_stack([
        df[["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]].values,
        np.sin(theta), np.cos(theta), np.sin(2*theta), np.cos(2*theta),
    ])
    return clf.predict_proba(X)[:, 1]


def fit_ebm_heavy(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10, max_rounds=4000, min_samples_leaf=10,
        max_bins=128, smoothing_rounds=4000,
        interaction_smoothing_rounds=1000, random_state=SEED,
    ).fit(X, y)


def cv_all(df):
    """Produce OOF for A1, triple (as usual), and OOF clamp probability."""
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(df)) for k in
           ["a1", "lin_x4", "ebm_x9", "ebm_full", "clamp_prob"]}
    oof["clamp_prob"][:] = np.nan  # only filled for x4<0, x8<0 rows

    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # A1 (deterministic)
        oof["a1"][va] = approach1_predict(sub_va, x5m)

        # LIN_x4 locked_b
        X_tr_lin = design_matrix(sub_tr, x5m, True, False)
        X_va_lin = design_matrix(sub_va, x5m, True, False)
        locked = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
        resid = sub_tr["target"].values - X_tr_lin @ locked
        oof["lin_x4"][va] = X_va_lin @ locked + resid.mean()

        # EBM_x9
        feats = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(sub_tr, feats, x5m); X_va = preprocess(sub_va, feats, x5m)
        oof["ebm_x9"][va] = fit_ebm_heavy(X_tr, sub_tr["target"].values).predict(X_va)

        # EBM_full
        feats = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats, x5m); X_va = preprocess(sub_va, feats, x5m)
        oof["ebm_full"][va] = fit_ebm_heavy(X_tr, sub_tr["target"].values).predict(X_va)

        # Clamp classifier on training-fold's clamp-candidate rows
        q_tr = clamp_candidate(sub_tr)
        a1_tr = approach1_predict(sub_tr, x5m)
        abs_resid_tr = np.abs(sub_tr["target"].values - a1_tr)
        is_bad_tr = (abs_resid_tr > 0.1).astype(int)
        clf = fit_clamp_classifier(sub_tr[q_tr], is_bad_tr[q_tr])

        # Apply to validation rows that are clamp candidates
        q_va = clamp_candidate(sub_va)
        if q_va.sum() > 0:
            probs = predict_clamp_prob(clf, sub_va[q_va])
            va_idx = np.where(q_va)[0]
            for local_i, global_i in enumerate(va_idx):
                oof["clamp_prob"][va[global_i]] = probs[local_i]

        print(f"  fold {fold+1}/{N_SPLITS} in {time.time()-t0:.0f}s  "
              f"clamp-val rows: {q_va.sum()}")
    return oof


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).to_numpy()
    safe = safe_mask(df)
    q = clamp_candidate(df)

    print("5-fold CV on base + clamp classifier")
    print("=" * 78)
    oof = cv_all(df)

    triple = 0.25 * oof["lin_x4"] + 0.25 * oof["ebm_x9"] + 0.50 * oof["ebm_full"]

    # Evaluate classifier
    a1_full = approach1_predict(df, float(df.loc[df["x5"] != SENTINEL, "x5"].median()))
    abs_resid_full = np.abs(df["target"].values - a1_full)
    is_bad = (abs_resid_full > 0.1).astype(int)
    from sklearn.metrics import roc_auc_score
    cp = oof["clamp_prob"]
    mask = ~np.isnan(cp)
    auc = roc_auc_score(is_bad[mask], cp[mask])
    print(f"\nOOF clamp classifier AUC on quadrant: {auc:.3f}  (n={mask.sum()})")

    # ---- Routing variants ----
    print("\n" + "=" * 78)
    print("Routing variants")
    print("=" * 78)

    # Base: router_A1_triple (safe → A1, rest → triple)
    base_router = np.where(safe, oof["a1"], triple)
    print(f"  base router (safe→A1, rest→triple):     "
          f"overall={mae(base_router, y):.3f}  non-sent={mae(base_router[~is_sent], y[~is_sent]):.3f}")

    # Soft clamp routing: within x4<0,x8<0 and non-sent, blend A1 and triple by clamp prob
    soft_routed = oof["a1"].copy()  # start with A1 for all safe rows
    # Unsafe non-sent → triple (same as base)
    soft_routed = np.where(safe | ((df["x4"] < 0) & (df["x8"] < 0) & ~is_sent), soft_routed, triple)
    # Now replace clamp-candidate rows with blend
    q_and_nonnan = q & ~np.isnan(cp)
    p = oof["clamp_prob"][q_and_nonnan]
    soft_routed[q_and_nonnan] = (1 - p) * oof["a1"][q_and_nonnan] + p * triple[q_and_nonnan]
    # Sentinel rows → triple
    soft_routed = np.where(is_sent, triple, soft_routed)
    print(f"  soft clamp routing (p-weighted blend):  "
          f"overall={mae(soft_routed, y):.3f}  non-sent={mae(soft_routed[~is_sent], y[~is_sent]):.3f}")

    # Hard clamp routing at different thresholds
    for thresh in [0.30, 0.40, 0.50, 0.60]:
        r = oof["a1"].copy()
        r = np.where(safe | ((df["x4"] < 0) & (df["x8"] < 0) & ~is_sent), r, triple)
        # Within q rows: route to triple if prob > thresh, else A1 (already)
        high_clamp = q_and_nonnan & (cp > thresh)
        r[high_clamp] = triple[high_clamp]
        r = np.where(is_sent, triple, r)
        print(f"  hard clamp routing (p>{thresh}):          "
              f"overall={mae(r, y):.3f}  non-sent={mae(r[~is_sent], y[~is_sent]):.3f}  "
              f"({high_clamp.sum()} routed to triple)")

    # ---- Build submission for the best variant ----
    print("\n" + "=" * 78)
    print("Building submission for best variant")
    print("=" * 78)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # A1 on test
    a1_test = approach1_predict(test, x5m_full)

    # Triple on test (refit on full)
    X_tr = design_matrix(df, x5m_full, True, False)
    X_te = design_matrix(test, x5m_full, True, False)
    locked = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
    resid = df["target"].values - X_tr @ locked
    lin_x4_test = X_te @ locked + resid.mean()

    feats = ebm_features(with_x4=False, with_x9=True)
    ebm_x9 = fit_ebm_heavy(preprocess(df, feats, x5m_full), df["target"].values)
    ebm_x9_test = ebm_x9.predict(preprocess(test, feats, x5m_full))

    feats = ebm_features(with_x4=True, with_x9=True)
    ebm_full = fit_ebm_heavy(preprocess(df, feats, x5m_full), df["target"].values)
    ebm_full_test = ebm_full.predict(preprocess(test, feats, x5m_full))

    triple_test = 0.25 * lin_x4_test + 0.25 * ebm_x9_test + 0.50 * ebm_full_test

    # Clamp classifier on full training data, applied to test clamp-candidates
    q_tr_full = clamp_candidate(df)
    clf = fit_clamp_classifier(df[q_tr_full], is_bad[q_tr_full])
    test_q = clamp_candidate(test)
    test_safe = safe_mask(test)
    test_sent = (test["x5"] == SENTINEL).to_numpy()

    # SOFT clamp routing
    soft_pred = a1_test.copy()
    soft_pred = np.where(test_safe | ((test["x4"] < 0) & (test["x8"] < 0) & ~test_sent),
                          soft_pred, triple_test)
    if test_q.sum() > 0:
        p_test = predict_clamp_prob(clf, test[test_q])
        idx = np.where(test_q)[0]
        soft_pred[idx] = (1 - p_test) * a1_test[idx] + p_test * triple_test[idx]
    soft_pred = np.where(test_sent, triple_test, soft_pred)

    pd.DataFrame({"id": test["id"], "target": soft_pred}).to_csv(
        SUBS / "submission_router_A1_clamp_soft.csv", index=False)
    print(f"  wrote submission_router_A1_clamp_soft.csv  "
          f"mean={soft_pred.mean():+.3f}  range=[{soft_pred.min():+.2f}, {soft_pred.max():+.2f}]")
    print(f"  clamp-candidate rows in test: {test_q.sum()}")


if __name__ == "__main__":
    main()
