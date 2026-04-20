"""Soft-clamp correction layered onto cross_LE.

Clamp archaeology (v1/v2) established:
- Correction on 86/368 quadrant rows: residual ≈ -15·x8 + 1  (R²=0.994)
- No observable trigger reaches sens/spec > 0.9; AUC ceiling ≈ 0.76

Strategy: instead of a hard trigger, learn a calibrated
probability p(clamp | features) restricted to the x4<0, x8<0
quadrant, then correct LIN_x4's x8 contribution by:

    correction = p · (-15·x8 + 1)

Expected contribution from x8 in the DGP:
    E[x8-contrib] = (1 - p) · (15·x8) + p · (1)

A1-style baseline used 15·x8. Subtracting p·(15·x8 - 1) aligns
with the mixture expectation. Only applies inside the quadrant;
elsewhere leave cross_LE untouched.

Pipeline
--------
1. 5-fold OOF on dataset.csv:
   - LIN_x4 (pure linear, no x9)  — cross_LE's linear leg
   - EBM_x9 (no x4)               — cross_LE's nonparametric leg
   - p_clamp  (LightGBM classifier on quadrant rows only, OOF)
2. cross_LE_base   = 0.5 · (LIN_x4 + EBM_x9)
3. cross_LE_clamp  = base + correction (only for quadrant rows)
4. Report CV MAE + non-sent MAE for both, compare.
5. Build full-data submission applying the correction.
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
from cv_ebm_variants import SENTINEL, SEED  # noqa: E402
from cv_x4_x9_swap_ensemble import fit_ebm, design_matrix  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5


def clamp_features(df: pd.DataFrame) -> pd.DataFrame:
    theta = np.arctan2(df["x7"], df["x6"])
    feats = pd.DataFrame({
        "x1": df["x1"], "x2": df["x2"], "x4": df["x4"], "x5": df["x5"].replace(SENTINEL, np.nan),
        "x6": df["x6"], "x7": df["x7"], "x8": df["x8"], "x9": df["x9"],
        "x10": df["x10"], "x11": df["x11"], "theta": theta,
        "sin_theta": np.sin(theta), "cos_theta": np.cos(theta),
        "sin_2theta": np.sin(2 * theta), "cos_2theta": np.cos(2 * theta),
        "x4_plus_x8": df["x4"] + df["x8"],
        "x4_minus_x8": df["x4"] - df["x8"],
        "x4_x8": df["x4"] * df["x8"],
        "abs_x8_minus_abs_x4": df["x8"].abs() - df["x4"].abs(),
        "x8_centered": np.abs(df["x8"] + 0.3),
        "x8_x9": df["x8"] * df["x9"],
        "x8_x10x11": df["x8"] * df["x10"] * df["x11"],
        "x4_x10x11": df["x4"] * df["x10"] * df["x11"],
        "x10x11": df["x10"] * df["x11"],
    })
    feats["x5"] = feats["x5"].fillna(feats["x5"].median())
    feats["city"] = (df["City"] == "Zaragoza").astype(float).values
    return feats


def quadrant_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["x4"] < 0) & (df["x8"] < 0) & (df["x5"] != SENTINEL)).values


def a1_residual(df: pd.DataFrame, x5_med: float) -> np.ndarray:
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_med).values
    is_zar = (df["City"] == "Zaragoza").astype(float).values
    a1 = (
        -100 * df["x1"] ** 2
        + 10 * np.cos(5 * np.pi * df["x2"])
        + 15 * df["x4"]
        - 8 * x5
        + 15 * df["x8"]
        - 4 * df["x9"]
        + df["x10"] * df["x11"]
        - 25 * is_zar
        + 20 * (df["x4"] > 0)
        + 92.5
    )
    return df["target"].values - a1.values


def fit_clamp_classifier(df_train: pd.DataFrame) -> "lightgbm.LGBMClassifier":
    import lightgbm as lgb
    x5_med = df_train.loc[df_train["x5"] != SENTINEL, "x5"].median()
    resid = a1_residual(df_train, x5_med)
    q = quadrant_mask(df_train)
    sub = df_train[q].copy()
    y = (np.abs(resid[q]) > 1.0).astype(int)
    if y.sum() < 10 or y.sum() > len(y) - 10:
        raise RuntimeError("Degenerate clamp class counts")
    X = clamp_features(sub)
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        num_leaves=16, min_data_in_leaf=8, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )
    clf.fit(X, y)
    return clf


def predict_clamp_prob(clf, df: pd.DataFrame) -> np.ndarray:
    """Return p(clamp) for every row. Zero outside quadrant."""
    q = quadrant_mask(df)
    p = np.zeros(len(df))
    if q.sum() > 0:
        X = clamp_features(df[q])
        p_q = clf.predict_proba(X)[:, 1]
        p[q] = p_q
    return p


def lin_x4_free(df_tr, df_va, x5m):
    X_tr = design_matrix(df_tr, x5m, include_x4=True, include_x9=False)
    X_va = design_matrix(df_va, x5m, include_x4=True, include_x9=False)
    return LinearRegression().fit(X_tr, df_tr["target"].values).predict(X_va)


def ebm_x9(df_tr, df_va, x5m):
    X_tr = design_matrix(df_tr, x5m, include_x4=False, include_x9=True)
    X_va = design_matrix(df_va, x5m, include_x4=False, include_x9=True)
    return fit_ebm(X_tr, df_tr["target"].values).predict(X_va)


def mae(p, y, mask=None):
    if mask is None:
        return float(np.mean(np.abs(p - y)))
    return float(np.mean(np.abs(p[mask] - y[mask])))


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    q_train = quadrant_mask(df)
    q_test = quadrant_mask(test)

    print(f"Train quadrant (x4<0, x8<0, x5 non-sent): {int(q_train.sum())}  /  "
          f"Test quadrant: {int(q_test.sum())}")

    # ---- 5-fold OOF ------------------------------------------------------
    print(f"\n5-fold OOF (seed {SEED})")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lin = np.zeros(len(df)); oof_ebm = np.zeros(len(df)); oof_p = np.zeros(len(df))
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        oof_lin[va] = lin_x4_free(sub_tr, sub_va, x5m)
        oof_ebm[va] = ebm_x9(sub_tr, sub_va, x5m)
        clf = fit_clamp_classifier(sub_tr)
        oof_p[va] = predict_clamp_prob(clf, sub_va)
        print(f"  fold {fold+1}/{N_SPLITS}  {time.time()-t0:.0f}s  "
              f"clamp-auc-proxy on this fold: (see summary)")

    # ---- Report base model CV -------------------------------------------
    print("\nBase legs:")
    print(f"  LIN_x4  CV={mae(oof_lin, y):.3f}  non-sent={mae(oof_lin, y, ~is_sent):.3f}")
    print(f"  EBM_x9  CV={mae(oof_ebm, y):.3f}  non-sent={mae(oof_ebm, y, ~is_sent):.3f}")

    # ---- Classifier diagnostics -----------------------------------------
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    resid = a1_residual(df, x5m_full)
    true_clamp = (np.abs(resid) > 1.0) & q_train
    from sklearn.metrics import roc_auc_score, average_precision_score
    oof_p_quadrant = oof_p[q_train]
    tc_quadrant = true_clamp[q_train]
    print(f"\nClamp classifier (quadrant): AUC={roc_auc_score(tc_quadrant, oof_p_quadrant):.3f}  "
          f"AP={average_precision_score(tc_quadrant, oof_p_quadrant):.3f}  "
          f"base rate={tc_quadrant.mean():.3f}")

    # ---- Cross_LE with and without clamp correction ---------------------
    base = 0.5 * (oof_lin + oof_ebm)
    correction = np.zeros(len(df))
    correction[q_train] = oof_p[q_train] * (-15 * df["x8"].values[q_train] + 1.0)
    # cross_LE blend: LIN_x4 contributes 0.5 weight, so apply half correction
    corrected = base - 0.5 * correction  # Subtract from LIN_x4 side (half-weight)
    # Alternative: apply full correction once (assumes EBM_x9 does NOT clamp)
    corrected_full = base - correction

    print("\nEnsembles:")
    print(f"  cross_LE (base, LB 2.94)   CV={mae(base, y):.3f}  "
          f"non-sent={mae(base, y, ~is_sent):.3f}")
    print(f"  cross_LE + half correction CV={mae(corrected, y):.3f}  "
          f"non-sent={mae(corrected, y, ~is_sent):.3f}")
    print(f"  cross_LE + full correction CV={mae(corrected_full, y):.3f}  "
          f"non-sent={mae(corrected_full, y, ~is_sent):.3f}")

    # Measure specifically on quadrant rows
    print("\nQuadrant-only MAE (368 train rows):")
    print(f"  cross_LE base            MAE={mae(base, y, q_train):.3f}")
    print(f"  cross_LE + half corr     MAE={mae(corrected, y, q_train):.3f}")
    print(f"  cross_LE + full corr     MAE={mae(corrected_full, y, q_train):.3f}")

    # ---- Full-data submission --------------------------------------------
    print("\nBuilding full-data submission...")
    # LIN_x4 full
    X_tr = design_matrix(df, x5m_full, include_x4=True, include_x9=False)
    X_te = design_matrix(test, x5m_full, include_x4=True, include_x9=False)
    lin_test = LinearRegression().fit(X_tr, y).predict(X_te)
    # EBM_x9 full
    X_tr = design_matrix(df, x5m_full, include_x4=False, include_x9=True)
    X_te = design_matrix(test, x5m_full, include_x4=False, include_x9=True)
    print("  training EBM_x9 on full data...")
    t0 = time.time()
    ebm_test = fit_ebm(X_tr, y).predict(X_te)
    print(f"  EBM_x9 trained in {time.time()-t0:.0f}s")
    # Clamp classifier full
    clf_full = fit_clamp_classifier(df)
    p_test = predict_clamp_prob(clf_full, test)
    print(f"  test-side clamp probabilities: mean={p_test[q_test].mean():.3f} "
          f"(expected ≈ 0.23)")

    cross_LE_test = 0.5 * (lin_test + ebm_test)
    correction_test = np.zeros(len(test))
    correction_test[q_test] = p_test[q_test] * (-15 * test["x8"].values[q_test] + 1.0)
    pred_half = cross_LE_test - 0.5 * correction_test
    pred_full = cross_LE_test - correction_test

    out = {
        "submission_cross_LE_soft_clamp_half.csv": pred_half,
        "submission_cross_LE_soft_clamp_full.csv": pred_full,
    }
    for name, p in out.items():
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(SUBS / name, index=False)
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
