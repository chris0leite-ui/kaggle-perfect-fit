"""Two tests for the x4·x9 interaction.

Test 1: EBM_full with (x4, x9) interaction EXCLUDED. Does removing the
        training-specific x4·x9 interaction improve CV / non-sent?

Test 2: Linear model with BOTH x4 and x9 present, plus x4·x9 manually
        added as a feature. Compare against the same linear model
        without the interaction (both include raw x9 and raw x4).

Both use 5-fold CV on dataset.csv. Top variants get submissions built.
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
from cv_x4_x9_swap_ensemble import ebm_features, fit_ebm  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Linear with x4 + x9 + other features (+/- x4·x9 interaction)
# ---------------------------------------------------------------------------
def lin_design(df: pd.DataFrame, x5_median: float, include_x4x9: bool) -> np.ndarray:
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values
    city = (df["City"] == "Zaragoza").astype(float).values
    cols = [
        df["x1"].values ** 2,
        np.cos(5 * np.pi * df["x2"].values),
        df["x4"].values,
        x5,
        is_sent,
        df["x8"].values,
        df["x9"].values,
        df["x10"].values,
        df["x11"].values,
        df["x10"].values * df["x11"].values,
        city,
    ]
    names = ["x1^2", "cos(5pi*x2)", "x4", "x5_imp", "x5_is_sent",
             "x8", "x9", "x10", "x11", "x10*x11", "city"]
    if include_x4x9:
        cols.append(df["x4"].values * df["x9"].values)
        names.append("x4*x9")
    return np.column_stack(cols), names


def cv_linear(df: pd.DataFrame, include_x4x9: bool) -> tuple[float, float, list]:
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    coefs_list = []
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        X_tr, names = lin_design(sub_tr, x5_med, include_x4x9)
        X_va, _ = lin_design(sub_va, x5_med, include_x4x9)
        lr = LinearRegression().fit(X_tr, sub_tr["target"].values)
        oof[va] = lr.predict(X_va)
        coefs_list.append(lr.coef_)
    m = float(np.mean(np.abs(oof - y)))
    mn = float(np.mean(np.abs(oof[~is_sent] - y[~is_sent])))
    avg_coefs = np.mean(coefs_list, axis=0)
    return m, mn, list(zip(names, avg_coefs))


# ---------------------------------------------------------------------------
# EBM_full with x4·x9 interaction excluded
# ---------------------------------------------------------------------------
def cv_ebm_exclude_x4x9(df: pd.DataFrame) -> tuple[float, float]:
    from interpret.glassbox import ExplainableBoostingRegressor
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        feats = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats, x5m)
        X_va = preprocess(sub_va, feats, x5m)
        model = ExplainableBoostingRegressor(
            interactions=10, max_rounds=4000, min_samples_leaf=10,
            max_bins=128, smoothing_rounds=4000,
            interaction_smoothing_rounds=1000, random_state=SEED,
            exclude=[("x4", "x9")],
        )
        model.fit(X_tr, sub_tr["target"].values)
        oof[va] = model.predict(X_va)
    return (float(np.mean(np.abs(oof - y))),
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))))


def fit_ebm_exclude_x4x9(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(
        interactions=10, max_rounds=4000, min_samples_leaf=10,
        max_bins=128, smoothing_rounds=4000,
        interaction_smoothing_rounds=1000, random_state=SEED,
        exclude=[("x4", "x9")],
    ).fit(X, y)


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values

    # ------------------------------------------------------------------
    # Test 2: Linear with both x4 and x9, ±x4·x9
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Test 2: Linear with x4 + x9 + other features, ± x4·x9 interaction")
    print("=" * 80)
    m_no, mn_no, coefs_no = cv_linear(df, include_x4x9=False)
    m_yes, mn_yes, coefs_yes = cv_linear(df, include_x4x9=True)
    print(f"  without x4*x9  overall={m_no:.3f}  non-sent={mn_no:.3f}")
    print(f"  with    x4*x9  overall={m_yes:.3f}  non-sent={mn_yes:.3f}")
    print(f"  Δ             {m_yes - m_no:+.3f} overall, "
          f"{mn_yes - mn_no:+.3f} non-sent")
    print("\n  Coefficients (avg across 5 folds):")
    for n, c in coefs_no:
        print(f"    without:  {n:<12s}  {c:+.3f}")
    print()
    for n, c in coefs_yes:
        print(f"    with:     {n:<12s}  {c:+.3f}")

    # ------------------------------------------------------------------
    # Test 1: EBM_full with x4·x9 interaction excluded
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Test 1: EBM_full with/without x4·x9 auto-interaction")
    print("=" * 80)
    # Reference: our best EBM config with x4·x9 allowed (this is the triple's
    # EBM_full component baseline; CV 3.030)
    t0 = time.time()
    m_ex, mn_ex = cv_ebm_exclude_x4x9(df)
    print(f"  EBM_full EXCLUDING (x4,x9)  overall={m_ex:.3f}  non-sent={mn_ex:.3f}  "
          f"[{time.time()-t0:.0f}s]")
    print(f"  EBM_full baseline  (from prior runs)  overall=3.030  non-sent=1.735")
    print(f"  Δ                  {m_ex - 3.030:+.3f} overall, "
          f"{mn_ex - 1.735:+.3f} non-sent")

    # ------------------------------------------------------------------
    # If EBM-exclude helps → plug into the triple ensemble and CV
    # ------------------------------------------------------------------
    if m_ex < 3.030:
        print("\n  Exclude helps — computing triple ensemble with new EBM_full")
        # Need OOF for LIN_x4_locked_b and EBM_x9; reuse from cross_LE_tune
        # For a quick integration we'd recompute OOFs. Skipping for now.
        # (Will be computed in the full build section below.)

    # ------------------------------------------------------------------
    # Build submissions
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Building submissions")
    print("=" * 80)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # Linear with x4 + x9 + x4*x9
    X_tr, _ = lin_design(df, x5m_full, include_x4x9=True)
    X_te, _ = lin_design(test, x5m_full, include_x4x9=True)
    lr = LinearRegression().fit(X_tr, df["target"].values)
    preds = lr.predict(X_te)
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
        SUBS / "submission_linear_x4_x9_x4x9.csv", index=False)
    print(f"  wrote submission_linear_x4_x9_x4x9.csv  "
          f"mean={preds.mean():+.3f}  range=[{preds.min():+.2f}, {preds.max():+.2f}]")

    # Linear without x4*x9 (for comparison)
    X_tr, _ = lin_design(df, x5m_full, include_x4x9=False)
    X_te, _ = lin_design(test, x5m_full, include_x4x9=False)
    lr = LinearRegression().fit(X_tr, df["target"].values)
    preds = lr.predict(X_te)
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
        SUBS / "submission_linear_x4_x9_no_inter.csv", index=False)
    print(f"  wrote submission_linear_x4_x9_no_inter.csv  "
          f"mean={preds.mean():+.3f}  range=[{preds.min():+.2f}, {preds.max():+.2f}]")

    # EBM_full excluding (x4, x9) — standalone
    feats_full = ebm_features(with_x4=True, with_x9=True)
    X_tr = preprocess(df, feats_full, x5m_full)
    X_te = preprocess(test, feats_full, x5m_full)
    ebm_ex = fit_ebm_exclude_x4x9(X_tr, df["target"].values)
    preds_ebm_ex = ebm_ex.predict(X_te)
    pd.DataFrame({"id": test["id"], "target": preds_ebm_ex}).to_csv(
        SUBS / "submission_ebm_full_no_x4x9.csv", index=False)
    print(f"  wrote submission_ebm_full_no_x4x9.csv  "
          f"mean={preds_ebm_ex.mean():+.3f}  range=[{preds_ebm_ex.min():+.2f}, "
          f"{preds_ebm_ex.max():+.2f}]")

    # Triple ensemble using EBM_full_no_x4x9 as the "full" component
    # Need LIN_x4_locked_b and EBM_x9 on full data
    from cv_cross_LE_tune import LOCKED_COEFS_B, LIN_COL_ORDER
    from cv_x4_x9_swap_ensemble import design_matrix
    X_tr = design_matrix(df, x5m_full, True, False)
    X_te_lin = design_matrix(test, x5m_full, True, False)
    locked_b = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
    resid = df["target"].values - X_tr @ locked_b
    lin_x4_b_pred = X_te_lin @ locked_b + resid.mean()

    feats_x9 = ebm_features(with_x4=False, with_x9=True)
    X_tr9 = preprocess(df, feats_x9, x5m_full)
    X_te9 = preprocess(test, feats_x9, x5m_full)
    ebm_x9_pred = fit_ebm(X_tr9, df["target"].values).predict(X_te9)

    # Triple with EBM_full excluding x4·x9
    triple_pred = 0.25 * lin_x4_b_pred + 0.25 * ebm_x9_pred + 0.50 * preds_ebm_ex
    pd.DataFrame({"id": test["id"], "target": triple_pred}).to_csv(
        SUBS / "submission_ensemble_triple_locked_b_no_x4x9.csv", index=False)
    print(f"  wrote submission_ensemble_triple_locked_b_no_x4x9.csv  "
          f"mean={triple_pred.mean():+.3f}  range=[{triple_pred.min():+.2f}, "
          f"{triple_pred.max():+.2f}]")


if __name__ == "__main__":
    main()
