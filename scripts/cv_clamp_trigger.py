"""Find the A1 clamp trigger — TabPFN / HGB / TabM on the x4<0 ∧ x8<0 quadrant.

A1 closed-form is perfect on 93 % of non-sentinel training rows. The other
7 % live exclusively in the x4 < 0 ∧ x8 < 0 quadrant with residual
≈ −18.4 · x8 (std 0.76). Prior attempts to find the trigger capped at
LightGBM AUC ~0.76 and a simple rule search ~22-26 % accuracy (flat).

This script raises the ceiling:

  * TabPFN v2 (ICL) classifier on the 368-row subset — whole-dataset prior
  * HistGBM with polynomial + sinusoidal interactions
  * TabPFN + HGB mean-probability ensemble
  * Regression variant: predict the residual directly (not classify);
    zero residual for non-clamp ⇒ MAE is the target metric
  * Impact simulation: if we identify clamp rows with probability p and
    route them to A1 + (−18.4·x8) when p > threshold, what would the
    safe-row MAE drop to?

Runs in /tmp/fm_env (needs tabpfn + sklearn already installed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_formulas import approach1_predict  # noqa: E402
from cv_ebm_variants import SENTINEL, SEED  # noqa: E402

DATA = REPO / "data"
OUT = REPO / "plots" / "a1_clamp"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
TABPFN_CLF_PATH = "/root/.cache/tabpfn/tabpfn-v2-classifier.ckpt"


# ---------------------------------------------------------------------------
# Feature engineering — raise the signal above the LightGBM-0.76 ceiling
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, x5_median: float) -> pd.DataFrame:
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median)
    out = pd.DataFrame({
        "x1":  df["x1"].values,
        "x2":  df["x2"].values,
        "x4":  df["x4"].values,
        "x5":  x5.values,
        "x6":  df["x6"].values,
        "x7":  df["x7"].values,
        "x8":  df["x8"].values,
        "x9":  df["x9"].values,
        "x10": df["x10"].values,
        "x11": df["x11"].values,
        "city":(df["City"] == "Zaragoza").astype(float).values,
        # Pair interactions the prior search flagged
        "x4x8":   df["x4"] * df["x8"],
        "x4_x8":  df["x4"] - df["x8"],
        "x5x8":   x5 * df["x8"],
        "x8x9":   df["x8"] * df["x9"],
        "x8_sq":  df["x8"] ** 2,
        "x8_abs": df["x8"].abs(),
        # Angle in x6/x7 — uniform but might still interact nonlinearly
        "theta_sin": np.sin(np.arctan2(df["x7"], df["x6"])),
        "theta_cos": np.cos(np.arctan2(df["x7"], df["x6"])),
    })
    return out


def mae(p, y):
    return float(np.mean(np.abs(p - y))) if len(y) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    y = df["target"].values
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    is_sent = (df["x5"].values == SENTINEL)

    # -----------------------------------------------------------------
    # Compute A1 residual on non-sentinel rows
    # -----------------------------------------------------------------
    a1 = approach1_predict(df, x5m)
    resid = y - a1

    quadrant = (df["x4"].values < 0) & (df["x8"].values < 0) & (~is_sent)
    q_idx = np.where(quadrant)[0]
    print(f"x4<0 & x8<0 & non-sent: {len(q_idx)} rows")

    # Clamp label — residual magnitude > 3 (clamp ≈ −18.4·x8, |x8|~0.5 → ~9;
    # perfect-fit rows have residual ~0; threshold well below 9 and above 0
    # is fine; std of residual in perfect rows is < 0.1).
    is_clamp = (np.abs(resid[q_idx]) > 3.0).astype(int)
    print(f"  clamp rows (|resid| > 3): {int(is_clamp.sum())} "
          f"({is_clamp.mean():.1%})")
    print(f"  median |resid| clamp  : {np.median(np.abs(resid[q_idx][is_clamp==1])):.2f}")
    print(f"  median |resid| perfect: {np.median(np.abs(resid[q_idx][is_clamp==0])):.4f}")

    X_all = build_features(df, x5m)
    X_q = X_all.iloc[q_idx].reset_index(drop=True)
    y_bin = is_clamp
    y_resid = resid[q_idx]

    # -----------------------------------------------------------------
    # 5-fold CV — three classifiers + residual regressor
    # -----------------------------------------------------------------
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(X_q)) for k in
           ["tabpfn", "hgb", "ensemble", "hgb_reg"]}

    print("\n" + "=" * 78)
    print(f"5-fold CV on the {len(X_q)}-row clamp quadrant")
    print("=" * 78)

    from tabpfn import TabPFNClassifier
    for fold, (tr, va) in enumerate(kf.split(X_q)):
        t0 = time.time()

        X_tr, X_va = X_q.iloc[tr].values, X_q.iloc[va].values
        y_tr, y_va = y_bin[tr], y_bin[va]
        yr_tr, yr_va = y_resid[tr], y_resid[va]

        # TabPFN classifier
        t1 = time.time()
        m_pfn = TabPFNClassifier(
            n_estimators=8, device="cpu",
            ignore_pretraining_limits=True, random_state=SEED,
            model_path=TABPFN_CLF_PATH,
        ).fit(X_tr, y_tr)
        cls = list(m_pfn.classes_)
        if 1 in cls:
            oof["tabpfn"][va] = m_pfn.predict_proba(X_va)[:, cls.index(1)]
        t_pfn = time.time() - t1

        # HistGBM classifier (balanced)
        t1 = time.time()
        m_hgb = HistGradientBoostingClassifier(
            max_iter=500, max_depth=5, learning_rate=0.05,
            l2_regularization=1.0, random_state=SEED, class_weight="balanced",
        ).fit(X_tr, y_tr)
        oof["hgb"][va] = m_hgb.predict_proba(X_va)[:, 1]
        t_hgb = time.time() - t1

        # Ensemble (mean prob)
        oof["ensemble"][va] = 0.5 * (oof["tabpfn"][va] + oof["hgb"][va])

        # Residual regressor (useful if we want probability-weighted correction)
        m_reg = HistGradientBoostingRegressor(
            max_iter=500, max_depth=5, learning_rate=0.05,
            l2_regularization=1.0, random_state=SEED,
        ).fit(X_tr, yr_tr)
        oof["hgb_reg"][va] = m_reg.predict(X_va)

        print(f"  fold {fold+1}/{N_SPLITS}  tabpfn={t_pfn:.0f}s  hgb={t_hgb:.1f}s  "
              f"fold AUC: pfn={roc_auc_score(y_va, oof['tabpfn'][va]):.3f}  "
              f"hgb={roc_auc_score(y_va, oof['hgb'][va]):.3f}  "
              f"[{time.time()-t0:.0f}s]")

    # -----------------------------------------------------------------
    # Overall AUC + impact simulation
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Classifier results (quadrant only)")
    print("=" * 78)
    for name in ["tabpfn", "hgb", "ensemble"]:
        auc = roc_auc_score(y_bin, oof[name])
        ap  = average_precision_score(y_bin, oof[name])
        print(f"  {name:<10s}  AUC={auc:.3f}  AP={ap:.3f}  "
              f"(baseline AP = {y_bin.mean():.3f})")

    # Residual regressor MAE vs. a zero baseline
    mae_zero = mae(np.zeros_like(y_resid), y_resid)
    mae_reg  = mae(oof["hgb_reg"], y_resid)
    print(f"\nResidual regression on the quadrant:")
    print(f"  zero baseline: MAE = {mae_zero:.3f}")
    print(f"  HGB regressor: MAE = {mae_reg:.3f}  "
          f"(reduction = {100*(1 - mae_reg/mae_zero):.1f}%)")

    # -----------------------------------------------------------------
    # Impact simulation on the FULL safe-row MAE
    # Current router safe MAE = 0.377 (all safe → A1). The 86 clamp rows
    # have ~9 MAE each; 282 non-clamp have ~0. If we correctly route clamp
    # rows to A1 + (−18.4 · x8) = target_true, safe MAE drops proportionally.
    # We simulate "at threshold t, predict A1 + correction for rows with
    # prob(clamp) > t; else predict A1".
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Simulated safe-row MAE by classifier threshold")
    print("=" * 78)
    a1_q = a1[q_idx]
    y_q  = y[q_idx]
    print(f"{'model':<10s}  {'threshold':>9s}  "
          f"{'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'quadrant MAE':>14s}")
    for name in ["tabpfn", "hgb", "ensemble"]:
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_label = (oof[name] > thr).astype(int)
            # Correction: for rows flagged, subtract 18.4 * x8 from A1
            pred = a1_q.copy()
            # Use the regressor's prediction as the correction instead of a fixed formula
            # If we're confident in the −18.4·x8 rule, replace with that; here we use regressor
            pred_flagged = a1_q + oof["hgb_reg"]
            pred = np.where(pred_label == 1, pred_flagged, a1_q)
            tp = int(((pred_label == 1) & (y_bin == 1)).sum())
            fp = int(((pred_label == 1) & (y_bin == 0)).sum())
            fn = int(((pred_label == 0) & (y_bin == 1)).sum())
            m = mae(pred, y_q)
            print(f"  {name:<10s}  {thr:>9.2f}  "
                  f"{tp:>4d} {fp:>4d} {fn:>4d}  {m:>14.3f}")

    # Baseline: no correction
    print(f"\n  {'A1 alone (no correction)':<26s}  "
          f"quadrant MAE = {mae(a1_q, y_q):.3f}")
    # Oracle: use the regressor prediction everywhere
    pred_oracle = a1_q + np.where(is_clamp == 1, oof['hgb_reg'], 0.0)
    print(f"  {'oracle (perfect label)':<26s}  "
          f"quadrant MAE = {mae(pred_oracle, y_q):.3f}")

    # Save
    pd.DataFrame({**oof, "y_bin": y_bin, "y_resid": y_resid,
                  "a1": a1_q, "x4": df.iloc[q_idx]["x4"].values,
                  "x8": df.iloc[q_idx]["x8"].values}).to_csv(
        OUT / "clamp_search_oof.csv", index=False)
    print(f"\nWrote {OUT / 'clamp_search_oof.csv'}")


if __name__ == "__main__":
    main()
