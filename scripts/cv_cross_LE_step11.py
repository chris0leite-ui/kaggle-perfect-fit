"""cross_LE with a corrected-step LIN_x4 component.

Current cross_LE (LB 2.94) uses LIN_x4 with pure-linear x4 (β≈+30).
We've shown β=30 is an x4-only-slope compromise that under-fits the
real training pattern at x4≈0. A cleaner decomposition is:

    target ≈ 15·x4 + 11·1{x4>0} + ...   (no x9)

where the 11·step replaces ~8 of the linear slope and ~8 of the
(miss-attributed) x9 contribution. Replacing LIN_x4_free with this
step11 model in cross_LE should preserve the ensemble robustness
while adding calibration on the 508 gap-zone test rows.

This script:

1. 5-fold CV for EBM_x9 and a new LIN_x4_step11 (no x9).
2. Reuses prior cross_LE recipe: 0.5 * (LIN_x4_step11 + EBM_x9).
3. Also tests the smoothed-step (tanh scale=0.15) variant to hedge
   against the step being unreal at x4=0.
4. Writes LB-ready submissions to submissions/.

Computes CV for:
  - LIN_x4_step11 solo
  - LIN_x4_tanh   solo
  - EBM_x9 solo
  - cross_LE_free (baseline, CV 2.97 / LB 2.94)
  - cross_LE_step11 (new)
  - cross_LE_tanh  (new, smoothed step hedge)
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
from cv_x4_x9_swap_ensemble import fit_ebm, design_matrix  # noqa: E402
from cv_ebm_variants import SENTINEL, SEED, preprocess  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5


def step11_design(df: pd.DataFrame, x5_med: float, shape: str) -> np.ndarray:
    """LIN_x4 design with step/tanh basis and NO x9."""
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_med).values
    city = (df["City"] == "Zaragoza").astype(float).values
    x4 = df["x4"].values

    cols = [
        df["x1"].values ** 2,
        np.cos(5 * np.pi * df["x2"].values),
        x4,
        x5,
        is_sent,
        df["x8"].values,
        df["x10"].values,
        df["x11"].values,
        df["x10"].values * df["x11"].values,
        city,
    ]
    if shape == "step":
        cols.append((x4 > 0).astype(float))
    elif shape == "tanh":
        cols.append(np.tanh(x4 / 0.15))
    else:
        raise ValueError(shape)
    return np.column_stack(cols)


def fit_step(df_tr, df_va, shape: str):
    x5m = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = step11_design(df_tr, x5m, shape)
    X_va = step11_design(df_va, x5m, shape)
    return LinearRegression().fit(X_tr, df_tr["target"].values).predict(X_va)


def fit_ebm_x9(df_tr, df_va):
    """EBM on full feature set EXCEPT x4 (matches cross_LE's EBM_x9)."""
    from cv_x4_x9_swap_ensemble import design_matrix  # noqa
    x5m = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5m, include_x4=False, include_x9=True)
    X_va = design_matrix(df_va, x5m, include_x4=False, include_x9=True)
    return fit_ebm(X_tr, df_tr["target"].values).predict(X_va)


def fit_lin_x4_free(df_tr, df_va):
    """Current cross_LE's LIN_x4 (pure linear, no step, no x9)."""
    x5m = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5m, include_x4=True, include_x9=False)
    X_va = design_matrix(df_va, x5m, include_x4=True, include_x9=False)
    return LinearRegression().fit(X_tr, df_tr["target"].values).predict(X_va)


def mae(p, y, mask=None):
    if mask is None:
        return float(np.mean(np.abs(p - y)))
    return float(np.mean(np.abs(p[mask] - y[mask])))


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values

    print("=" * 70)
    print("5-fold OOF for LIN_x4_{free, step, tanh} and EBM_x9")
    print("=" * 70)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(df)) for k in
           ["lin_free", "lin_step", "lin_tanh", "ebm_x9"]}
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        oof["lin_free"][va] = fit_lin_x4_free(sub_tr, sub_va)
        oof["lin_step"][va] = fit_step(sub_tr, sub_va, "step")
        oof["lin_tanh"][va] = fit_step(sub_tr, sub_va, "tanh")
        oof["ebm_x9"][va] = fit_ebm_x9(sub_tr, sub_va)
        print(f"  fold {fold+1}/{N_SPLITS}  {time.time()-t0:.0f}s")

    print("\nBase-model CV MAE:")
    for k in ["lin_free", "lin_step", "lin_tanh", "ebm_x9"]:
        print(f"  {k:<10s} CV={mae(oof[k], y):.3f}  "
              f"non-sent={mae(oof[k], y, ~is_sent):.3f}  "
              f"sent={mae(oof[k], y, is_sent):.3f}")

    print("\nEnsembles (0.5 blend each):")
    ensembles = {
        "cross_LE_free":   0.5 * (oof["lin_free"] + oof["ebm_x9"]),
        "cross_LE_step11": 0.5 * (oof["lin_step"] + oof["ebm_x9"]),
        "cross_LE_tanh":   0.5 * (oof["lin_tanh"] + oof["ebm_x9"]),
    }
    for name, p in ensembles.items():
        print(f"  {name:<18s} CV={mae(p, y):.3f}  "
              f"non-sent={mae(p, y, ~is_sent):.3f}  "
              f"sent={mae(p, y, is_sent):.3f}")

    # ------------------------------------------------------------------
    # Build submissions on FULL data
    # ------------------------------------------------------------------
    print("\nBuilding submissions on full data...")
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # LIN_x4_step
    X_tr = step11_design(df, x5m_full, "step")
    X_te = step11_design(test, x5m_full, "step")
    lin_step_test = LinearRegression().fit(X_tr, y).predict(X_te)

    # LIN_x4_tanh
    X_tr = step11_design(df, x5m_full, "tanh")
    X_te = step11_design(test, x5m_full, "tanh")
    lin_tanh_test = LinearRegression().fit(X_tr, y).predict(X_te)

    # EBM_x9
    X_tr = design_matrix(df, x5m_full, include_x4=False, include_x9=True)
    X_te = design_matrix(test, x5m_full, include_x4=False, include_x9=True)
    print("  training EBM_x9 on full data...")
    t0 = time.time()
    ebm_x9_test = fit_ebm(X_tr, y).predict(X_te)
    print(f"  EBM_x9 done in {time.time()-t0:.0f}s")

    SUBS.mkdir(exist_ok=True)
    out = {
        "submission_cross_LE_step11.csv": 0.5 * (lin_step_test + ebm_x9_test),
        "submission_cross_LE_tanh.csv":   0.5 * (lin_tanh_test + ebm_x9_test),
    }
    for name, p in out.items():
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(SUBS / name, index=False)
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
