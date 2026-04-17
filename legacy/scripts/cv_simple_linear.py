"""Fit a simple linear model on the user-specified feature set and CV it.

Feature set:
  - x1**2  (quadratic, mirrors A1)
  - cos(5*pi*x2)  (A1 basis)
  - x4 (linear)
  - x5_imputed (sentinel -> median)
  - x5_is_sentinel (binary indicator)
  - x8
  - x10
  - x11
  - city (Zaragoza=1)
  - x10*x11 (optional — EBM's top interaction)
  - intercept

Runs two variants: with and without the x10*x11 interaction. 5-fold KFold
(shuffle=True, seed=42) on dataset.csv (1500 rows). Reports overall, non-
sentinel, and sentinel MAE per fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

SENTINEL = 999.0
SEED = 42
N_SPLITS = 5


def design_matrix(df: pd.DataFrame, x5_median: float, include_interaction: bool) -> np.ndarray:
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values
    city = (df["City"] == "Zaragoza").astype(float).values
    cols = [
        df["x1"].values ** 2,                     # x1^2
        np.cos(5 * np.pi * df["x2"].values),      # cos(5*pi*x2)
        df["x4"].values,                          # x4
        x5,                                       # x5 imputed
        is_sent,                                  # x5_is_sentinel
        df["x8"].values,                          # x8
        df["x10"].values,                         # x10
        df["x11"].values,                         # x11
        city,                                     # city
    ]
    names = ["x1^2", "cos5piX2", "x4", "x5_imp", "x5_is_sent",
             "x8", "x10", "x11", "city"]
    if include_interaction:
        cols.append(df["x10"].values * df["x11"].values)
        names.append("x10*x11")
    return np.column_stack(cols), names


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def run_cv(df: pd.DataFrame, include_interaction: bool) -> dict:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    oof = np.zeros(len(df))
    per_fold = []
    coefs_list = []
    for k, (tr, va) in enumerate(kf.split(df)):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        X_tr, names = design_matrix(sub_tr, x5_med, include_interaction)
        X_va, _ = design_matrix(sub_va, x5_med, include_interaction)
        lr = LinearRegression().fit(X_tr, sub_tr["target"].values)
        oof[va] = lr.predict(X_va)
        coefs_list.append(lr.coef_)
        per_fold.append(mae(oof[va], sub_va["target"].values))
    overall = mae(oof, y)
    ns = mae(oof[~is_sent], y[~is_sent])
    sm = mae(oof[is_sent], y[is_sent])
    avg_coefs = np.mean(coefs_list, axis=0)
    return {
        "include_interaction": include_interaction,
        "names": names,
        "avg_coefs": avg_coefs,
        "per_fold": per_fold,
        "overall": overall,
        "non_sentinel": ns,
        "sentinel": sm,
    }


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    print(f"dataset: {df.shape}")
    print(f"sentinels: {(df['x5']==SENTINEL).sum()} ({100*(df['x5']==SENTINEL).mean():.1f}%)")

    print("\n" + "=" * 78)
    print("Variant A: without x10*x11 interaction (as specified)")
    print("=" * 78)
    res_a = run_cv(df, include_interaction=False)
    print(f"  overall CV MAE:      {res_a['overall']:.3f}")
    print(f"  non-sentinel MAE:    {res_a['non_sentinel']:.3f}")
    print(f"  sentinel MAE:        {res_a['sentinel']:.3f}")
    print(f"  per-fold:            {[round(x,3) for x in res_a['per_fold']]}")
    print(f"  avg coefs:")
    for n, c in zip(res_a["names"], res_a["avg_coefs"]):
        print(f"    {n:>12s}: {c:+.3f}")

    print("\n" + "=" * 78)
    print("Variant B: with x10*x11 interaction added")
    print("=" * 78)
    res_b = run_cv(df, include_interaction=True)
    print(f"  overall CV MAE:      {res_b['overall']:.3f}")
    print(f"  non-sentinel MAE:    {res_b['non_sentinel']:.3f}")
    print(f"  sentinel MAE:        {res_b['sentinel']:.3f}")
    print(f"  per-fold:            {[round(x,3) for x in res_b['per_fold']]}")
    print(f"  avg coefs:")
    for n, c in zip(res_b["names"], res_b["avg_coefs"]):
        print(f"    {n:>12s}: {c:+.3f}")

    print("\n" + "=" * 78)
    print(f"Delta from adding x10*x11: {res_a['overall']:.3f} -> {res_b['overall']:.3f} "
          f"({res_b['overall']-res_a['overall']:+.3f})")
    print("Reference models (from earlier CV runs):")
    print(f"  A1 closed form (CV):            1.80")
    print(f"  A2 ClosedFormModel (CV):        3.49")
    print(f"  EBM (R2 tuned, CV):             3.11")
    print(f"  EBM+GAM 70/30 (CV):             2.91")


if __name__ == "__main__":
    main()
