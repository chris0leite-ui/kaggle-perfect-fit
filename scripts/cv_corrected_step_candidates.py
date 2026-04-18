"""CV corrected-step candidates (follow-up to x4 functional-form oracle).

We established that A1's +20 step at x4=0 double-counts x9's training
cluster contrast. After decorrelating x9 (via x9_wc or dropping it),
the residual step is +11.

This script CV-scores several candidate models that combine:
  - step coefficient: {0, +11, +20}
  - x4 shape:        {linear, step-sharp, step-tanh-smoothed}
  - x9 treatment:    {drop, raw, x9_wc}

Purpose: identify the honest CV ranking under each combination. The
resulting best-in-class submissions are written to submissions/ for
LB testing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUB = REPO / "submissions"
SENTINEL = 999.0
RNG = 42


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["x5_is_sent"] = (df["x5"] == SENTINEL).astype(float)
        x5 = df["x5"].where(df["x5"] != SENTINEL, np.nan)
        df["x5_imp"] = x5.fillna(x5.median())
        df["city_code"] = (df["City"] == "Zaragoza").astype(float)
        df["x1sq"] = df["x1"] ** 2
        df["cos5pi_x2"] = np.cos(5 * np.pi * df["x2"])
        df["x10x11"] = df["x10"] * df["x11"]
        df["sign_x4"] = (df["x4"] > 0).astype(float)

    # Compute training cluster means for x9; apply to test via sign(x4)
    means = train.groupby("sign_x4")["x9"].mean()
    train["x9_wc"] = train["x9"] - train["sign_x4"].map(means)
    test["x9_wc"] = test["x9"] - test["sign_x4"].map(means)
    # Smooth step feature — same scale as oracle's 'tanh_mid' (CV 2.23 there).
    train["tanh_step"] = np.tanh(train["x4"] / 0.15)
    test["tanh_step"] = np.tanh(test["x4"] / 0.15)
    return train, test


def build_design(df: pd.DataFrame, shape: str, x9: str) -> tuple[np.ndarray, list[str]]:
    fixed = ["x1sq", "cos5pi_x2", "x5_imp", "x5_is_sent",
             "x8", "x10", "x11", "x10x11", "city_code", "x4"]
    cols = list(fixed)
    if shape == "linear":
        pass
    elif shape == "step_sharp":
        cols.append("sign_x4")
    elif shape == "step_tanh":
        cols.append("tanh_step")
    else:
        raise ValueError(shape)
    if x9 == "raw":
        cols.append("x9")
    elif x9 == "x9_wc":
        cols.append("x9_wc")
    elif x9 == "drop":
        pass
    else:
        raise ValueError(x9)
    return df[cols].to_numpy(), cols


def cv_and_fit(train: pd.DataFrame, shape: str, x9: str) -> dict:
    X, cols = build_design(train, shape, x9)
    y = train["target"].to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    errs, ns, sn = [], [], []
    is_sent = train["x5_is_sent"].to_numpy() > 0
    for tr, va in kf.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        p = m.predict(X[va])
        e = np.abs(p - y[va])
        errs.append(e.mean())
        ns.append(e[~is_sent[va]].mean())
        sn.append(e[is_sent[va]].mean())
    # Full-data refit for coefficient reporting
    m = LinearRegression().fit(X, y)
    return {
        "cv_mae": float(np.mean(errs)),
        "cv_non_sent": float(np.mean(ns)),
        "cv_sent": float(np.mean(sn)),
        "cols": cols,
        "coefs": dict(zip(cols, m.coef_)),
        "intercept": float(m.intercept_),
    }


def predict_and_save(train: pd.DataFrame, test: pd.DataFrame,
                     shape: str, x9: str, filename: str) -> None:
    Xtr, _ = build_design(train, shape, x9)
    Xte, _ = build_design(test, shape, x9)
    m = LinearRegression().fit(Xtr, train["target"].to_numpy())
    pred = m.predict(Xte)
    out = pd.DataFrame({"id": test["id"], "target": pred})
    out.to_csv(SUB / filename, index=False)
    print(f"  wrote {filename}  (n={len(out)})")


def main() -> None:
    train, test = load()

    # ------------------------------------------------------------------
    # 1. Grid of (shape, x9)
    # ------------------------------------------------------------------
    GRID = [
        ("linear",      "drop"),
        ("linear",      "raw"),
        ("linear",      "x9_wc"),
        ("step_sharp",  "drop"),
        ("step_sharp",  "raw"),
        ("step_sharp",  "x9_wc"),
        ("step_tanh",   "drop"),
        ("step_tanh",   "raw"),
        ("step_tanh",   "x9_wc"),
    ]
    print("=" * 92)
    print(f"{'shape':<12s} {'x9':<7s} {'CV':>7s} {'nonsent':>8s} {'sent':>6s} "
          f"{'β_x4':>7s} {'β_step':>8s} {'β_x9':>8s}")
    print("=" * 92)
    rows = []
    for shape, x9 in GRID:
        r = cv_and_fit(train, shape, x9)
        step_key = "sign_x4" if shape == "step_sharp" else ("tanh_step" if shape == "step_tanh" else None)
        x9_key = "x9" if x9 == "raw" else ("x9_wc" if x9 == "x9_wc" else None)
        print(f"{shape:<12s} {x9:<7s} {r['cv_mae']:>7.3f} "
              f"{r['cv_non_sent']:>8.3f} {r['cv_sent']:>6.2f} "
              f"{r['coefs']['x4']:>+7.2f} "
              f"{(r['coefs'].get(step_key) if step_key else float('nan')):>+8.2f} "
              f"{(r['coefs'].get(x9_key) if x9_key else float('nan')):>+8.2f}")
        rows.append({"shape": shape, "x9": x9, **{k: r[k] for k in ("cv_mae", "cv_non_sent", "cv_sent")},
                     "beta_step": r['coefs'].get(step_key) if step_key else np.nan,
                     "beta_x9": r['coefs'].get(x9_key) if x9_key else np.nan})
    pd.DataFrame(rows).to_csv(REPO / "plots" / "x4_oracle" / "corrected_step_cv.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Submissions for the three high-leverage candidates:
    #     a. step_sharp + drop x9          — corrected +11 step, no x9
    #     b. step_tanh  + drop x9          — smoothed +11 step, no x9
    #     c. linear     + drop x9          — baseline (prior simple_linear_interact was LB 7.38)
    # ------------------------------------------------------------------
    print("\nBuilding submissions:")
    predict_and_save(train, test, "step_sharp", "drop",
                     "submission_linear_step11_nox9.csv")
    predict_and_save(train, test, "step_tanh",  "drop",
                     "submission_linear_tanh_step_nox9.csv")
    predict_and_save(train, test, "linear",     "drop",
                     "submission_linear_nox9_baseline.csv")


if __name__ == "__main__":
    main()
