"""x8 clamp archaeology — simplicity-first brute-force trigger search.

Prior:
------
86 of 368 training rows in the x4<0 & x8<0 quadrant have
A1 residual ≈ -18.4·x8 (std 0.76); the other 282 have residual ≈ 0.
A LightGBM classifier on rich features hit AUC 0.76 but no simple
rule tested so far achieves clean separation.

The dataset is hand-crafted, so the trigger is likely a *single*
arithmetic expression on {x1..x11, City, θ=atan2(x7,x6)} with
integer/half-integer thresholds. This script exhaustively searches:

A. Single-feature thresholds.      (x_j > t, x_j < t)
B. Pairwise sums / differences.    (x_i + x_j, x_i - x_j)
C. Pairwise products / ratios.     (x_i * x_j, x_i / x_j)
D. Parity / modular rules.         (floor(x_i) mod k)
E. Angular-region cuts.            (sin(θ), cos(θ), θ in quadrant)
F. Triple combinations of the strongest pair-winners.

Scoring
-------
For every candidate rule we compute on the 368-row quadrant:
    sensitivity  = P(rule | clamp)          target ≥ 0.9
    specificity  = P(¬rule | ¬clamp)        target ≥ 0.9
    accuracy, AUC (if continuous)

We also check that the same rule holds on the x4>0 side
(predicting 0 clamp rows there) and that it does NOT fire on
non-sentinel-quadrant rows.

Correction form
---------------
On clamp rows only, we fit:
    residual = α · x8 + β · x4 + γ · 1 + ...  (single-term first)
with simplicity selection (smallest coefficients, integer-friendly).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "clamp_search"
OUT.mkdir(parents=True, exist_ok=True)
SENTINEL = 999.0

CLAMP_THR = 1.0     # |residual| > CLAMP_THR defines a clamped row
TARGET_SENS = 0.90
TARGET_SPEC = 0.90


def a1_predict(df: pd.DataFrame, x5_med: float) -> np.ndarray:
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_med).values
    is_zar = (df["City"] == "Zaragoza").astype(float).values
    return (
        -100 * df["x1"].values ** 2
        + 10 * np.cos(5 * np.pi * df["x2"].values)
        + 15 * df["x4"].values
        - 8 * x5
        + 15 * df["x8"].values
        - 4 * df["x9"].values
        + df["x10"].values * df["x11"].values
        - 25 * is_zar
        + 20 * (df["x4"].values > 0)
        + 92.5
    )


def load() -> pd.DataFrame:
    train = pd.read_csv(DATA / "dataset.csv")
    train["theta"] = np.arctan2(train["x7"], train["x6"])
    train["r_xy"] = np.sqrt(train["x6"] ** 2 + train["x7"] ** 2)  # ≈18
    train["city_code"] = (train["City"] == "Zaragoza").astype(float)
    x5m = train.loc[train["x5"] != SENTINEL, "x5"].median()
    train["x5_imp"] = train["x5"].where(train["x5"] != SENTINEL, x5m)
    train["x5_is_sent"] = (train["x5"] == SENTINEL).astype(float)
    train["a1"] = a1_predict(train, x5m)
    train["a1_resid"] = train["target"] - train["a1"]
    return train


def quadrant_slice(train: pd.DataFrame) -> pd.DataFrame:
    q = train[(train["x4"] < 0) & (train["x8"] < 0) & ~train["x5_is_sent"].astype(bool)].copy()
    q["is_clamp"] = q["a1_resid"].abs() > CLAMP_THR
    return q


def correction_form(q: pd.DataFrame) -> None:
    """Fit several simple corrections on clamp rows to identify shape."""
    clamp = q[q["is_clamp"]]
    print(f"\n=== Correction shape on {len(clamp)} clamp rows (x4<0, x8<0) ===")
    y = clamp["a1_resid"].values

    candidates = {
        "α · x8":               clamp[["x8"]].values,
        "α · x4":               clamp[["x4"]].values,
        "α · (x4 + x8)":        clamp[["x4", "x8"]].values,   # LS finds separate coefs
        "α · |x8|":             np.c_[np.abs(clamp["x8"].values)],
        "α · x8 + β · x4":      clamp[["x8", "x4"]].values,
        "α · x8 · x4":          np.c_[clamp["x8"].values * clamp["x4"].values],
        "α · min(x4, x8)":      np.c_[np.minimum(clamp["x4"].values, clamp["x8"].values)],
        "α · max(x4, x8)":      np.c_[np.maximum(clamp["x4"].values, clamp["x8"].values)],
        "α · (x4 − x8)":        np.c_[clamp["x4"].values - clamp["x8"].values],
        "α · (x4 + x8) (sum)":  np.c_[clamp["x4"].values + clamp["x8"].values],
        "α · x8² (sign(x8))":   np.c_[clamp["x8"].values ** 2],
    }
    for name, X in candidates.items():
        m = LinearRegression().fit(X, y)
        p = m.predict(X)
        r2 = 1 - np.var(y - p) / np.var(y)
        resid_std = float(np.std(y - p))
        coefs = ", ".join(f"{c:+.2f}" for c in m.coef_)
        print(f"  {name:<25s}  R²={r2:+.3f}  resid_std={resid_std:.2f}  "
              f"coef=({coefs})  intercept={m.intercept_:+.2f}")


def score_rule(q: pd.DataFrame, mask: np.ndarray) -> dict:
    """Contingency against is_clamp."""
    is_clamp = q["is_clamp"].values
    tp = int((mask & is_clamp).sum())
    fp = int((mask & ~is_clamp).sum())
    fn = int((~mask & is_clamp).sum())
    tn = int((~mask & ~is_clamp).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sens": sens, "spec": spec, "acc": acc}


def scan_single_feature(q: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """For every feature, scan thresholds and record the best rule."""
    rows = []
    for f in features:
        vals = q[f].values
        # candidate thresholds: quantiles + integer/half-integer near range
        qs = np.unique(np.concatenate([
            np.quantile(vals, np.linspace(0.05, 0.95, 19)),
            np.arange(np.floor(vals.min() * 2) / 2, np.ceil(vals.max() * 2) / 2 + 0.5, 0.5),
        ]))
        for t in qs:
            for direction in ("<", ">"):
                mask = (vals < t) if direction == "<" else (vals > t)
                if not (20 <= mask.sum() <= len(q) - 20):
                    continue
                s = score_rule(q, mask)
                # simplicity: prefer integer thresholds
                is_int = abs(t - round(t)) < 1e-6
                rows.append({
                    "rule": f"{f} {direction} {t:.3f}",
                    "feature": f, "thr": t, "dir": direction,
                    "is_int_thr": is_int,
                    **s,
                })
    return pd.DataFrame(rows)


def scan_pairwise(q: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Scan pairwise derived features (sum, diff, product, ratio)."""
    rows = []
    OPS = [
        ("+",   lambda a, b: a + b),
        ("-",   lambda a, b: a - b),
        ("*",   lambda a, b: a * b),
        ("/",   lambda a, b: a / np.where(np.abs(b) < 0.01, np.nan, b)),
    ]
    for i, j in combinations(features, 2):
        a = q[i].values
        b = q[j].values
        for sym, op in OPS:
            v = op(a, b)
            if np.any(~np.isfinite(v)):
                continue
            qs = np.quantile(v, np.linspace(0.05, 0.95, 19))
            qs = np.unique(np.concatenate([
                qs,
                np.arange(np.floor(v.min() * 2) / 2, np.ceil(v.max() * 2) / 2 + 0.5, 0.5),
            ]))
            for t in qs:
                for direction in ("<", ">"):
                    mask = (v < t) if direction == "<" else (v > t)
                    if not (20 <= mask.sum() <= len(q) - 20):
                        continue
                    s = score_rule(q, mask)
                    is_int = abs(t - round(t)) < 1e-6
                    rows.append({
                        "rule": f"({i} {sym} {j}) {direction} {t:.3f}",
                        "feature": f"{i}{sym}{j}", "thr": t, "dir": direction,
                        "is_int_thr": is_int,
                        **s,
                    })
    return pd.DataFrame(rows)


def scan_angular(q: pd.DataFrame) -> pd.DataFrame:
    """θ-based rules and θ-combined rules."""
    rows = []
    theta = q["theta"].values
    bases = {
        "sin(θ)": np.sin(theta),
        "cos(θ)": np.cos(theta),
        "sin(2θ)": np.sin(2 * theta),
        "cos(2θ)": np.cos(2 * theta),
        "θ": theta,
        "|θ|": np.abs(theta),
        "x8·cos(θ)": q["x8"].values * np.cos(theta),
        "x8·sin(θ)": q["x8"].values * np.sin(theta),
        "x4·cos(θ)": q["x4"].values * np.cos(theta),
        "x4·sin(θ)": q["x4"].values * np.sin(theta),
    }
    for name, v in bases.items():
        if np.any(~np.isfinite(v)):
            continue
        qs = np.quantile(v, np.linspace(0.05, 0.95, 19))
        for t in qs:
            for direction in ("<", ">"):
                mask = (v < t) if direction == "<" else (v > t)
                if not (20 <= mask.sum() <= len(q) - 20):
                    continue
                s = score_rule(q, mask)
                rows.append({
                    "rule": f"{name} {direction} {t:.3f}",
                    "feature": name, "thr": t, "dir": direction,
                    "is_int_thr": False,
                    **s,
                })
    return pd.DataFrame(rows)


def report_top(results: pd.DataFrame, label: str, top: int = 10) -> None:
    score = (results["sens"] + results["spec"]) / 2
    tiebreak = results["is_int_thr"].astype(int) * 0.01
    ranked = results.assign(score=score + tiebreak).sort_values("score", ascending=False)
    print(f"\n=== Top {top} in {label} ===")
    print(ranked.head(top)[["rule", "tp", "fp", "fn", "tn", "sens", "spec", "acc"]]
          .to_string(index=False,
                     formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                                 "acc": "{:.3f}".format}))


def main() -> None:
    print("=" * 72)
    print("x8 clamp archaeology — simple-rule brute-force")
    print("=" * 72)
    train = load()
    q = quadrant_slice(train)
    print(f"Quadrant size: {len(q)}  clamp: {q['is_clamp'].sum()}  "
          f"({100*q['is_clamp'].mean():.1f}%)")

    # 1. Shape of the correction ----------------------------------------
    correction_form(q)

    # 2. Simple-rule scans ----------------------------------------------
    feats_numeric = ["x1", "x2", "x4", "x5_imp", "x6", "x7", "x8",
                     "x9", "x10", "x11", "r_xy", "theta"]
    sf = scan_single_feature(q, feats_numeric)
    sf["is_single"] = True
    report_top(sf, "single-feature thresholds")

    pw = scan_pairwise(q, feats_numeric)
    pw["is_single"] = False
    report_top(pw, "pairwise sum/diff/product/ratio thresholds")

    ag = scan_angular(q)
    report_top(ag, "angular rules (θ-based)")

    # Combine all, save ------------------------------------------------
    allres = pd.concat([sf, pw, ag], ignore_index=True)
    allres["score"] = (allres["sens"] + allres["spec"]) / 2
    allres.sort_values("score", ascending=False).to_csv(OUT / "all_rules.csv", index=False)

    # 3. Report the absolute best rules --------------------------------
    print("\n" + "=" * 72)
    print("TOP 20 across all rules (by 0.5·(sens+spec))")
    print("=" * 72)
    top = allres.sort_values("score", ascending=False).head(20)
    print(top[["rule", "tp", "fp", "fn", "tn", "sens", "spec", "acc"]]
          .to_string(index=False,
                     formatters={"sens": "{:.3f}".format,
                                 "spec": "{:.3f}".format,
                                 "acc": "{:.3f}".format}))

    # 4. Print any rule passing the 0.9/0.9 target ---------------------
    good = allres[(allres["sens"] >= TARGET_SENS) & (allres["spec"] >= TARGET_SPEC)]
    print(f"\nRules meeting sens≥{TARGET_SENS} AND spec≥{TARGET_SPEC}: {len(good)}")
    if len(good):
        print(good.sort_values("acc", ascending=False).head(20).to_string(index=False,
              formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                          "acc": "{:.3f}".format}))

    print(f"\nFull ranking written to {OUT / 'all_rules.csv'}")


if __name__ == "__main__":
    main()
