"""Corrected A1 — leveraging the clamp-is-training-artifact finding.

On training rows ≥ 100 (and all rows in non-(x4<0,x8<0) quadrants),
A1's formula fits target EXACTLY:

    target = -100·x1² + 10·cos(5π·x2) + 15·x4 + 15·x8 - 8·x5
           + x10·x11 - 4·x9 - 25·zaragoza + 20·1{x4>0} + 92.5

A1 scored LB 10.80 because on test (x4 ⊥ x9) the -4·x9 term is pure
noise and the +20 step is partly x9-contamination. Combined with idea
#1 (pooled-feature rediscovery) and idea #2 (step = +11, not +20):

Corrected-A1 hypotheses for the test DGP, each using only training
rows ≥ 100 (dropping the 86 clamp artifacts) to fit the intercept:

  V1 "drop x9":     A1 - 4·x9 removed; step +20 kept; intercept refit
  V2 "step11":      A1 with step changed to +11; -4·x9 removed;
                    intercept refit
  V3 "step11_free": V2 but allow linear fit of all coefs on rows ≥ 100
                    with β_x9 constrained to 0
  V4 "pure linear": no step at all; linear x4; β_x9 = 0

All four use the SAME integer coefficient set from A1 for x1², cos,
x5, x8, x10·x11, city. Only step and x9 differ.

Side check: since rows 0-99 are training-only artefacts, drop them
from the fit. This gives unbiased intercepts for the test DGP.

Builds six submissions for LB testing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUBS = REPO / "submissions"
SENTINEL = 999.0


def load():
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        is_sent = (df["x5"] == SENTINEL).astype(float)
        df["is_sent"] = is_sent
        df["x5_imp"] = df["x5"].where(df["x5"] != SENTINEL, np.nan)
    x5m = float(train["x5_imp"].median())
    for df in (train, test):
        df["x5_imp"] = df["x5_imp"].fillna(x5m)
        df["city_code"] = (df["City"] == "Zaragoza").astype(float)
        df["step"] = (df["x4"] > 0).astype(float)
        df["x1sq"] = df["x1"] ** 2
        df["cos5pi"] = np.cos(5 * np.pi * df["x2"])
        df["x10x11"] = df["x10"] * df["x11"]
    return train, test, x5m


def a1_without_x9_fixed(df: pd.DataFrame, step_coef: float, intercept: float) -> np.ndarray:
    """A1 minus the -4·x9 term. Integer coefs preserved, step and intercept tunable."""
    return (
        -100 * df["x1sq"]
        + 10 * df["cos5pi"]
        + 15 * df["x4"]
        + 15 * df["x8"]
        - 8 * df["x5_imp"]
        + df["x10x11"]
        - 25 * df["city_code"]
        + step_coef * df["step"]
        + intercept
    )


def fit_intercept(df_fit: pd.DataFrame, step_coef: float) -> float:
    """Match training target mean (on selected rows) given fixed non-intercept coefs."""
    pred_no_intercept = a1_without_x9_fixed(df_fit, step_coef, 0.0)
    return float((df_fit["target"] - pred_no_intercept).mean())


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def cv_constrained(df: pd.DataFrame, step_coef: float, drop_clamp: bool) -> tuple[float, float]:
    """5-fold CV with integer coefs fixed, intercept refit per fold."""
    from sklearn.model_selection import KFold
    use = df[df["id"] >= 100] if drop_clamp else df
    y = use["target"].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errs, nsents = [], []
    for tr, va in kf.split(use):
        trdf = use.iloc[tr]
        vadf = use.iloc[va]
        b = fit_intercept(trdf, step_coef)
        p = a1_without_x9_fixed(vadf, step_coef, b).values
        errs.append(mae(p, vadf["target"].values))
        nsents.append(mae(p[~vadf["is_sent"].astype(bool).values],
                          vadf["target"].values[~vadf["is_sent"].astype(bool).values]))
    return float(np.mean(errs)), float(np.mean(nsents))


def cv_a1_as_is(df: pd.DataFrame, drop_clamp: bool) -> tuple[float, float]:
    """Sanity check: A1 unchanged on train."""
    use = df[df["id"] >= 100] if drop_clamp else df
    p = (
        -100 * use["x1sq"]
        + 10 * use["cos5pi"]
        + 15 * use["x4"]
        + 15 * use["x8"]
        - 8 * use["x5_imp"]
        + use["x10x11"]
        - 25 * use["city_code"]
        + 20 * use["step"]
        - 4 * use["x9"]
        + 92.5
    ).values
    y = use["target"].values
    e = np.abs(p - y)
    ns = e[~use["is_sent"].astype(bool).values]
    return float(e.mean()), float(ns.mean())


def main():
    train, test, x5m = load()
    y = train["target"].values

    print("=" * 74)
    print("A1 ground-truth check: A1 fits rows id≥100 exactly")
    print("=" * 74)
    mae_all, ns_all = cv_a1_as_is(train, drop_clamp=False)
    mae_100, ns_100 = cv_a1_as_is(train, drop_clamp=True)
    print(f"  A1 on all rows:       MAE={mae_all:.4f}  non-sent={ns_all:.4f}")
    print(f"  A1 on rows id≥100:    MAE={mae_100:.4f}  non-sent={ns_100:.4f}")
    print("  (The ~0 non-sent MAE confirms A1 is the TRAINING DGP for id≥100.)")

    # Explore corrections. CV on rows ≥ 100 (clean subset).
    print("\n" + "=" * 74)
    print("Corrected-A1 variants  (fit on id≥100, intercept-only refit)")
    print("=" * 74)
    variants = {
        "V1 step=+20 (drop x9)":    (20, True),
        "V2 step=+11 (drop x9)":    (11, True),
        "V3 step=+15 (drop x9)":    (15, True),
        "V4 step=+10 (drop x9)":    (10, True),
        "V5 step=+12 (drop x9)":    (12, True),
        "V6 step=+8  (drop x9)":    (8,  True),
        "V7 step=0   (pure linear)":(0,  True),
    }
    for name, (sc, drop) in variants.items():
        m, ns = cv_constrained(train, sc, drop)
        b = fit_intercept(train[train["id"] >= 100] if drop else train, sc)
        print(f"  {name:<32s} CV MAE={m:.3f}  non-sent={ns:.3f}  intercept={b:+.2f}")

    # Build submissions --------------------------------------------------
    print("\n" + "=" * 74)
    print("Building submissions")
    print("=" * 74)
    clean = train[train["id"] >= 100]
    to_write = {}
    for step_coef, tag in [(20, "step20"), (11, "step11"), (10, "step10"),
                            (12, "step12"), (8, "step8"), (0, "nostep")]:
        b = fit_intercept(clean, step_coef)
        pred = a1_without_x9_fixed(test, step_coef, b).values
        fname = f"submission_a1_nox9_{tag}.csv"
        to_write[fname] = pred
    # Also build A1-as-is on test for reference (already exists as subm_A1 in legacy)
    for name, p in to_write.items():
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(SUBS / name, index=False)
        print(f"  wrote {name}  mean={p.mean():+.2f}  std={p.std():.2f}")

    # Blend with cross_LE for safety -------------------------------------
    print("\n" + "=" * 74)
    print("Blends of corrected-A1 × cross_LE (for robustness)")
    print("=" * 74)
    cross_LE_path = SUBS / "submission_ensemble_cross_LE.csv"
    if cross_LE_path.exists():
        cross_LE = pd.read_csv(cross_LE_path).set_index("id")["target"]
        cross_LE = cross_LE.reindex(test["id"].values).values
        for tag in ["step11", "step10", "step12", "step20"]:
            a1_corr = to_write[f"submission_a1_nox9_{tag}.csv"]
            for w in [0.3, 0.5, 0.7]:
                blend = w * a1_corr + (1 - w) * cross_LE
                fname = f"submission_blend_a1{tag}_crossLE_{int(w*100)}.csv"
                pd.DataFrame({"id": test["id"], "target": blend}).to_csv(SUBS / fname, index=False)
                print(f"  {fname}  mean={blend.mean():+.2f}  std={blend.std():.2f}")


if __name__ == "__main__":
    main()
