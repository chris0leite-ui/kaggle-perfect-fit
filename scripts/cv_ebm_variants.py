"""5-fold CV grid for EBM variants — drop-x9, smoothing, regularization.

Each variant is fit 5 times (one per fold) on dataset.csv with
KFold(shuffle=True, random_state=42). Reports overall, non-sentinel and
sentinel MAE and saves OOF predictions for later ensembling.

Variants:
  1. baseline      — our Round 2 tuned params (current submission)
  2. no_x9         — same params but drop x9 from features
  3. heavy_smooth  — smoothing_rounds=2000, interaction_smoothing_rounds=500
  4. heavy_smooth_no_x9
  5. high_reg      — reg_alpha=1.0, reg_lambda=1.0, min_samples_leaf=30
  6. more_inter    — interactions=20, max_interaction_bins=64
  7. fewer_inter   — interactions=5
  8. default       — all EBM defaults (reference)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0
SEED = 42
N_SPLITS = 5

FEATURES_ALL = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]
FEATURES_NO_X9 = ["x1", "x2", "x4", "x5", "x8", "x10", "x11"]


def preprocess(df: pd.DataFrame, features: list[str], x5_median: float) -> pd.DataFrame:
    out = df[features].copy()
    out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
    out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
    out["city"] = (df["City"] == "Zaragoza").astype(float)
    return out


def fit_variant(tr: pd.DataFrame, va: pd.DataFrame, features: list[str],
                ebm_kwargs: dict) -> np.ndarray:
    from interpret.glassbox import ExplainableBoostingRegressor
    x5_median = float(tr.loc[tr["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(tr, features, x5_median)
    X_va = preprocess(va, features, x5_median)
    model = ExplainableBoostingRegressor(random_state=SEED, **ebm_kwargs)
    model.fit(X_tr, tr["target"].values)
    return model.predict(X_va)


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


VARIANTS = [
    {
        "name": "baseline",
        "features": FEATURES_ALL,
        "kwargs": dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128),
    },
    {
        "name": "no_x9",
        "features": FEATURES_NO_X9,
        "kwargs": dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128),
    },
    {
        "name": "heavy_smooth",
        "features": FEATURES_ALL,
        "kwargs": dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128,
                       smoothing_rounds=2000, interaction_smoothing_rounds=500),
    },
    {
        "name": "heavy_smooth_no_x9",
        "features": FEATURES_NO_X9,
        "kwargs": dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128,
                       smoothing_rounds=2000, interaction_smoothing_rounds=500),
    },
    {
        "name": "high_reg",
        "features": FEATURES_ALL,
        "kwargs": dict(interactions=10, max_rounds=2000, min_samples_leaf=30, max_bins=128,
                       reg_alpha=1.0, reg_lambda=1.0),
    },
    {
        "name": "more_inter",
        "features": FEATURES_ALL,
        "kwargs": dict(interactions=20, max_rounds=2000, min_samples_leaf=10, max_bins=128,
                       max_interaction_bins=64),
    },
    {
        "name": "fewer_inter",
        "features": FEATURES_ALL,
        "kwargs": dict(interactions=5, max_rounds=2000, min_samples_leaf=10, max_bins=128),
    },
    {
        "name": "default",
        "features": FEATURES_ALL,
        "kwargs": dict(),  # all defaults
    },
]


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    is_sent = (df["x5"] == SENTINEL).values
    y = df["target"].values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_oof = {}
    rows = []

    for v in VARIANTS:
        t0 = time.time()
        oof = np.zeros(len(df))
        per_fold = []
        for k, (tr_idx, va_idx) in enumerate(kf.split(df)):
            tr = df.iloc[tr_idx].reset_index(drop=True)
            va = df.iloc[va_idx].reset_index(drop=True)
            oof[va_idx] = fit_variant(tr, va, v["features"], v["kwargs"])
            per_fold.append(mae(oof[va_idx], va["target"].values))
        t = time.time() - t0
        overall = mae(oof, y)
        ns = mae(oof[~is_sent], y[~is_sent])
        sm = mae(oof[is_sent], y[is_sent])
        rows.append({
            "variant": v["name"], "features": "all" if "x9" in v["features"] else "no_x9",
            "overall": overall, "non_sentinel": ns, "sentinel": sm,
            "fold_min": min(per_fold), "fold_max": max(per_fold), "seconds": t,
        })
        all_oof[v["name"]] = oof
        print(f"  {v['name']:22s}  overall={overall:.3f}  non-sent={ns:.3f}  "
              f"sent={sm:.3f}  [{t:.0f}s]  folds={[round(x,3) for x in per_fold]}")

    results = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)
    print("\n" + "=" * 78)
    print("EBM variants ranked by CV MAE:")
    print("=" * 78)
    print(results.to_string(index=False))
    results.to_csv(OUT / "cv_ebm_variants.csv", index=False)
    oof_df = pd.DataFrame(all_oof)
    oof_df["target"] = y
    oof_df.to_csv(OUT / "cv_ebm_variants_oof.csv", index=False)


if __name__ == "__main__":
    main()
