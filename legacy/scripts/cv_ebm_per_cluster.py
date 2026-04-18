"""Per-cluster EBMs — train one model per x4 cluster and combine.

Training has only TWO non-empty (x4-sign, x9-level) blocks because of
selection bias: (x4>0, x9≈5.97) and (x4<0, x9≈4.02). So at most two
per-cluster models are possible.

Strategies tested (5-fold CV on dataset.csv):

  1. Matched routing    — predict with the model matching the row's x4 sign
  2. Average (always)   — average predictions from both models on every row
  3. x9-based routing   — route by x9 level (>5 → hi, <5 → lo): each
                          per-cluster model stays in its x9-training range
  4. Full-model average — 50/50 blend of full-data EBM and matched-routing

Each per-cluster model is a standard EBM (4k/0 config, our current best).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import FEATURES_ALL, SENTINEL, SEED, preprocess  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5

BASE = dict(
    interactions=10, max_rounds=4000, min_samples_leaf=10, max_bins=128,
    smoothing_rounds=4000, interaction_smoothing_rounds=1000,
    random_state=SEED,
)


def fit_ebm(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(**BASE).fit(X, y)


def cv_per_cluster(df: pd.DataFrame):
    """Return OOF predictions for four strategies and the full-data model.
    Also returns metadata (per-fold x5 median etc.) for submission building.
    """
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_matched = np.zeros(len(df))
    oof_avg = np.zeros(len(df))
    oof_x9route = np.zeros(len(df))
    oof_full = np.zeros(len(df))

    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        X_tr = preprocess(sub_tr, FEATURES_ALL, x5m)
        X_va = preprocess(sub_va, FEATURES_ALL, x5m)

        hi_mask = (sub_tr["x4"] > 0).values
        lo_mask = (sub_tr["x4"] < 0).values

        ebm_hi = fit_ebm(X_tr[hi_mask].reset_index(drop=True),
                          sub_tr.loc[hi_mask, "target"].values)
        ebm_lo = fit_ebm(X_tr[lo_mask].reset_index(drop=True),
                          sub_tr.loc[lo_mask, "target"].values)
        ebm_full = fit_ebm(X_tr, sub_tr["target"].values)

        p_hi = ebm_hi.predict(X_va)
        p_lo = ebm_lo.predict(X_va)
        p_full = ebm_full.predict(X_va)

        # matched routing: by x4 sign
        va_x4_hi = (sub_va["x4"].values > 0)
        oof_matched[va] = np.where(va_x4_hi, p_hi, p_lo)

        # always-average
        oof_avg[va] = 0.5 * (p_hi + p_lo)

        # x9-based routing: x9 > 5 → hi model, else → lo
        va_x9_hi = (sub_va["x9"].values > 5.0)
        oof_x9route[va] = np.where(va_x9_hi, p_hi, p_lo)

        # full-data reference
        oof_full[va] = p_full

        print(f"  fold {fold + 1}/{N_SPLITS} done in {time.time()-t0:.0f}s "
              f"(hi n={hi_mask.sum()}, lo n={lo_mask.sum()})")

    def score(oof):
        return (float(np.mean(np.abs(oof - y))),
                float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))))

    return {
        "matched":   (score(oof_matched)   + (oof_matched,)),
        "avg":       (score(oof_avg)       + (oof_avg,)),
        "x9_route":  (score(oof_x9route)   + (oof_x9route,)),
        "full":      (score(oof_full)      + (oof_full,)),
        "matched_plus_full_50": (score(0.5 * (oof_matched + oof_full)) + (0.5*(oof_matched+oof_full),)),
        "avg_plus_full_50":     (score(0.5 * (oof_avg + oof_full))     + (0.5*(oof_avg+oof_full),)),
    }


def build_per_cluster(df: pd.DataFrame, test: pd.DataFrame):
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(df, FEATURES_ALL, x5m)
    X_te = preprocess(test, FEATURES_ALL, x5m)

    hi_mask = (df["x4"] > 0).values
    lo_mask = (df["x4"] < 0).values

    print("  fitting EBM on x4>0 subset...")
    ebm_hi = fit_ebm(X_tr[hi_mask].reset_index(drop=True),
                      df.loc[hi_mask, "target"].values)
    print("  fitting EBM on x4<0 subset...")
    ebm_lo = fit_ebm(X_tr[lo_mask].reset_index(drop=True),
                      df.loc[lo_mask, "target"].values)
    print("  fitting EBM on full data...")
    ebm_full = fit_ebm(X_tr, df["target"].values)

    p_hi = ebm_hi.predict(X_te)
    p_lo = ebm_lo.predict(X_te)
    p_full = ebm_full.predict(X_te)
    te_x4_hi = (test["x4"].values > 0)
    te_x9_hi = (test["x9"].values > 5.0)

    variants = {
        "percluster_matched":    np.where(te_x4_hi, p_hi, p_lo),
        "percluster_avg":        0.5 * (p_hi + p_lo),
        "percluster_x9route":    np.where(te_x9_hi, p_hi, p_lo),
        "percluster_matched_plus_full_50": 0.5 * (np.where(te_x4_hi, p_hi, p_lo) + p_full),
        "percluster_avg_plus_full_50":     0.5 * (0.5*(p_hi + p_lo) + p_full),
    }
    for name, preds in variants.items():
        pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
            SUBS / f"submission_ebm_{name}.csv", index=False)
        print(f"  wrote submission_ebm_{name}.csv  mean={preds.mean():+.3f}  "
              f"range=[{preds.min():+.2f}, {preds.max():+.2f}]")


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    print(f"train: n={len(df)}  hi={(df['x4']>0).sum()}  lo={(df['x4']<0).sum()}")
    print(f"test:  n={len(test)}  hi={(test['x4']>0).sum()}  lo={(test['x4']<0).sum()}  "
          f"x9>5 n={(test['x9']>5).sum()}  x9<5 n={(test['x9']<=5).sum()}")

    print("\n" + "=" * 75)
    print("5-fold CV (each fold trains 2 per-cluster EBMs + 1 full EBM)")
    print("=" * 75)
    results = cv_per_cluster(df)

    print("\n" + "=" * 75)
    print(f"{'strategy':<40s}  {'overall':>8s}  {'non-sent':>8s}")
    print("=" * 75)
    for name in ["full", "matched", "avg", "x9_route",
                 "matched_plus_full_50", "avg_plus_full_50"]:
        (m, mn), _ = results[name][:2], results[name][2]
        m, mn = results[name][0], results[name][1]
        print(f"{name:<40s}  {m:8.3f}  {mn:8.3f}")

    print("\n" + "=" * 75)
    print("Building all per-cluster variants on full dataset")
    print("=" * 75)
    build_per_cluster(df, test)


if __name__ == "__main__":
    main()
