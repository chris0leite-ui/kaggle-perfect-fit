"""Three moves to try breaking the EBM LB plateau at 4.9:

  1. Monotone constraints  — force +/- signs on linear features (x4+, x5-,
     x8+, city-). EBM wiggle within its smoothed shapes might be
     training-specific; monotonicity caps it to a valid prior.
  2. interactions=0        — purely additive EBM. If interaction
     detection is picking up training-specific spurious pairs
     (beyond the real x10·x11), dropping it could help LB.
  3. 20-seed bag           — our 5-seed bag barely moved CV; maybe
     variance reduction at 20 seeds finally shows up.

All three use the current CV-best base (4k smooth, 0 post, defaults).
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


def fit_ebm(X, y, **overrides):
    from interpret.glassbox import ExplainableBoostingRegressor
    kw = {**BASE, **overrides}
    return ExplainableBoostingRegressor(**kw).fit(X, y)


def cv_variant(df, **overrides):
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        X_tr = preprocess(sub_tr, FEATURES_ALL, x5m)
        X_va = preprocess(sub_va, FEATURES_ALL, x5m)
        oof[va] = fit_ebm(X_tr, sub_tr["target"].values, **overrides).predict(X_va)
    return (float(np.mean(np.abs(oof - y))),
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))))


def cv_bagged(df, seeds, **overrides):
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        X_tr = preprocess(sub_tr, FEATURES_ALL, x5m)
        X_va = preprocess(sub_va, FEATURES_ALL, x5m)
        preds = np.zeros(len(va))
        for s in seeds:
            preds += fit_ebm(X_tr, sub_tr["target"].values, random_state=s, **overrides).predict(X_va)
        oof[va] = preds / len(seeds)
    return (float(np.mean(np.abs(oof - y))),
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))))


def monotone_kwargs(feature_cols):
    """Return monotone_constraints based on known signs (x4+, x5-, x8+, city-).
    0 = no constraint, +1 = increasing, -1 = decreasing."""
    sign = {"x4": +1, "x5": -1, "x8": +1, "city": -1}
    return [sign.get(c, 0) for c in feature_cols]


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    sample_cols = preprocess(df.head(5), FEATURES_ALL, x5m).columns.tolist()
    mono = monotone_kwargs(sample_cols)
    print(f"feature columns: {sample_cols}")
    print(f"monotone mask:   {mono}")

    print("=" * 80)
    print(f"{'variant':<35s}  {'overall':>8s}  {'non-sent':>8s}  time")
    print("=" * 80)
    results = []

    t0 = time.time()
    m, mn = cv_variant(df)
    print(f"{'ref: 4k/0 (CV best, LB 4.9)':<35s}  {m:8.3f}  {mn:8.3f}  [{time.time()-t0:.0f}s]")
    results.append(("ref", m, mn, {}))

    t0 = time.time()
    m, mn = cv_variant(df, monotone_constraints=mono)
    print(f"{'+ monotone constraints':<35s}  {m:8.3f}  {mn:8.3f}  [{time.time()-t0:.0f}s]")
    results.append(("monotone", m, mn, {"monotone_constraints": mono}))

    t0 = time.time()
    m, mn = cv_variant(df, interactions=0)
    print(f"{'interactions=0 (purely additive)':<35s}  {m:8.3f}  {mn:8.3f}  [{time.time()-t0:.0f}s]")
    results.append(("no_inter", m, mn, {"interactions": 0}))

    t0 = time.time()
    m, mn = cv_variant(df, interactions=0, monotone_constraints=mono)
    print(f"{'no_inter + monotone':<35s}  {m:8.3f}  {mn:8.3f}  [{time.time()-t0:.0f}s]")
    results.append(("no_inter_mono", m, mn,
                    {"interactions": 0, "monotone_constraints": mono}))

    t0 = time.time()
    seeds20 = list(range(20))
    m, mn = cv_bagged(df, seeds20)
    print(f"{'20-seed bag (ref config)':<35s}  {m:8.3f}  {mn:8.3f}  [{time.time()-t0:.0f}s]")
    results.append(("bag20", m, mn, {"_bag": seeds20}))

    print("\n" + "=" * 80)
    print("Building submissions (top 2 by CV, plus all plateau-breakers)")
    print("=" * 80)

    def build(name, bag=None, **kw):
        X_tr = preprocess(df, FEATURES_ALL, x5m)
        X_te = preprocess(test, FEATURES_ALL, x5m)
        if bag:
            preds = np.zeros(len(test))
            for s in bag:
                preds += fit_ebm(X_tr, df["target"].values, random_state=s, **kw).predict(X_te)
            preds /= len(bag)
        else:
            preds = fit_ebm(X_tr, df["target"].values, **kw).predict(X_te)
        pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
            SUBS / f"submission_{name}.csv", index=False)
        print(f"  wrote submission_{name}.csv  mean={preds.mean():+.3f}  "
              f"range=[{preds.min():+.2f}, {preds.max():+.2f}]")

    build("ebm_monotone", **{"monotone_constraints": mono})
    build("ebm_no_interactions", interactions=0)
    build("ebm_no_inter_monotone", interactions=0,
          **{"monotone_constraints": mono})
    build("ebm_bag20", bag=seeds20)


if __name__ == "__main__":
    main()
