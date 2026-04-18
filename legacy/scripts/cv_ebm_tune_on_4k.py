"""Hyperparameter tuning on top of the 4k/1k smoothing baseline.

The 4k smoothing variant (CV 3.053) is the current best on CV and the
LB-best ancestor (heavy_smooth at CV 3.081 → LB 4.9) demonstrated a 25x
CV→LB gain multiplier.

Tune on top of the 4k/1k setting:
  - interactions:     5, 10 (default), 15, 20
  - learning_rate:    0.005, 0.01 (default), 0.02
  - max_rounds:       2000 (default), 4000
  - max_leaves:       2, 3 (default)
  - bagging:          5 seeds, averaged

Only test promising axes one at a time (coord descent) to keep runtime
under ~30 min. Uses 5-fold CV and writes submissions for the top 3.
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
    interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128,
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
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))),
            float(np.mean(np.abs(oof[is_sent] - y[is_sent]))))


def cv_bagged(df, seeds: list[int], **overrides):
    """Average predictions from multiple seeds. OOF computed per fold."""
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
            kw = {**overrides, "random_state": s}
            preds += fit_ebm(X_tr, sub_tr["target"].values, **kw).predict(X_va)
        oof[va] = preds / len(seeds)
    return (float(np.mean(np.abs(oof - y))),
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))),
            float(np.mean(np.abs(oof[is_sent] - y[is_sent]))))


def build_submission(df, test, name, bagged_seeds=None, **overrides):
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(df, FEATURES_ALL, x5m)
    X_te = preprocess(test, FEATURES_ALL, x5m)
    if bagged_seeds:
        preds = np.zeros(len(test))
        for s in bagged_seeds:
            kw = {**overrides, "random_state": s}
            preds += fit_ebm(X_tr, df["target"].values, **kw).predict(X_te)
        preds /= len(bagged_seeds)
    else:
        preds = fit_ebm(X_tr, df["target"].values, **overrides).predict(X_te)
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
        SUBS / f"submission_{name}.csv", index=False)
    print(f"  wrote submission_{name}.csv  mean={preds.mean():+.3f}  "
          f"range=[{preds.min():+.2f}, {preds.max():+.2f}]")


VARIANTS: list[tuple[str, dict]] = [
    ("ref_4k",               {}),                              # base
    ("inter5",               {"interactions": 5}),
    ("inter15",              {"interactions": 15}),
    ("inter20",              {"interactions": 20}),
    ("lr_slow",              {"learning_rate": 0.005}),
    ("lr_fast",              {"learning_rate": 0.02}),
    ("max_rounds_4k",        {"max_rounds": 4000}),
    ("leaf_20",              {"min_samples_leaf": 20}),
    ("leaf_5",               {"min_samples_leaf": 5}),
]


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    print("=" * 80)
    print(f"{'variant':<25s}  {'overall':>8s}  {'non-sent':>8s}  {'sent':>7s}   time")
    print("=" * 80)
    results = []
    for name, kw in VARIANTS:
        t0 = time.time()
        m, mn, ms = cv_variant(df, **kw)
        results.append({"variant": name, "overall": m, "non_sent": mn, "sent": ms,
                        "kwargs": kw})
        print(f"{name:<25s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  [{time.time()-t0:.0f}s]")

    # Bagged variant on top of the best setting
    best = min(results, key=lambda r: r["overall"])
    print(f"\nBest single-seed: {best['variant']} @ {best['overall']:.3f}")
    t0 = time.time()
    bag_seeds = [1, 7, 42, 123, 2024]
    m, mn, ms = cv_bagged(df, bag_seeds, **best["kwargs"])
    print(f"{'bagged_5seeds_' + best['variant']:<25s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  "
          f"[{time.time()-t0:.0f}s]")
    results.append({"variant": f"bagged_5seeds_{best['variant']}",
                    "overall": m, "non_sent": mn, "sent": ms,
                    "kwargs": {**best["kwargs"], "bagged_seeds": bag_seeds}})

    pd.DataFrame(results).to_csv(REPO / "plots" / "cv_ebm_tune_on_4k.csv", index=False)

    # Build submissions for the top-3 CV variants
    top3 = sorted(results, key=lambda r: r["overall"])[:3]
    print("\n" + "=" * 80)
    print(f"Top-3 CV: {[(r['variant'], round(r['overall'], 3)) for r in top3]}")
    print("Building submissions for each")
    print("=" * 80)
    for r in top3:
        kw = {k: v for k, v in r["kwargs"].items() if k != "bagged_seeds"}
        bagged = r["kwargs"].get("bagged_seeds")
        build_submission(df, test, f"ebm_tune_{r['variant']}",
                         bagged_seeds=bagged, **kw)


if __name__ == "__main__":
    main()
