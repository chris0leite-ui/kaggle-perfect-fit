"""Separate smoothing rounds from post-smoothing refinement rounds.

EBM docs clarify: smoothing_rounds = number of INITIAL smoothed rounds;
max_rounds = TOTAL rounds. Post-smoothing rounds = max_rounds − smoothing_rounds.

Our previous "best" configs had max_rounds == smoothing_rounds (all rounds
smoothed, zero refinement). The LB-4.9 heavy_smooth had 500 smoothed +
1500 refinement. Test whether refinement on top of smoothing helps.

Grid:
  smoothing_rounds ∈ {500, 2000, 4000}
  post-smoothing rounds (refinement) ∈ {0, 1500, 4000}

Total = max_rounds = smoothing_rounds + post_rounds.

Also disables early_stopping so max_rounds is actually reached.
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


def fit_ebm(X, y, smoothing_rounds, post_rounds,
            interaction_smoothing_rounds=None,
            early_stopping=True):
    from interpret.glassbox import ExplainableBoostingRegressor
    max_rounds = smoothing_rounds + post_rounds
    if interaction_smoothing_rounds is None:
        interaction_smoothing_rounds = max(smoothing_rounds // 4, 100)
    kw = dict(
        interactions=10,
        max_rounds=max_rounds,
        min_samples_leaf=10,
        max_bins=128,
        smoothing_rounds=smoothing_rounds,
        interaction_smoothing_rounds=interaction_smoothing_rounds,
        random_state=SEED,
    )
    if not early_stopping:
        kw["early_stopping_rounds"] = 0
    return ExplainableBoostingRegressor(**kw).fit(X, y)


def cv_variant(df, smoothing_rounds, post_rounds,
               interaction_smoothing_rounds=None, early_stopping=True):
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
        model = fit_ebm(X_tr, sub_tr["target"].values,
                        smoothing_rounds, post_rounds,
                        interaction_smoothing_rounds, early_stopping)
        oof[va] = model.predict(X_va)
    return (float(np.mean(np.abs(oof - y))),
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))),
            float(np.mean(np.abs(oof[is_sent] - y[is_sent]))))


def build(df, test, name, smoothing_rounds, post_rounds,
          interaction_smoothing_rounds=None, early_stopping=True):
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(df, FEATURES_ALL, x5m)
    X_te = preprocess(test, FEATURES_ALL, x5m)
    model = fit_ebm(X_tr, df["target"].values, smoothing_rounds, post_rounds,
                    interaction_smoothing_rounds, early_stopping)
    preds = model.predict(X_te)
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
        SUBS / f"submission_{name}.csv", index=False
    )
    print(f"  wrote submission_{name}.csv  mean={preds.mean():+.3f}  "
          f"range=[{preds.min():+.2f}, {preds.max():+.2f}]")


GRID = [
    # (name, smoothing_rounds, post_rounds, early_stopping)
    ("smooth500_post1500_es",   500,  1500, True),   # heavy_smooth LB=4.9 baseline
    ("smooth500_post1500_noes", 500,  1500, False),  # same but forced full rounds
    ("smooth2000_post0",        2000, 0,    True),
    ("smooth2000_post1500",     2000, 1500, True),
    ("smooth2000_post4000",     2000, 4000, True),
    ("smooth4000_post0",        4000, 0,    True),   # max_rounds_4k baseline
    ("smooth4000_post1500",     4000, 1500, True),
    ("smooth4000_post4000",     4000, 4000, True),
    ("smooth6000_post0",        6000, 0,    True),
    ("smooth6000_post2000",     6000, 2000, True),
]


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    print("=" * 95)
    print(f"{'variant':<30s}  {'smooth':>7s}  {'post':>5s}  {'overall':>8s}  {'non-sent':>9s}  time")
    print("=" * 95)
    results = []
    for name, sr, pr, es in GRID:
        t0 = time.time()
        m, mn, ms = cv_variant(df, sr, pr, early_stopping=es)
        results.append({"name": name, "smooth": sr, "post": pr, "es": es,
                        "overall": m, "non_sent": mn, "sent": ms})
        print(f"{name:<30s}  {sr:>7d}  {pr:>5d}  {m:8.3f}  {mn:9.3f}  [{time.time()-t0:.0f}s]")

    pd.DataFrame(results).to_csv(REPO / "plots" / "cv_ebm_smooth_vs_refine.csv",
                                  index=False)

    print("\n" + "=" * 95)
    print("Top-3 variants by CV:")
    top3 = sorted(results, key=lambda r: r["overall"])[:3]
    for r in top3:
        print(f"  {r['name']:<30s}  CV {r['overall']:.3f}")
    print("Building submissions")
    print("=" * 95)
    for r in top3:
        build(df, test, f"ebm_{r['name']}", r["smooth"], r["post"],
              early_stopping=r["es"])


if __name__ == "__main__":
    main()
