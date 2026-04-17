"""Push EBM smoothing further — heavy_smooth scored LB 4.9 (from 5.66
baseline), with the CV→LB gain ratio indicating smoothing directly
regularises training-specific fit that doesn't transfer to test.
Try stronger settings to see whether the trend continues.

Variants (5-fold CV + submission):
  1. smoothing_rounds=2000, inter=500   — current LB-best (4.9)
  2. smoothing_rounds=4000, inter=1000  — 2x
  3. smoothing_rounds=6000, inter=2000  — 3x
  4. smoothing_rounds=4000, inter=1000, min_samples_leaf=20 — plus leaf reg
  5. smoothing_rounds=2000, inter=500, max_bins=64  — coarser bins
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


def fit_ebm(X, y, **extra_kwargs):
    from interpret.glassbox import ExplainableBoostingRegressor
    kw = dict(interactions=10, max_rounds=2000, min_samples_leaf=10,
              max_bins=128, random_state=SEED)
    kw.update(extra_kwargs)
    return ExplainableBoostingRegressor(**kw).fit(X, y)


def cv_variant(df: pd.DataFrame, name: str, **ebm_kwargs) -> tuple[float, float, float]:
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
        oof[va] = fit_ebm(X_tr, sub_tr["target"].values, **ebm_kwargs).predict(X_va)
    return (float(np.mean(np.abs(oof - y))),
            float(np.mean(np.abs(oof[~is_sent] - y[~is_sent]))),
            float(np.mean(np.abs(oof[is_sent] - y[is_sent]))))


def build_submission(df, test, name, **ebm_kwargs):
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(df, FEATURES_ALL, x5m)
    X_te = preprocess(test, FEATURES_ALL, x5m)
    ebm = fit_ebm(X_tr, df["target"].values, **ebm_kwargs)
    preds = ebm.predict(X_te)
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
        SUBS / f"submission_{name}.csv", index=False)
    print(f"  wrote submission_{name}.csv  mean={preds.mean():+.3f}  "
          f"range=[{preds.min():+.2f}, {preds.max():+.2f}]")


VARIANTS = [
    ("heavy_smooth_ref",
     dict(smoothing_rounds=2000, interaction_smoothing_rounds=500)),
    ("extra_smooth_4k",
     dict(smoothing_rounds=4000, interaction_smoothing_rounds=1000)),
    ("extra_smooth_6k",
     dict(smoothing_rounds=6000, interaction_smoothing_rounds=2000)),
    ("extra_smooth_leaf20",
     dict(smoothing_rounds=4000, interaction_smoothing_rounds=1000,
          min_samples_leaf=20)),
    ("extra_smooth_bins64",
     dict(smoothing_rounds=2000, interaction_smoothing_rounds=500,
          max_bins=64)),
    ("extra_smooth_bins64_4k",
     dict(smoothing_rounds=4000, interaction_smoothing_rounds=1000,
          max_bins=64)),
]


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    print("=" * 80)
    print(f"{'variant':<25s}  {'overall':>8s}  {'non-sent':>8s}  {'sent':>7s}   time")
    print("=" * 80)
    rows = []
    for name, kw in VARIANTS:
        t0 = time.time()
        m, mn, ms = cv_variant(df, name, **kw)
        rows.append({"variant": name, "overall": m, "non_sent": mn, "sent": ms})
        print(f"{name:<25s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  [{time.time()-t0:.0f}s]")

    pd.DataFrame(rows).to_csv(REPO / "plots" / "cv_ebm_extra_smooth.csv", index=False)

    print("\n" + "=" * 80)
    print("Building submissions (skipping reference variant — already exists)")
    print("=" * 80)
    for name, kw in VARIANTS[1:]:  # skip the heavy_smooth reference
        build_submission(df, test, name, **kw)


if __name__ == "__main__":
    main()
