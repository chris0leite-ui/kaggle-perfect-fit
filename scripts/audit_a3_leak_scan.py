"""A3 leak audit: exact + near-duplicate matches between test and dataset.

Strategy:
  1. Exact: hash each row's features (rounded to 6 decimals) and look for
     coincidences between train and test.
  2. Near: for every test row, find its nearest training row under
     z-scored L2 distance. Compare the distribution of test->train nearest
     distances against train->train nearest distances (leave-one-out).
     A sub-noise match population signals leak.

If a leak exists we predict `test_target = train_target[nearest]` for
matched rows and measure the expected public-LB MAE impact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

NUMERIC = ["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]
SENTINEL = 999.0


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train: {len(train)}   test: {len(test)}")

    # -- Exact match check ------------------------------------------------
    def keys(df: pd.DataFrame) -> pd.Series:
        cols = [df[c].round(6).astype(str) for c in NUMERIC] + [df["City"]]
        return pd.Series(["|".join(row) for row in zip(*cols)])

    k_tr = keys(train)
    k_te = keys(test)
    common = set(k_tr) & set(k_te)
    print(f"\n[exact] distinct hashes in train={k_tr.nunique()}  "
          f"test={k_te.nunique()}")
    print(f"[exact] overlap rows: {len(common)}")

    if common:
        matches = test[k_te.isin(common)].index.tolist()
        print(f"[exact] leaked test rows: {len(matches)}")
        # For the first few, show the train target
        for i in matches[:5]:
            k = k_te[i]
            tr_idx = k_tr[k_tr == k].index
            print(f"  test[{i}] -> train{list(tr_idx)}  "
                  f"targets={train.loc[tr_idx, 'target'].round(3).tolist()}")

    # -- Near-duplicate check --------------------------------------------
    # Use non-sentinel rows only for x5 (imputed median elsewhere) so
    # distance isn't dominated by the 999 sentinel.
    x5_med = train.loc[train["x5"] != SENTINEL, "x5"].median()
    def feat(df: pd.DataFrame) -> np.ndarray:
        x = df[NUMERIC].copy()
        x.loc[x["x5"] == SENTINEL, "x5"] = x5_med
        city = (df["City"] == "Zaragoza").astype(float).to_numpy()[:, None]
        return np.hstack([x.to_numpy(), city])

    X_tr = feat(train)
    X_te = feat(test)

    scaler = StandardScaler().fit(np.vstack([X_tr, X_te]))
    X_tr_z = scaler.transform(X_tr)
    X_te_z = scaler.transform(X_te)

    # test -> nearest train
    nn_te = NearestNeighbors(n_neighbors=1).fit(X_tr_z)
    d_te, i_te = nn_te.kneighbors(X_te_z)
    d_te = d_te.flatten()

    # train -> nearest train (leave-one-out via k=2)
    nn_tr = NearestNeighbors(n_neighbors=2).fit(X_tr_z)
    d_tr, _ = nn_tr.kneighbors(X_tr_z)
    d_tr = d_tr[:, 1]  # skip self

    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print("\n[near] nearest-neighbour distance quantiles (z-scored L2):")
    print(f"  {'q':>4s}  {'train->train':>14s}  {'test->train':>14s}")
    for q in qs:
        print(f"  {q:>3d}%  {np.percentile(d_tr, q):>14.4f}  "
              f"{np.percentile(d_te, q):>14.4f}")

    # Leak signature: test->train nearest smaller than train->train 5th %ile
    thr = np.percentile(d_tr, 5)
    leaked = d_te < thr
    print(f"\n[near] threshold (5th %ile of train-train NN dist): {thr:.4f}")
    print(f"[near] test rows below threshold: {leaked.sum()} / {len(test)} "
          f"({100.0 * leaked.mean():.1f}%)")

    # Also flag suspiciously tight matches
    for level_name, level in [("<0.01", 0.01), ("<0.05", 0.05), ("<0.1", 0.1),
                              ("<0.5", 0.5)]:
        n = int((d_te < level).sum())
        print(f"[near] test rows with NN dist {level_name:>5s}: {n}")

    # For the tightest 10 test rows, show the candidate train targets
    order = np.argsort(d_te)[:10]
    print("\n[near] 10 tightest test->train matches:")
    print(f"  {'test_i':>6s}  {'d_NN':>8s}  {'train_i':>8s}  {'train_target':>14s}")
    for i in order:
        print(f"  {i:>6d}  {d_te[i]:>8.4f}  {int(i_te[i, 0]):>8d}  "
              f"{train.loc[int(i_te[i, 0]), 'target']:>14.4f}")

    # -- LB-impact estimate ----------------------------------------------
    # If we naively copy train target for the leaked rows and use cross_LE
    # for the rest, what's the maximum possible MAE improvement?
    # Assumption: leaked rows become "~0 error" (MAE 0) vs their current
    # cross_LE error (~1.67 non-sent, ~10 sentinel).
    # Public LB is 50% of test.
    sentinel_rate = 0.152
    cross_LE_lb = 2.94
    if leaked.sum() > 0:
        test_sent = (test["x5"] == SENTINEL).to_numpy()
        leaked_sent = int((leaked & test_sent).sum())
        leaked_nonsent = int((leaked & ~test_sent).sum())
        saved = leaked_sent * 10.0 + leaked_nonsent * 1.67
        new_lb = (cross_LE_lb * 1500 - saved) / 1500
        print(f"\n[impact] leaked sentinel rows: {leaked_sent}  "
              f"leaked non-sent: {leaked_nonsent}")
        print(f"[impact] best-case absolute-error saved: {saved:.1f}")
        print(f"[impact] projected LB if all leaks perfect: "
              f"{cross_LE_lb:.3f} -> {new_lb:.3f}")


if __name__ == "__main__":
    main()
