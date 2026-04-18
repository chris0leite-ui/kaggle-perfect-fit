"""5-fold CV for k-nearest-neighbours regression on the target.

Tests kNN with several neighbour counts, both with and without x9 in the
feature set. Features are standardised per fold.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0
SEED = 42
N_SPLITS = 5

FEATURES_ALL = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]
FEATURES_NO_X9 = ["x1", "x2", "x4", "x5", "x8", "x10", "x11"]


def design(df: pd.DataFrame, features: list[str], x5_median: float) -> np.ndarray:
    out = df[features].copy()
    out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
    out["x5_is_sent"] = (df["x5"] == SENTINEL).astype(float)
    out["city"] = (df["City"] == "Zaragoza").astype(float)
    return out.values


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def run_knn(df: pd.DataFrame, features: list[str], k: int, weighted: bool) -> dict:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    is_sent = (df["x5"] == SENTINEL).values
    y = df["target"].values
    oof = np.zeros(len(df))
    per_fold = []
    t0 = time.time()
    for tr_idx, va_idx in kf.split(df):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        x5m = float(tr.loc[tr["x5"] != SENTINEL, "x5"].median())
        X_tr = design(tr, features, x5m)
        X_va = design(va, features, x5m)
        sc = StandardScaler().fit(X_tr)
        knn = KNeighborsRegressor(n_neighbors=k,
                                  weights="distance" if weighted else "uniform")
        knn.fit(sc.transform(X_tr), tr["target"].values)
        oof[va_idx] = knn.predict(sc.transform(X_va))
        per_fold.append(mae(oof[va_idx], va["target"].values))
    return {
        "k": k, "weighted": weighted,
        "features": "all" if "x9" in features else "no_x9",
        "overall": mae(oof, y),
        "non_sentinel": mae(oof[~is_sent], y[~is_sent]),
        "sentinel": mae(oof[is_sent], y[is_sent]),
        "seconds": time.time() - t0, "per_fold": per_fold, "oof": oof,
    }


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    rows = []
    oof_cols = {}
    for features in [FEATURES_ALL, FEATURES_NO_X9]:
        for k in [5, 15, 30, 50, 100]:
            for weighted in [False, True]:
                r = run_knn(df, features, k, weighted)
                key = f"knn_k{k}_{'w' if weighted else 'u'}_{r['features']}"
                oof_cols[key] = r.pop("oof")
                rows.append(r)
                print(f"  {key:24s}  overall={r['overall']:.3f}  "
                      f"non-sent={r['non_sentinel']:.3f}  sent={r['sentinel']:.3f}  "
                      f"[{r['seconds']:.0f}s]")
    results = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)
    print("\n" + "=" * 78)
    print("kNN variants ranked by CV MAE:")
    print("=" * 78)
    print(results[["k", "weighted", "features", "overall",
                   "non_sentinel", "sentinel"]].to_string(index=False))
    results.drop(columns=["per_fold"]).to_csv(OUT / "cv_knn.csv", index=False)

    oof_df = pd.DataFrame(oof_cols)
    oof_df["target"] = df["target"].values
    oof_df.to_csv(OUT / "cv_knn_oof.csv", index=False)


if __name__ == "__main__":
    main()
