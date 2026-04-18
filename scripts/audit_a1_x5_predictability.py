"""A1: can we predict x5 from other features better than the Uniform(7,12)
median floor (1.25 MAE)?

If yes, we beat the 1.52 sentinel floor on the public LB.

Uses non-sentinel rows from BOTH dataset.csv and test.csv as supervised
training data (x5 is observed there), regardless of whether target is
known. Evaluates models via 5-fold CV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SEED = 42
SENTINEL = 999.0
FEATS = ["x1", "x2", "x4", "x6", "x7", "x8", "x9", "x10", "x11"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATS].copy()
    out["city"] = (df["City"] == "Zaragoza").astype(float)
    out["theta"] = np.arctan2(df["x7"].values, df["x6"].values)
    out["sin_theta"] = np.sin(out["theta"])
    out["cos_theta"] = np.cos(out["theta"])
    return out


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")

    # Pool non-sentinel rows from train+test
    tr_ns = train[train["x5"] != SENTINEL].reset_index(drop=True)
    te_ns = test[test["x5"] != SENTINEL].reset_index(drop=True)
    pool = pd.concat([tr_ns, te_ns], ignore_index=True)
    X = build_features(pool).to_numpy()
    y = pool["x5"].to_numpy()
    print(f"pool: {len(pool)} rows  (train_ns={len(tr_ns)}, test_ns={len(te_ns)})")
    print(f"x5 stats: min={y.min():.3f} max={y.max():.3f} "
          f"mean={y.mean():.3f} median={np.median(y):.3f} std={y.std():.3f}")

    # Theoretical floor: MAE when predicting the median for Uniform(7,12)
    # With actual empirical distribution:
    const_mae = mean_absolute_error(y, np.full_like(y, np.median(y)))
    print(f"\nconstant-median MAE (empirical floor): {const_mae:.4f}")
    print(f"theoretical Uniform(7,12) median floor: 1.2500\n")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "constant_median":   None,  # special
        "linear":            LinearRegression(),
        "knn_k1":            KNeighborsRegressor(n_neighbors=1),
        "knn_k5":            KNeighborsRegressor(n_neighbors=5),
        "knn_k50":           KNeighborsRegressor(n_neighbors=50),
        "histgbr_default":   HistGradientBoostingRegressor(random_state=SEED),
        "histgbr_deep":      HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.05, max_iter=500, random_state=SEED),
        "histgbr_shallow":   HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=1000,
            l2_regularization=1.0, random_state=SEED),
    }

    # Try LightGBM if available
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            random_state=SEED, verbose=-1)
    except ImportError:
        print("(lightgbm not available, skipping)")

    # Try EBM if available
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        models["ebm"] = ExplainableBoostingRegressor(
            interactions=10, max_rounds=2000, random_state=SEED)
    except ImportError:
        print("(interpret-core not available, skipping)")

    print(f"{'model':<22s} {'OOF MAE':>10s} {'delta_vs_1.25':>15s}")
    print("-" * 50)
    results = {}
    for name, model in models.items():
        oof = np.zeros(len(y))
        for tr_idx, va_idx in kf.split(X):
            if name == "constant_median":
                oof[va_idx] = np.median(y[tr_idx])
            else:
                m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
                m.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = m.predict(X[va_idx])
        mae = mean_absolute_error(y, oof)
        results[name] = mae
        delta = mae - 1.25
        flag = " <-- BEATS FLOOR" if mae < 1.25 else ""
        print(f"{name:<22s} {mae:>10.4f} {delta:>+15.4f}{flag}")

    # Sentinel impact simulation
    best_name = min(results, key=results.get)
    best_mae = results[best_name]
    print(f"\nbest non-sentinel predictor: {best_name}  MAE={best_mae:.4f}")
    if best_mae < 1.25:
        # Estimate LB improvement: each sentinel row gained ~ 8 * (1.25 - best_mae)
        per_row_save = 8.0 * (1.25 - best_mae)
        total_save = 228 * per_row_save
        new_lb = (2.94 * 1500 - total_save) / 1500
        print(f"\n>>> LEAK CANDIDATE: x5 predictable to MAE {best_mae:.4f}")
        print(f">>> Expected per-sentinel-row absolute-error reduction: "
              f"{per_row_save:.3f}")
        print(f">>> Projected LB: 2.94 -> {new_lb:.3f}")
    else:
        print(f"\n>>> No predictor beats 1.25 floor. Delta from best: "
              f"+{best_mae - 1.25:.4f}")
        print(f">>> The sentinel floor at 1.52 LB is confirmed.")


if __name__ == "__main__":
    main()
