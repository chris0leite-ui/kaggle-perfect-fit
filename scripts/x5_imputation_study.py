"""Investigate whether x5 can be imputed better than by median.

Leverages A1's formula — assumed to be the true data-generating process — to
back-solve x5 on sentinel rows in training data:

    target = -100*x1^2 + 10*cos(5*pi*x2) + 15*x4 - 8*x5 + 15*x8 - 4*x9
             + x10*x11 - 25*is_zaragoza + 20*1(x4>0) + 92.5
    =>  x5_true = [(all non-x5 terms) - target] / 8

This gives us "ground-truth" x5 for the 222 sentinel training rows and lets us
measure how close any imputation strategy comes to it. We test:

    1. Median (current baseline)
    2. kNN (k=5, k=15, k=50) on feature similarity
    3. LinearRegression on (x1,x2,x4,x8,x9,x10,x11,City)
    4. LightGBM on the same features
    5. Ordering by id (is there a temporal / sorted pattern?)

Also quantifies the downstream impact: replacing median-imputation with the
best alternative in A1's formula — does total MAE drop?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "formulas"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


def non_x5_contribution(df: pd.DataFrame) -> np.ndarray:
    """All of A1 except the -8*x5 term and the constant 92.5."""
    is_zar = (df["City"] == "Zaragoza").astype(float).values
    x4_pos = (df["x4"].values > 0).astype(float)
    return (
        -100 * df["x1"].values ** 2
        + 10 * np.cos(5 * np.pi * df["x2"].values)
        + 15 * df["x4"].values
        + 15 * df["x8"].values
        - 4 * df["x9"].values
        + df["x10"].values * df["x11"].values
        - 25 * is_zar
        + 20 * x4_pos
        + 92.5
    )


def backsolve_x5(df: pd.DataFrame) -> np.ndarray:
    """Return the x5 implied by A1's formula and the observed target."""
    rest = non_x5_contribution(df)
    return (rest - df["target"].values) / 8.0


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    sent_mask = (df["x5"] == SENTINEL).values
    print(f"dataset: {df.shape}")
    print(f"sentinel rows: {sent_mask.sum()} ({100*sent_mask.mean():.1f}%)")

    # True x5 for sentinel rows (back-solved from A1)
    x5_true_sent = backsolve_x5(df.loc[sent_mask].reset_index(drop=True))
    print(f"\nBack-solved x5 for sentinels:")
    print(f"  n={len(x5_true_sent)}")
    print(f"  mean={x5_true_sent.mean():.3f}, std={x5_true_sent.std():.3f}")
    print(f"  min={x5_true_sent.min():.3f}, max={x5_true_sent.max():.3f}")

    x5_nonsent = df.loc[~sent_mask, "x5"].values
    print(f"\nObserved non-sentinel x5:")
    print(f"  n={len(x5_nonsent)}")
    print(f"  mean={x5_nonsent.mean():.3f}, std={x5_nonsent.std():.3f}")
    print(f"  min={x5_nonsent.min():.3f}, max={x5_nonsent.max():.3f}")

    # Validate: back-solved x5 for NON-sentinels should match the observed value
    x5_check = backsolve_x5(df.loc[~sent_mask].reset_index(drop=True))
    residual = x5_nonsent - x5_check
    print(f"\nBack-solved x5 validation on non-sentinels (should be ~0 if A1 is exact):")
    print(f"  residual mean={residual.mean():+.3f}, std={residual.std():.3f}")
    print(f"  exact matches (|resid|<0.01): {np.mean(np.abs(residual) < 0.01)*100:.1f}%")

    # Build feature matrix for imputation — everything EXCEPT x5
    FEATURES = ["x1", "x2", "x4", "x8", "x9", "x10", "x11"]
    X_all = df[FEATURES].copy()
    X_all["city"] = (df["City"] == "Zaragoza").astype(float)
    X_all["x10x11"] = df["x10"] * df["x11"]

    X_train = X_all.loc[~sent_mask].values
    y_train = x5_nonsent
    X_test = X_all.loc[sent_mask].values  # we'll predict for sentinels
    y_test = x5_true_sent  # "ground truth" via back-solve

    # --- Baseline: median ---
    median_val = float(np.median(y_train))
    median_pred = np.full_like(y_test, median_val)
    print(f"\n--- Imputation strategies: MAE on back-solved x5 ---")
    results = [("Median (current)", median_pred, median_val)]
    print(f"  Median (current):                      MAE={mae(median_pred, y_test):.3f}  (imputes {median_val:.2f})")

    # --- Mean ---
    mean_val = float(np.mean(y_train))
    mean_pred = np.full_like(y_test, mean_val)
    results.append(("Mean", mean_pred, None))
    print(f"  Mean:                                  MAE={mae(mean_pred, y_test):.3f}  (imputes {mean_val:.2f})")

    # --- kNN (several k, normalised features) ---
    scaler = StandardScaler().fit(X_train)
    Xs_tr = scaler.transform(X_train)
    Xs_te = scaler.transform(X_test)
    for k in [5, 15, 50, 100]:
        knn = KNeighborsRegressor(n_neighbors=k).fit(Xs_tr, y_train)
        pred = knn.predict(Xs_te)
        results.append((f"kNN k={k}", pred, None))
        print(f"  kNN k={k:3d}:                             MAE={mae(pred, y_test):.3f}")

    # --- Linear regression ---
    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results.append(("Linear regression", pred_lr, None))
    print(f"  Linear regression:                     MAE={mae(pred_lr, y_test):.3f}")
    print(f"    coefs: {dict(zip(list(X_all.columns), np.round(lr.coef_, 3).tolist()))}")
    print(f"    intercept: {lr.intercept_:+.3f}")
    print(f"    train R^2: {lr.score(X_train, y_train):.4f}")

    # --- Random Forest / LightGBM ---
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                  min_child_samples=20, verbose=-1)
        lgbm.fit(X_train, y_train)
        pred_lgb = lgbm.predict(X_test)
        results.append(("LightGBM", pred_lgb, None))
        print(f"  LightGBM:                              MAE={mae(pred_lgb, y_test):.3f}")
    except ImportError:
        print("  LightGBM not available — skipping")

    # --- ID proximity (nearest non-sentinel id) ---
    # For each sentinel, find the non-sentinel with closest id
    nonsent_ids = df.loc[~sent_mask, "id"].values
    nonsent_x5 = df.loc[~sent_mask, "x5"].values
    sent_ids = df.loc[sent_mask, "id"].values
    order = np.argsort(nonsent_ids)
    sorted_ids = nonsent_ids[order]
    sorted_x5 = nonsent_x5[order]
    pred_id = []
    for sid in sent_ids:
        idx = np.searchsorted(sorted_ids, sid)
        candidates = []
        if idx > 0: candidates.append(sorted_x5[idx - 1])
        if idx < len(sorted_ids): candidates.append(sorted_x5[idx])
        pred_id.append(np.mean(candidates))
    pred_id = np.array(pred_id)
    results.append(("Nearest-id neighbour", pred_id, None))
    print(f"  Nearest-id neighbour:                  MAE={mae(pred_id, y_test):.3f}")

    # --- Correlation diagnostic: which features predict x5? ---
    print(f"\nPearson r(x5, feature) on non-sentinel rows:")
    for col in list(X_all.columns):
        r = np.corrcoef(X_all[col].loc[~sent_mask].values, y_train)[0, 1]
        print(f"    {col:8s}: {r:+.4f}")

    # --- Scatter: back-solved sentinel x5 vs feature ---
    X_sent = X_all.loc[sent_mask]
    print(f"\nPearson r(back-solved sentinel x5, feature):")
    for col in X_all.columns:
        r = np.corrcoef(X_sent[col].values, y_test)[0, 1]
        print(f"    {col:8s}: {r:+.4f}")

    # --- Downstream effect: if we plug imputed x5 into A1, how does target MAE change? ---
    print(f"\n--- Effect on A1 target MAE ---")
    n_total = len(df)
    # Start from A1 predictions with median imputation (status quo)
    def a1_target(df_inner, x5_vec):
        is_zar = (df_inner["City"] == "Zaragoza").astype(float).values
        x4_pos = (df_inner["x4"].values > 0).astype(float)
        return (
            -100 * df_inner["x1"].values ** 2
            + 10 * np.cos(5 * np.pi * df_inner["x2"].values)
            + 15 * df_inner["x4"].values
            - 8 * x5_vec
            + 15 * df_inner["x8"].values
            - 4 * df_inner["x9"].values
            + df_inner["x10"].values * df_inner["x11"].values
            - 25 * is_zar
            + 20 * x4_pos
            + 92.5
        )

    y = df["target"].values
    # For non-sentinel rows, just use the real x5
    for name, pred_x5, _ in results:
        x5_full = df["x5"].values.copy()
        x5_full[sent_mask] = pred_x5
        pred_target = a1_target(df, x5_full)
        mae_all = mae(pred_target, y)
        mae_sent = mae(pred_target[sent_mask], y[sent_mask])
        mae_nons = mae(pred_target[~sent_mask], y[~sent_mask])
        print(f"  {name:25s}  overall={mae_all:.3f}   non-sent={mae_nons:.3f}   sent={mae_sent:.3f}")

    # Plots
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(min(x5_nonsent.min(), y_test.min()),
                       max(x5_nonsent.max(), y_test.max()), 40)
    ax.hist(x5_nonsent, bins=bins, alpha=0.5, label=f"non-sentinel x5 (n={len(x5_nonsent)})", color="steelblue")
    ax.hist(y_test, bins=bins, alpha=0.6, label=f"back-solved sentinel x5 (n={len(y_test)})", color="firebrick")
    ax.axvline(median_val, color="black", ls="--", lw=1, label=f"median imputation ({median_val:.1f})")
    ax.set_xlabel("x5")
    ax.set_ylabel("count")
    ax.set_title("Back-solved x5 vs observed x5 — do sentinels come from the same distribution?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "x5_imputation_distribution.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r[0] for r in results]
    maes = [mae(r[1], y_test) for r in results]
    order = np.argsort(maes)
    ax.barh([names[i] for i in order], [maes[i] for i in order], color="steelblue")
    for i, v in enumerate([maes[j] for j in order]):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("MAE on back-solved sentinel x5")
    ax.set_title("Imputation quality — lower is better")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "x5_imputation_comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
