"""cross_LE trained on UNCLAMPED + clean-x5 training data.

The 100 id<100 rows have target = A1 + (-15·x8 + 1 + ε). Subtracting the
known clamp correction gives the "pure" DGP target, turning them into
clean training samples. Two variants:

  v6: unclamped (100 rows corrected, all 1500 kept)
  v7: dropped   (only 1400 training rows used)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_x4_x9_swap_ensemble import fit_ebm  # noqa
from cv_ebm_variants import SENTINEL  # noqa

DATA = REPO / "data"
SUBS = REPO / "submissions"


def load_corrected():
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")

    # Seed-recover x5
    rs = np.random.RandomState(4242)
    for _ in range(4):
        rs.uniform(0, 1, 3000)
    x5_rec = rs.uniform(7, 12, 3000)

    train_c = train.copy()
    test_c = test.copy()
    tr_sent = (train["x5"] == SENTINEL).values
    te_sent = (test["x5"] == SENTINEL).values
    train_c.loc[tr_sent, "x5"] = x5_rec[:1500][tr_sent]
    test_c.loc[te_sent, "x5"] = x5_rec[1500:][te_sent]

    # Un-clamp training targets for id<100 rows
    clamp_mask = ((train["id"] < 100) & (train["x4"] < 0) & (train["x8"] < 0)).values
    train_unclamped = train_c.copy()
    train_unclamped.loc[clamp_mask, "target"] = (
        train_c.loc[clamp_mask, "target"].values
        - (-15 * train_c.loc[clamp_mask, "x8"].values + 1.0)
    )
    n_unclamped = int(clamp_mask.sum())
    print(f"Un-clamped {n_unclamped} training rows")
    return train_unclamped, test_c, clamp_mask


def design(df, include_x4, include_x9):
    city = (df["City"] == "Zaragoza").astype(float).values
    cols = [df["x1"].values ** 2, np.cos(5 * np.pi * df["x2"].values)]
    if include_x4:
        cols.append(df["x4"].values)
    cols.append(df["x5"].values)
    cols.append(df["x8"].values)
    cols.append(df["x10"].values)
    cols.append(df["x11"].values)
    cols.append(df["x10"].values * df["x11"].values)
    cols.append(city)
    if include_x9:
        cols.append(df["x9"].values)
    return np.column_stack(cols)


def train_and_predict(train, test):
    y = train["target"].values
    # LIN_x4
    X_tr_lin = design(train, True, False)
    X_te_lin = design(test, True, False)
    lin_pred = LinearRegression().fit(X_tr_lin, y).predict(X_te_lin)
    # EBM_x9
    X_tr_ebm = design(train, False, True)
    X_te_ebm = design(test, False, True)
    import time
    t0 = time.time()
    ebm_pred = fit_ebm(X_tr_ebm, y).predict(X_te_ebm)
    print(f"  EBM fit in {time.time()-t0:.0f}s")
    return 0.5 * (lin_pred + ebm_pred)


def main():
    train_unclamped, test_c, clamp_mask = load_corrected()

    print("\n=== v6: unclamped (all 1500 rows) ===")
    pred_v6 = train_and_predict(train_unclamped, test_c)
    out = SUBS / "submission_closed_form_v6_unclamped.csv"
    pd.DataFrame({"id": test_c["id"], "target": pred_v6}).to_csv(out, index=False)
    print(f"  wrote {out.name}  mean={pred_v6.mean():+.3f}")

    print("\n=== v7: dropped (only 1400 training rows) ===")
    train_dropped = train_unclamped.loc[~clamp_mask].reset_index(drop=True)
    print(f"  training rows: {len(train_dropped)}")
    pred_v7 = train_and_predict(train_dropped, test_c)
    out = SUBS / "submission_closed_form_v7_dropped.csv"
    pd.DataFrame({"id": test_c["id"], "target": pred_v7}).to_csv(out, index=False)
    print(f"  wrote {out.name}  mean={pred_v7.mean():+.3f}")

    # Diff against v5
    v5 = pd.read_csv(SUBS / "submission_closed_form_v5.csv").set_index("id")["target"].reindex(test_c["id"].values).values
    print(f"\nv6 vs v5 MAE: {np.mean(np.abs(pred_v6 - v5)):.4f}")
    print(f"v7 vs v5 MAE: {np.mean(np.abs(pred_v7 - v5)):.4f}")


if __name__ == "__main__":
    main()
