"""Mirror of cv_linear_4_ensemble.py but using EBM instead of linear.

  EBM_x4     : EBM, drops x9
  EBM_x9     : EBM, drops x4
  EBM_block1 : EBM, full features, trained on sign(x4)>0 subset
  EBM_block2 : EBM, full features, trained on sign(x4)<0 subset

Key structural difference vs linear: EBM's bounded shape functions pin
to the nearest training bin on extrapolation, so EBM_block_avg (raw,
unrouted) and EBM_x9 alone are not catastrophic the way LIN variants
were.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from interpret.glassbox import ExplainableBoostingRegressor

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUBS = REPO / "submissions"
SEED = 42
SENTINEL = 999.0
FEATS_ALL = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]

EBM_KW = dict(
    interactions=10, max_rounds=4000, min_samples_leaf=10, max_bins=128,
    smoothing_rounds=4000, interaction_smoothing_rounds=1000,
    random_state=SEED,
)


def preprocess(df, features, x5m):
    out = df[features].copy()
    out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5m)
    out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
    out["city"] = (df["City"] == "Zaragoza").astype(float)
    return out


def ebm_feats(x4, x9):
    f = list(FEATS_ALL)
    if not x4: f.remove("x4")
    if not x9: f.remove("x9")
    return f


def design_matrix(df, x5m, x4, x9):
    is_s = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5m).values
    city = (df["City"] == "Zaragoza").astype(float).values
    cols = [df["x1"].values ** 2, np.cos(5 * np.pi * df["x2"].values)]
    if x4: cols.append(df["x4"].values)
    cols += [x5, is_s, df["x8"].values, df["x10"].values, df["x11"].values,
             df["x10"].values * df["x11"].values, city]
    if x9: cols.append(df["x9"].values)
    return np.column_stack(cols)


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def fit_ebm(X, y):
    return ExplainableBoostingRegressor(**EBM_KW).fit(X, y)


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test  = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = train["target"].values
    is_sent = (train["x5"] == SENTINEL).to_numpy()
    n = len(train)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_ebm_x4  = np.zeros(n)
    oof_ebm_x9  = np.zeros(n)
    oof_ebm_b1  = np.zeros(n)
    oof_ebm_b2  = np.zeros(n)
    oof_ebm_bs  = np.zeros(n)
    oof_lin_x4  = np.zeros(n)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        sub_tr = train.iloc[tr_idx].reset_index(drop=True)
        sub_va = train.iloc[va_idx].reset_index(drop=True)
        y_tr = sub_tr["target"].values
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # EBM_x4 (drop x9)
        f = ebm_feats(True, False)
        oof_ebm_x4[va_idx] = fit_ebm(preprocess(sub_tr, f, x5m), y_tr) \
                             .predict(preprocess(sub_va, f, x5m))

        # EBM_x9 (drop x4)
        f = ebm_feats(False, True)
        oof_ebm_x9[va_idx] = fit_ebm(preprocess(sub_tr, f, x5m), y_tr) \
                             .predict(preprocess(sub_va, f, x5m))

        # EBM_block1, EBM_block2 (full features, within sign(x4))
        f = ebm_feats(True, True)
        m_b1 = (sub_tr["x4"] > 0).values
        m_b2 = (sub_tr["x4"] < 0).values
        e_b1 = fit_ebm(preprocess(sub_tr[m_b1], f, x5m), y_tr[m_b1])
        e_b2 = fit_ebm(preprocess(sub_tr[m_b2], f, x5m), y_tr[m_b2])
        X_va = preprocess(sub_va, f, x5m)
        oof_ebm_b1[va_idx] = e_b1.predict(X_va)
        oof_ebm_b2[va_idx] = e_b2.predict(X_va)
        route_va = (sub_va["x4"].values > 0)
        oof_ebm_bs[va_idx] = np.where(route_va, oof_ebm_b1[va_idx], oof_ebm_b2[va_idx])

        # LIN_x4 (for cross_LE comparator)
        Xtr = design_matrix(sub_tr, x5m, True, False)
        Xva = design_matrix(sub_va, x5m, True, False)
        oof_lin_x4[va_idx] = LinearRegression().fit(Xtr, y_tr).predict(Xva)

        print(f"  fold {fold+1}/5 done")

    strategies = {
        "EBM_x4 alone":              oof_ebm_x4,
        "EBM_x9 alone (LB 5.66)":    oof_ebm_x9,
        "EBM_block_s alone":         oof_ebm_bs,
        "EBM_block_avg alone":       0.5 * oof_ebm_b1 + 0.5 * oof_ebm_b2,
        "ebm4_avg (user request)":   0.25 * (oof_ebm_x4 + oof_ebm_x9 + oof_ebm_b1 + oof_ebm_b2),
        "ebm3_routed":               (oof_ebm_x4 + oof_ebm_x9 + oof_ebm_bs) / 3.0,
        "ebm_x4 + ebm_x9":           0.5 * oof_ebm_x4 + 0.5 * oof_ebm_x9,
        "ebm_x4 + ebm_block_s":      0.5 * oof_ebm_x4 + 0.5 * oof_ebm_bs,
        "ebm3 (block + x4 + x9)":    (oof_ebm_bs + oof_ebm_x4 + oof_ebm_x9) / 3.0,
        # Baselines
        "cross_LE (LB 2.94)":        0.5 * oof_lin_x4 + 0.5 * oof_ebm_x9,
        "triple_view (CV 2.92)":     (oof_ebm_bs + oof_lin_x4 + oof_ebm_x9) / 3.0,
    }

    x4 = train["x4"].to_numpy()
    print(f"\n{'strategy':<30s} {'CV':>8s} {'non-sent':>10s} {'sent':>8s} {'b1':>7s} {'b2':>7s}")
    print("-" * 76)
    rows = []
    for name, pred in strategies.items():
        ov = mae(pred, y); ns = mae(pred[~is_sent], y[~is_sent]); se = mae(pred[is_sent], y[is_sent])
        b1 = mae(pred[x4>0], y[x4>0]); b2 = mae(pred[x4<0], y[x4<0])
        rows.append((name, ov, ns, se, b1, b2))
        flag = " <-- beats triple_view" if ov < 2.9172 else ""
        print(f"{name:<30s} {ov:>8.4f} {ns:>10.4f} {se:>8.4f} {b1:>7.4f} {b2:>7.4f}{flag}")

    out = REPO / "plots" / "block_ensemble" / "cv_ebm_4.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["strategy", "cv", "non_sent", "sent", "b1", "b2"]).to_csv(out, index=False)
    print(f"\nCV table -> {out}")

    # Full-data submission for the user's ebm4_avg
    best = min(rows, key=lambda r: r[1])
    if best[0] == "ebm4_avg (user request)" or best[1] < 2.9172:
        print(f"\nBuilding full-data submission for ebm4_avg...")
        x5mf = float(train.loc[train["x5"] != SENTINEL, "x5"].median())

        f = ebm_feats(True, False)
        p_x4 = fit_ebm(preprocess(train, f, x5mf), y).predict(preprocess(test, f, x5mf))
        f = ebm_feats(False, True)
        p_x9 = fit_ebm(preprocess(train, f, x5mf), y).predict(preprocess(test, f, x5mf))

        f = ebm_feats(True, True)
        m_b1 = (train["x4"] > 0).values
        m_b2 = (train["x4"] < 0).values
        e_b1 = fit_ebm(preprocess(train[m_b1], f, x5mf), y[m_b1])
        e_b2 = fit_ebm(preprocess(train[m_b2], f, x5mf), y[m_b2])
        X_te = preprocess(test, f, x5mf)
        p_b1 = e_b1.predict(X_te); p_b2 = e_b2.predict(X_te)

        p = 0.25 * (p_x4 + p_x9 + p_b1 + p_b2)
        fn = "submission_ebm4_avg.csv"
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(SUBS / fn, index=False)
        print(f"  wrote {fn}  mean={p.mean():+.3f}")


if __name__ == "__main__":
    main()
