"""CV the 4-linear ensemble:

  LIN_x4     : linear, drops x9
  LIN_x9     : linear, drops x4
  LIN_block1 : linear, full features, trained on sign(x4)>0 subset
  LIN_block2 : linear, full features, trained on sign(x4)<0 subset

Ensembles tested:

  lin4_avg     = mean(LIN_x4, LIN_x9, LIN_block1, LIN_block2)
  lin3_routed  = mean(LIN_x4, LIN_x9, LIN_block_s)   where block_s is
                 per-row routed by sign(x4)
  lin4_avg+EBM_x9  (adds EBM_x9 as the sole EBM anchor, equal weights)
  lin4_avg+cross_LE
  lin4_avg+triple_view

Also records per-quadrant MAE (on-diagonal vs off-diagonal in training
is irrelevant — all training is on-diagonal — but we report block1/block2
split so the user can see which half the error comes from).
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


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test  = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = train["target"].values
    is_sent = (train["x5"] == SENTINEL).to_numpy()
    n = len(train)
    print(f"train={n}  test={len(test)}  sentinels={is_sent.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_lin_x4    = np.zeros(n)
    oof_lin_x9    = np.zeros(n)
    oof_lin_b1    = np.zeros(n)   # LIN_block1 applied to every val row
    oof_lin_b2    = np.zeros(n)
    oof_lin_bs    = np.zeros(n)   # routed by sign(x4) of the val row
    oof_ebm_x9    = np.zeros(n)
    oof_ebm_block = np.zeros(n)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        sub_tr = train.iloc[tr_idx].reset_index(drop=True)
        sub_va = train.iloc[va_idx].reset_index(drop=True)
        y_tr = sub_tr["target"].values
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # LIN_x4: drop x9
        Xtr = design_matrix(sub_tr, x5m, True, False)
        Xva = design_matrix(sub_va, x5m, True, False)
        oof_lin_x4[va_idx] = LinearRegression().fit(Xtr, y_tr).predict(Xva)

        # LIN_x9: drop x4
        Xtr = design_matrix(sub_tr, x5m, False, True)
        Xva = design_matrix(sub_va, x5m, False, True)
        oof_lin_x9[va_idx] = LinearRegression().fit(Xtr, y_tr).predict(Xva)

        # LIN block1/block2 (full features, within-cluster)
        m_b1 = (sub_tr["x4"] > 0).values
        m_b2 = (sub_tr["x4"] < 0).values
        Xtr_full = design_matrix(sub_tr, x5m, True, True)
        Xva_full = design_matrix(sub_va, x5m, True, True)
        m1 = LinearRegression().fit(Xtr_full[m_b1], y_tr[m_b1])
        m2 = LinearRegression().fit(Xtr_full[m_b2], y_tr[m_b2])
        oof_lin_b1[va_idx] = m1.predict(Xva_full)
        oof_lin_b2[va_idx] = m2.predict(Xva_full)
        route_va = (sub_va["x4"].values > 0)
        oof_lin_bs[va_idx] = np.where(route_va, oof_lin_b1[va_idx], oof_lin_b2[va_idx])

        # EBM_x9 and EBM_block_s for comparator ensembles
        f = ebm_feats(False, True)
        oof_ebm_x9[va_idx] = ExplainableBoostingRegressor(**EBM_KW).fit(
            preprocess(sub_tr, f, x5m), y_tr).predict(preprocess(sub_va, f, x5m))

        f = ebm_feats(True, True)
        e_b1 = ExplainableBoostingRegressor(**EBM_KW).fit(
            preprocess(sub_tr[m_b1], f, x5m), y_tr[m_b1])
        e_b2 = ExplainableBoostingRegressor(**EBM_KW).fit(
            preprocess(sub_tr[m_b2], f, x5m), y_tr[m_b2])
        Xva_ebm = preprocess(sub_va, f, x5m)
        p1 = e_b1.predict(Xva_ebm); p2 = e_b2.predict(Xva_ebm)
        oof_ebm_block[va_idx] = np.where(route_va, p1, p2)

        print(f"  fold {fold+1}/5 done")

    # Ensembles
    strategies = {
        "LIN_x4 alone":              oof_lin_x4,
        "LIN_x9 alone":              oof_lin_x9,
        "LIN_block_s alone":         oof_lin_bs,
        "LIN_block_avg alone":       0.5 * oof_lin_b1 + 0.5 * oof_lin_b2,
        "lin4_avg (user request)":   0.25 * (oof_lin_x4 + oof_lin_x9 + oof_lin_b1 + oof_lin_b2),
        "lin3_routed":               (oof_lin_x4 + oof_lin_x9 + oof_lin_bs) / 3.0,
        "lin4_no_x9":                (oof_lin_x4 + oof_lin_b1 + oof_lin_b2) / 3.0,
        "lin_x4 + lin_block_s":      0.5 * oof_lin_x4 + 0.5 * oof_lin_bs,
        "lin_x4 + lin_x9 + ebm_x9":  (oof_lin_x4 + oof_lin_x9 + oof_ebm_x9) / 3.0,
        "lin4 + cross_LE":           0.5 * (0.25 * (oof_lin_x4 + oof_lin_x9 + oof_lin_b1 + oof_lin_b2))
                                      + 0.5 * (0.5 * oof_lin_x4 + 0.5 * oof_ebm_x9),
        "lin4 + triple_view":        0.5 * (0.25 * (oof_lin_x4 + oof_lin_x9 + oof_lin_b1 + oof_lin_b2))
                                      + 0.5 * ((oof_ebm_block + oof_lin_x4 + oof_ebm_x9) / 3.0),
        "lin4 + EBM_x9":             0.8 * (0.25 * (oof_lin_x4 + oof_lin_x9 + oof_lin_b1 + oof_lin_b2))
                                      + 0.2 * oof_ebm_x9,
        # baselines for reference
        "cross_LE (LB 2.94)":        0.5 * oof_lin_x4 + 0.5 * oof_ebm_x9,
        "triple_view (submitted)":   (oof_ebm_block + oof_lin_x4 + oof_ebm_x9) / 3.0,
    }

    x4 = train["x4"].to_numpy()
    print(f"\n{'strategy':<30s} {'CV':>8s} {'non-sent':>10s} {'sent':>8s} {'b1':>7s} {'b2':>7s}")
    print("-" * 76)
    results = []
    for name, pred in strategies.items():
        ov = mae(pred, y); ns = mae(pred[~is_sent], y[~is_sent]); se = mae(pred[is_sent], y[is_sent])
        b1 = mae(pred[x4>0], y[x4>0]); b2 = mae(pred[x4<0], y[x4<0])
        results.append((name, ov, ns, se, b1, b2))
        print(f"{name:<30s} {ov:>8.4f} {ns:>10.4f} {se:>8.4f} {b1:>7.4f} {b2:>7.4f}")

    # Full-data training + submission for the user's 4-linear ensemble
    print(f"\nBuilding full-data submission for lin4_avg (user request)...")
    x5mf = float(train.loc[train["x5"] != SENTINEL, "x5"].median())

    X_tr_no_x9 = design_matrix(train, x5mf, True, False)
    X_te_no_x9 = design_matrix(test,  x5mf, True, False)
    p_lin_x4 = LinearRegression().fit(X_tr_no_x9, y).predict(X_te_no_x9)

    X_tr_no_x4 = design_matrix(train, x5mf, False, True)
    X_te_no_x4 = design_matrix(test,  x5mf, False, True)
    p_lin_x9 = LinearRegression().fit(X_tr_no_x4, y).predict(X_te_no_x4)

    X_tr_full = design_matrix(train, x5mf, True, True)
    X_te_full = design_matrix(test,  x5mf, True, True)
    m_b1 = (train["x4"] > 0).values
    m_b2 = (train["x4"] < 0).values
    p_lin_b1 = LinearRegression().fit(X_tr_full[m_b1], y[m_b1]).predict(X_te_full)
    p_lin_b2 = LinearRegression().fit(X_tr_full[m_b2], y[m_b2]).predict(X_te_full)

    lin4_avg = 0.25 * (p_lin_x4 + p_lin_x9 + p_lin_b1 + p_lin_b2)
    fn = "submission_lin4_avg.csv"
    pd.DataFrame({"id": test["id"], "target": lin4_avg}).to_csv(SUBS / fn, index=False)
    print(f"  wrote {fn}  mean={lin4_avg.mean():+.3f}  range=[{lin4_avg.min():+.2f}, {lin4_avg.max():+.2f}]")

    out = REPO / "plots" / "block_ensemble" / "cv_linear_4.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=["strategy", "cv", "non_sent", "sent", "b1", "b2"]).to_csv(out, index=False)
    print(f"\nCV table -> {out}")


if __name__ == "__main__":
    main()
