"""Block-ensemble v2: add linear variants for all combinations.

Linear models extrapolate smoothly (unlike EBM's bounded shapes), which
helps on off-diagonal test rows where EBM pins to the nearest training
bin. We add:

  LIN_x9, LIN_full           : linear single-view variants using x9
  LIN_block1, LIN_block2     : within-cluster linear, clean x9 partial
  LIN_block_s (routed)       : per-row block-routed linear
  LIN_q11 ... LIN_q22        : 2x2 quadrant linear
  LIN_quad_s (routed)        : per-row quadrant-routed linear

Ensemble strategies tested:

  triple_view  (v1 winner)   : (EBM_block_s + LIN_x4 + EBM_x9) / 3
  lin_triple                 : (LIN_block_s + LIN_x4 + LIN_x9) / 3
  mixed_block                : (EBM_block_s + LIN_block_s + LIN_x4 + EBM_x9) / 4
  five_view_overall          : triple_view half + (LIN_block_s + EBM_full) / 2
  all_views                  : 7-way equal blend of block + single-view + overall
  lin_heavy                  : linear-dominant blend with EBM_x9 for x9 signal
  anchored                   : 0.4 EBM_full + 0.3 triple_view + 0.3 lin_block_s
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
    """A2-style hand-crafted linear basis."""
    is_s = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5m).values
    city = (df["City"] == "Zaragoza").astype(float).values
    cols = [df["x1"].values ** 2, np.cos(5 * np.pi * df["x2"].values)]
    if x4: cols.append(df["x4"].values)
    cols += [x5, is_s, df["x8"].values, df["x10"].values, df["x11"].values,
             df["x10"].values * df["x11"].values, city]
    if x9: cols.append(df["x9"].values)
    return np.column_stack(cols)


def fit_lin(df_tr, df_va, x5m, x4, x9, mask=None):
    X_tr = design_matrix(df_tr, x5m, x4, x9)
    X_va = design_matrix(df_va, x5m, x4, x9)
    y_tr = df_tr["target"].values
    if mask is not None:
        X_tr = X_tr[mask]; y_tr = y_tr[mask]
    return LinearRegression().fit(X_tr, y_tr).predict(X_va)


def fit_ebm(X, y):
    return ExplainableBoostingRegressor(**EBM_KW).fit(X, y)


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = train["target"].values
    is_sent = (train["x5"] == SENTINEL).to_numpy()
    n = len(train)
    print(f"train={n}  test={len(test)}  sentinels={is_sent.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    keys = [
        # EBMs
        "ebm_block_s", "ebm_block1", "ebm_block2",
        "ebm_x4", "ebm_x9", "ebm_full",
        "ebm_q_s",
        # Linears
        "lin_x4", "lin_x9", "lin_full",
        "lin_block_s", "lin_block1", "lin_block2",
        "lin_q_s",
    ]
    oof = {k: np.zeros(n) for k in keys}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        sub_tr = train.iloc[tr_idx].reset_index(drop=True)
        sub_va = train.iloc[va_idx].reset_index(drop=True)
        y_tr = sub_tr["target"].values
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # ---- Single-view EBMs ----
        for k, (x4, x9) in [("ebm_x4", (True, False)),
                            ("ebm_x9", (False, True)),
                            ("ebm_full", (True, True))]:
            f = ebm_feats(x4, x9)
            m = fit_ebm(preprocess(sub_tr, f, x5m), y_tr)
            oof[k][va_idx] = m.predict(preprocess(sub_va, f, x5m))

        # ---- Single-view linears ----
        oof["lin_x4"][va_idx]  = fit_lin(sub_tr, sub_va, x5m, True,  False)
        oof["lin_x9"][va_idx]  = fit_lin(sub_tr, sub_va, x5m, False, True)
        oof["lin_full"][va_idx]= fit_lin(sub_tr, sub_va, x5m, True,  True)

        # ---- 2-block (sign(x4)) EBM + linear ----
        feats_full = ebm_feats(True, True)
        m_b1_tr = (sub_tr["x4"] > 0).values
        m_b2_tr = (sub_tr["x4"] < 0).values

        # EBM block models
        ebm_b1 = fit_ebm(preprocess(sub_tr[m_b1_tr], feats_full, x5m), y_tr[m_b1_tr])
        ebm_b2 = fit_ebm(preprocess(sub_tr[m_b2_tr], feats_full, x5m), y_tr[m_b2_tr])
        X_va_full = preprocess(sub_va, feats_full, x5m)
        oof["ebm_block1"][va_idx] = ebm_b1.predict(X_va_full)
        oof["ebm_block2"][va_idx] = ebm_b2.predict(X_va_full)
        route_va = (sub_va["x4"].values > 0)
        oof["ebm_block_s"][va_idx] = np.where(
            route_va, oof["ebm_block1"][va_idx], oof["ebm_block2"][va_idx])

        # Linear block models
        oof["lin_block1"][va_idx] = fit_lin(sub_tr, sub_va, x5m, True, True, mask=m_b1_tr)
        oof["lin_block2"][va_idx] = fit_lin(sub_tr, sub_va, x5m, True, True, mask=m_b2_tr)
        oof["lin_block_s"][va_idx] = np.where(
            route_va, oof["lin_block1"][va_idx], oof["lin_block2"][va_idx])

        # ---- 2x2 quadrant EBM + linear ----
        quad_masks_tr = {
            "q11": (sub_tr["x4"] > 0) & (sub_tr["x8"] > 0),
            "q12": (sub_tr["x4"] > 0) & (sub_tr["x8"] < 0),
            "q21": (sub_tr["x4"] < 0) & (sub_tr["x8"] > 0),
            "q22": (sub_tr["x4"] < 0) & (sub_tr["x8"] < 0),
        }
        q_va = np.where(
            sub_va["x4"].values > 0,
            np.where(sub_va["x8"].values > 0, "q11", "q12"),
            np.where(sub_va["x8"].values > 0, "q21", "q22"),
        )
        for q, mask in quad_masks_tr.items():
            m_arr = mask.values
            e = fit_ebm(preprocess(sub_tr[m_arr], feats_full, x5m), y_tr[m_arr])
            p_ebm = e.predict(X_va_full)
            p_lin = fit_lin(sub_tr, sub_va, x5m, True, True, mask=m_arr)
            sel = (q_va == q)
            oof["ebm_q_s"][va_idx[sel]] = p_ebm[sel]
            oof["lin_q_s"][va_idx[sel]] = p_lin[sel]

        print(f"  fold {fold+1}/5 done")

    # -------- strategies --------
    s = {
        # v1 bests
        "cross_LE":            0.5 * oof["lin_x4"] + 0.5 * oof["ebm_x9"],
        "triple_view":         (oof["ebm_block_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 3.0,
        # all-linear
        "lin_triple":          (oof["lin_block_s"] + oof["lin_x4"] + oof["lin_x9"]) / 3.0,
        "lin_block_only":      oof["lin_block_s"],
        # mixed block (EBM block + LIN block)
        "mixed_block_3":       (oof["ebm_block_s"] + oof["lin_block_s"] + oof["lin_x4"]) / 3.0,
        "mixed_block_4":       (oof["ebm_block_s"] + oof["lin_block_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 4.0,
        # with overall anchor
        "triple+full":         (oof["ebm_block_s"] + oof["lin_x4"] + oof["ebm_x9"] + oof["ebm_full"]) / 4.0,
        "five_view_overall":   0.5 * (oof["ebm_block_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 3.0
                                + 0.5 * (oof["lin_block_s"] + oof["ebm_full"]) / 2.0,
        "anchored":            0.4 * oof["ebm_full"]
                                + 0.3 * (oof["ebm_block_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 3.0
                                + 0.3 * oof["lin_block_s"],
        # big mixtures
        "all_views":           (oof["ebm_block_s"] + oof["lin_block_s"]
                                + oof["lin_x4"] + oof["ebm_x9"]
                                + oof["lin_x9"] + oof["ebm_x4"]
                                + oof["ebm_full"]) / 7.0,
        "block_heavy":         0.4 * oof["ebm_block_s"] + 0.3 * oof["lin_block_s"]
                                + 0.15 * oof["lin_x4"] + 0.15 * oof["ebm_x9"],
        "lin_heavy":           0.25 * oof["lin_block_s"] + 0.25 * oof["lin_x4"]
                                + 0.25 * oof["lin_x9"] + 0.25 * oof["ebm_x9"],
        # quadrant variants
        "quad_lin_triple":     (oof["lin_q_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 3.0,
        "quad_mixed_4":        (oof["ebm_q_s"] + oof["lin_q_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 4.0,
        # bases for reference
        "lin_x4_alone":        oof["lin_x4"],
        "lin_x9_alone":        oof["lin_x9"],
        "lin_full_alone":      oof["lin_full"],
        "lin_block_s_alone":   oof["lin_block_s"],
        "ebm_block_s_alone":   oof["ebm_block_s"],
        "ebm_full_alone":      oof["ebm_full"],
    }

    x4 = train["x4"].to_numpy()
    print(f"\n{'strategy':<22s} {'CV MAE':>8s} {'non-sent':>10s} {'sent':>8s}"
          f" {'b1':>7s} {'b2':>7s}")
    print("-" * 70)
    rows = []
    best_cv = 2.9172  # v1 triple_view
    for name, pred in s.items():
        ov = mae(pred, y)
        ns = mae(pred[~is_sent], y[~is_sent])
        se = mae(pred[is_sent], y[is_sent])
        b1 = mae(pred[x4 > 0], y[x4 > 0])
        b2 = mae(pred[x4 < 0], y[x4 < 0])
        rows.append((name, ov, ns, se, b1, b2))
        flag = " <-- beats triple_view" if ov < best_cv else ""
        print(f"{name:<22s} {ov:>8.4f} {ns:>10.4f} {se:>8.4f} {b1:>7.4f} {b2:>7.4f}{flag}")

    out_dir = REPO / "plots" / "block_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["strategy", "cv", "non_sent", "sent", "block1", "block2"]) \
      .to_csv(out_dir / "cv_results_v2.csv", index=False)
    print(f"\nCV table -> {out_dir/'cv_results_v2.csv'}")

    winners = [r for r in rows if r[1] < best_cv]
    if not winners:
        print(f"\nNo strategy beats triple_view (CV {best_cv}). No new submissions.")
        return

    print(f"\nBuilding submissions for {len(winners)} strategy(ies)...")
    x5mf = float(train.loc[train["x5"] != SENTINEL, "x5"].median())

    # Full-data base models
    print("  training full-data bases...")
    preds: dict[str, np.ndarray] = {}

    # Single-view EBMs
    for k, (x4f, x9f) in [("ebm_x4", (True, False)),
                          ("ebm_x9", (False, True)),
                          ("ebm_full", (True, True))]:
        f = ebm_feats(x4f, x9f)
        preds[k] = fit_ebm(preprocess(train, f, x5mf), y).predict(preprocess(test, f, x5mf))

    # Single-view linears
    def lin_full_data(x4f, x9f, mask=None):
        X_tr = design_matrix(train, x5mf, x4f, x9f)
        X_te = design_matrix(test,  x5mf, x4f, x9f)
        y_tr = y
        if mask is not None:
            X_tr = X_tr[mask]; y_tr = y[mask]
        return LinearRegression().fit(X_tr, y_tr).predict(X_te)

    preds["lin_x4"]    = lin_full_data(True,  False)
    preds["lin_x9"]    = lin_full_data(False, True)
    preds["lin_full"]  = lin_full_data(True,  True)

    # Block models
    m_b1_tr = (train["x4"] > 0).values
    m_b2_tr = (train["x4"] < 0).values
    feats_full = ebm_feats(True, True)
    ebm_b1 = fit_ebm(preprocess(train[m_b1_tr], feats_full, x5mf), y[m_b1_tr])
    ebm_b2 = fit_ebm(preprocess(train[m_b2_tr], feats_full, x5mf), y[m_b2_tr])
    X_te_full = preprocess(test, feats_full, x5mf)
    preds["ebm_block1"] = ebm_b1.predict(X_te_full)
    preds["ebm_block2"] = ebm_b2.predict(X_te_full)
    route_te = (test["x4"].values > 0)
    preds["ebm_block_s"] = np.where(route_te, preds["ebm_block1"], preds["ebm_block2"])

    preds["lin_block1"] = lin_full_data(True, True, mask=m_b1_tr)
    preds["lin_block2"] = lin_full_data(True, True, mask=m_b2_tr)
    preds["lin_block_s"] = np.where(route_te, preds["lin_block1"], preds["lin_block2"])

    # Quadrant models
    q_te = np.where(
        test["x4"].values > 0,
        np.where(test["x8"].values > 0, "q11", "q12"),
        np.where(test["x8"].values > 0, "q21", "q22"),
    )
    preds["ebm_q_s"] = np.empty(len(test))
    preds["lin_q_s"] = np.empty(len(test))
    quad_masks = {
        "q11": (train["x4"] > 0) & (train["x8"] > 0),
        "q12": (train["x4"] > 0) & (train["x8"] < 0),
        "q21": (train["x4"] < 0) & (train["x8"] > 0),
        "q22": (train["x4"] < 0) & (train["x8"] < 0),
    }
    for q, mask in quad_masks.items():
        m_arr = mask.values
        print(f"  training full-data {q} ebm+lin (n={m_arr.sum()})...")
        e = fit_ebm(preprocess(train[m_arr], feats_full, x5mf), y[m_arr])
        p_ebm = e.predict(X_te_full)
        p_lin = lin_full_data(True, True, mask=m_arr)
        sel = (q_te == q)
        preds["ebm_q_s"][sel] = p_ebm[sel]
        preds["lin_q_s"][sel] = p_lin[sel]

    strat_full = {
        "triple_view":         (preds["ebm_block_s"] + preds["lin_x4"] + preds["ebm_x9"]) / 3.0,
        "lin_triple":          (preds["lin_block_s"] + preds["lin_x4"] + preds["lin_x9"]) / 3.0,
        "mixed_block_3":       (preds["ebm_block_s"] + preds["lin_block_s"] + preds["lin_x4"]) / 3.0,
        "mixed_block_4":       (preds["ebm_block_s"] + preds["lin_block_s"] + preds["lin_x4"] + preds["ebm_x9"]) / 4.0,
        "triple+full":         (preds["ebm_block_s"] + preds["lin_x4"] + preds["ebm_x9"] + preds["ebm_full"]) / 4.0,
        "five_view_overall":   0.5 * (preds["ebm_block_s"] + preds["lin_x4"] + preds["ebm_x9"]) / 3.0
                                + 0.5 * (preds["lin_block_s"] + preds["ebm_full"]) / 2.0,
        "anchored":            0.4 * preds["ebm_full"]
                                + 0.3 * (preds["ebm_block_s"] + preds["lin_x4"] + preds["ebm_x9"]) / 3.0
                                + 0.3 * preds["lin_block_s"],
        "all_views":           (preds["ebm_block_s"] + preds["lin_block_s"]
                                + preds["lin_x4"] + preds["ebm_x9"]
                                + preds["lin_x9"] + preds["ebm_x4"]
                                + preds["ebm_full"]) / 7.0,
        "block_heavy":         0.4 * preds["ebm_block_s"] + 0.3 * preds["lin_block_s"]
                                + 0.15 * preds["lin_x4"] + 0.15 * preds["ebm_x9"],
        "lin_heavy":           0.25 * preds["lin_block_s"] + 0.25 * preds["lin_x4"]
                                + 0.25 * preds["lin_x9"] + 0.25 * preds["ebm_x9"],
        "quad_lin_triple":     (preds["lin_q_s"] + preds["lin_x4"] + preds["ebm_x9"]) / 3.0,
        "quad_mixed_4":        (preds["ebm_q_s"] + preds["lin_q_s"] + preds["lin_x4"] + preds["ebm_x9"]) / 4.0,
    }

    for name, cv_val, *_ in winners:
        if name not in strat_full:
            continue
        p = strat_full[name]
        fn = f"submission_{name}.csv"
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(SUBS / fn, index=False)
        print(f"  wrote {fn}  CV={cv_val:.4f}  mean={p.mean():+.3f}")


if __name__ == "__main__":
    main()
