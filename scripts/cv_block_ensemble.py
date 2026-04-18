"""Block-ensemble CV + submission builder.

Three complementary "views" that each escape the x4-x9 selection shift:

  block view   : EBM trained within sign(x4) subset. Full feature set.
                 Within each block the training joint (x4, x9) has r=~0,
                 matching test. Per-row prediction routes by sign(x4).
  no-x9 view   : LIN_x4 (A2-style basis) or EBM_x4 — can't exploit the
                 spurious x4-x9 coupling because x9 isn't visible.
  no-x4 view   : EBM_x9 — can't exploit because x4 isn't visible.

5-fold CV on dataset.csv. Strategies tested:

  S1  block_route   EBM_block_{sign(x4)}
  S2  block_avg     0.5*EBM_block1 + 0.5*EBM_block2
  S3  cross_LE      0.5*LIN_x4 + 0.5*EBM_x9   (current LB 2.94 winner)
  S4  triple_view   (1/3)*EBM_block_s + (1/3)*LIN_x4 + (1/3)*EBM_x9
  S5  four_view     equal-weight {block_s, LIN_x4, EBM_x4, EBM_x9}
  S6  block_cross   0.5*EBM_block_s + 0.5*cross_LE
  S7  block_avg+x_  0.5*block_avg + 0.25*LIN_x4 + 0.25*EBM_x9
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError:
    raise SystemExit("interpret-core not installed. `pip install interpret-core`.")

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


def preprocess(df: pd.DataFrame, features: list[str], x5_median: float) -> pd.DataFrame:
    out = df[features].copy()
    out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
    out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
    out["city"] = (df["City"] == "Zaragoza").astype(float)
    return out


def ebm_features(with_x4: bool, with_x9: bool) -> list[str]:
    feats = list(FEATS_ALL)
    if not with_x4: feats.remove("x4")
    if not with_x9: feats.remove("x9")
    return feats


def design_matrix(df: pd.DataFrame, x5_median: float,
                  include_x4: bool, include_x9: bool) -> np.ndarray:
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values
    city = (df["City"] == "Zaragoza").astype(float).values
    cols = [df["x1"].values ** 2,
            np.cos(5 * np.pi * df["x2"].values)]
    if include_x4: cols.append(df["x4"].values)
    cols += [x5, is_sent, df["x8"].values, df["x10"].values, df["x11"].values,
             df["x10"].values * df["x11"].values, city]
    if include_x9: cols.append(df["x9"].values)
    return np.column_stack(cols)


def fit_ebm(X: pd.DataFrame, y: np.ndarray) -> ExplainableBoostingRegressor:
    return ExplainableBoostingRegressor(**EBM_KW).fit(X, y)


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test  = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y     = train["target"].values
    is_sent = (train["x5"] == SENTINEL).to_numpy()
    sign_x4 = np.sign(train["x4"].to_numpy())
    print(f"train: {len(train)}   block1(x4>0): {(sign_x4 > 0).sum()}   "
          f"block2(x4<0): {(sign_x4 < 0).sum()}   sentinels: {is_sent.sum()}")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    n = len(train)

    # OOF containers
    oof = {k: np.zeros(n) for k in
           ["block_s", "block1", "block2", "lin_x4", "ebm_x4", "ebm_x9", "ebm_full",
            "q11", "q12", "q21", "q22", "quad_s"]}

    for fold, (tr, va) in enumerate(kf.split(train)):
        sub_tr = train.iloc[tr].reset_index(drop=True)
        sub_va = train.iloc[va].reset_index(drop=True)
        y_tr   = sub_tr["target"].values
        x5m    = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # --- Block models (within sign(x4)) -------------------------------
        mask_b1 = (sub_tr["x4"] > 0).values
        mask_b2 = (sub_tr["x4"] < 0).values
        feats_full = ebm_features(with_x4=True, with_x9=True)

        X_tr_b1 = preprocess(sub_tr[mask_b1], feats_full, x5m)
        X_tr_b2 = preprocess(sub_tr[mask_b2], feats_full, x5m)
        X_va_full = preprocess(sub_va, feats_full, x5m)

        m_b1 = fit_ebm(X_tr_b1, y_tr[mask_b1])
        m_b2 = fit_ebm(X_tr_b2, y_tr[mask_b2])
        oof["block1"][va] = m_b1.predict(X_va_full)
        oof["block2"][va] = m_b2.predict(X_va_full)
        # Routed: pick by sign(x4) of the val row
        route = (sub_va["x4"].values > 0)
        oof["block_s"][va] = np.where(route, oof["block1"][va], oof["block2"][va])

        # --- LIN_x4 (no x9) free-fit --------------------------------------
        X_tr_lin = design_matrix(sub_tr, x5m, include_x4=True, include_x9=False)
        X_va_lin = design_matrix(sub_va, x5m, include_x4=True, include_x9=False)
        oof["lin_x4"][va] = LinearRegression().fit(X_tr_lin, y_tr).predict(X_va_lin)

        # --- EBM_x4 (no x9) -----------------------------------------------
        feats = ebm_features(with_x4=True, with_x9=False)
        m = fit_ebm(preprocess(sub_tr, feats, x5m), y_tr)
        oof["ebm_x4"][va] = m.predict(preprocess(sub_va, feats, x5m))

        # --- EBM_x9 (no x4) -----------------------------------------------
        feats = ebm_features(with_x4=False, with_x9=True)
        m = fit_ebm(preprocess(sub_tr, feats, x5m), y_tr)
        oof["ebm_x9"][va] = m.predict(preprocess(sub_va, feats, x5m))

        # --- EBM_full (reference) -----------------------------------------
        feats = ebm_features(with_x4=True, with_x9=True)
        m = fit_ebm(preprocess(sub_tr, feats, x5m), y_tr)
        oof["ebm_full"][va] = m.predict(preprocess(sub_va, feats, x5m))

        # --- 2x2 quadrant blocks (sign(x4), sign(x8)) ---------------------
        # q11: x4>0, x8>0   q12: x4>0, x8<0
        # q21: x4<0, x8>0   q22: x4<0, x8<0 (contains the A1 clamp)
        quad_masks_tr = {
            "q11": (sub_tr["x4"] > 0) & (sub_tr["x8"] > 0),
            "q12": (sub_tr["x4"] > 0) & (sub_tr["x8"] < 0),
            "q21": (sub_tr["x4"] < 0) & (sub_tr["x8"] > 0),
            "q22": (sub_tr["x4"] < 0) & (sub_tr["x8"] < 0),
        }
        quad_models = {}
        for q, m_tr in quad_masks_tr.items():
            m_tr_arr = m_tr.values
            quad_models[q] = fit_ebm(
                preprocess(sub_tr[m_tr_arr], feats_full, x5m), y_tr[m_tr_arr])
            oof[q][va] = quad_models[q].predict(X_va_full)
        # Route val rows by their own (sign(x4), sign(x8))
        q_va = np.where(
            sub_va["x4"].values > 0,
            np.where(sub_va["x8"].values > 0, "q11", "q12"),
            np.where(sub_va["x8"].values > 0, "q21", "q22"),
        )
        for q in ("q11", "q12", "q21", "q22"):
            sel = (q_va == q)
            oof["quad_s"][va[sel]] = oof[q][va[sel]]

        print(f"  fold {fold+1}/5 done")

    # -------- Assemble strategies --------------------------------------------
    strategies = {
        "block_route":      oof["block_s"],
        "block_avg":        0.5 * oof["block1"] + 0.5 * oof["block2"],
        "cross_LE":         0.5 * oof["lin_x4"] + 0.5 * oof["ebm_x9"],
        "triple_view":      (oof["block_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 3.0,
        "four_view":        (oof["block_s"] + oof["lin_x4"] + oof["ebm_x4"] + oof["ebm_x9"]) / 4.0,
        "block_cross":      0.5 * oof["block_s"] + 0.25 * oof["lin_x4"] + 0.25 * oof["ebm_x9"],
        "block_avg+cross":  0.25 * oof["block1"] + 0.25 * oof["block2"]
                             + 0.25 * oof["lin_x4"] + 0.25 * oof["ebm_x9"],
        # 2x2 quadrant-block strategies
        "quad_route":       oof["quad_s"],
        "quad_avg":         0.25 * (oof["q11"] + oof["q12"] + oof["q21"] + oof["q22"]),
        "quad_cross":       0.5 * oof["quad_s"] + 0.25 * oof["lin_x4"] + 0.25 * oof["ebm_x9"],
        "quad_triple":      (oof["quad_s"] + oof["lin_x4"] + oof["ebm_x9"]) / 3.0,
        "quad_four_view":   (oof["quad_s"] + oof["lin_x4"] + oof["ebm_x4"] + oof["ebm_x9"]) / 4.0,
        "quad_avg+cross":   0.5 * (0.25 * (oof["q11"] + oof["q12"] + oof["q21"] + oof["q22"]))
                             + 0.25 * oof["lin_x4"] + 0.25 * oof["ebm_x9"],
        # Bases for reference
        "lin_x4_alone":     oof["lin_x4"],
        "ebm_x4_alone":     oof["ebm_x4"],
        "ebm_x9_alone":     oof["ebm_x9"],
        "ebm_full_alone":   oof["ebm_full"],
    }

    # Per-quadrant diagnostic
    x4 = train["x4"].to_numpy()
    x9 = train["x9"].to_numpy()
    # "safe" = on-diagonal for training (always true here), tag by sign(x4)
    print(f"\n{'strategy':<20s} {'CV MAE':>8s} {'non-sent':>10s} {'sent':>8s}"
          f" {'b1_cv':>8s} {'b2_cv':>8s}")
    print("-" * 70)
    rows = []
    for name, pred in strategies.items():
        overall = mae(pred, y)
        ns      = mae(pred[~is_sent], y[~is_sent])
        se      = mae(pred[is_sent],  y[is_sent])
        b1      = mae(pred[x4 > 0],   y[x4 > 0])
        b2      = mae(pred[x4 < 0],   y[x4 < 0])
        rows.append((name, overall, ns, se, b1, b2))
        flag = " <-- beats cross_LE" if (overall < 2.97 and name != "cross_LE") else ""
        print(f"{name:<20s} {overall:>8.4f} {ns:>10.4f} {se:>8.4f}"
              f" {b1:>8.4f} {b2:>8.4f}{flag}")

    results = pd.DataFrame(rows, columns=["strategy", "cv", "non_sent", "sent", "block1", "block2"])
    out_dir = REPO / "plots" / "block_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "cv_results.csv", index=False)
    print(f"\nCV table -> {out_dir/'cv_results.csv'}")

    # -------- Submissions for strategies that beat cross_LE -------------------
    winners = [r for r in rows if r[1] < 2.97 and r[0] != "cross_LE"]
    if not winners:
        print("\nNo strategy beats cross_LE (CV 2.97). No submissions built.")
        return

    print(f"\nBuilding submissions for {len(winners)} strategy(ies) "
          f"that beat cross_LE...")
    x5m_full = float(train.loc[train["x5"] != SENTINEL, "x5"].median())

    # Train every base model on full data
    print("  training full-data base models...")
    mask_b1 = (train["x4"] > 0).values
    mask_b2 = (train["x4"] < 0).values
    feats_full = ebm_features(with_x4=True, with_x9=True)
    m_b1 = fit_ebm(preprocess(train[mask_b1], feats_full, x5m_full), y[mask_b1])
    m_b2 = fit_ebm(preprocess(train[mask_b2], feats_full, x5m_full), y[mask_b2])

    X_tr_lin = design_matrix(train, x5m_full, include_x4=True, include_x9=False)
    X_te_lin = design_matrix(test,  x5m_full, include_x4=True, include_x9=False)
    preds_lin_x4 = LinearRegression().fit(X_tr_lin, y).predict(X_te_lin)

    feats = ebm_features(with_x4=True, with_x9=False)
    preds_ebm_x4 = fit_ebm(preprocess(train, feats, x5m_full), y).predict(
        preprocess(test, feats, x5m_full))

    feats = ebm_features(with_x4=False, with_x9=True)
    preds_ebm_x9 = fit_ebm(preprocess(train, feats, x5m_full), y).predict(
        preprocess(test, feats, x5m_full))

    X_te_full = preprocess(test, feats_full, x5m_full)
    preds_b1 = m_b1.predict(X_te_full)
    preds_b2 = m_b2.predict(X_te_full)
    route_te = (test["x4"].values > 0)
    preds_block_s = np.where(route_te, preds_b1, preds_b2)

    # 2x2 quadrant models on full data
    quad_masks = {
        "q11": (train["x4"] > 0) & (train["x8"] > 0),
        "q12": (train["x4"] > 0) & (train["x8"] < 0),
        "q21": (train["x4"] < 0) & (train["x8"] > 0),
        "q22": (train["x4"] < 0) & (train["x8"] < 0),
    }
    preds_q = {}
    for q, m_tr in quad_masks.items():
        m_tr_arr = m_tr.values
        print(f"  training full-data {q} (n={m_tr_arr.sum()})...")
        m_q = fit_ebm(preprocess(train[m_tr_arr], feats_full, x5m_full), y[m_tr_arr])
        preds_q[q] = m_q.predict(X_te_full)
    q_te = np.where(
        test["x4"].values > 0,
        np.where(test["x8"].values > 0, "q11", "q12"),
        np.where(test["x8"].values > 0, "q21", "q22"),
    )
    preds_quad_s = np.empty(len(test))
    for q in ("q11", "q12", "q21", "q22"):
        sel = (q_te == q)
        preds_quad_s[sel] = preds_q[q][sel]

    strat_test = {
        "block_route":      preds_block_s,
        "block_avg":        0.5 * preds_b1 + 0.5 * preds_b2,
        "triple_view":      (preds_block_s + preds_lin_x4 + preds_ebm_x9) / 3.0,
        "four_view":        (preds_block_s + preds_lin_x4 + preds_ebm_x4 + preds_ebm_x9) / 4.0,
        "block_cross":      0.5 * preds_block_s + 0.25 * preds_lin_x4 + 0.25 * preds_ebm_x9,
        "block_avg+cross":  0.25 * preds_b1 + 0.25 * preds_b2
                             + 0.25 * preds_lin_x4 + 0.25 * preds_ebm_x9,
        "quad_route":       preds_quad_s,
        "quad_avg":         0.25 * (preds_q["q11"] + preds_q["q12"] + preds_q["q21"] + preds_q["q22"]),
        "quad_cross":       0.5 * preds_quad_s + 0.25 * preds_lin_x4 + 0.25 * preds_ebm_x9,
        "quad_triple":      (preds_quad_s + preds_lin_x4 + preds_ebm_x9) / 3.0,
        "quad_four_view":   (preds_quad_s + preds_lin_x4 + preds_ebm_x4 + preds_ebm_x9) / 4.0,
        "quad_avg+cross":   0.5 * (0.25 * (preds_q["q11"] + preds_q["q12"] + preds_q["q21"] + preds_q["q22"]))
                             + 0.25 * preds_lin_x4 + 0.25 * preds_ebm_x9,
    }
    for name, cv, *_ in winners:
        if name not in strat_test:
            continue
        p = strat_test[name]
        fn = f"submission_{name}.csv"
        pd.DataFrame({"id": test["id"], "target": p}).to_csv(SUBS / fn, index=False)
        print(f"  wrote {fn}  CV={cv:.4f}  mean={p.mean():+.3f}")


if __name__ == "__main__":
    main()
