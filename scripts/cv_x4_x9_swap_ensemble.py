"""Four single-view models + cross-ensembles.

Because x4 and x9 are highly correlated in training (r=0.83) but
independent in test (r=0.001), models built on ONLY x4 or ONLY x9
should learn similar training-target patterns but disagree on off-
diagonal test rows. Ensembling them may reduce error if their
training-fit signals partially cancel out on the test shift.

Four base models:
  EBM_x4 — full feature set EXCEPT x9
  EBM_x9 — full feature set EXCEPT x4
  LIN_x4 — A2-shape linear EXCEPT x9 (x1^2 + cos(5π*x2) + x4 + ...)
  LIN_x9 — A2-shape linear EXCEPT x4 (x1^2 + cos(5π*x2) + x9 + ...)

Ensembles:
  EBM_avg      = 0.5 * (EBM_x4 + EBM_x9)
  LIN_avg      = 0.5 * (LIN_x4 + LIN_x9)
  ALL_avg      = 0.25 * (all four)
  cross_EL     = 0.5 * (EBM_x4 + LIN_x9)
  cross_LE     = 0.5 * (LIN_x4 + EBM_x9)

Plus reference: full EBM with both (our current LB-4.9 config).

5-fold CV on dataset.csv with KFold(shuffle, seed=42). Submissions
built for all ensembles that improve over the full-EBM baseline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import FEATURES_ALL, SENTINEL, SEED, preprocess  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
N_SPLITS = 5
EBM_KW = dict(interactions=10, max_rounds=4000, min_samples_leaf=10,
              max_bins=128, smoothing_rounds=4000,
              interaction_smoothing_rounds=1000, random_state=SEED)


def fit_ebm(X, y):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(**EBM_KW).fit(X, y)


def design_matrix(df: pd.DataFrame, x5_median: float,
                  include_x4: bool, include_x9: bool) -> np.ndarray:
    """A2-style design matrix, optionally dropping x4 and/or x9."""
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values
    city = (df["City"] == "Zaragoza").astype(float).values

    cols = [
        df["x1"].values ** 2,
        np.cos(5 * np.pi * df["x2"].values),
    ]
    if include_x4:
        cols.append(df["x4"].values)
    cols.append(x5)
    cols.append(is_sent)
    cols.append(df["x8"].values)
    cols.append(df["x10"].values)
    cols.append(df["x11"].values)
    cols.append(df["x10"].values * df["x11"].values)
    cols.append(city)
    if include_x9:
        cols.append(df["x9"].values)
    return np.column_stack(cols)


def ebm_features(with_x4: bool, with_x9: bool) -> list[str]:
    feats = list(FEATURES_ALL)  # ['x1','x2','x4','x5','x8','x9','x10','x11']
    if not with_x4:
        feats.remove("x4")
    if not with_x9:
        feats.remove("x9")
    return feats


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Four base OOF buffers
    oof = {"ebm_x4": np.zeros(len(df)), "ebm_x9": np.zeros(len(df)),
           "lin_x4": np.zeros(len(df)), "lin_x9": np.zeros(len(df)),
           "ebm_full": np.zeros(len(df))}

    print("=" * 70)
    print("5-fold CV — fitting 5 models per fold")
    print("=" * 70)
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # ---- EBM with x4, no x9
        feats_x4 = ebm_features(with_x4=True, with_x9=False)
        X_tr = preprocess(sub_tr, feats_x4, x5m)
        X_va = preprocess(sub_va, feats_x4, x5m)
        oof["ebm_x4"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        # ---- EBM with x9, no x4
        feats_x9 = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(sub_tr, feats_x9, x5m)
        X_va = preprocess(sub_va, feats_x9, x5m)
        oof["ebm_x9"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        # ---- EBM full reference
        feats_full = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats_full, x5m)
        X_va = preprocess(sub_va, feats_full, x5m)
        oof["ebm_full"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        # ---- Linear with x4, no x9
        X_tr_lin = design_matrix(sub_tr, x5m, include_x4=True, include_x9=False)
        X_va_lin = design_matrix(sub_va, x5m, include_x4=True, include_x9=False)
        lr = LinearRegression().fit(X_tr_lin, sub_tr["target"].values)
        oof["lin_x4"][va] = lr.predict(X_va_lin)

        # ---- Linear with x9, no x4
        X_tr_lin = design_matrix(sub_tr, x5m, include_x4=False, include_x9=True)
        X_va_lin = design_matrix(sub_va, x5m, include_x4=False, include_x9=True)
        lr = LinearRegression().fit(X_tr_lin, sub_tr["target"].values)
        oof["lin_x9"][va] = lr.predict(X_va_lin)

        print(f"  fold {fold+1}/{N_SPLITS} done in {time.time()-t0:.0f}s")

    # ---- Base-model scores
    print("\n" + "=" * 70)
    print(f"{'model':<40s}  {'overall':>8s}  {'non-sent':>9s}")
    print("=" * 70)
    scores = {}
    for name in ["ebm_full", "ebm_x4", "ebm_x9", "lin_x4", "lin_x9"]:
        m = mae(oof[name], y)
        mn = mae(oof[name][~is_sent], y[~is_sent])
        scores[name] = (m, mn)
        print(f"{name:<40s}  {m:8.3f}  {mn:9.3f}")

    # ---- Ensembles
    print("\n" + "=" * 70)
    print("Ensembles")
    print("=" * 70)
    ensembles = {
        "EBM_avg (x4 + x9 EBMs)":            0.5 * (oof["ebm_x4"] + oof["ebm_x9"]),
        "LIN_avg (x4 + x9 linears)":          0.5 * (oof["lin_x4"] + oof["lin_x9"]),
        "ALL4_avg (all four models)":         0.25 * (oof["ebm_x4"] + oof["ebm_x9"]
                                                      + oof["lin_x4"] + oof["lin_x9"]),
        "cross_EL (EBM_x4 + LIN_x9)":         0.5 * (oof["ebm_x4"] + oof["lin_x9"]),
        "cross_LE (LIN_x4 + EBM_x9)":         0.5 * (oof["lin_x4"] + oof["ebm_x9"]),
        "full_and_EBMavg (50/50)":            0.5 * (oof["ebm_full"] + 0.5 * (oof["ebm_x4"] + oof["ebm_x9"])),
    }
    for name, pred in ensembles.items():
        m = mae(pred, y); mn = mae(pred[~is_sent], y[~is_sent])
        scores[name] = (m, mn)
        print(f"{name:<40s}  {m:8.3f}  {mn:9.3f}")

    # ---- Disagreement diagnostic
    print("\n" + "=" * 70)
    print("Model-pair disagreement (std of OOF prediction differences)")
    print("=" * 70)
    for a, b in [("ebm_x4", "ebm_x9"), ("lin_x4", "lin_x9"),
                 ("ebm_full", "ebm_x4"), ("ebm_full", "ebm_x9")]:
        d = oof[a] - oof[b]
        print(f"  {a} vs {b}:  mean={d.mean():+.3f}  std={d.std():.3f}  "
              f"|d|.mean={np.abs(d).mean():.3f}")

    # ---- Build submissions on the full data for the TOP ensembles
    print("\n" + "=" * 70)
    print("Building submissions for top-CV ensembles (vs full EBM baseline)")
    print("=" * 70)
    x5m_full = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    def fit_all_full(feats_x4, feats_x9, feats_full):
        out = {}
        # EBMs
        X_tr = preprocess(df, feats_x4, x5m_full)
        X_te = preprocess(test, feats_x4, x5m_full)
        out["ebm_x4"] = fit_ebm(X_tr, df["target"].values).predict(X_te)
        X_tr = preprocess(df, feats_x9, x5m_full)
        X_te = preprocess(test, feats_x9, x5m_full)
        out["ebm_x9"] = fit_ebm(X_tr, df["target"].values).predict(X_te)
        X_tr = preprocess(df, feats_full, x5m_full)
        X_te = preprocess(test, feats_full, x5m_full)
        out["ebm_full"] = fit_ebm(X_tr, df["target"].values).predict(X_te)
        # Linears
        X_tr = design_matrix(df, x5m_full, True, False)
        X_te = design_matrix(test, x5m_full, True, False)
        out["lin_x4"] = LinearRegression().fit(X_tr, df["target"].values).predict(X_te)
        X_tr = design_matrix(df, x5m_full, False, True)
        X_te = design_matrix(test, x5m_full, False, True)
        out["lin_x9"] = LinearRegression().fit(X_tr, df["target"].values).predict(X_te)
        return out

    preds = fit_all_full(
        ebm_features(True, False),
        ebm_features(False, True),
        ebm_features(True, True),
    )
    ens_submissions = {
        "ebm_avg_x4_x9":        0.5 * (preds["ebm_x4"] + preds["ebm_x9"]),
        "lin_avg_x4_x9":        0.5 * (preds["lin_x4"] + preds["lin_x9"]),
        "all4_avg_x4_x9":       0.25 * (preds["ebm_x4"] + preds["ebm_x9"]
                                        + preds["lin_x4"] + preds["lin_x9"]),
        "full_and_ebmavg_50_50":0.5 * (preds["ebm_full"] + 0.5 * (preds["ebm_x4"] + preds["ebm_x9"])),
    }
    for name, pred in ens_submissions.items():
        pd.DataFrame({"id": test["id"], "target": pred}).to_csv(
            SUBS / f"submission_ensemble_{name}.csv", index=False)
        print(f"  wrote submission_ensemble_{name}.csv  "
              f"mean={pred.mean():+.3f}  range=[{pred.min():+.2f}, {pred.max():+.2f}]")


if __name__ == "__main__":
    main()
