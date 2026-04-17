"""EBM with x4 forced to a linear term.

EBM learns a nonparametric shape per feature. Our diagnostics show x4's
true relationship with target is cleanly linear (slope ≈ +30.5), but
EBM's fitted shape has no mechanism to prefer linearity — it may carry
a subtle training-specific artifact near the x4 ≈ 0 gap (which has
zero training observations, so EBM's bin interpolation is unconstrained
there). Test has 34% of rows inside that gap.

Workaround: residualise x4 via a linear fit, then fit EBM on the
remaining features only. At predict time add β_x4 · x4_test back.

Three variants:
  1. free β_x4  — fitted once on the whole training fold via OLS with
                   all other features present (partial β_x4).
  2. locked β_x4 = +30  — the A1/A2 declared integer (validated by our
                           earlier locked-integer CV experiment).
  3. locked β_x4 = +31  — the slight-disagreement alternative.

Baseline: ordinary EBM (same hyperparams), unchanged, for CV reference.
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


def fit_ebm(X_tr: pd.DataFrame, y_tr: np.ndarray,
            smoothing: str = "heavy"):
    from interpret.glassbox import ExplainableBoostingRegressor
    kw = dict(interactions=10, max_rounds=2000, min_samples_leaf=10,
              max_bins=128, random_state=SEED)
    if smoothing == "heavy":
        kw.update(smoothing_rounds=2000, interaction_smoothing_rounds=500)
    return ExplainableBoostingRegressor(**kw).fit(X_tr, y_tr)


def cv_variant(df: pd.DataFrame, locked_beta_x4: float | None,
               smoothing: str = "heavy") -> tuple[float, float, float, list[float]]:
    """5-fold CV. locked_beta_x4 = None -> fit beta_x4 per fold via OLS.
    Returns (overall MAE, non-sent MAE, sent MAE, per-fold betas)."""
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    betas = []
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        X_tr_full = preprocess(sub_tr, FEATURES_ALL, x5_med)
        X_va_full = preprocess(sub_va, FEATURES_ALL, x5_med)

        if locked_beta_x4 is None:
            # Fit partial beta_x4 via OLS on ALL features, to get the
            # conditional slope (controlling for everything else).
            lr = LinearRegression().fit(X_tr_full, sub_tr["target"].values)
            beta_x4 = float(lr.coef_[X_tr_full.columns.get_loc("x4")])
        else:
            beta_x4 = float(locked_beta_x4)
        betas.append(beta_x4)

        resid_tr = sub_tr["target"].values - beta_x4 * sub_tr["x4"].values

        X_tr_no_x4 = X_tr_full.drop(columns=["x4"])
        X_va_no_x4 = X_va_full.drop(columns=["x4"])

        ebm = fit_ebm(X_tr_no_x4, resid_tr, smoothing=smoothing)
        oof[va] = ebm.predict(X_va_no_x4) + beta_x4 * sub_va["x4"].values

    m = float(np.mean(np.abs(oof - y)))
    m_ns = float(np.mean(np.abs(oof[~is_sent] - y[~is_sent])))
    m_sn = float(np.mean(np.abs(oof[is_sent] - y[is_sent])))
    return m, m_ns, m_sn, betas


def cv_baseline(df: pd.DataFrame, smoothing: str = "heavy"):
    """Plain EBM (no residualisation)."""
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        X_tr = preprocess(sub_tr, FEATURES_ALL, x5_med)
        X_va = preprocess(sub_va, FEATURES_ALL, x5_med)
        oof[va] = fit_ebm(X_tr, sub_tr["target"].values, smoothing).predict(X_va)
    m = float(np.mean(np.abs(oof - y)))
    m_ns = float(np.mean(np.abs(oof[~is_sent] - y[~is_sent])))
    m_sn = float(np.mean(np.abs(oof[is_sent] - y[is_sent])))
    return m, m_ns, m_sn


def build_submission(df: pd.DataFrame, test: pd.DataFrame,
                     beta_x4: float | None, name: str,
                     smoothing: str = "heavy"):
    x5_med = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    X_tr_full = preprocess(df, FEATURES_ALL, x5_med)
    X_te_full = preprocess(test, FEATURES_ALL, x5_med)

    if beta_x4 is None:
        lr = LinearRegression().fit(X_tr_full, df["target"].values)
        beta_x4 = float(lr.coef_[X_tr_full.columns.get_loc("x4")])
    print(f"  using beta_x4 = {beta_x4:+.3f}")

    resid = df["target"].values - beta_x4 * df["x4"].values
    X_tr = X_tr_full.drop(columns=["x4"])
    X_te = X_te_full.drop(columns=["x4"])
    ebm = fit_ebm(X_tr, resid, smoothing)
    preds = ebm.predict(X_te) + beta_x4 * test["x4"].values
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(
        SUBS / f"submission_{name}.csv", index=False
    )
    print(f"  wrote submission_{name}.csv  mean={preds.mean():+.3f}  "
          f"range=[{preds.min():+.2f}, {preds.max():+.2f}]")


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    print("=" * 90)
    print(f"{'variant':<45s}  {'overall':>8s}  {'non-sent':>8s}  {'sent':>7s}  {'beta_x4':>9s}")
    print("=" * 90)

    t0 = time.time()
    m, mn, ms = cv_baseline(df, smoothing="heavy")
    print(f"{'EBM heavy_smooth (baseline)':<45s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  {'nonparam':>9s}  [{time.time()-t0:.0f}s]")

    t0 = time.time()
    m, mn, ms = cv_baseline(df, smoothing="light")
    print(f"{'EBM R2 tuned (LB 5.66)':<45s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  {'nonparam':>9s}  [{time.time()-t0:.0f}s]")

    for beta_desc, beta in [("x4 linear FREE β", None),
                             ("x4 linear LOCKED β=+30", +30.0),
                             ("x4 linear LOCKED β=+31", +31.0)]:
        t0 = time.time()
        m, mn, ms, betas = cv_variant(df, beta, smoothing="heavy")
        beta_str = f"{np.mean(betas):+.2f}" if betas else "n/a"
        print(f"{'EBM heavy + ' + beta_desc:<45s}  {m:8.3f}  {mn:8.3f}  {ms:7.3f}  {beta_str:>9s}  [{time.time()-t0:.0f}s]")

    print("\n" + "=" * 90)
    print("Building submissions")
    print("=" * 90)
    for name, beta in [("ebm_heavy_x4linear_free", None),
                       ("ebm_heavy_x4linear_30",   +30.0)]:
        build_submission(df, test, beta, name, smoothing="heavy")


if __name__ == "__main__":
    main()
