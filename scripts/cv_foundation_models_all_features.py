"""5-fold CV for TabPFN v2 + TabM on the FULL feature set (adds x6, x7, angle).

The baseline run in cv_foundation_models.py dropped x6 and x7 per the
Round 2 / post-submission finding that x6^2 + x7^2 = 324 exactly and
theta = atan2(x7, x6) is uniform on [-pi, pi] and orthogonal to the
target (corr(theta, residual) = +0.01).

That finding was reached with EBM. Maybe TabPFN v2 / TabM can extract
signal from x6, x7 that EBM could not. This CV run answers that.

Feature set vs. baseline:

  baseline (cv_foundation_models.py):  x1 x2 x4 x5 x8 x9 x10 x11 + sent + city
  ALL FEATURES (this script):          + x6 x7 + sin(theta) cos(theta)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL, SEED, preprocess  # noqa: E402
from cv_x4_x9_swap_ensemble import ebm_features, design_matrix, fit_ebm  # noqa: E402
from cv_cross_LE_tune import LOCKED_COEFS_B, LIN_COL_ORDER  # noqa: E402
from cv_foundation_models import (  # noqa: E402
    fit_predict_tabpfn, fit_predict_tabm,
    lin_x4_free_predict, lin_x4_locked_predict, mae,
)

DATA = REPO / "data"
OUT = REPO / "plots" / "foundation_models"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5

NUM_COLS_ALL = ["x1", "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"]


def fm_arrays_all(df: pd.DataFrame, x5_median: float):
    """Return X_num (12 cols: 10 numeric + sent + sin/cos angle) and x_cat (city)."""
    df_c = df.copy()
    df_c["x5"] = df_c["x5"].where(df_c["x5"] != SENTINEL, x5_median)

    x5_is_sent = (df["x5"] == SENTINEL).astype(np.float32).values.reshape(-1, 1)

    x6 = df["x6"].values.astype(np.float32)
    x7 = df["x7"].values.astype(np.float32)
    theta = np.arctan2(x7, x6)
    sin_t = np.sin(theta).astype(np.float32).reshape(-1, 1)
    cos_t = np.cos(theta).astype(np.float32).reshape(-1, 1)

    X_num = np.concatenate([
        df_c[NUM_COLS_ALL].values.astype(np.float32),
        x5_is_sent,
        sin_t, cos_t,
    ], axis=1)  # 10 + 1 + 2 = 13 numeric columns
    x_cat = (df["City"] == "Zaragoza").astype(np.int64).values.reshape(-1, 1)
    return X_num, x_cat


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = {k: np.zeros(len(df)) for k in
           ["tabpfn_all", "tabm_all",
            "lin_x4_free", "lin_x4_b", "ebm_x9", "ebm_full"]}

    print("=" * 78)
    print("5-fold CV — TabPFN v2 + TabM with ALL features (incl. x6, x7, sin/cos(theta))")
    print("=" * 78)
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        X_num_tr, x_cat_tr = fm_arrays_all(sub_tr, x5m)
        X_num_va, x_cat_va = fm_arrays_all(sub_va, x5m)
        X_flat_tr = np.concatenate([X_num_tr, x_cat_tr.astype(np.float32)], axis=1)
        X_flat_va = np.concatenate([X_num_va, x_cat_va.astype(np.float32)], axis=1)

        t1 = time.time()
        oof["tabpfn_all"][va] = fit_predict_tabpfn(X_flat_tr, sub_tr["target"].values, X_flat_va)
        t_pfn = time.time() - t1

        t1 = time.time()
        oof["tabm_all"][va] = fit_predict_tabm(
            X_num_tr, x_cat_tr, sub_tr["target"].values,
            X_num_va, x_cat_va,
        )
        t_tabm = time.time() - t1

        # Reference baselines (unchanged — use original EBM/linear features)
        oof["lin_x4_free"][va] = lin_x4_free_predict(sub_tr, sub_va)
        oof["lin_x4_b"][va]    = lin_x4_locked_predict(sub_tr, sub_va)

        feats_x9 = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(sub_tr, feats_x9, x5m); X_va = preprocess(sub_va, feats_x9, x5m)
        oof["ebm_x9"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        feats_full = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats_full, x5m); X_va = preprocess(sub_va, feats_full, x5m)
        oof["ebm_full"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        print(f"  fold {fold+1}/{N_SPLITS}  tabpfn={t_pfn:.0f}s  tabm={t_tabm:.0f}s  "
              f"total={time.time()-t0:.0f}s")

    cross_LE = 0.5 * (oof["lin_x4_free"] + oof["ebm_x9"])
    cross_LE_b = 0.5 * (oof["lin_x4_b"] + oof["ebm_x9"])
    triple = 0.5 * cross_LE_b + 0.5 * oof["ebm_full"]

    fm_avg = 0.5 * (oof["tabpfn_all"] + oof["tabm_all"])

    rows = []

    def record(name, pred):
        rows.append({
            "model": name,
            "overall": mae(pred, y),
            "non_sent": mae(pred[~is_sent], y[~is_sent]),
            "sent":     mae(pred[is_sent], y[is_sent]),
        })

    record("TabPFN v2 (all feats)",              oof["tabpfn_all"])
    record("TabM (all feats)",                   oof["tabm_all"])
    record("cross_LE (ref, LB 2.94)",            cross_LE)
    record("triple (ref, CV 2.824)",             triple)
    record("FM_avg = 0.5*(TabPFN+TabM), all",    fm_avg)

    for w in [0.3, 0.5, 0.7]:
        record(f"{w:.1f}*TabPFN_all + {1-w:.1f}*cross_LE",
               w * oof["tabpfn_all"] + (1-w) * cross_LE)
        record(f"{w:.1f}*TabPFN_all + {1-w:.1f}*triple",
               w * oof["tabpfn_all"] + (1-w) * triple)
        record(f"{w:.1f}*FM_avg_all + {1-w:.1f}*triple",
               w * fm_avg + (1-w) * triple)

    results = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)
    print("\n" + "=" * 78)
    print("Ranked by CV MAE (overall)  —  ALL FEATURES")
    print("=" * 78)
    for _, r in results.iterrows():
        print(f"  {r['model']:<50s}  "
              f"overall={r['overall']:6.3f}  "
              f"non-sent={r['non_sent']:6.3f}  "
              f"sent={r['sent']:6.3f}")

    results.to_csv(OUT / "cv_foundation_models_all_features.csv", index=False)
    pd.DataFrame({**oof, "target": y, "is_sent": is_sent.astype(int)}).to_csv(
        OUT / "cv_foundation_models_all_features_oof.csv", index=False)

    # -----------------------------------------------------------------
    # Direct comparison: all-features TabPFN vs baseline TabPFN CV (3.280)
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("All-features vs baseline  —  foundation models")
    print("=" * 78)
    print(f"  TabPFN v2 baseline (no x6/x7):  CV 3.280  non-sent 2.003")
    print(f"  TabPFN v2 all-features:         CV {mae(oof['tabpfn_all'], y):.3f}  "
          f"non-sent {mae(oof['tabpfn_all'][~is_sent], y[~is_sent]):.3f}")
    print(f"  TabM baseline (no x6/x7):       CV 3.935  non-sent 2.670")
    print(f"  TabM all-features:              CV {mae(oof['tabm_all'], y):.3f}  "
          f"non-sent {mae(oof['tabm_all'][~is_sent], y[~is_sent]):.3f}")


if __name__ == "__main__":
    main()
