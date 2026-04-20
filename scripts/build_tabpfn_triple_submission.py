"""Build the TabPFN + triple blend submission — our best CV so far.

CV MAE from cv_foundation_models.py:
   0.3 * TabPFN v2 + 0.7 * triple  =  2.793  (best overall)
   triple (ref)                    =  2.824
   cross_LE (LB 2.94)              =  2.973

Triple is:
   triple = 0.5 * cross_LE_locked_b + 0.5 * EBM_full
   cross_LE_locked_b = 0.5 * (LIN_x4_locked_b + EBM_x9)

So full blend:
   pred = 0.3 * tabpfn
        + 0.7 * ( 0.5 * 0.5 * (LIN_x4_locked_b + EBM_x9)
                + 0.5 * EBM_full )
        = 0.3  * tabpfn
        + 0.175 * LIN_x4_locked_b
        + 0.175 * EBM_x9
        + 0.35  * EBM_full
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL  # noqa: E402
from cv_x4_x9_swap_ensemble import ebm_features, design_matrix, fit_ebm  # noqa: E402
from cv_ebm_variants import preprocess  # noqa: E402
from cv_cross_LE_tune import LOCKED_COEFS_B, LIN_COL_ORDER  # noqa: E402
from cv_foundation_models import fit_predict_tabpfn, fm_arrays  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # -------------- TabPFN on full data --------------
    X_num_tr, x_cat_tr, _ = fm_arrays(df, x5m)
    X_num_te, x_cat_te, _ = fm_arrays(test, x5m)
    X_flat_tr = np.concatenate([X_num_tr, x_cat_tr.astype(np.float32)], axis=1)
    X_flat_te = np.concatenate([X_num_te, x_cat_te.astype(np.float32)], axis=1)
    print("Fitting TabPFN v2 on full 1500 rows...")
    tabpfn_pred = fit_predict_tabpfn(X_flat_tr, y, X_flat_te)
    print(f"  tabpfn mean={tabpfn_pred.mean():+.3f}  range=[{tabpfn_pred.min():+.2f}, {tabpfn_pred.max():+.2f}]")

    # -------------- LIN_x4 locked_b on full data --------------
    print("Building LIN_x4 (locked_b)...")
    X_tr_lin = design_matrix(df,   x5m, include_x4=True, include_x9=False)
    X_te_lin = design_matrix(test, x5m, include_x4=True, include_x9=False)
    lock = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
    intercept = (y - X_tr_lin @ lock).mean()
    lin_pred = X_te_lin @ lock + intercept
    print(f"  lin_x4 mean={lin_pred.mean():+.3f}  intercept={intercept:+.3f}")

    # -------------- EBM_x9 (no x4) on full data --------------
    print("Fitting EBM_x9 (heavy-smooth) on full 1500 rows...")
    feats9 = ebm_features(with_x4=False, with_x9=True)
    X_tr = preprocess(df, feats9, x5m); X_te = preprocess(test, feats9, x5m)
    ebm_x9_pred = fit_ebm(X_tr, y).predict(X_te)

    # -------------- EBM_full on full data --------------
    print("Fitting EBM_full (heavy-smooth) on full 1500 rows...")
    feats_full = ebm_features(with_x4=True, with_x9=True)
    X_tr = preprocess(df, feats_full, x5m); X_te = preprocess(test, feats_full, x5m)
    ebm_full_pred = fit_ebm(X_tr, y).predict(X_te)

    # -------------- Blend --------------
    triple = 0.5 * (0.5 * (lin_pred + ebm_x9_pred)) + 0.5 * ebm_full_pred
    blend  = 0.3 * tabpfn_pred + 0.7 * triple

    # Also persist the pure triple for comparison (already committed as
    # submission_ensemble_triple_locked_b_lambda50.csv, but rebuild fresh in case)
    out_blend = SUBS / "submission_tabpfn_triple_blend.csv"
    pd.DataFrame({"id": test["id"], "target": blend}).to_csv(out_blend, index=False)
    print(f"\nWrote {out_blend.name}")
    print(f"  mean={blend.mean():+.3f}  range=[{blend.min():+.2f}, {blend.max():+.2f}]")
    print(f"  tabpfn contribution: 30%")
    print(f"  triple contribution: 70%  (17.5% lin_x4 + 17.5% ebm_x9 + 35% ebm_full)")


if __name__ == "__main__":
    main()
