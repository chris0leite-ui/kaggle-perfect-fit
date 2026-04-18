"""Router submission: A1 on safe rows, cross_LE elsewhere.

Router_A1_triple (submitted) scored LB 3.35 on the public board. Back-
solving the segment contributions from the test composition:

    test:   27.9 % safe non-sent  +  56.9 % unsafe non-sent  +  15.2 % sent
    3.35 =  0 · 0.279            +  U · 0.569               +  10 · 0.152
    =>  U (triple MAE on unsafe non-sent) ≈ 3.22

A1 on safe rows is essentially perfect (as on train). The bottleneck is
the backup model on unsafe non-sent + sentinel. cross_LE has the best
LB-confirmed behaviour on non-sent rows (2.94 total → 1.68 on non-sent
averaged). Swap triple → cross_LE as the router backup:

    proj:   0 · 0.279  +  1.75 · 0.569  +  10 · 0.152  ≈  2.52

Even if cross_LE runs ~25 % worse on unsafe-only (say 2.2 vs 1.75) we
still land ~2.8 — below cross_LE alone (2.94).

Safe rule (same as cv_router_A1.py):
    safe iff  x5 ≠ 999   AND   |x4| > 0.167
           AND ((x4 > 0 AND x9 > 5) OR (x4 < 0 AND x9 < 5))
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL, preprocess  # noqa: E402
from cv_x4_x9_swap_ensemble import design_matrix, ebm_features, fit_ebm  # noqa: E402
from cv_router_A1 import safe_mask  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].values
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())

    # ------------------ A1 closed form on test ------------------
    a1_pred = approach1_predict(test, x5m)

    # ------------------ cross_LE = 0.5 * (LIN_x4_free + EBM_x9_heavy) ------------------
    X_tr_lin = design_matrix(df,   x5m, include_x4=True, include_x9=False)
    X_te_lin = design_matrix(test, x5m, include_x4=True, include_x9=False)
    lin_x4_pred = LinearRegression().fit(X_tr_lin, y).predict(X_te_lin)

    feats_x9 = ebm_features(with_x4=False, with_x9=True)
    X_tr = preprocess(df,   feats_x9, x5m)
    X_te = preprocess(test, feats_x9, x5m)
    print("Fitting EBM_x9 (heavy-smooth) on full 1500 rows...")
    ebm_x9_pred = fit_ebm(X_tr, y).predict(X_te)

    cross_LE = 0.5 * (lin_x4_pred + ebm_x9_pred)

    # ------------------ Router: safe → A1, else → cross_LE ------------------
    safe = safe_mask(test)
    pred = np.where(safe, a1_pred, cross_LE)

    n = len(test)
    n_safe = int(safe.sum())
    n_sent = int((test["x5"] == SENTINEL).sum())
    n_unsafe_ns = n - n_safe - n_sent
    print(f"\nTest composition ({n} rows):")
    print(f"  safe (→ A1):            {n_safe:>4d}  ({n_safe/n:6.2%})")
    print(f"  unsafe non-sent (→ cross_LE): {n_unsafe_ns:>4d}  ({n_unsafe_ns/n:6.2%})")
    print(f"  sentinel (→ cross_LE):  {n_sent:>4d}  ({n_sent/n:6.2%})")

    out = SUBS / "submission_router_A1_cross_LE.csv"
    pd.DataFrame({"id": test["id"], "target": pred}).to_csv(out, index=False)
    print(f"\nWrote {out.name}")
    print(f"  mean={pred.mean():+.3f}  range=[{pred.min():+.2f}, {pred.max():+.2f}]")
    print(f"  A1 contribution:       {n_safe/n:6.2%} of rows")
    print(f"  cross_LE contribution: {(n-n_safe)/n:6.2%} of rows")


if __name__ == "__main__":
    main()
