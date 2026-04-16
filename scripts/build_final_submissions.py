"""Build final submissions using models trained on the full 1500-row dataset.

The CV analysis in cv_ensemble_eval.py ranked these as the top candidates:
    1. A1 closed form alone                 CV MAE 1.80
    2. NNLS EBM+GAM+A1 stacked ensemble     CV MAE 2.02
    3. Stacked EBM+GAM+A1 (Ridge)            CV MAE 2.04
    4. EBM+GAM 70/30 (old Round 2 best)     CV MAE 2.91

A1 has no learnable parameters, so "train on full data" is a no-op for it. For
every other model — and for the meta-learner weights — we refit on all 1500
rows of dataset.csv before predicting on test.csv.

Meta-learner note: stacked ensemble weights are fit on **out-of-fold** (OOF)
predictions (same 5-fold split as cv_ensemble_eval.py). The base models
themselves are refit on all 1500 rows for the final submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

from cv_ensemble_eval import (
    ClosedFormModel, a1_predict, build_oof, ebm_preprocess, fit_ebm,
    fit_gam, gam_design,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUBS = REPO / "submissions"
SUBS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


def write(preds: np.ndarray, ids: pd.Series, path: Path) -> None:
    pd.DataFrame({"id": ids, "target": preds}).to_csv(path, index=False)
    print(f"  wrote {path.name}  (n={len(preds)}, mean={preds.mean():+.3f}, "
          f"min={preds.min():+.2f}, max={preds.max():+.2f})")


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    print(f"train: {train.shape}, test: {test.shape}")

    x5_median = float(train.loc[train["x5"] != SENTINEL, "x5"].median())
    print(f"x5 median (from full train): {x5_median:.4f}")

    # --- A1: no fit ---
    print("\n[A1] closed form — no parameters to train.")
    p_a1 = a1_predict(test, x5_median)

    # --- A2: fit on full train ---
    print("\n[A2] ClosedFormModel — fitting on 1500 rows ...")
    m2 = ClosedFormModel().fit(train, train["target"].values)
    p_a2 = m2.predict(test)
    coef_names = ["City", "x4", "x8", "x5", "x10*x11",
                  "cos(pi*x1)", "cos(5pi*x2)", "x9_resid", "x5_is_sentinel", "const"]
    for name, c in zip(coef_names, m2.coef_):
        print(f"    {name:>16s}: {c:+.3f}")

    # --- EBM + GAM: fit on full train ---
    print("\n[EBM] fitting on 1500 rows ...")
    ebm = fit_ebm(train, x5_median)
    p_ebm_full = ebm.predict(ebm_preprocess(test, x5_median))

    print("[GAM] fitting on 1500 rows ...")
    gam, gam_names = fit_gam(train, x5_median)
    p_gam_full = gam.predict(gam_design(test, x5_median)[gam_names].values)

    # --- Meta-learner weights from OOF predictions ---
    print("\nComputing OOF predictions to fit meta-learners ...")
    oof = build_oof(train)
    y = oof["y"].values

    # NNLS (non-negative weights, no intercept) EBM+GAM+A1
    X_oof = oof[["ebm", "gam", "a1"]].values
    w_nnls, _ = nnls(X_oof, y)
    print(f"NNLS weights (EBM, GAM, A1): {np.round(w_nnls, 3).tolist()}")

    # Ridge stacker EBM+GAM+A1 (with intercept, allows negative weights)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_oof, y)
    print(f"Ridge coefs (EBM, GAM, A1): {np.round(ridge.coef_, 3).tolist()}, "
          f"intercept={ridge.intercept_:+.3f}")

    # Apply meta-weights to base model predictions on test
    X_test = np.column_stack([p_ebm_full, p_gam_full, p_a1])
    p_nnls = X_test @ w_nnls
    p_ridge = ridge.predict(X_test)

    # EBM+GAM 70/30 (previous Round 2 best)
    p_eg7030 = 0.7 * p_ebm_full + 0.3 * p_gam_full

    # --- Write submissions ---
    print("\nWriting submissions:")
    write(p_a1, test["id"], SUBS / "submission_a1_closedform.csv")
    write(p_a2, test["id"], SUBS / "submission_a2_closedformmodel.csv")
    write(p_ebm_full, test["id"], SUBS / "submission_ebm.csv")
    write(p_eg7030, test["id"], SUBS / "submission_ebm_gam_70_30.csv")
    write(p_ridge, test["id"], SUBS / "submission_stacked_ebm_gam_a1.csv")
    write(p_nnls, test["id"], SUBS / "submission_nnls_ebm_gam_a1.csv")

    # Combined predictions table for analysis
    combined = pd.DataFrame({
        "id": test["id"],
        "x4": test["x4"],
        "x5_is_sentinel": (test["x5"] == SENTINEL).astype(int),
        "City": test["City"],
        "a1_closedform": p_a1,
        "a2_closedformmodel": p_a2,
        "ebm": p_ebm_full,
        "gam": p_gam_full,
        "ebm_gam_70_30": p_eg7030,
        "stacked_ridge_ebm_gam_a1": p_ridge,
        "nnls_ebm_gam_a1": p_nnls,
    })
    combined.to_csv(SUBS / "all_submissions_combined.csv", index=False)
    print(f"\n  wrote all_submissions_combined.csv (includes x4, sentinel flag)")

    # Sanity: pairwise agreement summary
    cols = ["a1_closedform", "a2_closedformmodel", "ebm", "ebm_gam_70_30",
            "stacked_ridge_ebm_gam_a1", "nnls_ebm_gam_a1"]
    print("\nPairwise mean |prediction diff| (should be small among near-optimal models):")
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = float(np.mean(np.abs(combined[a] - combined[b])))
            print(f"    {a:30s} vs {b:30s}: {d:.3f}")


if __name__ == "__main__":
    main()
