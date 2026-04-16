"""Train the trimmed linear model on full dataset.csv and write a Kaggle submission.

Feature set (Variant B from cv_simple_linear.py):
  x1**2, cos(5*pi*x2), x4, x5_imputed, x5_is_sentinel, x8, x10, x11,
  city=Zaragoza, x10*x11

5-fold CV of this model: MAE 3.70 (non-sentinel 2.54, sentinel 10.36).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cv_simple_linear import design_matrix, SENTINEL

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUBS = REPO / "submissions"


def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    print(f"train: {train.shape}, test: {test.shape}")

    x5_median = float(train.loc[train["x5"] != SENTINEL, "x5"].median())
    print(f"x5 median (non-sentinel): {x5_median:.4f}")

    X_train, names = design_matrix(train, x5_median, include_interaction=True)
    X_test, _ = design_matrix(test, x5_median, include_interaction=True)

    lr = LinearRegression().fit(X_train, train["target"].values)
    print(f"\nFitted on all 1500 rows.")
    print(f"Intercept: {lr.intercept_:+.3f}")
    for n, c in zip(names, lr.coef_):
        print(f"  {n:>12s}: {c:+.3f}")

    preds = lr.predict(X_test)
    out = pd.DataFrame({"id": test["id"], "target": preds})
    out_path = SUBS / "submission_simple_linear_interact.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  (n={len(out)}, mean={preds.mean():+.3f}, "
          f"min={preds.min():+.2f}, max={preds.max():+.2f})")


if __name__ == "__main__":
    main()
