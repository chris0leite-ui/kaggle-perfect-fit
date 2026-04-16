"""Train best EBM variants on full dataset.csv and write Kaggle submissions.

Two variants:
  - heavy_smooth  — best CV (3.08) across all tested variants
  - no_x9          — worse CV (3.83) but drops the feature whose train->test
                     correlation with x4 collapsed (hypothesis test)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cv_ebm_variants import preprocess, FEATURES_ALL, FEATURES_NO_X9, SENTINEL, SEED

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUBS = REPO / "submissions"


def build(name: str, features: list[str], kwargs: dict) -> None:
    from interpret.glassbox import ExplainableBoostingRegressor
    train = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    x5m = float(train.loc[train["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(train, features, x5m)
    X_te = preprocess(test, features, x5m)
    model = ExplainableBoostingRegressor(random_state=SEED, **kwargs)
    model.fit(X_tr, train["target"].values)
    preds = model.predict(X_te)
    out = pd.DataFrame({"id": test["id"], "target": preds})
    p = SUBS / f"submission_{name}.csv"
    out.to_csv(p, index=False)
    print(f"  wrote {p.name}  (mean={preds.mean():+.3f}, "
          f"min={preds.min():+.2f}, max={preds.max():+.2f})")


def main():
    build(
        name="ebm_heavy_smooth",
        features=FEATURES_ALL,
        kwargs=dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128,
                    smoothing_rounds=2000, interaction_smoothing_rounds=500),
    )
    build(
        name="ebm_no_x9",
        features=FEATURES_NO_X9,
        kwargs=dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128),
    )
    build(
        name="ebm_heavy_smooth_no_x9",
        features=FEATURES_NO_X9,
        kwargs=dict(interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128,
                    smoothing_rounds=2000, interaction_smoothing_rounds=500),
    )


if __name__ == "__main__":
    main()
