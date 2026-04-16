"""Reverse-engineered formula for the Kaggle 'Perfect Fit' target.

Discovered structure
--------------------
target = -100·x1² + 10·cos(5π·x2) + 15·x4 − 8·x5 + 15·x8
         − 4·x9 + x10·x11 − 25·is_zaragoza + 20·𝟙(x4>0) + 92.5

Key observations
~~~~~~~~~~~~~~~~
* Country is constant ("Spain") → ignored.
* City is binary: Zaragoza → −25 offset.
* x6, x7 are noise features with no predictive power.
* x5 = 999 is a sentinel; original x5 unknown for those rows.
* sign(x4) = sign(x9 − 5) in the dataset — x4 > 0 iff x9 ≥ 5.
* Formula is exact (residual = 0) for all x4 > 0 rows and ~87 % of
  x4 < 0 rows.  ~86 rows with x4 < 0 and x8 < 0 have a small
  residual (the x8 contribution appears clamped to ≈ 1 instead of
  15·x8); the precise condition is still under investigation.
"""

import numpy as np
import pandas as pd

NOISE_FEATURES = ("x6", "x7")
SENTINEL_VALUE = 999


def predict(df: pd.DataFrame) -> np.ndarray:
    """Apply the reverse-engineered formula to *df* and return predictions.

    Parameters
    ----------
    df : DataFrame with columns x1, x2, x4, x5, x8, x9, x10, x11, City.

    Returns
    -------
    numpy array of predicted target values.
    """
    is_zaragoza = (df["City"] == "Zaragoza").astype(float).values
    x4_positive = (df["x4"].values > 0).astype(float)

    return (
        -100 * df["x1"].values ** 2
        + 10 * np.cos(5 * np.pi * df["x2"].values)
        + 15 * df["x4"].values
        - 8 * df["x5"].values
        + 15 * df["x8"].values
        - 4 * df["x9"].values
        + df["x10"].values * df["x11"].values
        - 25 * is_zaragoza
        + 20 * x4_positive
        + 92.5
    )
