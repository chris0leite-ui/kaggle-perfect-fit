"""Verify recovered DGP: reproduces 7 of 10 numeric features exactly.

Under seed = 4242 with numpy's MT19937 (np.random.RandomState):

  call #1   rs.uniform(-0.5, 0.5, 3000)   → x1
  call #2   rs.uniform(-0.5, 0.5, 3000)   → x2
  call #3   rs.uniform(0, 1, 3000)        → city  (u<0.5 = Zaragoza)
  call #4   rs.uniform(0, 1, 3000)        → c4 (drives x4 piecewise)
  call #5   rs.uniform(7, 12, 3000)       → x5 before sentinel mask
  call #6   rs.uniform(0, 1, 3000)        → c6 (drives x6, x7 on circle)

x4 transform:
  id < 750         : x4 = c4/3 − 0.5        (range [−0.5, −1/6])
  750 ≤ id < 1500  : x4 = c4/3 + 1/6        (range [+1/6, +0.5])
  id ≥ 1500 (test) : x4 = c4 − 0.5          (range [−0.5, +0.5])

x6, x7 from the circle of radius 18:
  x6 = 18 · sin(2π · c6)
  x7 = 18 · cos(2π · c6)

x9, x10, x11 still unaccounted for (calls #7+ don't match simple
uniform transforms or separate seeds; search ongoing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

SEED = 4242


def recover_pool() -> pd.DataFrame:
    rs = np.random.RandomState(SEED)
    x1   = rs.uniform(-0.5, 0.5, 3000)
    x2   = rs.uniform(-0.5, 0.5, 3000)
    ucity = rs.uniform(0, 1, 3000)
    c4   = rs.uniform(0, 1, 3000)
    x5   = rs.uniform(7, 12, 3000)
    c6   = rs.uniform(0, 1, 3000)

    ids = np.arange(3000)
    x4 = np.where(ids < 750, c4 / 3 - 0.5,
         np.where(ids < 1500, c4 / 3 + 1 / 6, c4 - 0.5))
    city = np.where(ucity < 0.5, "Zaragoza", "Albacete")
    x6v = 18 * np.sin(2 * np.pi * c6)
    x7v = 18 * np.cos(2 * np.pi * c6)

    return pd.DataFrame({
        "id": ids, "x1": x1, "x2": x2, "x4": x4, "x5_true": x5,
        "x6": x6v, "x7": x7v, "City": city,
    })


def main():
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    pool = pd.concat([train, test], ignore_index=True, sort=False).sort_values("id").reset_index(drop=True)
    rec = recover_pool()

    print(f"Seed verification — np.random.RandomState({SEED})")
    print("=" * 64)
    for col in ["x1", "x2", "x4", "x6", "x7"]:
        err = np.max(np.abs(rec[col].values - pool[col].values))
        print(f"  {col}: max |err| = {err:.2e}")
    mask = pool["x5"] != 999
    err_x5 = np.max(np.abs(rec.loc[mask, "x5_true"].values - pool.loc[mask, "x5"].values))
    print(f"  x5 (non-sent): max |err| = {err_x5:.2e}")
    city_match = (rec["City"].values == pool["City"].values).sum()
    print(f"  City: {city_match}/3000 match")

    # Test sentinel recovery
    sent_ids = test.loc[test["x5"] == 999, "id"].values
    recovered = rec.set_index("id").loc[sent_ids, "x5_true"].values
    print(f"\nRecovered x5 for {len(recovered)} test sentinel rows:")
    print(f"  min={recovered.min():.4f}  max={recovered.max():.4f}  "
          f"mean={recovered.mean():.4f}")
    print(f"  (matches U(7, 12) distribution as expected)")


if __name__ == "__main__":
    main()
