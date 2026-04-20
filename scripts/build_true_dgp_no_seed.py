"""TRUE_DGP submission WITHOUT seed hacking.

Applies the closed-form recovered DGP to test.csv, but imputes sentinel
x5 rows (x5 == 999) with the training-set median of non-sentinel x5 —
no RNG recovery, no row-level x5 reconstruction.

Formula:
    target = −100·x1² + 10·cos(5π·x2) + 15·x4 − 8·x5 + 15·x8 − 4·x9
           + x10·x11 − 25·zaragoza + 20·1(x9 > 5) + 92.5

Expected public LB: ~1.52 (the sentinel noise floor). Non-sentinel rows
are predicted exactly (formula is the test DGP). Sentinel rows carry
the 8·|median−true_x5| error, where true_x5 ~ Uniform(7, 12).
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path('data')
SUBS = Path('submissions')
SENTINEL = 999.0

train = pd.read_csv(DATA / 'dataset.csv')
test = pd.read_csv(DATA / 'test.csv')

x5_median = train.loc[train['x5'] != SENTINEL, 'x5'].median()
print(f'training non-sentinel x5 median: {x5_median:.4f}')
print(f'test sentinel rows: {(test["x5"] == SENTINEL).sum()} / {len(test)}')

x5 = test['x5'].where(test['x5'] != SENTINEL, x5_median).values
x1 = test['x1'].values
x2 = test['x2'].values
x4 = test['x4'].values
x8 = test['x8'].values
x9 = test['x9'].values
x10 = test['x10'].values
x11 = test['x11'].values
zaragoza = (test['City'] == 'Zaragoza').astype(float).values

pred = (
    -100 * x1**2
    + 10 * np.cos(5 * np.pi * x2)
    + 15 * x4
    - 8 * x5
    + 15 * x8
    - 4 * x9
    + x10 * x11
    - 25 * zaragoza
    + 20 * (x9 > 5).astype(float)
    + 92.5
)

out = SUBS / 'submission_true_dgp_no_seed.csv'
pd.DataFrame({'id': test['id'], 'target': pred}).to_csv(out, index=False)
print(f'wrote {out}')
print(f'prediction stats: mean={pred.mean():+.3f} std={pred.std():.3f} '
      f'min={pred.min():+.3f} max={pred.max():+.3f}')
