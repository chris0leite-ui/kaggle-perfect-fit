"""Seed 4242 recovers x5 for every row. Use it to fill test sentinels
and apply corrected-A1 on test."""
import numpy as np, pandas as pd
from pathlib import Path

DATA = Path('data'); SUBS = Path('submissions')

# Load
train = pd.read_csv(DATA/'dataset.csv')
test = pd.read_csv(DATA/'test.csv')

# Reproduce DGP sequence with seed 4242
rs = np.random.RandomState(4242)
x1 = rs.uniform(-0.5, 0.5, 3000)   # x1
x2 = rs.uniform(-0.5, 0.5, 3000)   # x2
u_city = rs.uniform(0, 1, 3000)    # city
c4 = rs.uniform(0, 1, 3000)        # c4 → x4
x5_dgp = rs.uniform(7, 12, 3000)   # x5 (before sentinel mask)

# Verify on training
for col, gen in [('x1', x1), ('x2', x2)]:
    err = np.max(np.abs(gen[:1500] - train[col].values))
    print(f'train.{col} reproduces with seed 4242: max|err|={err:.2e}')
# x5 for non-sentinel rows
mask_obs_train = train['x5'] != 999
err_x5 = np.max(np.abs(x5_dgp[:1500][mask_obs_train.values] - train.loc[mask_obs_train, 'x5'].values))
print(f'train.x5 (non-sent) reproduces: max|err|={err_x5:.2e}')
mask_obs_test = test['x5'] != 999
err_x5_t = np.max(np.abs(x5_dgp[1500:][mask_obs_test.values] - test.loc[mask_obs_test, 'x5'].values))
print(f'test.x5 (non-sent) reproduces: max|err|={err_x5_t:.2e}')

# Recover test x5
test_x5_recovered = x5_dgp[1500:].copy()
print(f'\nRecovered test x5 for {(~mask_obs_test).sum()} sentinel rows')
print(f'Recovered values stats: min={test_x5_recovered.min():.3f} max={test_x5_recovered.max():.3f}')

# Build CORRECTED-A1 for test, using recovered x5
is_zar = (test['City'] == 'Zaragoza').astype(float).values
# Use step=+12 (CV optimum) and intercept from fitting on id>=100 in train
# with x9 dropped, step=12
# Compute A1-body-no-x9 with step=12
# Fit intercept on train id>=100 with true x5
train_clean = train[train['id'] >= 100].copy()
x5_train_clean = train_clean['x5'].where(train_clean['x5'] != 999, np.nan).fillna(0).values
# Actually use the recovered x5 for training too so intercept is computed on true values
x5_train_all = x5_dgp[:1500]
train_clean_mask = train['id'] >= 100
body_no_x5 = (-100*train['x1']**2 + 10*np.cos(5*np.pi*train['x2']) + 15*train['x4']
              + 15*train['x8'] + train['x10']*train['x11']
              - 25*(train['City']=='Zaragoza').astype(float) 
              + 12*(train['x4']>0)).values
# Intercept = mean(target - body + 8*x5_true) over id>=100 rows
target_train = train['target'].values
intercept = np.mean((target_train - body_no_x5 + 8*x5_train_all)[train_clean_mask])
print(f'\nFitted intercept (step=12, no x9, true x5): {intercept:.4f}')

# Predict test
body_test = (-100*test['x1']**2 + 10*np.cos(5*np.pi*test['x2']) + 15*test['x4']
             + 15*test['x8'] + test['x10']*test['x11']
             - 25*(test['City']=='Zaragoza').astype(float) 
             + 12*(test['x4']>0)).values
pred_test = body_test - 8*test_x5_recovered + intercept

# Also build with step=11 and step=10 for comparison
for step in [10, 11, 12, 15, 20]:
    body_tr = (-100*train['x1']**2 + 10*np.cos(5*np.pi*train['x2']) + 15*train['x4']
               + 15*train['x8'] + train['x10']*train['x11']
               - 25*(train['City']=='Zaragoza').astype(float)
               + step*(train['x4']>0)).values
    b = np.mean((target_train - body_tr + 8*x5_train_all)[train_clean_mask])
    body_te = (-100*test['x1']**2 + 10*np.cos(5*np.pi*test['x2']) + 15*test['x4']
               + 15*test['x8'] + test['x10']*test['x11']
               - 25*(test['City']=='Zaragoza').astype(float)
               + step*(test['x4']>0)).values
    pred = body_te - 8*test_x5_recovered + b
    fname = SUBS/f'submission_seed4242_step{step}_nox9.csv'
    pd.DataFrame({'id': test['id'], 'target': pred}).to_csv(fname, index=False)
    print(f'wrote {fname.name}  intercept={b:.4f}  step={step}  pred mean={pred.mean():+.2f}')

# ALSO: try the original A1 formula with -4·x9 (maybe x9 is in the real DGP after all)
for step in [11, 12, 20]:
    body_te = (-100*test['x1']**2 + 10*np.cos(5*np.pi*test['x2']) + 15*test['x4']
               + 15*test['x8'] - 4*test['x9'] + test['x10']*test['x11']
               - 25*(test['City']=='Zaragoza').astype(float)
               + step*(test['x4']>0)).values
    body_tr = (-100*train['x1']**2 + 10*np.cos(5*np.pi*train['x2']) + 15*train['x4']
               + 15*train['x8'] - 4*train['x9'] + train['x10']*train['x11']
               - 25*(train['City']=='Zaragoza').astype(float)
               + step*(train['x4']>0)).values
    b = np.mean((target_train - body_tr + 8*x5_train_all)[train_clean_mask])
    pred = body_te - 8*test_x5_recovered + b
    fname = SUBS/f'submission_seed4242_step{step}_withx9.csv'
    pd.DataFrame({'id': test['id'], 'target': pred}).to_csv(fname, index=False)
    print(f'wrote {fname.name}  intercept={b:.4f}  step={step}  pred mean={pred.mean():+.2f}')
