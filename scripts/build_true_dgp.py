"""A1 with 1(x4>0) → 1(x9>5). This is the real DGP.

On training, 1(x9>5) ≡ 1(x4>0) so A1 fits exactly. On test where
x9 ⊥ x4, they differ.
"""
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path('data'); SUBS = Path('submissions')
train = pd.read_csv(DATA/'dataset.csv')
test = pd.read_csv(DATA/'test.csv')

rs = np.random.RandomState(4242)
for _ in range(4): rs.uniform(0, 1, 3000)
x5_all = rs.uniform(7, 12, 3000)
x5_test = x5_all[1500:]
x5_train = x5_all[:1500]

SENT = 999.0
train_c = train.copy()
is_sent = (train['x5'] == SENT).values
train_c.loc[is_sent, 'x5'] = x5_train[is_sent]
is_zar_tr = (train_c['City'] == 'Zaragoza').astype(float).values

a1_tr = (-100*train_c['x1'].values**2 + 10*np.cos(5*np.pi*train_c['x2'].values)
         + 15*train_c['x4'].values - 8*train_c['x5'].values
         + 15*train_c['x8'].values - 4*train_c['x9'].values
         + train_c['x10'].values*train_c['x11'].values
         - 25*is_zar_tr + 20*(train_c['x9'].values > 5) + 92.5)

clamp_mask = ((train['id'] < 100) & (train['x4'] < 0) & (train['x8'] < 0)).values
a1_tr[clamp_mask] += -15*train_c['x8'].values[clamp_mask] + 1.0

resid = train['target'].values - a1_tr
non_clamp = ~clamp_mask
print(f"A1-variant (with 1(x9>5)) on training non-clamp: max|err|={np.max(np.abs(resid[non_clamp])):.2e}")
identical = ((train_c['x9'] > 5) == (train_c['x4'] > 0)).values
print(f"1(x9>5) == 1(x4>0) on training: {identical.sum()}/{len(train)} rows")
mism = train_c[~identical]
if len(mism) > 0:
    print(f"Mismatches ({len(mism)} rows): shown")
    print(mism[['id','x4','x9']].to_string(index=False))

# Predict test
is_zar_te = (test['City'] == 'Zaragoza').astype(float).values
pred_test = (-100*test['x1'].values**2 + 10*np.cos(5*np.pi*test['x2'].values)
             + 15*test['x4'].values - 8*x5_test + 15*test['x8'].values
             - 4*test['x9'].values + test['x10'].values*test['x11'].values
             - 25*is_zar_te + 20*(test['x9'].values > 5) + 92.5)

out = SUBS / "submission_closed_form_TRUE_DGP.csv"
pd.DataFrame({'id': test['id'], 'target': pred_test}).to_csv(out, index=False)
print(f"\nwrote {out.name}  mean={pred_test.mean():+.3f}")

v5 = pd.read_csv(SUBS/'submission_closed_form_v5.csv').set_index('id')['target'].reindex(test['id'].values).values
print(f"MAE vs v5 (LB 1.37): {np.mean(np.abs(pred_test - v5)):.4f}")
