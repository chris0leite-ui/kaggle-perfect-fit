# Pooled-feature rediscovery — results

Idea #1 from the DGP-archaeology plan.
Script: `scripts/pooled_feature_rediscovery.py`.

## Motivation

PC/LiNGAM were earlier run on training features only (1500 rows).
That pool is selection-biased — most visibly `r(x4, x9) = +0.832`
in train but `+0.001` in test. If other pairs had the same problem,
downstream models would carry additional hidden train-only structure.
Pooling train + test features (3000 rows, no targets leaked) lets us
spot every such case.

## Result — x4-x9 is the only shifted pair, nothing else

Ranked Pearson shift |r_train − r_pool| across all 55 pairs of the
11 numeric+city features:

| a  | b     | r_train | r_test | r_pool | shift  |
|----|-------|--------:|-------:|-------:|-------:|
| x4 | x9    | +0.832  | +0.001 | +0.447 | **+0.384** |
| x2 | City  | −0.030  | +0.045 | +0.008 | −0.038 |
| x7 | x9    | −0.041  | +0.026 | −0.007 | −0.034 |
| x5 | x6    | +0.050  | −0.015 | +0.019 | +0.031 |
| x4 | x8    | +0.049  | −0.017 | +0.019 | +0.030 |
| …  | …     | …       | …      | …      | < 0.03 |

With n=1500 per split the standard error on any true-null r is
≈ 1 / √(1498) = 0.026, so the 3-σ cut-off is 0.08. **Every pair
except x4-x9 is consistent with zero shift.** Spearman gives the
same ranking (x4-x9 at 0.319, everything else < 0.04).

PC consensus edges:
- `pc_train`  → {(x4, x9)}
- `pc_pool`   → {(x4, x9), (x11, x2), (x11, x9)}

The two extra pool-only edges (x11-x2, x11-x9) have pooled Pearson
r in the range [-0.035, +0.005], so they are likely artefacts of the
larger pool's extra statistical power under the Fisher-Z test. No
pool-only edge has a pairwise |r| above the 0.05 noise floor.

## Implication for modelling

1. **The selection-bias contamination is narrowly concentrated on x4-x9.**
   No hidden coupling between x1-x9, x2-x11, x8-x10, etc. — training
   structure for every other pair is trustworthy and transfers to test.
2. **Every CV→LB regression we've seen traces back to x9.** If a model
   uses x9 in a way that relies on its training coupling with x4 (linear
   β, Simpson-corrected x9_wc, EBM main effect extrapolated from the
   [+5, +7] / [+3, +5] training range into gap-violating test rows),
   it will regress on LB. The cross-LE winner at LB 2.94 works because
   it doesn't let x9 drive x4's contribution (LIN_x4 carries x4 alone).
3. **There is no hidden structure to discover** beyond what EBM's
   pairwise interaction scan already reported. The 0.95 baseline-R²
   from the earlier interaction search matches this: training correlations
   other than x4-x9 contain only the x10·x11 interaction, already
   exploited.

## Artefacts

| File                         | Purpose |
|------------------------------|---------|
| `correlation_table.csv`      | All 55 pairs, r_train/r_test/r_pool, sorted by shift |
| `suspect_edges.csv`          | Pairs with |shift| > 0.10 (just x4-x9) |
| `whitelist.csv`              | Pool-stable pairs (every other pair) |
| `heatmaps.png`               | Train / test / pool Pearson heatmaps |
| `partial_heatmaps.png`       | Partial correlations controlling for x4 + City |
| `shift_bars.png`             | Top-20 pairs by |r_train − r_pool| |
| `causal_edges.txt`           | PC edge sets (train vs pool) |
