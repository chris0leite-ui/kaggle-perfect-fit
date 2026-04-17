# The Perfect Fit — Work Report

A concise narrative of the Kaggle "Perfect Fit" competition: what the
data was, what generalised, what didn't, and why.

**Terminology** (used throughout):

- **MAE** — Mean Absolute Error. Our evaluation metric.
- **CV** — 5-fold Cross-Validation on `dataset.csv`, using
  `KFold(shuffle=True, random_state=42)`. We always compute out-of-fold
  predictions explicitly so we can slice them by subgroup.
- **LB** — Public Leaderboard (Kaggle).
- **DGP** — Data-Generating Process (the true underlying function from
  features to target).
- **EBM** — Explainable Boosting Machine (interpret-core's
  `ExplainableBoostingRegressor`): a gradient-boosted additive model
  with bounded shape functions per feature + selected 2-way interactions.
- **GAM** — Generalised Additive Model (`pygam`'s cubic-spline family).
- **Sentinel** — a magic value that encodes "missing". Here,
  `x5 = 999.0` on ~15% of rows.

## 1. Competition and data

- Tabular regression, metric **MAE**. Submission = `(id, target)` on
  1,500 test rows.
- Training: 1,500 rows in `dataset.csv`. Features
  `x1, x2, x4, x5, x6, x7, x8, x9, x10, x11` (the competition skips
  `x3`) plus categoricals `Country` and `City`. Target continuous,
  `std ≈ 24.1`.
- Test: 1,500 rows in `test.csv`, same schema without `target`.

## 2. Data quirks we exploited

- **`Country` is constant at `Spain`.** Drop it.
- **`City` is binary** (Zaragoza / Albacete). Encode as a 0/1 column;
  its coefficient is always ~24–25 (large effect).
- **`x5` has a hard sentinel at 999.0.** Non-sentinel values are
  Uniform(7, 12). Imputation strategies (mean, median, k-Nearest-
  Neighbours, gradient boosting) all hit the same `~1.25` per-row
  Mean-Absolute-Error on sentinel rows — the Uniform(7, 12) median-
  optimal bound. **Nothing beats median imputation for x5.**
- **`x6² + x7² = 18²` exactly** on every row. Only the angle
  `θ = atan2(x7, x6)` varies. θ is uniform on `[−π, π]` and
  independent of every other feature and of the sentinel indicator.
  x6 and x7 are noise.
- **`x4` is bimodal with a gap at `[−0.167, 0.167]`.** Training has
  zero observations in the gap; the test set has 508 rows (33.9%)
  inside it.

## 3. Training vs test distribution shift

The single most consequential finding. On `dataset.csv` vs `test.csv`:

- `corr(x4, x9) = +0.832` in training, **`+0.001` in test**.
- Training `(x4, x9)` forms two disjoint clusters
  (`x4>0 → x9 ~ N(5.97, 0.57)`, `x4<0 → x9 ~ N(4.02, 0.57)`).
  Within-cluster correlation is zero.
- Test draws `x4` and `x9` independently; 734 test rows (48.9%)
  land in the off-diagonal quadrants `{x4>0, x9<5}` and
  `{x4<0, x9>5}` that are empty in training.
- All other pairwise correlations are `< 0.06` on both sets.
- Sentinel rate is preserved (14.8% train, 15.2% test).

**Consequence.** Any model that learns an x4 ↔ x9 relationship — an
interaction term, a residualised `x9_resid = x9 − β·x4`, a cluster-
centered `x9_wc`, or an inflated linear `β_x9` — extrapolates wrongly
on the 49% of test rows in the empty quadrants. Density-ratio
reweighting cannot fix this: reweighting redistributes mass over
observed rows, and the off-diagonal quadrants have no observed rows.

## 4. Signal discovery

Marginal linear correlations with target (by decreasing magnitude):
`City, x4, x5, x8, x9, x10, x11`. Additional findings:

- **`x1` and `x2` carry only nonlinear signal.** x1 has an
  inverted-U shape (GAM R² 0.109); x2 oscillates, fit well by
  `cos(5π·x2)` (GAM R² 0.068).
- **`x10 × x11` is a genuine pairwise interaction** (pure-interaction
  Root-Mean-Square 3.02 after an additive baseline; noise floor
  ~1.71). No other pair exceeds the noise floor.
- **Causal-discovery tools (the PC algorithm and DirectLiNGAM —
  linear-acyclic causal-graph recovery methods) returned a false
  positive `x4 → x9` edge.** Both assume linear causal structure and
  no selection bias; the selection-biased training correlation
  satisfied their independence tests. The `0.001` test correlation
  falsifies the causal claim.

## 5. The A1 closed-form formula and its failure

A sibling branch reverse-engineered a zero-parameter closed-form
(we call it **A1**) that fits most training rows to machine precision:

```
target = −100·x1² + 10·cos(5π·x2)
       + 15·x4 + 20·𝟙(x4 > 0)
       − 8·x5_imp + 15·x8 − 4·x9
       + x10·x11 − 25·𝟙(City = Zaragoza) + 92.5
```

- Training CV: overall MAE 1.80; non-sentinel 0.38; sentinel 10.0.
- **93.3% of non-sentinel training rows have `|residual| < 0.01`.**
  The 6.7% imperfect rows all lie in the quadrant `x4 < 0 AND
  x8 < 0`, where residual ≈ `−18.4·x8` (std 0.76). We call this
  subset the "clamp".
- **Public LB: MAE 10.80** — ~6× worse than its CV.

Two distinct failure modes explain the CV-LB gap:

1. `+20·𝟙(x4>0)` is a hard step that misfires on the 508 test rows
   inside the x4 gap (training had zero observations there, so the
   step was unfalsifiable during reverse-engineering).
2. `−4·x9` extrapolates on the 734 off-diagonal test rows.

Together these cover ~83% of the test set.

## 6. Models and CV→LB behaviour

| Family | Variant | CV MAE | LB MAE |
|---|---|---|---|
| Linear | baseline (City + x4 + x5 + x8 + x10 + x11) | 9.98 | — |
| Linear | + splines(x1, x2) + x10·x11, no x9 | 3.70 | 7.38 |
| Linear | locked-integer skeleton + cluster-centered x9 | 2.90 | 10.75 |
| Closed form | A1 (hand-engineered, step at x4=0) | 1.80 | 10.80 |
| Closed form | A2 (linear least-squares on hand basis) | 3.49 | 9.44 |
| GAM | tuned + x10·x11 | 3.56 | — |
| LightGBM | tuned | 4.66 | — |
| EBM | default | 3.24 | — |
| EBM | tuned (smoothing 2,000 rounds) | 3.11 | 5.66 |
| EBM | heavy smoothing (smoothing 2,000 + refine 1,500) | 3.08 | **4.90** |
| EBM | all-smoothed (smoothing 4,000, no refine) | 3.03 | — |
| Ensemble | EBM + GAM, 70/30 weighted | 2.91 | 6.47 |
| Ensemble | **cross_LE = 0.5·LIN(no x9) + 0.5·EBM(no x4)** | 2.97 | **2.94** |
| Ensemble | triple = 0.25·LIN + 0.25·EBM(no x4) + 0.5·EBM(full) | 2.82 | — |
| Router | safe → A1; else → triple ensemble | **1.84** | — |

Column names follow the convention **LIN** = hand-crafted linear
design matrix (`x1², cos(5π·x2), x4, x5_imp, x5_is_sent, x8, x10,
x11, x10·x11, City`, optionally `x9`). **EBM(no x4)** and
**EBM(no x9)** are EBM on the full feature set with one column
removed; **EBM(full)** keeps both.

Three patterns in the table:

1. **Parametric models catastrophically miss on LB** (A1, A2, the
   locked-integer linear). Their closed-form x9 coefficient encodes
   training joint structure that doesn't exist in test.
2. **EBM's bounded shape functions generalise.** EBM cannot
   extrapolate past its training bins, which is a feature: on
   off-diagonal test rows, EBM holds the nearest-bin value rather
   than projecting a trend. CV-to-LB degradation is ~1.8× (best
   among non-ensemble models).
3. **Ensembles that mix complementary failure modes help
   dramatically.** `cross_LE` pairs a linear model with no x9
   (immune to the shift) against an EBM with no x4 (handles
   everything else nonparametrically). Their errors on off-diagonal
   rows partially cancel. **CV 2.97 → LB 2.94** is unusually tight.

## 7. Final submissions

Four CSV files live in `submissions/`. The notebook
`notebooks/final_submissions.ipynb` rebuilds them byte-identically.

| File | CV MAE | LB MAE |
|---|---|---|
| `submission_ebm_heavy_smooth.csv` | 3.08 | **4.90** |
| `submission_ensemble_cross_LE.csv` | 2.97 | **2.94** |
| `submission_ensemble_triple_locked_b_lambda50.csv` | 2.82 | untested |
| `submission_router_A1_triple.csv` | 1.84 | untested |

## 8. Noise floor and placement

- **Sentinel x5 irreducible noise floor.** x5 has slope ≈ −8 on
  target; imputing a Uniform(7, 12) feature with its median leaves
  ≈ 1.25 per-row MAE. Weighted by the 15.2% sentinel rate in test:
  `0.152 × 8 × 1.25 = 1.52` MAE. **No submission can beat 1.52 on
  the public LB.**
- **LB top cluster** sits at 1.65–1.71 (within 0.2 of the floor),
  implying the leaders recovered the DGP to ≈ 0.15 MAE on non-
  sentinel rows.
- **Our best LB (2.94)** is ~1.4 MAE above the top cluster; our
  best non-sentinel MAE on CV was 1.72 (cross_LE). The gap to
  the top is approximately "A1's 93% non-sentinel perfection on the
  safe quadrants" that we couldn't safely route to, not a
  modelling-quality gap.

## 9. Ideas we tried and rejected (one line each)

- **Drop x9 from EBM.** CV 3.83 vs 3.08 — hurts CV; x9 carries real
  within-training signal that EBM extracts without over-committing.
- **Residualise x9 against x4** (A2 variant). Encodes the spurious
  x4-x9 edge; LB 9.44 vs EBM's 5.66.
- **Within-cluster-centered x9** (`x9_wc`). Same issue expressed
  differently; LB 10.75.
- **Monotone EBM constraints** on x4, x5, x8, City. CV 3.49.
- **EBM with interactions disabled.** CV 4.16.
- **EBM forced linear on x4 via residualisation.** CV 3.33–3.61.
- **Per-cluster EBM** (one model per sign-of-x4). CV 3.50.
- **LightGBM tuned.** CV 4.66 — shallow trees + heavy regularisation
  needed to not overfit; still worse than default EBM.
- **GAM + x10·x11 interaction.** CV 3.56 — best additive-only
  model, still worse than EBM.
- **Random Forest.** CV ~8.8 — can't fit x1/x2 nonlinearity
  smoothly.
- **Quantile regression with splines.** CV 6.96 — L1 loss +
  high-dim spline basis → optimisation issues.
- **Huber regression.** CV 5.19 — no effect vs OLS; our outliers
  are sentinel-driven, not leverage-type.
- **k-Nearest-Neighbours (20 configs).** Best CV 9.33 — local
  similarity misses `cos(5π·x2)` structure.
- **NNLS** (Non-Negative Least-Squares) **stacking of EBM + GAM
  + A1.** CV 2.02 — A1 got 0.83 weight and memorised training;
  confirmed on LB by A1-alone's 10.80.
- **Density-ratio reweighting to break x4-x9 correlation**
  (classifier-based, KDE, Gaussian-copula). Weighted correlation
  dropped from 0.83 to 0.74 best; downstream CV moved ≤ 0.08.
  Structurally impossible to generate mass in the empty quadrants.
- **Synthetic off-diagonal augmentation** (resample x9, recompute
  target via A1). CV 4.40 — using A1 as DGP oracle backfires
  because A1's `−4·x9` is itself wrong on test.
- **A1 + EBM residual corrector.** CV 2.04 — EBM overfits the
  residual noise on the 93% of rows A1 already fits exactly.
- **Classifier-based clamp trigger for the router.** LightGBM
  reached AUC 0.76 on the 23% hidden "clamp" rows within the
  `x4<0, x8<0` quadrant; routing variants (soft blend, hard
  thresholds at 0.3/0.4/0.5/0.6) gave CV 1.85–1.86, worse than
  the parameter-free base router (CV 1.84). AUC would need to
  reach ~0.9 before the 4-to-1 reward-to-cost ratio of routing
  pays off.
- **x6/x7 angle θ as a feature.** KS test `p = 0.89` against the
  clamp indicator; independent of every other feature. Adding θ
  hurt EBM CV 3.47 → 3.56.
- **Integer-locked coefficient hypothesis for A2's basis.** CV 2.90
  (identical to free fit) — the DGP is integer, but the formula
  itself still extrapolates wrongly on test (LB 10.75).

## 10. Open questions

- **Hidden clamp trigger.** The 23% of `x4<0 AND x8<0` rows with
  `residual ≈ −18.4·x8` look like a Bernoulli(0.23) hidden binary
  in the DGP. No observed feature or two-way interaction predicts
  it above AUC 0.76. Three-way feature interactions, unsupervised
  structure discovery, or a non-tabular side channel might push it
  to AUC 0.9.
- **DGP oracle for test-distribution augmentation.** Synthetic
  off-diagonal rows would close the train-test shift if we had the
  true target function. A1 is not that function on the off-diagonal
  side; we don't know what is.
