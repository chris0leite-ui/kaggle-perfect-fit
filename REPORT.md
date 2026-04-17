# The Perfect Fit — Work Report

Concise narrative of the Kaggle "Perfect Fit" competition. Each section
splits into **Observations** (facts, numbers verifiable against the
data) and **Discussion** (interpretation: what worked, what didn't,
why). Numbers in observation blocks match `CLAUDE.md`.

## About this work

This analysis was produced in collaboration between a human researcher
and **Claude Code** (Anthropic's command-line coding agent). The human
set the direction, chose which experiments were worth running, and
submitted to the Kaggle leaderboard. Claude Code handled the
implementation: exploratory data analysis, causal-discovery runs,
cross-validated modelling (linear / GAM / LightGBM / EBM / ensembles),
diagnostic plots, the reverse-engineering comparison, the cross-view
ensemble, the A1 router, and this report. Work progressed through
~10 alternating rounds of human-set objectives and Claude-run
experiments; `CLAUDE.md` keeps the full session log, `LEARNINGS.md`
distils portable patterns, and this file summarises the outcome.

**Terminology** (used throughout):

- **MAE** — Mean Absolute Error. Our evaluation metric.
- **CV** — 5-fold Cross-Validation on `dataset.csv`, using
  `KFold(shuffle=True, random_state=42)`. Out-of-fold predictions are
  computed explicitly so we can slice them by subgroup.
- **LB** — Public Leaderboard (Kaggle).
- **DGP** — Data-Generating Process (the true underlying function from
  features to target).
- **EBM** — Explainable Boosting Machine (interpret-core's
  `ExplainableBoostingRegressor`): a gradient-boosted additive model
  with bounded shape functions per feature plus selected 2-way
  interactions.
- **GAM** — Generalised Additive Model (cubic splines, `pygam`).
- **Sentinel** — a magic value encoding "missing". Here, `x5 = 999.0`
  on ~15% of rows.

## 1. Competition and data

### Observations

- Tabular regression, metric **MAE**. Submission is `(id, target)`
  on 1,500 test rows.
- Training: 1,500 rows in `dataset.csv`. Features
  `x1, x2, x4, x5, x6, x7, x8, x9, x10, x11` (competition skips `x3`)
  plus categoricals `Country` and `City`. Target continuous,
  `std ≈ 24.1`.
- Test: 1,500 rows in `test.csv`, same schema without `target`.

## 2. Data quirks

### Observations

- `Country` is constant at `Spain` on every row (train and test).
- `City` is binary: `Zaragoza` / `Albacete`.
- `x5` has 222 training rows and 228 test rows exactly equal to
  `999.0` (the sentinel). The remaining rows lie in `[7.0, 11.5]`
  with no values in `(11.5, 999)`.
- `x6² + x7² = 324` to numerical precision on every row, i.e.
  `(x6, x7)` lies on a circle of radius 18. The angle
  `θ = atan2(x7, x6)` is uniform on `[−π, π]`.
- `x4` is bimodal with a gap at `[−0.167, +0.167]`. Training has
  zero observations in this gap; test has 508 rows (33.9%) inside it.

![x4 bimodality — gap at 0](plots/diagnostics/x4_bimodality.png)

![x5 non-sentinel and back-solved sentinel distributions](plots/formulas/x5_imputation_distribution.png)

### Discussion

Constant `Country` drops out. Binary `City` becomes a ±1 indicator
with a large coefficient (~24–25). x5's sentinel pattern is clean:
non-sentinel values are Uniform(7, 12) and every imputation strategy
(mean, median, k-Nearest-Neighbours, gradient boosting) hits exactly
1.25 per-row MAE on sentinel rows — the Uniform-median optimal bound
— so nothing beats median imputation for x5. The x6/x7 circle means
the radial component carries zero information; the angle is uniform
and independent of everything, effectively noise. The x4 gap is the
most consequential quirk: any functional form for `target(x4)` on
the gap fits training equally well because there is no training
signal there, but 34% of test rows live inside it. This is how A1's
"+20 step at x4=0" survived reverse-engineering but failed on the LB.

## 3. Training vs test distribution shift

### Observations

- `corr(x4, x9) = +0.832` in training, **`+0.001` in test**.
- Training joint of `(x4, x9)` forms two disjoint clusters:
  `x4>0 → x9 ~ N(5.97, 0.57)`, `x4<0 → x9 ~ N(4.02, 0.57)`.
  Within-cluster `corr(x4, x9) ≈ 0`.
- Test contains 734 rows (48.9%) in the off-diagonal quadrants
  `{x4>0, x9<5}` and `{x4<0, x9>5}`, which are empty in training.
- x5 sentinel rate is preserved: 14.8% train, 15.2% test.
- All other pairwise feature correlations are `< 0.06` on both sets.

![(x4, x9) joint: training vs test](plots/reweight/x4_x9_joint_train_vs_test.png)

### Discussion

The x4-x9 correlation flip (0.83 → 0.001) is the single most
consequential shift in this dataset and is a textbook **selection
bias** artifact: training rows were drawn in a way that coupled x4
and x9 (two disjoint clusters), while the generative process used for
the test set samples them independently. The correlation is real in
the training sample and absent in the true data-generating process.
Any model that treats the training coupling as structural — an
interaction term, a residualised feature like A2's
`x9_resid = x9 − β·x4`, a cluster-centered `x9_wc`, or a linear
coefficient on x9 inflated by omitted-variable bias — extrapolates
wrongly on the 49% of test rows in the off-diagonal quadrants.

Our mitigation is the **cross-view ensemble** behind
`submission_ensemble_cross_LE.csv` (LB MAE 2.94, our best public
result). We fit the full feature set **twice**:

- one model drops `x9` and keeps `x4`;
- one model drops `x4` and keeps `x9`.

Neither single model can exploit the spurious x4-x9 coupling because
only one of the two features is visible to it. Averaging their
predictions `0.5·LIN(no x9) + 0.5·EBM(no x4)` cancels each view's
reliance on the training joint: on an off-diagonal test row, one
submodel sees only x4 (correct) and the other sees only x9
(correct); the selection-induced bias from each is uncorrelated and
partially averages out.

Density-ratio reweighting (our alternative attempt to "fix" the joint)
failed for a structural reason: reweighting only redistributes
probability mass across observed rows, and the off-diagonal quadrants
have zero mass in training — there is nothing to reweight toward.
Models that survived the shift without ensembling are the ones whose
x9 treatment cannot extrapolate: EBM's bounded shape functions hold
at the boundary bin value instead of projecting a trend.

## 4. Signal discovery

### Observations

- Marginal linear correlations with target, decreasing magnitude:
  `City, x4, x5, x8, x9, x10, x11`.
- `x1` has no linear signal (`r ≈ 0`) but a hump shape (GAM
  R² = 0.109).
- `x2` has no linear signal but oscillates (GAM R² = 0.068, fit
  by `cos(5π·x2)`).
- `x6, x7` individually have no univariate signal; θ is independent
  of every other feature and of the sentinel indicator.
- The **PC algorithm** and **DirectLiNGAM** (two linear-acyclic
  causal-graph recovery methods) consensus on training: a directed
  edge `x4 → x9` is detected.
- Per-pair interaction search (double-centred residual grid, 12×12
  bins): `x10 × x11` is the only pair above the noise floor
  (Root-Mean-Square 3.02 vs floor ~1.71).

![PC + DirectLiNGAM consensus causal graph](plots/causal/consensus_dag.png)

![Per-pair residual heatmap grid — only x10·x11 shows structure](plots/interactions/target_pairwise_residual.png)

### Discussion

Linear-correlation scans work well on `x4, x5, x8, x9, x10, x11,
City` and miss `x1, x2` entirely — both have `r ≈ 0` but a
pronounced nonlinear response. GAM and EBM shape functions recover
x1's hump and x2's sinusoid without manual coaxing. The **PC +
DirectLiNGAM `x4 → x9` edge was a false positive**: both algorithms
assume linear causal structure and no selection bias, so the strong
train-only correlation satisfied their conditional-independence
tests. The test-set correlation of 0.001 falsifies the causal claim
cheaply. Several rounds of modelling (Simpson's-paradox framing for
x9, `x9_resid`, `x9_wc`) that encoded the spurious edge then failed
on LB. The double-centred per-pair interaction search was
model-agnostic and correctly isolated `x10·x11` as the only real
interaction.

## 5. The A1 closed-form formula

### Observations

A sibling branch reverse-engineered this zero-parameter formula (we
call it **A1**):

```
target = −100·x1² + 10·cos(5π·x2)
       + 15·x4 + 20·𝟙(x4 > 0)
       − 8·x5_imp + 15·x8 − 4·x9
       + x10·x11 − 25·𝟙(City = Zaragoza) + 92.5
```

- Training CV: overall MAE 1.80, non-sentinel 0.38, sentinel 10.0.
- **93.3% of non-sentinel training rows** have `|residual| < 0.01`.
- The remaining 6.7% all lie in the quadrant `x4 < 0 AND x8 < 0`.
  No imperfect rows outside that quadrant.
- On those imperfect rows: `residual / x8` has mean −18.41, std 0.76,
  range [−21.29, −17.18]. We call this subset the "clamp".
- **Public LB: MAE 10.80.**

### Discussion

A1 is effectively the true DGP for ~93% of non-sentinel training
rows, explaining the numerical-precision fit (zero free parameters,
no overfitting possible). The 23% clamp subset inside `x4<0 AND
x8<0` has a strikingly consistent signature (residual ≈ `−18.4·x8`,
std 0.76), suggesting the DGP replaces `+15·x8` with something close
to `−3.4·x8` on a hidden subset. A1's LB 10.80 vs CV 1.80 is
explained by two failure modes: the `+20·𝟙(x4>0)` step misfires on
the 508 test rows inside the x4 gap (training had zero observations
there, so the step was unfalsifiable during reverse-engineering), and
`−4·x9` extrapolates badly on the 734 off-diagonal test rows. The
two failure regions together cover ~83% of test rows.

## 6. Models trained

### Observations

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
| EBM | heavy smoothing (smoothing 2k + refine 1.5k) | 3.08 | **4.90** |
| EBM | all-smoothed (smoothing 4,000, no refine) | 3.03 | — |
| Ensemble | EBM + GAM, 70/30 weighted | 2.91 | 6.47 |
| Ensemble | **cross_LE = 0.5·LIN(no x9) + 0.5·EBM(no x4)** | 2.97 | **2.94** |
| Ensemble | triple = 0.25·LIN + 0.25·EBM(no x4) + 0.5·EBM(full) | 2.82 | — |
| Router | safe → A1; else → triple ensemble | **1.84** | — |

Column-name conventions: **LIN** is the hand-crafted linear design
matrix (`x1², cos(5π·x2), x4, x5_imp, x5_is_sent, x8, x10, x11,
x10·x11, City`, optionally `x9`). **EBM(no x4)** and **EBM(no x9)**
are EBM on the full feature set with one column removed; **EBM(full)**
keeps both.

### Discussion

Three patterns dominate the CV-vs-LB table.

**First**, parametric models catastrophically miss on LB (A1
CV 1.80 → LB 10.80; A2 CV 3.49 → LB 9.44; locked-integer + `x9_wc`
CV 2.90 → LB 10.75). Their closed-form x9 coefficients — raw,
residualised against x4, or within-cluster-centered — all encode the
training joint structure that doesn't exist in test.

**Second**, EBM's bounded shape functions generalise (CV 3.1 → LB
5.66; heavy smoothing CV 3.08 → LB 4.90). EBM cannot extrapolate its
learned shape past training bins, which is a feature here: on
off-diagonal test rows with "impossible" (x4, x9) combinations, EBM
holds the nearest-bin value rather than projecting a trend.

**Third**, ensembles that mix complementary failure modes help
dramatically. `cross_LE` pairs LIN without x9 (immune to the shift)
with EBM without x4 (handles everything else nonparametrically);
errors on off-diagonal rows partially cancel, hitting LB 2.94.
Adding EBM(full) at 50% weight (`triple`) drops CV to 2.82. The
router exploits A1's zero-parameter exactness on the ~20% of test
rows where the safe predicate holds.

## 7. Final submissions

### Observations

Four CSV files live in `submissions/`. The notebook
`notebooks/final_submissions.ipynb` rebuilds them byte-identically.

| File | CV MAE | LB MAE |
|---|---|---|
| `submission_ebm_heavy_smooth.csv` | 3.08 | **4.90** |
| `submission_ensemble_cross_LE.csv` | 2.97 | **2.94** |
| `submission_ensemble_triple_locked_b_lambda50.csv` | 2.82 | untested |
| `submission_router_A1_triple.csv` | 1.84 | untested |

## 8. Noise floor and placement

### Observations

- x5 has slope ≈ −8 on target; imputing a Uniform(7, 12) feature
  with its median leaves ≈ 1.25 per-row MAE. Weighted by the 15.2%
  test-set sentinel rate: `0.152 × 8 × 1.25 = 1.52` MAE.
- LB top cluster sits at **1.65–1.71**, within 0.2 of the theoretical
  floor.
- Our best non-sentinel CV MAE was 1.72 (cross_LE).

### Discussion

**No submission can beat 1.52 MAE on the public LB.** The top cluster
at 1.65–1.71 implies those teams recovered the DGP to ≈ 0.15 MAE on
non-sentinel rows. Our best LB (2.94) is ~1.4 MAE above them. The gap
is approximately "A1's 93% non-sentinel perfection on the safe
quadrants that we couldn't safely route to", not a modelling-quality
gap — EBM-family submissions track CV well and are already near
their ceiling.

## 9. Rejected ideas (one line each)

- **Drop x9 from EBM.** CV 3.83 vs 3.08 — x9 carries real within-
  training signal that EBM extracts without over-committing.
- **Residualise x9 against x4 (A2's `x9_resid`).** Encodes the
  spurious x4-x9 edge; LB 9.44.
- **Within-cluster-centered `x9_wc`.** Same issue, different
  parameterisation; LB 10.75.
- **Monotone EBM constraints** on x4, x5, x8, City. CV 3.49.
- **EBM with interactions disabled.** CV 4.16.
- **EBM with x4 forced linear via residualisation.** CV 3.33–3.61.
- **Per-cluster EBM** (one model per `sign(x4)`). CV 3.50.
- **LightGBM tuned.** CV 4.66 — shallow trees + heavy regularisation
  still worse than default EBM.
- **GAM + x10·x11 interaction.** CV 3.56 — best additive-only model.
- **Random Forest.** CV ~8.8 — can't fit x1/x2 nonlinearity smoothly.
- **Quantile regression with splines.** CV 6.96 — L1 loss +
  high-dim spline basis caused optimisation issues.
- **Huber regression.** CV 5.19 — no effect vs OLS (outliers are
  sentinel-driven, not leverage-type).
- **k-Nearest-Neighbours (20 configs).** Best CV 9.33 — local
  similarity misses `cos(5π·x2)` structure.
- **NNLS** (Non-Negative Least-Squares) **stacking of EBM + GAM
  + A1.** CV 2.02 — A1 got 0.83 weight and memorised training; LB
  would follow A1's 10.80.
- **Density-ratio reweighting** to break the x4-x9 correlation
  (classifier, KDE = Kernel Density Estimator, Gaussian-copula
  analytic). Weighted correlation dropped 0.83 → 0.74 best;
  downstream CV moved ≤ 0.08. Structurally impossible to generate
  mass in the empty quadrants.
- **Synthetic off-diagonal augmentation** (resample x9, recompute
  target via A1). CV 4.40 — A1 as DGP oracle backfires because
  A1's `−4·x9` is itself wrong on test.
- **A1 + EBM residual corrector.** CV 2.04 — EBM overfits noise on
  the 93% of rows A1 already fits exactly.
- **Classifier-based clamp trigger for the router.** LightGBM
  reached AUC (Area Under the Receiver-Operating-Characteristic
  Curve) 0.76 on the clamp. Routing variants (soft blend, hard
  threshold 0.3/0.4/0.5/0.6) all gave CV 1.85–1.86, worse than
  the parameter-free base router (CV 1.84). Would need AUC ~0.9
  to pay for the 4:1 reward-to-cost routing asymmetry.
- **x6/x7 angle θ as a feature.** KS (Kolmogorov-Smirnov) `p = 0.89`
  against the clamp indicator; independent of every other feature.
  Adding θ hurt EBM CV 3.47 → 3.56.

  ![A1 perfect-fit vs x6/x7 angle — uniform across θ](plots/a1_clamp/a1_fit_vs_x6x7_angle.png)

- **Integer-locked coefficients** on A2's basis. CV 2.90 (identical
  to free fit) — the DGP is integer, but the formula still
  extrapolates wrongly on test (LB 10.75).

## 10. Open questions

### Discussion

- **Hidden clamp trigger.** The 23% of `x4<0 AND x8<0` rows with
  `residual ≈ −18.4·x8` look like a Bernoulli(0.23) draw in the DGP.
  No observed feature or two-way interaction predicts it above AUC
  0.76. Three-way interactions, unsupervised structure discovery,
  or a non-tabular side channel might push it to 0.9.
- **DGP oracle for test-distribution augmentation.** Synthetic
  off-diagonal rows would close the train-test shift if we had the
  true target function. A1 is not that function on the off-diagonal
  side; we don't know what is.
