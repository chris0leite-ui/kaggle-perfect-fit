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
| Ensemble | triple = 0.25·LIN + 0.25·EBM(no x4) + 0.5·EBM(full) | 2.82 | 3.71 |
| Router | safe → A1; else → triple ensemble | **1.84** | 3.35 |
| Ensemble | triple_view = (EBM_block_s + LIN_x4 + EBM_x9) / 3 | 2.92 | 4.66 |

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
**Subsequent extensions regressed on LB despite better CV**: adding
EBM(full) at 50% weight (`triple`, CV 2.82) landed at LB 3.71, a 1.32×
CV→LB degradation; routing "safe" rows to A1 (CV 1.84) landed at LB
3.35, a 1.82× degradation; adding a within-cluster block model to
cross_LE (`triple_view`, CV 2.92) landed at LB 4.66, a 1.60× degradation.
In every case the CV gain was apparent, not real — extra capacity
absorbed training-specific selection structure that doesn't transfer
to test. The plateau at **LB 2.94 is the real ceiling** under available
signal.

The block-model failure is instructive: it looked principled because
within each sign(x4) cluster, x4 and x9 are already independent in
training — exactly matching the test joint. But on off-diagonal test
rows (49% of test), the block EBM extrapolates: an x4>0 test row with
x9=3 gets routed to the block-1 model whose training x9 was
N(5.97, 0.57), so x9=3 is 5σ out-of-distribution and EBM pins it to
the lowest training bin. That pinning produces biased boundary values
that CV never stress-tests, because training is 100% on-diagonal by
construction.

## 7. Final submissions

### Observations

Four CSV files live in `submissions/`. The notebook
`notebooks/final_submissions.ipynb` rebuilds them byte-identically.

| File | CV MAE | LB MAE |
|---|---|---|
| `submission_ebm_heavy_smooth.csv` | 3.08 | **4.90** |
| `submission_ensemble_cross_LE.csv` | 2.97 | **2.94** |
| `submission_ensemble_triple_locked_b_lambda50.csv` | 2.82 | 3.71 |
| `submission_router_A1_triple.csv` | 1.84 | 3.35 |
| `submission_triple_view.csv` | 2.92 | 4.66 |

**Primary recommendation: `submission_ensemble_cross_LE.csv`** (LB 2.94).
Every candidate with lower CV regressed on LB — the CV→LB multiplier
grew with CV gain, confirming cross_LE is the plateau.

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
non-sentinel rows. Our best LB (2.94) is ~1.4 MAE above them.
Decomposing: cross_LE's non-sentinel LB MAE is
`(2.94 × 1500 − 228 × 10) / 1272 ≈ 1.67`, so the residual headroom
lives entirely in non-sentinel rows.

The A1 router's LB 3.35 implies non-sentinel MAE ≈ 2.16, ~0.9 above the
heuristic projection of `(419·0.38 + 853·1.7)/1272 = 1.27`. The broken
assumption was that A1's training non-sentinel MAE (0.38) transfers to
test. It does not. A1's `+15·x4 + 20·𝟙(x4 > 0)` step encodes the training
selection rule (empty x4 gap), not the true DGP, so even "safe" rows
(|x4| > 0.167, on-diagonal x4-x9) get systematic per-row error: at x4 =
0.5, A1 predicts `+27.5` while A2's pure-linear `+30.5·x4` predicts
`+15.25`. On test (where the gap selection doesn't hold) the A2-style
form is closer, which is why EBM's bounded shapes beat A1 despite worse
training fit.

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
- **Alternative functional forms for the x4 transition.** A1's step at
  x4 = 0 was unfalsifiable on training (zero observations in the gap).
  Plausible alternatives that also fit training near-perfectly but may
  extrapolate better:
  - **Smooth transition**: `+30·x4 + 20·tanh(k·x4)` with large k looks
    like A1's step on training yet passes smoothly through zero on test.
  - **Pure linear with cubic correction**: `+30·x4 − 2·x4³`.
  - **A2 form + explicit x8 clamp**: `+30·x4 + 14·x8` outside the clamp
    region, with the `−18.4·x8` adjustment on the clamped subset. Needs
    a clamp trigger at AUC ≫ 0.76.
  Each is cheap to encode but costs a submission slot to validate; none
  have been tested.
- **LB-driven ensembling.** We have three LB-validated base learners
  (cross_LE 2.94, EBM_heavy_smooth 4.90, EBM 5.66). Submitting 2–3
  weighted blends (e.g. `0.7·cross_LE + 0.3·EBM_smooth`) would let us
  triangulate optimal weights using real test signal rather than CV.
  Expected gain modest (~0.05–0.15 MAE) since cross_LE already
  dominates.
- **Adversarial validation.** Train a classifier to distinguish train
  from test; down-weight training rows with high "train-like" scores
  and refit EBM. Unlike our failed x4-x9 density-ratio reweighting
  (which could only redistribute mass across observed rows), adversarial
  weights use the full feature space and might partially close the
  shift. Not guaranteed to beat cross_LE.

## 11. Final reflection — this is hand-crafted synthetic data

The dataset is synthetic. A1's formula interpolates 93% of non-sentinel
training rows to numerical precision — that cannot happen on real data.
The DGP is some closed-form expression; the top of the leaderboard at
1.65–1.71 (within 0.2 of the 1.52 sentinel floor) means the winners
recovered it almost exactly. We did not.

Why we are stuck at LB 2.94 despite knowing the structure:

1. **Two of A1's ten terms are wrong on test.** The `+20·𝟙(x4 > 0)` step
   and `−4·x9` collectively account for A1's CV-1.80 → LB-10.80 collapse.
   The rest of the formula (`−100·x1² + 10·cos(5π·x2) − 8·x5 + 15·x8 +
   x10·x11 − 25·Zaragoza`) is correct.
2. **The x4 gap and x4-x9 selection bias make 2/10 terms unfalsifiable
   from training alone.** No amount of CV can tell us whether the x4
   transition is a step, a smooth sigmoid, or a cubic correction.
   Similarly for x9 — our β_x9 estimates are biased by the training
   joint in ways CV cannot detect.
3. **A 23% Bernoulli clamp inside `x4<0 AND x8<0` has no observable
   trigger up to AUC 0.76.** It is either truly stochastic in the DGP
   or triggered by something we cannot see.

**Simple solutions that would work if true — none of which we have
evidence for**:

- Submit A1 with `−4·x9` replaced by `0·x9` and the x4 step replaced
  by `+30·x4 + 20·tanh(20·x4)`. If the DGP is A2-like with a smooth
  x4 transition and x9 is noise, this one submission could score
  ≤ 2.0.
- Submit A2 with the clamp applied on the quadrant average
  (`+11.8·x8` inside x4<0, x8<0 rather than `+15·x8`). Handles the
  clamp via Bayes-optimal blending with no trigger needed.
- Combine both corrections in one submission.

Each is one linear-model fit, no ML. **We have not tried any of them
because every CV-informed probe regressed on LB, and these three are
CV-invisible (CV score would be ~3.5, LB could be ≤ 2.0 if right or ≥
10 if wrong).** They are genuine coin-flips — no evidence either way.

**Bottom line**: the final ceiling is 1.52 (sentinel floor). Our 2.94
is ~60% of that headroom above the floor. The leaderboard top sits
near the floor because they solved the hand-crafted formula. We did
not, and closing the gap requires one lucky guess at the remaining
two-term correction — not more modelling.

## 12. Post-plateau breakthrough — DGP archaeology

After section 11 concluded "we did not solve the hand-crafted formula",
a new round attacked the DGP directly. Four ideas in sequence cracked
the generation process.

### 12.1 Pooled-feature rediscovery

**Observations.** Re-ran pairwise correlations on (train alone), (test
alone), and (pooled 3000 rows). Across all 55 feature pairs + City:

| pair | r_train | r_test | shift |
|---|---|---|---|
| x4, x9 | +0.832 | +0.001 | **+0.831** |
| every other | < 0.05 | < 0.05 | < 0.04 |

Standard error for a true-null r at n=1500 is 0.026, so the 3σ
threshold is 0.08. Only x4-x9 crosses it.

**Discussion.** Every prior LB regression (A1, locked_b, router,
triple) traced back to x9 in some way. The pooled scan confirms the
selection-bias contamination is narrowly concentrated on x4-x9.
No hidden coupling among other features to worry about.

### 12.2 x4 functional-form oracle — step/x9 confound

**Observations.** 12 candidate f(x4) bases (step, linear, tanh,
sigmoid, polynomial, spline, knot-at-±0.17, abs-hinge) fit to
training. Every near-step basis (sigmoid_50, tanh_narrow, sharp step,
knots_0.17) ties at CV 2.19 — training has zero rows in the gap, so
they're indistinguishable on CV.

The cluster-bias audit across every feature:

| feature | t-stat (cluster mean diff) | conclusion |
|---|---|---|
| x9 | **−66.5** | strongly clustered by sign(x4) |
| every other | \|t\| < 2 | no cluster dependence |

Fitting linear_step with different x9 treatments:

| variant | β_step | β_x9 | CV |
|---|---|---|---|
| raw x9 (A1-ish) | **+19.54** | −4.26 | 2.19 |
| x9_wc (Simpson-corrected) | **+11.21** | −4.26 | 2.19 |
| drop x9 | **+11.38** | — | 3.39 |

**Discussion.** A1's +20 step was **double-counting** x9's cluster
gap of +1.96 × β_x9 = −4, inflating the real step by +7.8. True step
is ≈ +11. This finally explains why every A1/locked_b variant LB-flopped
around 10.75 — they used the wrong step AND the wrong β_x9.

Plugging the corrected step into cross_LE: solo LIN improved (CV 3.70
→ 3.39), but the ensemble regressed (2.95 → 3.04) because EBM_x9 was
already absorbing the cluster contrast implicitly. Net: no LB win
available from this correction alone, but the mechanism is now
understood.

### 12.3 Clamp archaeology — trigger is `id < 100`

**Observations.** Fit the residual correction on 86 clamp rows:
`residual = −15·x8 + 1` (R² = 0.994, std = 0.07). Brute-force over
1-feature thresholds, 2-feature sums/diffs/products/ratios, angular
cuts, and AND-rules plateaued at AUC 0.76 — no feature-observable
trigger. Checked other quadrants: all 910 non-quadrant non-sentinel
rows have A1 residual exactly zero.

Scanned id-100 buckets. The first 100 training rows contain all 86
clamp rows; every other bucket has zero clamp rows. `id < 100` gives
sens 1.000 and spec 1.000.

**Discussion.** The clamp is a **training-data artefact** added
post-hoc to the first 100 training rows. Test ids start at 1500 → no
test row is a clamp row. A1 is the exact training DGP for rows id≥100
(non-sent MAE = 0.0000 on 1192 rows).

A1's LB 10.80 failure comes entirely from the x4-x9 shift on x9,
NOT from the clamp. Combining the corrections from 12.2 and 12.3
gives a family of "corrected A1" submissions with integer-preserved
coefficients, −4·x9 dropped, step re-tuned on rows id≥100. Six step
variants written (+10, +11, +12, +15, +20, 0) plus 12 cross_LE-blend
hedges.

### 12.4 x5 archaeology — confirmed MCAR noise

**Observations.** Using the exact A1 fit on id≥100, back-solved
x5_true = (A1_body − target) / 8 for 222 training sentinel rows. Max
|err| = 5.3e-15 on 1192 validation rows (machine precision). Recovered
sentinel distribution: mean 9.47, std 1.45; KS test vs Uniform(7, 12)
p = 0.645.

Tested whether id or position structures the sentinels:
- Sentinel rate by id-100 bucket: [0.07, 0.24] at n=100 (binomial SE
  0.036, all within 3σ)
- mod-{2, 3, 5, 7, 10, 100} rates: all ≈ 0.148
- ACF of x5 sorted by id, lags 1–30: every |ρ| < 0.06 (95% CI ±0.057)
- Pearson r with 9 transforms (sin/cos of 2π·id/1500, golden-ratio
  drift, LCG hash, etc.): |r| < 0.05, p > 0.1

**Discussion.** Sentinel selection is MCAR; x5 is genuine noise. No
DGP-archaeology finding helps x5 imputation.

### 12.5 Seed recovered — `np.random.RandomState(4242)`

**Observations.** Brute-force scan of seeds 0..100 000 on both MT19937
and PCG64 APIs. Hit: seed 4242 with `np.random.RandomState` matches
x1's first 5 values to 5.55e-17, full-row max |err| = 9.89e-17.

RNG sequence reverse-engineered:

| call | formula | features | max \|err\| |
|---|---|---|---|
| 1 | `rs.uniform(-0.5, 0.5, 3000)` | x1 | 9.89e-17 |
| 2 | `rs.uniform(-0.5, 0.5, 3000)` | x2 | 9.89e-17 |
| 3 | `rs.uniform(0, 1, 3000)` | city (Zaragoza if < 0.5) | 3000/3000 |
| 4 | `rs.uniform(0, 1, 3000)` | c4 → x4 piecewise | 1.11e-16 |
| 5 | `rs.uniform(7, 12, 3000)` | x5 (pre-mask) | 1.78e-15 |
| 6 | `rs.uniform(0, 1, 3000)` | c6 → x6, x7 on circle | 1.78e-15 |

- x4 = (id<750) c4/3 − 0.5 ; (750≤id<1500) c4/3 + 1/6 ; (id≥1500) c4 − 0.5
- x6 = 18·sin(2π·c6); x7 = 18·cos(2π·c6)

Verified on train slice (ids 0–1499) AND test slice (ids 1500–2999)
under the same single seed 4242. **No separate test seed**.

x9, x10, x11 are NOT recovered. Brute force ruled out: continuation
of 4242 stream at any tested offset/distribution/size, separate
MT19937 seeds 0..2M, PCG64 seeds 0..1M, Python `random` seeds 0..100k,
2D/3D array calls, sorted-match under shuffle. Possible-but-untested:
seeds > 2M, non-uniform distributions, `rs.shuffle` between calls,
rejection sampling.

**Discussion.** The SENTINEL recovery is what matters for LB.

Test x5 for all 228 sentinel rows retrieved from call #5. The entire
**1.52-MAE sentinel floor collapses to 0**. Combined with the
corrected A1 formula from 12.2–12.3, we now have submissions that
encode every reverse-engineered DGP piece:

| submission | step | x9 | intercept | hypothesis |
|---|---|---|---|---|
| seed4242_step12_nox9 | +12 | dropped | +76.52 | corrected A1 |
| seed4242_step11_nox9 | +11 | dropped | +77.05 | idea-#2 value |
| **seed4242_step20_withx9** | +20 | −4·x9 | **+92.500** | A1 exact |
| (5 more variants + 12 blends) | | | | |

`step20_withx9` has the integer-exact A1 intercept of +92.5 — literally
A1 applied to test with perfect sentinel x5. If A1 is the test DGP,
projected LB ≈ 0; else LB ≈ 9.3 (prior A1 LB minus 1.52 sentinel
saving).

### 12.6 Projected LB

Given cross_LE's CV→LB ratio of 0.99 and simple_linear_interact's 2.0:

- **seed4242_step12_nox9** (CV 3.16): expected LB 1.5–2.5. Better than
  cross_LE's 2.94 by the 1.5 MAE sentinel correction, potentially more
  if the step is right.
- **seed4242_step20_withx9**: bimodal. LB ≈ 0 if A1 is the DGP, else
  ≈ 9.3. No middle ground.
- **Floor of what's achievable** with seed recovery: near 0 (since
  sentinel MAE is eliminated), limited only by DGP-formula correctness.

The leaderboard top at 1.65 was previously thought to be the
theoretical floor (sentinel floor 1.52 + ~0.1 non-sent). With x5
recovered, submissions that match the full test DGP can beat this.

## 13. 🎯 TRUE DGP — LB 0.00

The user raised a critical observation: the competition description states
test data is in the convex hull of training data. That implies a single
DGP across both splits. Since A1 fits training exactly (max |err| 4e-14)
but scored LB 9.79, A1 must be *equivalent-on-training* to the true DGP
via a feature-coupling coincidence — not the DGP itself.

**The substitution**: on training, `1(x4 > 0) ≡ 1(x9 > 5)` for all 1500
rows (the piecewise c4→x4 transform couples sign(x4) to the x9 cluster
via id ranges). On test where x4 ⊥ x9, the two indicators disagree on
~50% of rows.

**True DGP**:

    target = −100·x1² + 10·cos(5π·x2) + 15·x4 − 8·x5 + 15·x8 − 4·x9
           + x10·x11 − 25·zaragoza + 20·1(x9 > 5) + 92.5

**Verification** by projecting `20·(1(x9>5) − 1(x4>0))` onto the observed
v5-A1 delta basis on test (x9 ~ U(3, 7), x4 ⊥ x9):

|                   | predicted | observed |
|-------------------|----------:|---------:|
| β_x9              | +7.5      | +7.46    |
| β_1(x4>0)         | −20       | −19.78   |
| intercept         | −27.5     | −27.03   |

The three coefficients match to 1%. Same formula fits training non-clamp
rows to machine precision via either indicator (they are identical on
training, 1500/1500). This is the complete closed form.

**Final leaderboard**:

| submission | LB |
|---|---:|
| A1 literal | 9.79 |
| cross_LE | 2.94 |
| v4 (cross_LE + x5 patch) | 1.66 |
| v5 (clean-x5 retrain) | 1.37 |
| **TRUE_DGP** | **0.00** |

**What made the hand-crafted DGP survive for so long**:

The author's single non-obvious touch was using `1(x9 > 5)` rather than the
semantically equivalent `1(x4 > 0)`. On training, these are tautologically
equal — no fit, no CV, no residual-analysis tool can distinguish them. Only
the test-distribution change exposes the difference, and only if the analyst
hypothesises the substitution from structural reasoning (feature-coupling
arithmetic) rather than data-driven methods.

Every other archaeological step (seed recovery, clamp origin, pooled
rediscovery, step/x9 decomposition) merely narrowed the space so the
final substitution could be guessed. The convex-hull invariance principle
the user invoked was what made the closed form derivable.
