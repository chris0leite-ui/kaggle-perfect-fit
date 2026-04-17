# LEARNINGS

Portable lessons from the "perfect-fit" Kaggle competition that should
carry over to future tabular-regression problems. `CLAUDE.md` has the
narrative history; this file has the transferable patterns.

Best result on this competition: **LB 2.94** (cross_LE ensemble), against
a theoretical sentinel-noise floor of 1.52 and a leaderboard top cluster
of 1.65–1.71. Our sentinel-stripped non-sentinel MAE was about 1.7 —
roughly the quality of the top models, dominated by irreducible x5
sentinel noise.

## 1. Evaluation protocol

- Split a **holdout** early from the full dataset (seed-fixed, ~20% of
  rows, stratified by any categorical that might shift). Do not touch
  it during exploration.
- For everything else use `KFold(n_splits=5, shuffle=True, random_state=42)`.
  Fixed seed across every script. `cross_val_score` is too opaque —
  compute OOF predictions explicitly so you can slice them later.
- **Always report MAE by stratum**: overall, and non-sentinel if your
  data has a sentinel column. Two numbers beat one — many models look
  close on overall MAE but differ on the clean subset.
- Primary metric must match the leaderboard's. Don't optimise RMSE and
  hope an MAE-graded LB moves.

## 2. First-day EDA checklist

These five things, in this order, would have saved us weeks:

1. `df.nunique()` per column → catches constant features (`Country="Spain"`).
2. Marginal histograms per feature → catches bimodal gaps, sentinel
   spikes (we found x4 bimodal with a 0.33-wide gap at 0 and x5=999
   appearing 222× in 1,500 rows).
3. `np.hypot(col_a, col_b)` sanity for every numeric pair → catches
   algebraic constraints. We missed for weeks that `x6²+x7²=18²` exactly.
4. **Train vs test correlation heatmap** — Pearson r for every pair,
   computed separately on train and on test. We missed the x4–x9 shift
   (train r=+0.83, test r=0) for weeks. This single check would have
   redirected our modelling early.
5. `df[col].isna()` and `df[col]==SENTINEL` counts per column. Sentinels
   masquerading as real values destroy linear models silently.

## 3. Sentinels and the irreducible-noise floor

- **Impute with training median + keep a binary indicator column**. The
  indicator rarely contributes predictive power on its own but is cheap
  insurance and lets tree models express "is this row safe" to
  downstream terms.
- Compute the **sentinel noise floor** before modelling:
  `slope_of_feature × (feature_range / 4)` under a Uniform prior.
  For us this was `8 × 1.25 = 10 MAE` on sentinel rows. Multiplied
  by the sentinel ratio (15.2% on test) it yields **1.52 — the
  theoretical LB floor**. We verified the leaderboard top 4 sit within
  0.2 of this number. No amount of modelling crosses that line; chasing
  under-floor performance is wasted effort.
- Sentinel rows are a separate modelling regime from the rest.
  Track sentinel MAE and non-sentinel MAE as two metrics from day one;
  different models dominate each.

## 4. Selection bias is invisible to causal discovery

PC and LiNGAM both confidently inferred an `x4 → x9` edge from the
training correlation. Test data had `r(x4, x9) = 0.001`. Every
downstream decision that used the inferred edge (residualising x9 on
x4, treating x9 as a descendant, the `x4 & x9` EBM interaction) was
baking training-only structure into our models.

**Rule**: before trusting a causal edge or a high-correlation-driven
feature engineering choice, verify the correlation is **stable across
train and test**. If it isn't, the edge is a selection-bias artefact.

## 5. EBM as default tabular baseline

`interpret.glassbox.ExplainableBoostingRegressor` is our go-to for
tabular regression. It:

- discovers interactions automatically (`interactions=10` was plenty),
- gives interpretable shape plots for free,
- is more robust to distribution shift than LightGBM here (bounded
  shape functions can't extrapolate past training range),
- tunes cleanly along a **smoothing axis**: very small CV gains from
  heavier smoothing can translate into large LB gains under shift.

Config that worked across this competition:

```python
ExplainableBoostingRegressor(
    interactions=10, max_bins=128, min_samples_leaf=10,
    smoothing_rounds=2000, interaction_smoothing_rounds=500,   # "heavy"
    # or: smoothing_rounds=4000, interaction_smoothing_rounds=1000,  # "4k"
    max_rounds=2000,  # or 4000 for the 4k variant
    random_state=42,
)
```

The jump from `smoothing_rounds=500` (EBM default) to `2000` moved CV
from 3.3 to 3.1 (−0.2) but moved LB from 5.66 to 4.90 (−0.76) — a
**25× CV→LB multiplier**. Keep pushing smoothing until CV starts to
degrade, not just until it plateaus.

## 6. Reverse-engineered formulas: seductive and dangerous

We built a zero-parameter closed form ("A1") that fit 93% of training
rows **exactly** (CV 1.80, non-sent 0.38). On LB it scored 10.80. The
formula had memorised the training joint distribution (x4 ⊥ x9 was
false in train, so a `-4·x9` term absorbed training-specific structure
that didn't exist in test).

**Rules**:

- Never trust a CV number from a fitted-by-hand formula unless the
  fitting process can't see train-only correlations. Cross-view check:
  build the formula on one half of the data and evaluate on the other
  half; if CV collapses, the formula is memorising.
- Formulas with **integer coefficients** are a real possibility in
  designed-DGP Kaggle competitions. Lock the coefs to integers, refit
  only the intercept, and compare CV. If CV doesn't degrade, the DGP
  was integer — use the locked version, it generalises better.

## 7. Cross-view ensembling under distribution shift

When two features A and B are correlated in train but may be
independent in test, train **two models on disjoint subsets**:

- `MODEL_A` — includes A, excludes B.
- `MODEL_B` — includes B, excludes A.

Average them. On train both fit well; on test, where (A, B) can vary
freely, the average dampens each model's reliance on the A↔B coupling.

For us: `cross_LE = 0.5·LIN(x4, no x9) + 0.5·EBM(no x4, with x9)` was
our best LB submission (2.94), beating the full-feature EBM (5.66).
**Pairing a parametric model on one view with a non-parametric model
on the other helps further** — their error patterns are complementary.

Triple-ensemble with a "full" model at 0.5 weight helps CV and
typically LB too: `0.25·LIN_A + 0.25·EBM_B + 0.5·EBM_full`.

## 8. Router / memoriser pattern

If you find a model that is *near-perfect on a known-safe subset* but
catastrophic elsewhere, **route**, don't average:

```python
safe = (predicate_that_holds_in_DGP_invariants)
pred = np.where(safe, MEMORISER(X), ROBUST_ENSEMBLE(X))
```

"Safe" must be definable from **test-distribution invariants** (not
from training-only artefacts). For us: `x5 ≠ SENTINEL AND |x4| > gap
AND cluster(x4) matches cluster(x9)`. Any row that violates a known
invariant is routed to the robust side, never to the memoriser.

A classifier with AUC < ~0.9 is not sharp enough to replace a
hand-crafted predicate here — false positives cost more than true
positives gain when the memoriser fails catastrophically.

## 9. Don't let CV optimism mislead you

| Submission | CV MAE | LB MAE | CV→LB |
|---|---|---|---|
| A1 closed form | 1.80 | 10.80 | **6.0×** |
| A2 linear parametric | 3.49 | 9.44 | 2.7× |
| EBM alone (tuned) | 3.11 | 5.66 | 1.8× |
| cross_LE | 2.97 | 2.94 | 1.0× |

**CV→LB ratio is a model property** — it measures how much the model
memorises vs. generalises. Rank your candidates by LB, not CV, when
you can. When you can only see CV, **prefer models with bounded /
non-parametric components** — they have lower CV→LB ratios under
distribution shift.

## 10. Submission-day checklist

- Retrain every model on the **full** labelled dataset (not `train.csv`).
- Fix seeds everywhere (`np.random.seed`, `random_state=` on every
  estimator, `KFold(random_state=)`).
- Run the final notebook **from a clean kernel** — no leaking in-memory
  state from earlier experimentation.
- Commit the CSVs, the notebook, and a one-line LB log per submission.
  The LB log is the only ground truth you have.
- If you have multiple candidates, submit the **most conservative**
  (least extrapolating) one first. It sets a LB floor that lets you
  evaluate the riskier ones by delta.

---

## Code-review notes (specific to this repo)

Small observations from wrapping this up, as checklist items for next
time:

- **One `constants.py`**. `SENTINEL = 999.0` got redefined in at least
  ten scripts. A `src/constants.py` (or `src/_shared.py`) catches this.
- **Avoid script-as-library**. `scripts/cv_router_A1.py` imports from
  `scripts/cv_cross_LE_tune.py` which imports from
  `scripts/cv_x4_x9_swap_ensemble.py` which imports from
  `scripts/cv_ebm_variants.py`. That chain is hard to follow. Put
  reusable helpers under `src/` and keep `scripts/` as thin entry
  points.
- **Don't commit empty test scaffolds**. Five of the eleven test files
  in this repo contained zero tests. Either write them or don't
  commit the file.
- **Keep the `src/` ↔ exploratory boundary strict**. Modules like
  `causal.py`, `clusters.py`, `tuning.py` survived in `src/` but never
  entered the final pipeline. Push exploratory code to
  `scripts/` or `notebooks/` and reserve `src/` for what ships.
- **Label every submission CSV at birth**. We ended with seven CSVs
  and had to grep CLAUDE.md to remember which was which. A
  `submissions/README.md` table, or prefixing filenames with LB scores,
  would have saved time.
