# Clamp archaeology — the trigger is `id < 100`

Idea #3 from the DGP-archaeology plan, completed via three
diagnostic passes. The punch line is on line 1 of this header:
the clamp is **a training-data artifact**, not a DGP feature.

## The diagnostic chain

### 1. Correction shape (what the clamp DOES)

On 86 of 368 rows in the x4<0 & x8<0 quadrant, A1 has a non-zero
residual. Fitting `residual = α·x8 + β`:

    α = -14.89   β = +1.004   R² = 0.994   std(residuals) = 0.07

Equivalently, `residual = -15·x8 + 1` to three decimal places. For
clamp rows the DGP **drops x8's +15·x8 contribution and adds a
constant +1**. No other feature — not |x8|, not x4, not x8·x9, not
x4·x8 — improves the R² or reduces the residual std below 0.07.

### 2. The other three quadrants have zero A1 residual

|  quadrant      |   n  |   rows with |A1-resid|>1  | mean |resid| |
|----------------|-----:|--------------------------:|-------------:|
| x4<0, x8<0     |  368 |  **86 (23.4%)**           | 1.306        |
| x4<0, x8>0     |  285 |                         0 | **0.0000**   |
| x4>0, x8<0     |  331 |                         0 | **0.0000**   |
| x4>0, x8>0     |  294 |                         0 | **0.0000**   |

A1's formula fits **exactly** on 910 of 910 non-sentinel rows
outside the x4<0, x8<0 quadrant and on 282 of 368 rows inside it.
Across all of training it fits 1192 of 1278 non-sentinel rows to
machine precision.

### 3. The trigger is the row id

A brute-force search over single features, pairwise sums/diffs/
products/ratios, angular (θ = atan2(x7, x6)) cuts, AND-rules, and
band rules found AUC plateauing at ~0.76. No observable feature
produces a sens/spec ≥ 0.9 rule. But id does:

    id < 100   →  tp=86   fp=0   tn=282   fn=0     sens=1.000  spec=1.000

**All 86 clamp rows have id ∈ [0, 99]; all 282 non-clamp rows in
the quadrant have id ≥ 100.** Non-quadrant rows with id < 100 are
fit exactly by A1, so the id < 100 rule only fires inside the
x4<0, x8<0 quadrant.

Breakdown by id bucket (non-sentinel rows):

|  id range   |  n  |  |A1-resid| > 1  |
|-------------|----:|-----------------:|
| 0 – 99      |  86 |            **86** |
| 100 – 199   |  93 |                 0 |
| 200 – 299   |  85 |                 0 |
| 300 – 499   | 172 |                 0 |
| 500 – 799   | 257 |                 0 |
| 800 – 1499  | 585 |                 0 |

The clamp rows occupy **exactly** the first 100 ids of the training
file (86 of them land in the x4<0, x8<0 quadrant; the other 14 fall
in quadrants where A1's formula already collapses to the clamp
target by coincidence). Test ids start at 1500, so **no test row is
a clamp row**.

## Implication — A1 is the exact training DGP

Verified empirically:

    A1 on all 1500 train rows:        non-sent MAE = 0.3761
    A1 on train rows with id ≥ 100:   non-sent MAE = 0.0000

The first 100 training rows were modified after generation (x8's
contribution dropped, +1 added). Everything else in the training
file is an exact sample of the DGP A1 posits. The entire CV→LB gap
is attributable to the x4-x9 selection shift, **not** to the clamp.

## Corrected-A1 for test

With the clamp dispatched, the only remaining LB-relevant error
in A1 is its x9 term. Idea #1 established x4 ⊥ x9 in the true DGP
(train r = +0.832, test r = +0.001). Idea #2 showed A1's step +20
is half x9 cluster contrast: the residual step after decorrelating
x9 is +11. Combining the two fixes and fitting the intercept on
rows id ≥ 100:

    target = -100·x1² + 10·cos(5π·x2) + 15·x4 + 15·x8 - 8·x5_imp
             + x10·x11 - 25·zaragoza + step_c · 1{x4>0} + b

|  variant                         |  CV MAE |  non-sent |  intercept  |
|----------------------------------|--------:|----------:|------------:|
| V1 step = +20 (A1-style)         |   4.996 |     3.918 |     +72.08  |
| V6 step = +8                     |   3.658 |     2.525 |     +78.51  |
| V3 step = +15 (use x4 slope)     |   3.446 |     2.216 |     +74.76  |
| V4 step = +10                    |   3.294 |     2.096 |     +77.44  |
| **V2 step = +11** (idea #2 value)|   3.193 |     1.973 |     +76.90  |
| **V5 step = +12** (CV optimum)   |   3.158 |     1.923 |     +76.37  |
| V7 step = 0 (pure linear)        |   6.743 |     6.068 |     +82.80  |

The optimum lies at step ∈ [+11, +12] (the two are within CV noise).

### Projected LB

Existing cross_LE's CV→LB ratio is ≈ 0.99 (well calibrated).
simple_linear_interact (same structural family, pure linear
without step) had CV 3.695 → LB 7.38 (ratio 2.0). That 2× ratio
comes from the missing step. The corrected-A1 closes this gap:
the step is the true DGP feature, with β_x9 = 0 removing the
selection-shift contamination.

A best-case projection — the test DGP really is `A1 minus x9
term, step ≈ +12, clamp absent` — would put LB near the 1.52
sentinel floor. A worst-case (the step is still partly shift-
contaminated beyond what we've identified) keeps LB around 3–5.
Only a submission tells us which end we're at.

## Submissions written (all CV-tested, awaiting LB)

| File                                      |  step | CV MAE | non-sent |
|-------------------------------------------|------:|-------:|---------:|
| submission_a1_nox9_step20.csv             |   +20 |  4.996 |    3.918 |
| submission_a1_nox9_step15.csv (not built; see step12 similar) |  — |    — |       — |
| submission_a1_nox9_step12.csv             |   +12 |  3.158 |    1.923 |
| submission_a1_nox9_step11.csv             |   +11 |  3.193 |    1.973 |
| submission_a1_nox9_step10.csv             |   +10 |  3.294 |    2.096 |
| submission_a1_nox9_step8.csv              |    +8 |  3.658 |    2.525 |
| submission_a1_nox9_nostep.csv             |     0 |  6.743 |    6.068 |
| submission_blend_a1step{11,12}_crossLE_{30,50,70}.csv |  — |  — | — |

Robustness blends with cross_LE are hedges in case the corrected-A1
underperforms its CV projection. If step12 alone lands below 3.0 LB
we've confirmed the DGP; if above, the blend with cross_LE clips
the downside.

## Files

| Script                                         | Purpose |
|------------------------------------------------|---------|
| `scripts/clamp_archaeology.py`                 | Correction-shape fit + single/pair/angular trigger scan |
| `scripts/clamp_archaeology_v2.py`              | AND-rules and band-rule search (still failed) |
| `scripts/cv_soft_clamp_cross_LE.py`            | Soft-clamp correction on cross_LE (hurts CV) |
| `scripts/build_corrected_a1.py`                | Grid over step coef, fit intercept, write submissions |

| Artefact                                       | Contents |
|------------------------------------------------|----------|
| `plots/clamp_search/all_rules.csv`             | All triggering-rule scan results (pre-id discovery) |
| `plots/clamp_search/v2_*.csv`                  | AND-rule results |
