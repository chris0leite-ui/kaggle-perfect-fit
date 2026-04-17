# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition

Kaggle tabular regression: https://www.kaggle.com/competitions/the-perfect-fit

Dataset: 1500 rows, numeric features `x1, x2, x4–x11`, categorical features `Country` and `City`, target `target`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python3 -m pytest tests/

# Run a single test file
python3 -m pytest tests/test_data.py -v

# Run a single test by name
python3 -m pytest tests/test_data.py::test_split_sizes -v
```

Tests must be run from the repo root so that `src/` is importable.

## Architecture

```
src/
  data.py          # split_holdout(), load_train_holdout()
  eda.py           # detect_sentinels(), correlations(), scatter/partial-residual plots
  clusters.py      # assign_clusters(), replace_sentinels(), cluster_stats(), cluster plots
  causal.py        # run_pc(), run_direct_lingam(), consensus_graph(), bootstrap_edges()
  causal_plots.py  # plot_dag(), plot_adjacency_heatmap(), plot_edge_bootstrap()
  features.py      # CityEncoder, SentinelHandler, X9Residualizer, SplineBasisExpander, InteractionAdder, build_preprocessor()
  models.py        # GAMRegressor, EBMRegressor, AveragingEnsemble, WeightedEnsemble, StackedEnsemble, build_*() for 20+ models
  evaluate.py      # split_val_test(), cross_validate_model(), evaluate_on_holdout(), compare_models(), final_test_evaluation()
  diagnostics.py   # compute_shap_values(), compute_ks_tests(), compute_residuals(), EBM shape/interaction plots
  tuning.py        # grid_search_cv() — diagnostics-informed hyperparameter search
tests/             # mirrors src/ — one test file per module (115 tests total)
data/              # gitignored except .gitkeep; holds dataset.csv, train.csv, holdout.csv
plots/             # EDA, causal, cluster, and diagnostic visualizations + index.html viewer
  diagnostics/     # ~70 PNGs: SHAP, distribution shift, residuals, QQ, EBM shapes
  eda_round2/      # x10*x11 interaction analysis, x5 sentinel scatter
  round2/          # Round 2 results: comparison charts, residual plots, QQ plots, results.html
submissions/       # gitignored except .gitkeep
```

**Data split:** `data/dataset.csv` is the full dataset. `src/data.split_holdout()` produces `data/train.csv` (1200 rows) and `data/holdout.csv` (300 rows, seed=42). All exploration and model development use `train.csv` only; `holdout.csv` is reserved for final evaluation.

## Workflow

**Clarify before planning, plan before coding.** Always ask clarifying questions and go back and forth with the user until there is full alignment on goals, approach, and scope. Do not write plans or code until clarity is achieved. Explore options together, discuss trade-offs, and confirm the direction before proceeding.

Red-green TDD: write a failing test in `tests/`, implement the minimum in `src/` to pass, then refactor.

## Stack

scikit-learn · LightGBM · HistGradientBoostingRegressor · LinearRegression · pygam (GAMs) · interpret-core (EBM) · SHAP · causal-learn (PC, GES) · lingam (DirectLiNGAM) · networkx · scipy

## Learnings from EDA & Causal Discovery

### Variable roles (from PC + DirectLiNGAM consensus, bootstrap-validated)

| Variable | Role | Notes |
|----------|------|-------|
| **x4** | Primary cause of target | Weight +36.8. Train-only r=0.83 with x9 is a selection-bias artifact (see post-submission diagnostics); x4 and x9 are actually independent. |
| **City** | Strong direct cause | Binary (Zaragoza/Albacete); weight -23.2 on target |
| **x8** | Direct cause | Weight +12.3 |
| **x5** | Direct cause | Weight -8.1; Pearson r≈0 is misleading due to 222 sentinel values (999.0) |
| **x10** | Direct cause | Weight +2.6 |
| **x11** | Direct cause | Weight +2.8 |
| **x9** | Independent predictor (not a descendant of x4) | PC/LiNGAM wrongly inferred x4→x9 from a selection-biased training correlation (r=0.83). Test r(x4,x9)=+0.001. |
| **x1** | Nonlinear predictor | GAM R²=0.109 but linear R²≈0; hump-shaped relationship. Causal methods missed it (they assume linearity) |
| **x2** | Nonlinear predictor | GAM R²=0.068 but linear R²≈0; oscillating/wavy relationship. Same blind spot |
| **x6, x7** | Noise | No linear or nonlinear signal found |
| **Country** | Constant | Always "Spain" — drop it |

### Key gotchas

- **Sentinel values in x5**: 222 rows have x5=999.0. Must replace with median (or NaN + impute) before any analysis. Masquerades as zero-correlation noise if left in.
- **x4/x9 training correlation is selection bias, not causation**: r=0.83 in train, +0.001 in test. x4 and x9 are independent in the true DGP. Residualising x9 on x4 (or otherwise encoding the inferred x4→x9 edge) applies training-only structure and hurts test performance.
- **PC and LiNGAM assume linearity and no selection bias**: They cannot detect nonlinear causal relationships (x1, x2), and they mistake selection-induced associations for causal edges (x4, x9). Nonlinear independence tests (kernel-based, GAM-residual) and test-set validation are needed to complete the picture.
- **GES is too slow**: `local_score_CV_general` doesn't finish on 12 variables; `local_score_BDeu` gives unreliable results on continuous data. Stick with PC + LiNGAM.

### Cluster analysis (City × x4 interaction)

x4 is bimodal with **zero observations** in [-0.167, +0.167]. Combined with City, this creates 4 balanced clusters (~300 rows each in train.csv):

| Cluster | n | Target mean | Target std |
|---------|---|-------------|------------|
| Albacete_high | 296 | +19.6 | 17.5 |
| Albacete_low | 292 | -2.5 | 17.1 |
| Zaragoza_high | 300 | -4.4 | 19.0 |
| Zaragoza_low | 312 | -25.8 | 18.2 |

**Key findings:**

- **Additive, not interactive**: City effect ≈ +23.5 regardless of x4 group; x4 effect ≈ +21.7 regardless of City. F-test for interaction: F=0.44, p=0.51. No interaction term needed.
- **Cluster means alone → R²=0.449**: Nearly half the variance comes from just City + x4.
- **Single global x4 slope ≈ +31.5**: Within-cluster slopes are consistent (CIs overlap heavily). No cluster-dependent slope.
- **Other features (x1, x2, x5, x8, x10, x11) are independent of clusters**: Identical distributions across all 4 groups (Cohen's d < 0.1). They contribute additively to the remaining 55% of variance.
- **x9 Simpson's paradox**: Globally r(x9,target)=+0.35, but within each cluster r(x9,target)≈-0.1. The positive global correlation is entirely inherited from x4 via between-cluster structure. x9's independent contribution is tiny (R² +0.003, slope ≈ -4).

**Modeling implications:**

- Use x4 as **continuous** (not categorical clusters) — keeps within-group gradient.
- Model is cleanly additive: `target ≈ 23.6·City + 31.5·x4 + f(x1) + g(x2) + h(x5,x8,x10,x11) + noise`.
- x1, x2 need **nonlinear** treatment (GAM splines or tree-based).
- No per-cluster models needed — single model with all features is optimal.
- x9 is safe to include alongside x4 (partial effect is orthogonal), but gain is marginal.

### Cluster analysis code

- `src/clusters.py`: `find_x4_gap()`, `assign_clusters()`, `replace_sentinels()`, `cluster_stats()`, `plot_boxplots()`, `plot_scatter_x4_target()`, `plot_distributions()`, `plot_summary()`
- `tests/test_clusters.py`: 15 tests
- `plots/clusters/summary.png`: composite visualization (box plots, densities, scatter)

### Pre-modeling diagnostic analyses (completed)

1. **x5 sentinel missingness** — 178 sentinel rows (14.8% of train). Distribution across clusters is borderline non-uniform (chi2=7.67, p=0.053 — Albacete_low has only 29 vs ~50 in others). However, the sentinel indicator does NOT predict target beyond imputed x5 (coef=-1.15, p=0.51). **Conclusion:** Safe to impute with median; no need for a binary indicator feature.

2. **Residual analysis after City + x4** — Base model R²=0.451. Residuals are near-normal (skew=0.04, kurtosis=-0.44) but Shapiro-Wilk rejects at p=0.008 (expected with n=1200). Levene's test shows mild heteroscedasticity across clusters (p=0.027), suggesting slightly different residual variance per group. **Conclusion:** Additive assumption is valid but a robust/heteroscedastic model may gain marginally.

3. **x1/x2 nonlinear shapes within clusters** — Per-cluster GAM R² values are consistent: x1 ranges [0.178, 0.227], x2 ranges [0.136, 0.219]. Shapes are similar across all 4 clusters. **Conclusion:** A single global GAM shape for x1 and x2 is appropriate; no cluster-specific nonlinear terms needed.

4. **Total R² ceiling** — Full model R²=0.938 (adj=0.936), residual std=6.0. R² breakdown: City+x4: 0.451, +GAM(x1): 0.547, +GAM(x2): 0.626, +x5/x8/x10/x11: 0.938. **Conclusion:** Feature set captures ~94% of variance. Remaining ~6% is irreducible noise (std≈6).

5. **x4 bimodal origin** — Gap at [-0.167, +0.167], width=0.334, gap-ratio=868x median spacing. Zero observations in a 0.334-wide band. Split balanced (604 below, 596 above), identical across cities (KS p=0.61). **Conclusion:** x4 is a designed variable (likely abs(latent) or truncated), not a natural continuous feature.

### Confirmed modeling strategy

| Decision | Evidence |
|----------|----------|
| Single global model (no per-cluster splits) | Additive structure confirmed; no interaction (F=0.44, p=0.51); GAM shapes identical across clusters |
| x4 as continuous | Single slope ≈ +31.5; bimodality is a design artifact |
| City as binary feature | Consistent +23.5 effect |
| Spline/nonlinear treatment for x1, x2 | Combined +17.5% R² over linear |
| Linear treatment for x5, x8, x10, x11 | These 4 features add +31.2% R² |
| Impute x5 sentinels with median | Binary indicator not predictive (p=0.51) |
| Drop x6, x7, Country | No signal; Country constant |
| R² ceiling ≈ 0.938 | Residual std ≈ 6.0; ~6% irreducible noise |
| Mild heteroscedasticity | Levene p=0.027; may benefit from robust regression |

### Causal discovery code

- `src/causal.py`: `preprocess_for_causal()`, `run_pc()`, `run_direct_lingam()`, `run_ges()`, `adjacency_to_edges()`, `consensus_graph()`, `bootstrap_edges()`
- `src/causal_plots.py`: `plot_dag()`, `plot_adjacency_heatmap()`, `plot_edge_bootstrap()`
- `tests/test_causal.py`: 15 tests
- `plots/causal/`: DAGs, heatmaps, bootstrap charts
- `plots/index.html`: self-contained HTML viewer with all EDA + causal + diagnostic results

## Modeling Results (Round 1)

### Evaluation setup

- **Metric**: MAE (Mean Absolute Error)
- **CV**: 5-fold on train.csv (1200 rows), KFold(shuffle=True, seed=42)
- **Holdout**: holdout.csv (300 rows) split into val (150) + test (150), stratified by City
- **12 models**: 7 curated (hypothesis-driven features) + 5 all-variables variants

### Model comparison

| Model | CV MAE | Val MAE | Notes |
|-------|--------|---------|-------|
| **EBM** | **3.28** | **3.22** | Clear winner; auto-discovers interactions |
| Ensemble (EBM+HistGBR+LightGBM) | 4.65 | 4.21 | Dragged down by weaker tree models |
| GAM | 4.40 | 4.80 | Best additive model |
| LightGBM | 5.83 | 5.29 | Conservative defaults, needs tuning |
| HistGBR | 5.88 | 5.47 | Conservative defaults, needs tuning |
| Linear + splines | 5.18 | 5.54 | SplineTransformer for x1/x2 |
| Linear baseline | 9.98 | 9.50 | City+x4+x5+x8+x10+x11 only |

All-variables variants performed equal or slightly worse (x6/x7 confirmed as noise).

### Hypothesis validation

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| x6, x7 are noise | **Confirmed** | All-vars models equal or worse; x6 coef=-0.011, x7=-0.032 |
| x9 residualization correct | **Confirmed** | x9_resid coef=-1.81 (negative, as predicted by Simpson's paradox) |
| x4 dominant cause | **Confirmed** | Curated linear x4 coef=+31.9 (matches causal +36.8); City=-23.2 (exact) |
| Nonlinear >> linear | **Confirmed** | Linear MAE ~10 vs GAM 4.4 vs EBM 3.3 (55-67% reduction) |
| Additive structure | **Partially** | GAM 4.8 vs EBM 3.2 — EBM wins by ~25%, indicating meaningful interactions |

### Modeling code

- `src/features.py`: `CityEncoder`, `SentinelHandler`, `X9Residualizer`, `SplineBasisExpander`, `build_preprocessor()`
- `src/models.py`: `GAMRegressor`, `EBMRegressor`, `AveragingEnsemble`, `build_*()` functions, `build_all_models()`
- `src/evaluate.py`: `split_val_test()`, `cross_validate_model()`, `evaluate_on_holdout()`, `compare_models()`, `final_test_evaluation()`
- `src/data.py`: added `load_train_holdout()`
- `tests/test_features.py`: 21 tests
- `tests/test_evaluate.py`: 10 tests
- `tests/test_models.py`: 21 tests

## Diagnostics & Interpretability

### Distribution shift

No significant covariate shift detected. KS tests for all 8 features: p > 0.05 for both train-vs-val and train-vs-test. Models should generalize.

### EBM shape functions — EDA confirmation

EBM's learned shapes validate our EDA findings precisely:
- **x1**: Inverted-U (hump) — peak at ~0, dropping to -15 at extremes. Matches GAM R²=0.109.
- **x2**: Sinusoidal/oscillating — ~3 full cycles. Matches GAM "wavy" finding.
- **x4**: Roughly linear (+25 slope) with bimodal gap visible near 0.
- **City**: Binary step: Albacete +12.3, Zaragoza -11.8 (total gap ~24, matches -23.2 weight).

### EBM interactions — new discovery

| Interaction | Mean |Score| | Significance |
|-------------|----------------|--------------|
| **x10 & x11** | **1.60** | Strongest by 2.3x — NOT detected in linear EDA |
| x4 & x9 | 0.67 | Expected (x4 causes x9) |
| x1 & x5_is_sentinel | 0.36 | Sentinel indicator carries interaction info |
| x8 & x5_is_sentinel | 0.29 | Same pattern |
| x8 & x9 | 0.27 | Moderate |

The **x10 × x11 interaction** likely explains the 1.6 MAE gap between GAM (purely additive, 4.8) and EBM (with interactions, 3.2). Our F-test only checked City×x4 — it missed this entirely.

### Per-cluster MAE

| Model | Albacete_high | Albacete_low | Zaragoza_high | Zaragoza_low |
|-------|--------------|-------------|---------------|-------------|
| EBM | 3.02 | 2.97 | **4.10** | 2.71 |
| GAM | 4.94 | 4.92 | **5.44** | 3.87 |
| Ensemble | 4.23 | 3.89 | **4.65** | 4.07 |

**Zaragoza_high is hardest** for all models — higher residual variance (std ~19.0 vs ~17 for others).

### QQ plot analysis

- **Linear baseline**: Most Gaussian residuals (range -38 to +26). Clean but large errors from missed nonlinearity.
- **GAM**: Excellent normality in core; slight **left-heavy tail** (3-4 underpredictions at -15 to -17). Possible boundary spline artifacts.
- **EBM**: **Tightest core** of any model, but **heaviest tails** — 3 points below -20, one near +22. Leptokurtic pattern: extremely accurate for most observations, occasional large outliers.
- **LightGBM / HistGBR**: Moderate heavy tails on both sides. Core not as tight as GAM or EBM.

**Key insight**: EBM's weakness is outliers, not bias. Its median prediction is excellent, but ~3-5 large-error observations pull up the MAE.

### SHAP feature importance (all models agree)

Top ranking: **City > x4 > x5 > x10 ≈ x11 > x1 > x8 > x2 > x9_resid ≈ 0**. Consistent across linear, GAM, tree models.

### Diagnostics code

- `src/diagnostics.py`: `compute_ks_tests()`, `compute_residuals()`, `compute_cluster_mae()`, `compute_shap_values()`, `plot_*()` functions, `update_index_html()`
- `tests/test_diagnostics.py`: 6 tests
- `plots/diagnostics/`: ~70 PNGs (SHAP, distribution, residual, EBM shapes)

## Modeling Results (Round 2)

### x10 × x11 Interaction EDA

EBM's Round 1 diagnostics flagged x10 & x11 as the strongest pairwise interaction (mean |score| = 1.60). Round 2 EDA confirmed:

- **Pure multiplicative**: `x10*x11` is the best functional form (R²=0.258 of residual variance after main effects). Alternatives (quadratic, abs-diff, sum) are all weaker.
- **F-test**: F=56.3, p<1e-8 in full linear model. Coefficient = +0.88.
- **Heatmap**: Clean diagonal gradient from low-x10/low-x11 (-8 residual) to high-x10/high-x11 (+18 residual).

### Hyperparameter tuning results

| Model | Best Params | CV MAE | vs Round 1 |
|-------|-------------|--------|------------|
| **EBM** | min_samples_leaf=10, max_bins=128, max_rounds=2000 | **3.16** | 3.28→3.16 (-3.7%) |
| **LightGBM** | n_estimators=2000, lr=0.1, depth=4, min_child=40 | **4.66** | 5.83→4.66 (-20%) |
| **GAM** | lam=2.0, n_splines=20 | **4.39** | 4.40→4.39 (marginal) |

**Key finding**: EBM's biggest lever was `max_bins=128` (down from default ~256). Fewer bins = stronger regularization = fewer heavy-tailed outliers. `max_rounds` had no effect beyond 2000.

### Round 2 model comparison

| Model | CV MAE | Val MAE | Notes |
|-------|--------|---------|-------|
| **EBM+GAM (70/30)** | **2.99** | **2.94** | Best overall; breaks sub-3.0 MAE |
| EBM+GAM (50/50) | 3.02 | 3.02 | Also strong |
| R1 EBM (default) | 3.33 | 3.19 | Round 1 winner |
| EBM tuned | 3.16 | 3.19 | Better CV, same val |
| GAM + x10*x11 | 3.56 | 3.74 | Massive improvement over base GAM |
| LightGBM tuned | 4.66 | 4.14 | 20% improvement from tuning |
| Linear+splines+x10*x11 | 4.56 | 4.53 | Interaction helps linear too |
| GAM tuned | 4.39 | 4.77 | Near-optimal already |
| Huber+splines | 5.19 | 5.54 | No improvement over OLS |

### What worked and what didn't

| Approach | Verdict | Impact |
|----------|---------|--------|
| **EBM+GAM ensemble** | **Best approach** | Val 3.19→2.94 (-7.8%). Complementary error profiles. |
| **x10*x11 interaction** | **Major discovery** | GAM 4.40→3.56. Closed 48% of GAM-EBM gap. |
| EBM max_bins tuning | Helpful | 3.28→3.16 CV. Fewer bins tames outliers. |
| LightGBM tuning | Helpful | 5.83→4.66. Shallow trees + high regularization. |
| GAM regularization | Marginal | Already near-optimal. |
| Quantile regression | **Failed** | 6.96 — L1 loss + splines = optimization issues. |
| Huber regression | **No effect** | 5.19 vs 5.18 — outliers aren't leverage-type. |
| Random Forest | **Poor** | ~8.8 MAE. Can't capture smooth x1/x2 nonlinearity. |

### Per-cluster MAE (best model: EBM+GAM 70/30)

| Cluster | R1 EBM | R2 EBM+GAM | Improvement |
|---------|--------|------------|-------------|
| Albacete_high | 2.98 | 3.05 | -2.3% |
| Albacete_low | 3.02 | 2.50 | +17.2% |
| Zaragoza_high | 3.97 | 3.45 | +13.1% |
| Zaragoza_low | 2.69 | 2.80 | -4.1% |

Zaragoza_high remains hardest but improved significantly. The ensemble especially helps on the two clusters where EBM and GAM have different weaknesses.

### Stacked ensemble (Ridge meta-learner)

Ridge regression on out-of-fold (OOF) predictions learns optimal weights from data:

| Model | CV MAE | Val MAE | Learned Weights |
|-------|--------|---------|-----------------|
| Stacked (EBM+GAM, Ridge) | 3.05 | 2.94 | 0.608 EBM + 0.404 GAM |
| Stacked (EBM+GAM+LGBM, Ridge) | 3.05 | 2.96 | 0.634 EBM + 0.410 GAM - 0.035 LGBM |
| Weighted (70/30 manual) | 2.99 | 2.94 | 0.7 EBM + 0.3 GAM (fixed) |

**Key finding**: Ridge learned weights (~60/40) are close to our manual 70/30. LightGBM receives near-zero weight — it doesn't contribute. Alpha has no effect (problem is too simple for regularization to matter).

### Final test evaluation (unbiased)

Models retrained on train+val (1350 rows), evaluated on held-out test (150 rows, **never used during model selection**):

| Model | Test MAE | vs R1 EBM |
|-------|----------|-----------|
| **R2: Stacked (EBM+GAM, Ridge)** | **2.52** | **-10.4%** |
| **R2: Weighted EBM+GAM (70/30)** | **2.53** | **-10.1%** |
| R2: EBM tuned | 2.80 | -0.5% |
| R1: EBM (default) | 2.81 | baseline |
| R2: GAM + x10*x11 | 3.18 | +13.0% |

Test MAE (2.52) is better than val MAE (2.94), suggesting the model generalizes well and val was a conservative estimate.

### Round 2 code

- `src/tuning.py`: `grid_search_cv()` — diagnostics-informed hyperparameter search
- `src/features.py`: added `InteractionAdder`, new preprocessor flavors (`linear_interact`, `linear_spline_interact`)
- `src/models.py`: added `StackedEnsemble`, `WeightedEnsemble`, `build_ebm_tuned()`, `build_gam_tuned()`, `build_gam_interact()`, `build_huber_nonlinear()`, `build_quantile_nonlinear()`, `build_linear_nonlinear_interact()`, `build_lgbm_tuned()`, `build_histgbr_tuned()`, `build_rf()`, `build_ensemble_ebm_gam()`, `build_ensemble_ebm_gam_weighted()`, `build_stacked_ensemble()`
- `tests/test_tuning.py`: 4 tests
- `tests/test_round2_models.py`: 31 tests
- `plots/round2/results.html`: self-contained HTML with all Round 2 results
- `plots/round2/`: comparison charts, residual plots, QQ plots, cluster MAE
- `plots/eda_round2/`: x10*x11 interaction analysis

## Error Analysis & Noise Floor

### x5 sentinels are the dominant error source

All 15 worst ensemble predictions are x5 sentinel observations (x5=999.0):

| Group | n (val) | MAE | Median AE |
|-------|---------|-----|-----------|
| Non-sentinel | 123 (82%) | **1.53** | 1.26 |
| Sentinel | 27 (18%) | **9.39** | 7.62 |

Correlation of `|residual|` with `x5_is_sentinel`: **r=0.70** — no other feature exceeds r=0.12. The worst 15 observations account for **49% of total error**.

### Why sentinels are irreducible

x5 has slope -8.0 on target (strong, consistent across all clusters). When x5=999.0, the true value is missing, introducing ~8 units of random prediction error. Tested alternatives:

| Strategy | Overall MAE | Non-sentinel | Sentinel |
|----------|-------------|--------------|----------|
| **Current (NaN + indicator)** | **3.16** | **1.80** | 9.56 |
| NaN, no indicator | 3.26 | 1.92 | 9.45 |
| Drop x5, keep indicator | 10.54 | 10.71 | 8.38 |
| Drop x5 entirely | 10.54 | 10.73 | 8.18 |
| Forced sentinel interactions in EBM | 4.31 | 3.33 | 8.79 |

Dropping x5 destroys non-sentinel predictions (1.80→10.71). Forcing sentinel interactions trades worse non-sentinel for modest sentinel gain — net negative. The current approach is already optimal.

### No Simpson's paradox for x5

Unlike x9, x5 has no confounding or sign-flip:

- Global slope: -7.99 | Within-cluster slopes: -7.0 to -9.2 (same sign, strong)
- Partial r *increases* after controlling City+x4 (-0.48 → -0.65, no sign flip)
- x5 is uncorrelated with x4 (r=0.017) and City (r=0.013)
- Sentinel missingness is MCAR — no feature predicts it (all |r| < 0.06)

### Noise floor

- **Non-sentinel MAE ≈ 1.5**: True model quality for 82% of the data
- **Sentinel MAE ≈ 9.5**: Data limitation, not modeling gap
- Residual kurtosis = 7.38: Bimodal error distribution (tight core + sentinel outliers)
- EBM and GAM residuals correlate at r=0.79: both fail on the same sentinel observations

**Conclusion**: The ensemble is near-optimal. Further MAE improvement requires better x5 data, not better models.

## Next steps (TODO)

1. **Submission generation**: Train on full train.csv, predict on test set, generate Kaggle submission CSV.
2. **x4 bimodal origin**: Why zero observations near x4=0? Still unexplored.

## Round 3 — Reverse-engineered formulas vs Kaggle leaderboard

### Three reverse-engineered formulas compared

Two sibling branches had attempted to reverse-engineer the data-generating
process. We added our Round 2 ensemble as a third reference point and
compared all three on the competition test set.

| Approach | Source branch | x4 treatment | x1 / x2 basis | x10·x11 |
|---|---|---|---|---|
| **A1** closed form | `reverse-engineer-equation` | `15·x4 + 20·𝟙(x4>0)` (hard step at 0) | `−100·x1²`, `10·cos(5π·x2)` | `+1·x10·x11` |
| **A2** ClosedFormModel | `review-backwards-engineering` | `+30.5·x4` (pure linear) | `+25·cos(π·x1)`, `+10·cos(5π·x2)` | `+1·x10·x11` |
| **A3** EBM Round 2 | this branch | nonparametric (plateau in gap) | EBM shape functions | EBM interaction |

### x4 disambiguation: 508 test rows fall in the training gap

Training data has zero observations in x4 ∈ [−0.167, +0.167] (designed bimodal
gap). The competition test set has **508 rows (33.9%)** inside that gap — a
natural experiment for distinguishing A1's step from A2/A3's smooth treatments.
Pairwise prediction differences inside the gap are 2–4× larger than outside
(see `plots/formulas/preds_vs_x4.png`, `x4_marginal_curves.png`).

### x1 piecewise-linear hypothesis: rejected

Tested whether x1 might also have a step at 0 (analogous to x4). Result: no.
- `x1²` and `cos(π·x1)` tied at MAE ≈ 3.52 on the x1-residual.
- Piecewise-linear with step at 0 was worse (MAE ≈ 3.87).
- Fitted step coefficient was only +1.08 (vs +20 for x4) — effectively zero.
- x1 has no distribution gap (every histogram bin ≥ 22 observations), so any
  real step would be detectable.

x1 is smoothly symmetric around 0; the cos / quadratic forms are
indistinguishable on the observed range.

### 5-fold CV on dataset.csv (1500 rows)

| Rank | Model | CV MAE | Non-sent MAE | Sent MAE |
|---|---|---|---|---|
| 1 | **A1 closed form alone** | **1.80** | **0.38** | 10.00 |
| 2 | NNLS EBM+GAM+A1 | 2.02 | 0.64 | 9.99 |
| 3 | Stacked EBM+GAM+A1 (Ridge) | 2.04 | 0.66 | 10.02 |
| 4 | Stacked EBM+GAM+A2 | 2.85 | 1.58 | 10.19 |
| 5 | EBM+GAM 70/30 (Round 2 best) | 2.91 | 1.65 | 10.19 |
| 6 | EBM (R2 tuned, alone) | 3.11 | 1.84 | 10.39 |
| 7 | A2 ClosedFormModel (alone) | 3.49 | 2.30 | 10.28 |
| 8 | GAM (R2 tuned, alone) | 3.52 | 2.37 | 10.15 |

A1 alone won CV by ~38% over the previous Round 2 ensemble. NNLS allocated
0.83 weight to A1 and only 0.10/0.07 to EBM/GAM. The training data was almost
perfectly explained by A1's hand-engineered formula.

### x5 imputation study: cannot be improved beyond mean/median

A1 is exact enough to back-solve the true x5 for 222 sentinel training rows
from the observed target: x5_true = (rest_of_A1 − target) / 8. Validated on
non-sentinels: 93.3% match within 0.01 (the remaining 6.7% are the known
x4<0, x8<0 edge case where A1 clamps x8). Back-solved sentinel x5 distribution
is Uniform(7, 12), identical to observed non-sentinel x5.

| Strategy | MAE on back-solved x5 |
|---|---|
| Mean (9.40) | **1.248** |
| Median (9.34) | 1.251 |
| Linear regression | 1.254 |
| kNN k=100 | 1.256 |
| LightGBM | 1.309 |
| kNN k=5 | 1.349 |
| Nearest-id neighbour | 1.431 |

All Pearson r(x5, feature) < 0.13. Linear regression R² = 0.006. x5 is
genuinely independent random noise — the theoretical bound for imputing
Uniform(7, 12) with a constant is (12−7)/4 = **1.25**, exactly matching
mean/median results. **The 8 × 1.25 = 10.0 sentinel-row MAE floor is
mathematically irreducible.**

### Kaggle public-leaderboard reality check

CV optimism vs reality — the formulas DID NOT generalise:

| Model | CV MAE | Public LB MAE | Train→Test degradation |
|---|---|---|---|
| **EBM alone** | 3.11 | **5.66** | 2.7× |
| EBM+GAM 70/30 | 2.91 | 6.47 | 3.5× |
| A2 ClosedFormModel | 3.49 | 9.44 | 4.0× |
| A1 closed form | 1.80 | **10.80** | **29×** |

For comparison, the leaderboard top 4 cluster at 1.65–1.71 — within 0.2 of
the theoretical sentinel floor (1.52 = 228·10/1500), implying they recovered
the true DGP with ~0.15 non-sentinel MAE.

### Conclusions

1. **The reverse-engineered formulas (A1, A2) memorised training data.** A1's
   "exact" CV fit was an artifact: the cos / quadratic basis + x4 step
   happened to interpolate the training distribution but doesn't extrapolate.
2. **The x4 step at 0 is wrong.** Training data has no observations in the
   gap, so the +20 step couldn't be falsified during reverse-engineering.
   Test data does cover the gap and exposed it.
3. **Hand-engineered cos basis is also wrong.** Both A1 and A2 use it; both
   fail. Only EBM (no functional assumptions) survives the shift.
4. **GAM hurts EBM in the test regime.** Its rigid linear backbone for
   x4/x5/x8 contributes miscalibrated signal. Pure EBM beats the ensemble.
5. **Less inductive bias = better generalisation here.** EBM (5.66) >
   EBM+GAM (6.47) > A2 (9.44) > A1 (10.80). Direct ranking by formula rigidity.
6. **Sentinel noise floor confirmed.** 1.52 MAE is the theoretical floor;
   nobody on the leaderboard beats it. Top 4 are within 0.2 of it.
7. **Final position: ~43rd / public LB.** Honest CV-validated performance
   without DGP discovery.

### Round 3 code

- `scripts/compare_formulas.py` — predict from A1/A2/A3 on test.csv + visualise
- `scripts/test_x1_shape.py` — fit cos / quadratic / piecewise-linear x1 candidates
- `scripts/cv_ensemble_eval.py` — 5-fold OOF CV across all individual models and ensembles
- `scripts/cv_sentinel_breakdown.py` — same CV split by sentinel status
- `scripts/x5_imputation_study.py` — benchmark imputation strategies vs back-solved truth
- `scripts/build_final_submissions.py` — write Kaggle submissions trained on full data
- `submissions/submission_*.csv` — six committed Kaggle submissions
- `plots/formulas/` — 12 PNGs (x4 marginals, x1 candidates, CV bars, sentinel breakdown,
  x5 imputation, prediction differences, gap distribution)

## Post-submission diagnostics

### x4 and x9 are independent — training correlation is selection bias

The 5-fold CV and the Kaggle leaderboard diverged severely. A pairwise
correlation check on `dataset.csv` vs `test.csv` exposed the cause:

| Pair | Train r | Test r |
|---|---|---|
| x4 vs x9 | **+0.832** | **+0.001** |
| every other pair | < 0.05 | < 0.06 |

**x4 and x9 are actually independent in the true DGP.** The +0.83 training
correlation is a **relic of selection bias in the training sample**, not a
causal edge. Our earlier PC/LiNGAM consensus inferred an x4→x9 edge because
those methods cannot distinguish causation from selection-induced
association; test data (drawn without the same selection) reveals the
features are uncorrelated.

Every round of modelling downstream of that mistaken edge — Simpson's-paradox
framing for x9, `x9_resid`, "x4/x9 collinearity" warnings, the EBM x4 & x9
interaction of 0.67 — was modelling training-specific structure that the
test set doesn't share. This explains why A1 (raw x9 coef −4) and A2
(`x9_resid` coef −2.4) both failed hard, and why pure EBM (which learns x9's
partial effect from data rather than leaning on residualisation) generalised
better.

### x6 / x7 live on a circle of radius 18

`sqrt(x6² + x7²) = 18.000` exactly, with std 0 across all 1500 rows in both
train and test. Only the angle `θ = atan2(x7, x6)` carries information.
`θ` is uniform on [−π, π] and **independent of every other feature**,
including x5 and the x5 sentinel indicator:

- `r(θ, x5) = +0.012` on observed-x5 rows
- Same uniform angle distribution within the 222 sentinel rows

Matches the earlier `reverse-engineer-equation` finding that adding θ as a
feature hurt EBM CV (3.47 → 3.56). x6 and x7 are noise features dressed up
as a circle.

### x5 sentinel ratio is preserved

| | n | sentinels (x5=999) | % |
|---|---|---|---|
| train (`dataset.csv`) | 1500 | 222 | 14.8 |
| test (`test.csv`) | 1500 | 228 | 15.2 |

Essentially identical. The public-LB noise floor for *any* model is
0.152 · 10 = **1.52 MAE**, and leaderboard top 4 at 1.65–1.71 are within
0.2 of it.

### Post-submission code

- `scripts/plot_x6x7_angle_vs_x5.py` — angle-vs-x5 scatter
- `scripts/plot_test_pairwise.py` — 10×10 scatter matrix + Pearson heatmap (test.csv)
- `plots/formulas/x6x7_angle_vs_x5.png`
- `plots/formulas/test_pairwise_scatter.png`
- `plots/formulas/test_correlation_heatmap.png`

## Post-submission modelling experiments

### Simple linear model (A2 minus x9_resid)

Trimmed A2's basis to avoid the x4→x9 shift: `x1² + cos(5π·x2) + x4 +
x5_imp + x5_is_sent + x8 + x10 + x11 + x10·x11 + City`. Fitted coefficients
rediscover A2 (x4=+30.4, x5=−8.0, x8=+14.1, city=−24.8, x10·x11=+0.95).

| Variant | CV MAE | Public LB |
|---|---|---|
| without x10·x11 | 4.50 | — |
| **with x10·x11** | **3.70** | **7.38** |

Adding the x10·x11 interaction is worth 0.80 CV MAE (−18%). **Dropping
`x9_resid` cut A2's test MAE from 9.44 → 7.38 (−22%)** — strong evidence
that the train→test x4-x9 correlation shift was hurting A2's generalisation.

### EBM hyperparameter grid (CV)

| Variant | CV MAE | Non-sent | Sent | Notes |
|---|---|---|---|---|
| **heavy_smooth** | **3.08** | 1.79 | 10.50 | smoothing_rounds=2000, interaction_smoothing=500 |
| high_reg | 3.09 | 1.81 | 10.46 | reg_alpha=1, reg_lambda=1, min_leaf=30 |
| baseline (Round 2 tuned) | 3.11 | 1.84 | 10.39 | interactions=10, max_bins=128 |
| fewer_inter | 3.11 | 1.84 | 10.40 | |
| more_inter | 3.13 | 1.88 | 10.30 | |
| default | 3.24 | 1.99 | 10.43 | all EBM defaults |
| no_x9 | **3.83** | 2.67 | 10.53 | drop x9 from features |
| heavy_smooth_no_x9 | 3.94 | 2.77 | 10.65 | |

**Dropping x9 hurts EBM CV by +0.72 MAE** because x9 is legitimately
informative in training (r=0.83 with x4). CV cannot tell us whether x9 helps
or hurts *test* performance — only the leaderboard can. The linear model's
LB drop from 9.44 → 7.38 suggests x9 does hurt test; whether EBM extracts
cleaner partial effects that survive the shift is an open question.

Heavy smoothing gives a marginal 0.03 CV gain over baseline — negligible,
probably within noise.

### kNN is not competitive

20 configurations (k ∈ {5, 15, 30, 50, 100}, uniform/distance-weighted,
all features / no x9). Best is **k=5, distance-weighted, all features:
CV MAE 9.33** — worse than every linear variant and every EBM variant.
Local similarity cannot recover the cos(5π·x2) / x1² structure.

### Experiment code

- `scripts/cv_simple_linear.py` — trimmed linear model, 5-fold CV
- `scripts/build_simple_linear_submission.py` — writes submission_simple_linear_interact.csv
- `scripts/cv_ebm_variants.py` — 8-variant EBM grid
- `scripts/cv_knn.py` — 20-variant kNN grid
- `scripts/build_ebm_variant_submissions.py` — writes three EBM submissions

### Kaggle leaderboard tracker

| Submission | CV MAE | Public LB | Place |
|---|---|---|---|
| EBM alone (R2 tuned) | 3.11 | **5.66** | **~43** |
| EBM+GAM 70/30 | 2.91 | 6.47 | — |
| simple_linear_interact (no x9) | 3.70 | 7.38 | — |
| A2 ClosedFormModel | 3.49 | 9.44 | — |
| A1 closed form | 1.80 | 10.80 | — |

Top-of-leaderboard cluster sits at 1.65–1.71 (theoretical floor 1.52).

## Interaction search — no further pairs found

Concern: were we missing interactions beyond x10·x11? Ran a full
per-pair search over all 11 features (10 numeric + City) on dataset.csv.

For each pair (xi, xj):
1. Fit a simple additive baseline (cubic splines on x1, x2, x4, x8;
   linear x5 + sentinel indicator + x9 + x10 + x11; one-hot City).
2. Bin the residuals on a 12×12 grid per pair.
3. Double-centre the grid (subtract row and column means) so any
   remaining signal is pure interaction.
4. Score each pair by RMS of the double-centred grid.

Additive baseline R² ≈ 0.950 (residual std 5.40 vs target std 24.10).

Top 5 pairs by pure-interaction RMS:

| xi | xj | RMS |
|---|---|---|
| **x11** | **x10** | **3.02** |
| x10 | x8 | 1.71 |
| x8 | x7 | 1.64 |
| x9 | x8 | 1.63 |
| x11 | x2 | 1.63 |

**Noise floor calculation**: residual std / √(cells-per-bin) ≈ 5.40 / √10
≈ **1.71**. Every pair except x10·x11 sits at or below this floor.
Conclusion: **no additional interactions detectable** beyond x10·x11.
The ~6% irreducible-noise ceiling from earlier diagnostics is consistent
with this — there's no hidden pairwise structure to exploit.

### Interaction-search code

- `scripts/plot_target_pairwise_heatmaps.py` — per-pair residual
  heatmaps + double-centred RMS ranking
- `plots/interactions/target_pairwise_raw.png` — mean(target) per (xi, xj) bin
- `plots/interactions/target_pairwise_residual.png` — mean(residual) per (xi, xj) bin (interactions visible)
- `plots/interactions/interaction_ranking.csv` — ranked scores for all 55 pairs

## Reweighting attempt — cannot break the x4-x9 shift

Hypothesis: if x4 ⊥ x9 in the true DGP, reweight training rows by
`w = p(x4)·p(x9) / p(x4, x9)` so the training joint matches the test
joint, then refit candidate models with `sample_weight=w`.

**Finding: it cannot work here.** The training joint (x4, x9) is two
disjoint clusters — x4>0 rows have x9 ~ N(5.97, 0.57); x4<0 rows have
x9 ~ N(4.02, 0.57). Clusters overlap by less than 1σ in x9.
Within-cluster r(x4, x9) ≈ 0; the +0.83 correlation is entirely
between-cluster. Test (where x4 ⊥ x9) contains many rows in the
off-diagonal quadrants — (x4>0, x9<5) and (x4<0, x9>5) — that
**literally do not exist in training**. Reweighting cannot fabricate
missing rows; it can only redistribute mass across observed ones.
`plots/reweight/x4_x9_joint_train_vs_test.png` visualises the gap.

Three DRE estimators tried — all fail to decorrelate:

| Method | Weighted corr(x4, x9) | Notes |
|---|---|---|
| unweighted | +0.832 | baseline |
| HistGBM classifier DRE | +0.814 | shallow: can't separate joint from shuffled well |
| KDE ratio (Silverman bw) | +0.800 | oversmooths on 2-cluster joint |
| **Gaussian-copula analytical** | **+0.736** | best but still far from 0 |

CV on downstream models confirms — reweighting hurts slightly, never helps:

| Model | Unweighted CV MAE | Weighted CV MAE | Δ |
|---|---|---|---|
| linear (no x9) + x10·x11 | 3.70 | 3.71 | +0.02 |
| linear (with x9) + x10·x11 | 3.48 | 3.51 | +0.03 |
| EBM baseline | 3.11 | 3.19 | +0.08 |

Under a *weighted* validation metric (a test-distribution proxy), linear
models gain marginally (−0.10 for no_x9, −0.11 for with_x9) but EBM
loses (+0.12). Net: **no robust LB improvement expected** from
reweighting. Submissions were still built for reference
(`submission_*_reweighted*.csv`) but are not recommended.

**Remaining options** for the x4-x9 shift:

1. **Drop x9 entirely** — the only robust fix. `submission_ebm_no_x9.csv`
   already exists (Round 3); the honest LB test hasn't been sent yet.
   Expected CV hit ≈ +0.7 MAE vs EBM with x9, but should not degrade
   on LB as sharply as models that lean on the spurious edge.
2. **Constrained β_x9** in parametric models — pin x9's coefficient to
   its within-cluster value (~0, or very small negative) rather than
   the between-cluster-inflated slope. Not directly available in EBM.
3. **Live with the shift** — EBM alone at 5.66 LB is our best submission;
   it uses x9 but learns partial effects less aggressively than GAM
   or the hand-engineered formulas, which is why it generalises better
   than the ensembles.

### Reweighting code

- `scripts/reweight_x4x9.py` — three DRE estimators (classifier, KDE,
  Gaussian copula) + 5-fold CV for linear and EBM + submission builder
- `plots/reweight/weights_overview.png` — scatter of training rows
  coloured by weight, + weight distribution histogram
- `plots/reweight/x4_x9_joint_train_vs_test.png` — side-by-side scatter
  showing the empty training quadrants
- `plots/reweight/weights.csv` — per-row weights

## Simpson-corrected x9 — breakthrough CV MAE

Key insight: instead of dropping x9 or residualising against x4 (both of
which encode training-specific structure or throw away the true signal),
**center x9 per x4-cluster**:

    x9_wc = x9 − E[x9 | sign(x4)]

Cluster means from training: E[x9 | x4>0] = 5.971; E[x9 | x4<0] = 4.016.
By construction, x9_wc is uncorrelated with sign(x4) in training, so a
linear fit recovers only the within-cluster (Simpson's-paradox-true)
slope. Reverse-engineered A2's closed form plus x9_wc gives the cleanest
model we have:

| Model | CV MAE | Non-sent | x9 treatment |
|---|---|---|---|
| **linear parametric + x9_wc** | **2.934** | **1.684** | within-cluster center |
| **GAM + x9_wc** | **3.002** | 1.784 | within-cluster center |
| GAM parametric x1/x2 + x9_wc | 3.365 | 2.170 | within-cluster center |
| linear parametric + raw x9 | 3.477 | 2.296 | raw (β_x9 = −2.41, contaminated) |
| GAM + raw x9 | 3.519 | 2.367 | raw |
| GAM + x10·x11 (no x9) | 3.734 | 2.609 | dropped |
| linear parametric (no x9) — prev LB 7.38 | 3.695 | 2.537 | dropped |
| EBM alone (R2 tuned) — LB 5.66 | 3.11 | 1.84 | raw in EBM |
| EBM+GAM 70/30 — LB 6.47 | 2.91 | 1.65 | raw + residualised |

The enhanced linear model beats EBM on CV and ties the EBM+GAM ensemble
(−5% vs simple-linear-no-x9 baseline, −16% vs raw-x9 variant). β_x9_wc
= **−4.28**, matching the predicted within-cluster slope ≈ −4 almost
exactly.

### Coefficients vs reverse-engineered A2

| Feature | Learned on full dataset | A1/A2 closed form |
|---|---|---|
| x1² | −101.54 | −100 (A1) |
| cos(5π·x2) | +9.99 | +10 (A1) |
| x4 | +30.47 | +30.5 (A2) |
| x5_imp | −8.02 | −8 |
| x5_is_sent | −1.19 | (intercept offset) |
| x8 | +14.07 | +14.1 |
| x10 | +0.11 | +3 (small — dominated by x10·x11) |
| x11 | +0.05 | +2.5 |
| x10·x11 | +0.97 | +1 |
| city (Zaragoza=1) | −24.99 | −24.8 |
| **x9_wc** | **−4.28** | (A2 used x9_resid with coef −2.4 — contaminated) |

The model recovers A2's hand-engineered coefficients and replaces its one
broken piece (x9_resid) with the Simpson-corrected x9_wc. Since the +0.83
train correlation is removed by construction, the x9 contribution should
now survive the test-distribution shift.

### Why this may finally generalise

A2 failed on LB (9.44) because its x9_resid term relied on x4-x9 coupling
that doesn't exist in test. x9_wc sidesteps this: it is orthogonal to
sign(x4) by construction, so its coefficient reflects only the within-
cluster partial effect, which is invariant under train→test. The model
algebraically equals:

    prediction = β_x1·x1² + β_x2·cos(5π·x2) + β_x4·x4
               + β_x5·x5_imp + β_sent·x5_is_sent + β_x8·x8
               + β_x10·x10 + β_x11·x11 + β_x10x11·x10·x11
               + β_city·city + β_x9·x9 + β_cluster·sign(x4)

(the sign(x4) dummy is absorbed into the x4 coefficient, but mathematically
it is there). This is identical to A2's form with a clean β_x9 = −4.28
and a cluster-intercept correction that A2 lacked.

### Enhanced-GAM code

- `scripts/cv_gam_enhanced.py` — 8 variants (linear vs GAM × {square/spline}
  × {cos/spline} × {none/raw/wc} × {with/without x10·x11}), CV + submission
- `plots/gam_enhanced/cv_results.csv` — all variant scores
- `submissions/submission_linear_enhanced_x9wc.csv` — CV 2.934, free fit
- `submissions/submission_gam_enhanced_x9wc.csv` — CV 3.002

## Integer-coefficient locking — further CV gain

Testing whether the true DGP uses integer coefficients by fixing them
and fitting only the intercept.

| Variant | CV MAE | Non-sent | Free params |
|---|---|---|---|
| **C. learned-rounded (all locked)** | **2.897** | 1.664 | 1 |
| B. A1/A2 declared integers (all locked) | 2.900 | 1.661 | 1 |
| F. A1/A2 integers, x4=+31 variant | 2.904 | **1.653** | 4 |
| D. partial-lock (obvious integers only) | 2.910 | 1.662 | 6 |
| A. free fit baseline (all 11 coefs) | 2.934 | 1.684 | 11 |

**Rounding to integers improves CV by 0.03–0.04** even though the free fit
has 11× more parameters to absorb noise. The locked model uses just the
intercept + residual x5 sentinel offset, confirming the DGP coefficients
are genuinely integer. x1²=−100 (A1's value) vs −102 (learned-rounded)
are indistinguishable on CV (both give 2.900); x4=+30 vs +31 are also
within noise.

### Locked-coefficient submissions

- `submission_linear_enh_locked_c.csv` — CV 2.897, all coefs locked to
  learned-rounded (x1²=−102, cos(5π·x2)=+10, x4=+30, x5=−8, x8=+14,
  x10·x11=+1, city=−25, x9_wc=−4, x10=x11=0, sentinel=−1)
- `submission_linear_enh_locked_b.csv` — CV 2.900, A1/A2 declared integers
  (x1²=−100, everything else same as C)
- `submission_linear_enh_locked_f.csv` — CV 2.904, x4=+31 variant

### Rounded-coefficient code

- `scripts/cv_rounded_coefs.py` — seven lock configurations, CV + submissions
- `plots/gam_enhanced/cv_rounded_coefs.csv` — scores per config

## LB verdict on x9_wc — refuted

The x9_wc linear model **failed catastrophically on the leaderboard**:

| Submission | CV MAE | LB MAE |
|---|---|---|
| EBM alone (R2 tuned, with x9) | 3.11 | **5.66** ← still best |
| EBM heavy_smooth, no x9 | — | 7.57 |
| simple_linear_interact (no x9) | 3.70 | 7.38 |
| A2 (x9_resid) | 3.49 | 9.44 |
| **locked_b (x9_wc, β=−4)** | **2.900** | **10.75** |
| A1 (raw x9 + step) | 1.80 | 10.80 |

**Diagnosis** — the within-cluster Simpson slope β_x9=−4 is a *training-only*
artifact. On test, x4 ⊥ x9 creates off-diagonal rows (48.9% of test) that
don't exist in training. Our linear β=−4 extrapolates to x9_wc values up
to ±2, adding std=8.3 of systematic error for those rows — enough to
shift EBM's 5.66 baseline to ~10.7. EBM survives the same shift because
its bounded shape function cannot extrapolate; linear/parametric
coefficients have no such protection.

**The pattern** — x9 carries ~2 MAE of real signal that EBM extracts
nonparametrically but **no parametric form captures safely**:

- drop x9 entirely → costs 2 MAE (EBM: 5.66 → 7.57; linear: 7.38)
- parametric x9 (A1/A2/locked_b) → costs 3–5 MAE via extrapolation
- nonparametric x9 (EBM) → wins

All x9_wc-based submissions were expected to land ~10–11 on LB and
have been removed.

## Smoothing sweep — heavy smoothing is the big lever

`submission_ebm_heavy_smooth.csv` (smoothing_rounds=2000,
interaction_smoothing_rounds=500) **scored LB 4.9**, a 0.76 MAE
improvement over the plain EBM's 5.66 despite a CV gain of only 0.03
(3.11 → 3.08). The CV→LB gain multiplier was ~25×, telling us smoothing
directly regularises training-specific fit that doesn't transfer.

Subsequent CV-validated sweep on top of the 2k/500 baseline:

| Variant | Smoothing | max_rounds | Other | CV MAE |
|---|---|---|---|---|
| **ebm_tune_max_rounds_4k** | **4k / 1k** | **4000** | defaults | **3.030** |
| ebm_combined_B | 4k / 1k | 4000 | leaf_5 | 3.033 |
| ebm_tune_leaf_5 | 4k / 1k | 2000 | leaf_5 | 3.034 |
| bagged_5seeds_max_rounds_4k | 4k / 1k | 4000 | 5-seed bag | 3.035 |
| extra_smooth_4k (earlier) | 4k / 1k | 2000 | defaults | 3.053 |
| extra_smooth_6k | 6k / 2k | 2000 | defaults | 3.061 |
| heavy_smooth_ref (LB 4.9) | 2k / 500 | 2000 | defaults | 3.081 |
| bins64 variants | 2k–4k | 2000 | max_bins=64 | 3.12–3.14 (worse) |
| lr_slow (0.005), inter5, inter20 | — | — | — | 3.09–3.30 (worse) |

**Findings**:

- **4k smoothing dominates 2k** (CV 3.053 vs 3.081). 6k smoothing shows diminishing returns.
- **max_rounds=4000 + 4k smoothing** is the best single-lever combination (CV 3.030 vs 3.053, a further −0.023 CV).
- **Bagging 5 seeds gives ~0** beyond single-seed (3.035 vs 3.030) → the model is bias-limited, not variance-limited.
- **Combining leaf_5, inter15, larger smooth doesn't compound** — max_rounds=4000 alone is optimal.
- **max_bins=64 hurts** — coarser bins cost 0.05+ CV MAE.

If the 25× multiplier observed on the first smoothing jump partially
holds for subsequent CV gains, the 3.030 variant could plausibly land
4.3–4.7 on LB. More likely: diminishing returns, so expect 4.7–4.9.

### Remaining candidates in `submissions/` (4 files)

- `submission_ebm.csv` — LB 5.66 (reference baseline)
- `submission_ebm_heavy_smooth.csv` — **LB 4.9** (current best, confirmed)
- `submission_ebm_tune_max_rounds_4k.csv` — CV 3.030, **next submission candidate**
- `submission_ebm_combined_B.csv` — CV 3.033, alternative (adds leaf_5 on top of max_rounds)

### Smoothing + tuning code

- `scripts/cv_ebm_extra_smooth.py` — 2k/4k/6k smoothing + bins + leaf sweep
- `scripts/cv_ebm_tune_on_4k.py` — interactions / lr / rounds / leaf on 4k baseline
- `plots/cv_ebm_extra_smooth.csv`, `plots/cv_ebm_tune_on_4k.csv` — full CV tables


