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
  features.py      # CityEncoder, SentinelHandler, X9Residualizer, SplineBasisExpander, build_preprocessor()
  models.py        # GAMRegressor, EBMRegressor, AveragingEnsemble, build_*() for 12 models
  evaluate.py      # split_val_test(), cross_validate_model(), evaluate_on_holdout(), compare_models()
  diagnostics.py   # compute_shap_values(), compute_ks_tests(), compute_residuals(), EBM shape/interaction plots
tests/             # mirrors src/ — one test file per module (80 tests total)
data/              # gitignored except .gitkeep; holds dataset.csv, train.csv, holdout.csv
plots/             # EDA, causal, cluster, and diagnostic visualizations + index.html viewer
  diagnostics/     # ~70 PNGs: SHAP, distribution shift, residuals, QQ, EBM shapes
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
| **x4** | Primary cause of target | Weight +36.8; also causes x9 (explains r=0.83 between them) |
| **City** | Strong direct cause | Binary (Zaragoza/Albacete); weight -23.2 on target |
| **x8** | Direct cause | Weight +12.3 |
| **x5** | Direct cause | Weight -8.1; Pearson r≈0 is misleading due to 222 sentinel values (999.0) |
| **x10** | Direct cause | Weight +2.6 |
| **x11** | Direct cause | Weight +2.8 |
| **x9** | Descendant of x4 | NOT independent — its target correlation (r=0.35) is inherited from x4 |
| **x1** | Nonlinear predictor | GAM R²=0.109 but linear R²≈0; hump-shaped relationship. Causal methods missed it (they assume linearity) |
| **x2** | Nonlinear predictor | GAM R²=0.068 but linear R²≈0; oscillating/wavy relationship. Same blind spot |
| **x6, x7** | Noise | No linear or nonlinear signal found |
| **Country** | Constant | Always "Spain" — drop it |

### Key gotchas

- **Sentinel values in x5**: 222 rows have x5=999.0. Must replace with median (or NaN + impute) before any analysis. Masquerades as zero-correlation noise if left in.
- **x4/x9 collinearity**: r=0.83 but x4 causes x9. Don't include both raw — use x4 alone or x4 + residual(x9|x4).
- **PC and LiNGAM assume linearity**: They cannot detect nonlinear causal relationships (x1, x2). Nonlinear independence tests (kernel-based, GAM-residual) are needed to complete the picture.
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
