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
src/       # importable modules — one file per concern (data, features, models, …)
tests/     # mirrors src/ — one test file per module
data/      # gitignored except .gitkeep; holds dataset.csv, train.csv, holdout.csv
submissions/  # gitignored except .gitkeep
```

**Data split:** `data/dataset.csv` is the full dataset. `src/data.split_holdout()` produces `data/train.csv` (1200 rows) and `data/holdout.csv` (300 rows, seed=42). All exploration and model development use `train.csv` only; `holdout.csv` is reserved for final evaluation.

## Workflow

Red-green TDD: write a failing test in `tests/`, implement the minimum in `src/` to pass, then refactor.

## Stack

scikit-learn · LightGBM · HistGradientBoostingRegressor · LinearRegression · pygam (GAMs) · interpret-core (EBM) · causal-learn (PC, GES) · lingam (DirectLiNGAM) · networkx

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

### Diagnostic analyses (completed)

1. **x5 sentinel missingness** — 178 sentinel rows (14.8% of train). Distribution across clusters is borderline non-uniform (chi2=7.67, p=0.053 — Albacete_low has only 29 vs ~50 in others). However, the sentinel indicator does NOT predict target beyond imputed x5 (coef=-1.15, p=0.51). **Conclusion:** Safe to impute with median; no need for a binary indicator feature.

2. **Residual analysis after City + x4** — Base model R²=0.451. Residuals are near-normal (skew=0.04, kurtosis=-0.44) but Shapiro-Wilk rejects at p=0.008 (expected with n=1200). Levene's test shows mild heteroscedasticity across clusters (p=0.027), suggesting slightly different residual variance per group. **Conclusion:** Additive assumption is valid but a robust/heteroscedastic model may gain marginally.

3. **x1/x2 nonlinear shapes within clusters** — Per-cluster GAM R² values are consistent: x1 ranges [0.178, 0.227], x2 ranges [0.136, 0.219]. Shapes are similar across all 4 clusters. However, the pooled-vs-cluster F-test is significant (F≈35, p<0.001) because cluster-specific fits capture between-cluster variance (City+x4 effect), not because the nonlinear shapes differ. **Conclusion:** A single global GAM shape for x1 and x2 is appropriate; no cluster-specific nonlinear terms needed.

4. **Total R² ceiling** — Full model R²=0.938 (adj=0.936), residual std=6.0. R² breakdown:
   - City + x4: 0.451 (45.1%)
   - + GAM(x1): 0.547 (+9.7%)
   - + GAM(x2): 0.626 (+7.8%)
   - + x5,x8,x10,x11: 0.938 (+31.2%)
   **Conclusion:** The feature set captures ~94% of variance. Remaining ~6% is likely irreducible noise (std≈6). The linear features (x5,x8,x10,x11) contribute the largest marginal gain after the base.

5. **x4 bimodal origin** — Gap at [-0.167, +0.167], width=0.334, gap-ratio=868x the median inter-observation spacing. The gap is exact/manufactured — zero observations in a 0.334-wide band. Split is balanced: 604 below, 596 above. Distribution is identical across cities (KS stat=0.043, p=0.61). **Conclusion:** x4 is a designed variable (likely abs(latent) or truncated), not a natural continuous feature. The bimodality is an artifact of the data generation process, not a modeling concern.

### Diagnostics code

- `src/diagnostics.py`: `sentinel_indicator()`, `sentinel_cluster_crosstab()`, `sentinel_chi2_test()`, `sentinel_target_regression()`, `fit_city_x4_ols()`, `residual_normality_test()`, `residual_heteroscedasticity_test()`, `residual_stats()`, `fit_gam_per_cluster()`, `pooled_vs_cluster_gam_test()`, `fit_r2_ceiling()`, `feature_group_r2_breakdown()`, `x4_bimodality_test()`, `x4_city_distribution_test()`, `x4_gap_analysis()`
- `src/diagnostics_plots.py`: `plot_sentinel_distribution()`, `plot_residual_analysis()`, `plot_gam_per_cluster()`, `plot_r2_breakdown()`, `plot_ceiling_residuals()`, `plot_x4_bimodality()`
- `tests/test_diagnostics.py`: 35 tests
- `plots/diagnostics/`: sentinel distribution, residual analysis, per-cluster GAMs, R² breakdown, ceiling residuals, x4 bimodality

### Causal discovery code

- `src/causal.py`: `preprocess_for_causal()`, `run_pc()`, `run_direct_lingam()`, `run_ges()`, `adjacency_to_edges()`, `consensus_graph()`, `bootstrap_edges()`
- `src/causal_plots.py`: `plot_dag()`, `plot_adjacency_heatmap()`, `plot_edge_bootstrap()`
- `tests/test_causal.py`: 15 tests
- `plots/causal/`: DAGs, heatmaps, bootstrap charts
- `plots/index.html`: self-contained HTML viewer with all EDA + causal + diagnostic results
