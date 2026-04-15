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

### Causal discovery code

- `src/causal.py`: `preprocess_for_causal()`, `run_pc()`, `run_direct_lingam()`, `run_ges()`, `adjacency_to_edges()`, `consensus_graph()`, `bootstrap_edges()`
- `src/causal_plots.py`: `plot_dag()`, `plot_adjacency_heatmap()`, `plot_edge_bootstrap()`
- `tests/test_causal.py`: 15 tests
- `plots/causal/`: DAGs, heatmaps, bootstrap charts
- `plots/index.html`: self-contained HTML viewer with all EDA + causal results
