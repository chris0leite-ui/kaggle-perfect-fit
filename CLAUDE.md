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

scikit-learn · LightGBM · HistGradientBoostingRegressor · LinearRegression
