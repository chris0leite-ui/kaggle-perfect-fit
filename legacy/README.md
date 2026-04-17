# legacy/

Archived material from the competition run. Nothing here is required to
reproduce the final submissions — see `notebooks/final_submissions.ipynb`
at the repo root for that.

The full narrative of why each item is archived lives in `CLAUDE.md` at
the repo root.

## What's in here

### `legacy/scripts/` (29 files)

Experiments that informed the final approach but are not on its reproduction
path:

- **EBM hyperparameter sweeps** — `cv_ebm_extra_smooth.py`,
  `cv_ebm_tune_on_4k.py`, `cv_ebm_smooth_vs_refine.py`,
  `cv_ebm_plateau_breakers.py`, `cv_ebm_x4_linear.py`,
  `cv_ebm_per_cluster.py`.
  Summarised in CLAUDE.md § "Smoothing sweep". Winning params migrated
  into `scripts/cv_x4_x9_swap_ensemble.py`.
- **Alternative models** — `cv_gam_enhanced.py`, `cv_knn.py`,
  `cv_simple_linear.py`, `cv_rounded_coefs.py`, `cv_rounded_coefs_no_x9.py`.
  All dominated by the EBM-based ensembles on either CV or LB.
- **Rejected ideas** — `cv_a1_plus_ebm_residual.py`,
  `cv_router_ensemble.py` (first router attempt with `locked_b_full`),
  `cv_router_with_clamp_classifier.py`, `cv_synthetic_augmentation.py`,
  `reweight_x4x9.py`, `search_clamp_trigger.py`.
  See CLAUDE.md § "Creative-ideas pass" and "Reweighting attempt".
- **Diagnostic / EDA** — `cv_ensemble_eval.py`, `cv_sentinel_breakdown.py`,
  `cv_x4x9_interaction_tests.py`, `investigate_a1_clamp.py`,
  `test_x1_shape.py`, `x5_imputation_study.py`.
- **Build helpers for stale submissions** — `build_final_submissions.py`,
  `build_simple_linear_submission.py`.
- **Plotting-only** — `plot_a1_fit_vs_angle.py`, `plot_target_pairwise_heatmaps.py`,
  `plot_test_pairwise.py`, `plot_x6x7_angle_vs_x5.py`.

### `legacy/src/` (11 modules)

The original `src/` tree. None of the six kept scripts (in `scripts/`)
or the final notebook (`notebooks/final_submissions.ipynb`) import from
here — they are all self-contained. Archived as-is for reference:

- `data.py`, `evaluate.py` — used to be the eval harness; superseded by
  the notebook's built-in 5-fold CV.
- `features.py`, `models.py` — the scikit-learn-style wrappers
  (`CityEncoder`, `SentinelHandler`, `EBMRegressor`, `GAMRegressor`,
  `AveragingEnsemble`, …). None of them survived into the top submissions.
- `eda.py`, `diagnostics.py`, `causal.py`, `causal_plots.py`,
  `clusters.py`, `tuning.py` — exploratory analysis modules; their
  outputs live in the archived `plots/*/` subfolders and in CLAUDE.md.

### `legacy/tests/` (11 files, 37 passing tests)

Mirrors `legacy/src/`. Five of these files are empty scaffolds that
shipped with the project; the other six cover the exploratory modules.

### `legacy/plots/`

Plot subfolders that aren't referenced as critical in CLAUDE.md:
`causal/`, `ccpr/`, `avp/`, `clusters/`, `eda_round2/`, `round2/`,
`interactions/`, `reweight/`, `scatter/`.

The three kept under `plots/` at the repo root — `diagnostics/`,
`formulas/`, `a1_clamp/` — are the ones CLAUDE.md calls out as
high-signal.

### `legacy/submissions/` (3 CSVs)

Stale candidates that were superseded by the four files in `submissions/`
at the repo root:

- `submission_ebm.csv` — LB 5.66, superseded by `submission_ebm_heavy_smooth.csv`.
- `submission_ensemble_cross_LE_locked_c_50.csv` — untested alt of the LB 2.94 best.
- `submission_ensemble_triple_locked_b_lambda30.csv` — slightly worse CV than the λ=0.5 variant we kept.


