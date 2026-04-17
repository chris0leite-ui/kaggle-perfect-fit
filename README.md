# The Perfect Fit — Kaggle Competition

https://www.kaggle.com/competitions/the-perfect-fit

Tabular regression (1,500 rows, 10 numeric + 2 categorical features,
continuous target, MAE metric).

Best public LB: **MAE 2.94** with the `cross_LE` ensemble (ranked ~43).
The leaderboard-top cluster sits at 1.65–1.71, within 0.2 of the
theoretical noise floor of 1.52 imposed by the sentinel x5 column.

## Reproduce the submissions

```bash
pip install -r requirements.txt

# Place dataset.csv, test.csv, sample_submission.csv into data/
# (download from Kaggle — not shipped with the repo)

jupyter notebook notebooks/final_submissions.ipynb
```

Running the notebook top-to-bottom writes four CSVs into `submissions/`:

| File | 5-fold CV MAE | Public LB MAE | What it is |
|---|---|---|---|
| `submission_ebm_heavy_smooth.csv` | 3.08 | **4.90** | Single EBM, smoothing tuned for robustness. Simplest baseline. |
| `submission_ensemble_cross_LE.csv` | 2.97 | **2.94** | 0.5·LIN(no x9) + 0.5·EBM(no x4). **Best confirmed submission.** |
| `submission_ensemble_triple_locked_b_lambda50.csv` | 2.82 | untested | 0.25·LIN_locked_b + 0.25·EBM(no x4) + 0.5·EBM(full). |
| `submission_router_A1_triple.csv` | **1.84** | untested | Router: safe rows → A1 closed form, else → triple ensemble. |

## Repo layout

```
notebooks/
  final_submissions.ipynb   Self-contained. The only thing you need to reproduce top submissions.
scripts/                    Six scripts originally used to build the top submissions (kept for reference).
data/                       Competition data (gitignored; put dataset.csv, test.csv, sample_submission.csv here).
submissions/                Four competitive submission CSVs.
plots/                      High-signal diagnostics, formulas, and clamp-analysis plots.
legacy/                     Archive of exploratory code, stale plots, stale submissions, and rejected ideas.
CLAUDE.md                   Full development log — every decision, every dead end, every result.
LEARNINGS.md                Portable patterns for future tabular competitions.
REPORT.md                   Work report: data quirks, distribution shift, models tried, final results, rejected ideas.
```

## Key findings (TL;DR)

- **Sentinel-noise floor**. x5 has a `SENTINEL = 999.0` value on 15% of
  rows with slope ≈ −8 and a Uniform(7, 12) prior on the true value.
  Best achievable MAE on sentinel rows is `8 × 1.25 = 10`. Times the
  sentinel ratio: **1.52 MAE is the theoretical public-LB floor** — no
  leaderboard submission beats it.
- **Selection bias in training**. `r(x4, x9) = +0.83` in train but
  `+0.001` in test. Every model that leaned on that correlation
  (reverse-engineered formulas, x9 residualisation) generalised poorly.
- **EBM is resilient, linear/formulaic models extrapolate badly**. Our
  best submission averages a parametric model on x4-only with an EBM
  on x9-only — the two views disagree cleanly on the off-diagonal test
  rows where `x4 ⊥ x9`.

Details and other lessons: `CLAUDE.md` (narrative, 1,200 lines),
`LEARNINGS.md` (portable patterns, ~2 pages).
