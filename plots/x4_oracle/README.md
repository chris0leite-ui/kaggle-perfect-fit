# x4 functional-form oracle — results

Idea #2 from the DGP-archaeology plan.

Three scripts run under this directory:

| script                                          | purpose |
|-------------------------------------------------|---------|
| `scripts/x4_functional_form_oracle.py`          | 12 candidate bases, CV + anti-adversarial bin CV + KS on gap predictions |
| `scripts/x4_step_vs_x9_attribution.py`          | Does the +20 step come from x4 or x9? |
| `scripts/cv_corrected_step_candidates.py`       | CV a 3×3 grid: shape {linear, step, tanh} × x9 {drop, raw, wc} |
| `scripts/cv_cross_LE_step11.py`                 | Plug the corrected step into cross_LE |

## 1. Oracle ranking — step-family candidates tie at CV 2.19

| candidate         | description                          | CV MAE | near-gap MAE | gap KS |
|-------------------|--------------------------------------|-------:|-------------:|-------:|
| **linear_step** (A1) | x4 + 1{x4>0}                      |  2.191 |        2.142 |  0.049 |
| sigmoid_50        | x4 + σ(50·x4)                        |  2.191 |        2.142 |  0.051 |
| tanh_narrow       | x4 + tanh(x4/0.05)                   |  2.191 |        2.143 |  0.055 |
| knots_0.17        | x4 + (x4+0.17)+ + (x4−0.17)+         |  2.195 |        2.164 |  0.057 |
| tanh_mid          | x4 + tanh(x4/0.15)                   |  2.230 |        2.363 |  0.057 |
| cubic_spline_5k   | cubic spline, knots at ±0.25, 0      |  2.290 |        2.412 |  0.059 |
| tanh_wide         | x4 + tanh(x4/0.30)                   |  2.357 |        2.772 |  0.061 |
| step only         | 1{x4>0}                              |  2.820 |        2.988 |  0.047 |
| poly3             | x4 + x4² + x4³                       |  2.545 |        3.234 |  0.065 |
| linear            | x4                                   |  3.477 |        4.761 |  0.059 |

Any candidate with a near-step basis (sigmoid_50, tanh_narrow, sharp step,
knots at ±0.17) ties the A1 form on training. **Training cannot
distinguish among them** because the gap has zero observations. The
oracle's KS test is insensitive to step placement — we need a different
discriminator.

## 2. The key finding — A1's +20 step is *half x9 contamination*

Training cluster statistics:

|             | n   | mean x9 | mean target |
|-------------|-----|--------:|------------:|
| sign(x4)=0  | 750 |   +4.02 |      −14.12 |
| sign(x4)=1  | 750 |   +5.97 |       +7.31 |

Target contrast between clusters: **+21.43**.
Of this, 15·(mean x4 contrast = +0.670) = +10.04 is explained by the
linear x4 slope. The remaining +11.38 is the "jump at x4=0".

Fit a linear_step model with four x9 treatments:

| variant                           | β_x4  | β_step | β_x9  | CV   |
|-----------------------------------|------:|-------:|------:|-----:|
| raw x9 (A1-shape)                 | +15.04| **+19.54** | −4.26 | 2.19 |
| x9_wc (Simpson-corrected)         | +15.04| **+11.21** | −4.26 | 2.19 |
| drop x9                           | +14.78| **+11.38** |   —   | 3.39 |
| no step + raw x9 (A2-shape)       | +36.93|     —      | −2.41 | 3.48 |
| no step + x9_wc                   | +30.47|     —      | −4.28 | 2.93 |

**β_step drops from +19.5 to +11.2 when x9's cluster contrast is
removed.** The +8 difference is exactly accounted for by x9's cluster
gap (≈+1.96) times A1's β_x9 (≈−4): 1.96 · 4 ≈ +7.8.

Cluster-bias audit confirms only x9 has a significant cluster shift
(t = −66.5; every other feature |t| < 2 and |diff| < 0.1 std). So the
decomposition is clean: **true DGP step ≈ +11, β_x9 ≈ 0**. A1's +20
step was double-counting.

## 3. Corrected-step CV grid

| shape       | x9     | CV    | non-sent | β_x4  | β_step | β_x9  |
|-------------|--------|------:|---------:|------:|-------:|------:|
| step_sharp  | raw    | 2.191 |    0.829 | +15.04|  +19.54| −4.26 |
| step_sharp  | x9_wc  | 2.191 |    0.829 | +15.04|  +11.21| −4.26 |
| step_sharp  | drop   | 3.388 |    2.202 | +14.78|  +11.38|   —   |
| step_tanh   | drop   | 3.390 |    2.202 | +11.80|   +7.00|   —   |
| linear      | x9_wc  | 2.933 |    1.684 | +30.47|    —   | −4.28 |
| linear      | drop   | 3.695 |    2.538 | +30.44|    —   |   —   |

## 4. Cross_LE with corrected step — doesn't help

Blend-weight sweep (w·LIN + (1−w)·EBM_x9):

| LIN variant    | best w | best CV | vs cross_LE_free |
|----------------|-------:|--------:|-----------------:|
| **lin_free** (β_x4=+30, no step)       | 0.4 | **2.947** | LB 2.94 (known)   |
| lin_step (+11 step, no x9)             | 0.5 |  3.037    | CV +0.09          |
| lin_tanh (+7 tanh, no x9)              | 0.5 |  3.032    | CV +0.09          |

The step-corrected LIN improves as a solo model (CV 3.39 vs 3.70) but
**hurts the ensemble** by ~0.08 CV across every blend weight. EBM_x9
implicitly absorbs the cluster-contrast through x9's shape function;
adding an explicit step in LIN double-counts and reduces ensemble
diversity.

## 5. Takeaways

1. **A1's +20 step is wrong by a factor of ~2.** True step is +11. This
   was invisible in the train-only PC/LiNGAM analysis because the step
   and x9's cluster contrast are aliased 1:1 in the training design matrix.
2. **No LB improvement available via linear step correction.** Every
   variant that replaces cross_LE's lin_free component degrades the
   ensemble by ~0.08 CV — which typically translates to LB regression.
3. **Solo LIN candidate improves from CV 3.70 → 3.39** by adopting
   the +11 step. If submitted alone, projected LB is ~5–7 (vs
   simple_linear_interact LB 7.38). Written as
   `submission_linear_step11_nox9.csv` for optional LB testing.
4. **Next attack (idea #3) — the hidden x8 clamp** is now the dominant
   remaining source of non-sent error. Worth more than any further
   f(x4) refinement.

## Submissions written (all CV-tested, none LB-tested)

| File                                    | CV   | Notes |
|-----------------------------------------|-----:|-------|
| `submission_linear_step11_nox9.csv`     | 3.39 | Solo step-11 linear, no x9. Projected LB 5–7. |
| `submission_linear_tanh_step_nox9.csv`  | 3.39 | Same with smoothed step (scale 0.15). |
| `submission_linear_nox9_baseline.csv`   | 3.70 | Sanity-check replica of simple_linear_interact (LB 7.38). |
| `submission_cross_LE_step11.csv`        | 3.04 | Ensemble with step LIN. Projected LB 3.1–3.3 (worse than cross_LE). |
| `submission_cross_LE_tanh.csv`          | 3.03 | Same with tanh step. |
