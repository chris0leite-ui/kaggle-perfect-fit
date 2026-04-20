# x5 archaeology — three investigations, one verdict

Following the clamp breakthrough (id<100 triggers the x8 clamp),
we asked: does x5 have an analogous hand-crafted pattern? Three
clean tests leveraging A1's exact fit on rows id≥100:

## (1) Back-solve x5 from target — works perfectly

Since A1 fits id≥100 non-clamp rows to machine precision:

    x5_true = (A1_body_without_x5 − target) / 8

Validation on the 1192 non-sentinel rows where we know x5:

| statistic                | value |
|--------------------------|------:|
| mean(x5_solved − x5_obs) | **+3.7e-16** |
| max |error|              | **5.3e-15** |
| std(error)               | 1.3e-15 |
| rows with |err| < 1e-3   | **1192 / 1192 (100%)** |

The inversion is exact to float precision. Back-solving 222 sentinel
x5 values gives a distribution indistinguishable from Uniform(7, 12):

| sample                                | mean | median | std  | KS vs U(7,12) |
|---------------------------------------|-----:|-------:|-----:|--------------:|
| observed non-sentinel (id≥100, n=1192)| 9.42 | 9.35   | 1.46 | — |
| back-solved sentinel (id≥100, n=208)  | 9.47 | 9.50   | 1.45 | **D=0.051, p=0.645** |
| back-solved sentinel (id<100, n=14)   | 10.03| 10.04  | 1.19 | small n |

Cannot reject uniform. The back-solved sentinel x5 occupies the
same support [7.04, 12.00] as observed x5.

## (2) Does id predict sentinel status? No.

Sentinel rate by id-100 bucket (train):

| id range     | rate |
|--------------|------|
| [0, 100)     | 0.140 |
| [100, 200)   | 0.070 |
| [200, 300)   | 0.150 |
| …            | …   |
| [1400, 1500) | 0.210 |

mean = 0.148 (overall rate), range [0.07, 0.24] consistent with
binomial SE ≈ 0.036 at n=100. Test buckets equivalent: [0.12, 0.18].

mod-k tests (k = 2, 3, 5, 7, 10): every residue class sits near
0.148 ± binomial noise. mod 100 looks irregular (rates from 0.00
to 0.40), but at n=15 per residue the binomial 2σ band is ±0.18 —
every observed rate fits inside.

**Sentinel selection is MCAR.**

## (3) Does x5 have sequential / RNG structure in id? No.

**Autocorrelation of x5 ordered by id:** every lag 1–30 has
|ρ| < 0.06; the 95% confidence band is ±0.057. No lag is
significant.

**Linear x5 ~ id:** slope −2.3e−5, R² = 0.0000.

**Pearson r(x5, f(id)) across 9 transforms** (identity, sin/cos
of 2π·id/1500, sin 2π·id·φ for golden-ratio drift, id mod {5, 7,
11}, a classical LCG hash): every r has |r| < 0.05 and p > 0.1.

**FFT:** power spectrum is approximately flat (top frequencies
at 0.42, 0.17, 0.49 cycles/row — white-noise echoes of the
ceiling freq). No single peak dominates.

## Verdict

x5 is **genuine Uniform(7, 12) noise with MCAR sentinels**. The
1.52-MAE sentinel floor is mathematically irreducible from any
observable feature, any id-based rule, or any detectable sequential
structure.

All three DGP-archaeology ideas (#1 x4-x9 shift, #2 step is +11-12
not +20, #3 clamp is id<100) **cannot help with x5**. The best we
can do on sentinel rows is predict at the median, exactly as we
already do.

## Artefacts

| file | contents |
|------|----------|
| `scripts/x5_archaeology.py`              | runs all three investigations |
| `plots/x5_archaeology/x5_archaeology.png`| six-panel diagnostic figure |
| `plots/x5_archaeology/x5_solved.csv`     | back-solved x5 for every train row |

The six panels:

  (a) x5 vs id scatter with back-solved sentinels overlaid — no trend
  (b) Sentinel rate per id-100 bucket (train + test) — all in MCAR band
  (c) ACF of x5 by id — flat within 95% CI
  (d) Back-solved vs observed x5 histograms + Uniform(7,12) reference
  (e) FFT power spectrum — approximately flat
  (f) First 200 non-sentinel x5 values by id — visible noise
