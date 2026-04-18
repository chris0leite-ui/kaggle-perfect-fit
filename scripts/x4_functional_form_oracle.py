"""x4 functional-form oracle (idea #2 from the DGP-archaeology plan).

Why
---
A1's training CV is 1.80 thanks to a hard `+20·1{x4>0}` step at the
origin, which fits training perfectly — yet the step is untestable
from training alone because training has zero rows in x4 ∈
[-0.167, +0.167]. Test has 508 rows in that gap. The step costs
A1 ~9 MAE on the leaderboard; a smoother f(x4) that matches training
equally well should extrapolate better.

Method
------
Enumerate a dozen candidate bases for f(x4). For each candidate:

1. Fit a *full* OLS model on training:

       target ~ 1 + x1² + cos(5π·x2) + x5_imp + x5_is_sent + x8
                 + x10 + x11 + x10·x11 + city_code + x9 + basis(x4)

   (x9 is included raw; it is equally contaminated across candidates,
    so the comparison is fair.)

2. Training CV MAE — baseline fit quality.

3. **Anti-adversarial bin CV** — the key test. Bin training rows by
   |x4| and hold out the *most-adjacent-to-gap* bins one at a time.
   This directly measures each candidate's ability to extrapolate
   from training to points just inside the gap.

4. Predict on test (1500 rows including 508 gap rows):
   - Mean prediction by x4-bin (shows the learned f(x4) curve).
   - Gap-bin prediction distribution (KS vs interpolation).

Candidates
----------
- `linear`              : x4
- `step`                : 1{x4>0}             (pure step)
- `linear_step` (A1)    : x4 + 1{x4>0}
- `tanh_narrow`         : x4 + tanh(x4/0.05)  (near-step)
- `tanh_mid`            : x4 + tanh(x4/0.15)
- `tanh_wide`           : x4 + tanh(x4/0.3)
- `sigmoid_50`          : x4 + sigmoid(50·x4) (very sharp)
- `abs_hinge`           : x4 + |x4|
- `poly2`               : x4 + x4²
- `poly3`               : x4 + x4² + x4³
- `knots_0.17`          : x4 + (x4+0.17)+ + (x4-0.17)+
- `cubic_spline_5k`     : 5-knot natural cubic spline on x4

Outputs
-------
plots/x4_oracle/candidates_cv.csv       scored ranking
plots/x4_oracle/marginal_curves.png     learned f(x4) shapes
plots/x4_oracle/perbin_mae.png          MAE per |x4| bin
plots/x4_oracle/gap_predictions.png     gap-row prediction hists
plots/x4_oracle/README.md               written separately
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "x4_oracle"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0
X4_GAP = (-0.167, 0.167)
RNG = 42


# ----------------------------- candidates ----------------------------------


@dataclass
class Basis:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]   # returns (n, k) basis
    description: str


def _linear(x):    return np.c_[x]
def _step(x):      return np.c_[(x > 0).astype(float)]
def _linstep(x):   return np.c_[x, (x > 0).astype(float)]
def _tanh(scale):  return lambda x: np.c_[x, np.tanh(x / scale)]
def _sigmoid(k):   return lambda x: np.c_[x, 1.0 / (1.0 + np.exp(-k * x))]
def _abs(x):       return np.c_[x, np.abs(x)]
def _poly2(x):     return np.c_[x, x ** 2]
def _poly3(x):     return np.c_[x, x ** 2, x ** 3]
def _knots(x):
    return np.c_[x, np.maximum(x + 0.167, 0), np.maximum(x - 0.167, 0)]
def _cubic_spline(knots):
    def basis(x):
        # natural cubic spline with fixed knots; uses Wahba's parameterisation.
        # For robust behaviour we build a simple truncated-power basis.
        cols = [x, x ** 2, x ** 3]
        for k in knots:
            cols.append(np.maximum(x - k, 0) ** 3)
        return np.c_[tuple(cols)]
    return basis


CANDIDATES: list[Basis] = [
    Basis("linear",        _linear,                 "x4"),
    Basis("step",          _step,                   "1{x4>0}"),
    Basis("linear_step",   _linstep,                "x4 + 1{x4>0}  (A1)"),
    Basis("tanh_narrow",   _tanh(0.05),             "x4 + tanh(x4/0.05)"),
    Basis("tanh_mid",      _tanh(0.15),             "x4 + tanh(x4/0.15)"),
    Basis("tanh_wide",     _tanh(0.30),             "x4 + tanh(x4/0.30)"),
    Basis("sigmoid_50",    _sigmoid(50),            "x4 + σ(50·x4)"),
    Basis("abs_hinge",     _abs,                    "x4 + |x4|"),
    Basis("poly2",         _poly2,                  "x4 + x4²"),
    Basis("poly3",         _poly3,                  "x4 + x4² + x4³"),
    Basis("knots_0.17",    _knots,                  "x4 + (x4+0.17)+ + (x4-0.17)+"),
    Basis("cubic_spline_5k", _cubic_spline(np.linspace(-0.5, 0.5, 5)[1:-1]),
          "cubic spline, 3 interior knots at ±0.25 and 0"),
]


# ----------------------------- data ----------------------------------------


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["x5_is_sent"] = (df["x5"] == SENTINEL).astype(float)
        x5 = df["x5"].where(df["x5"] != SENTINEL, np.nan)
        med = x5.median()
        df["x5_imp"] = x5.fillna(med)
        df["city_code"] = (df["City"] == "Zaragoza").astype(float)
        df["x1sq"] = df["x1"] ** 2
        df["cos5pi_x2"] = np.cos(5 * np.pi * df["x2"])
        df["x10x11"] = df["x10"] * df["x11"]
    return train, test


FIXED_FEATURES = [
    "x1sq", "cos5pi_x2", "x5_imp", "x5_is_sent",
    "x8", "x10", "x11", "x10x11", "city_code", "x9",
]


def build_design(df: pd.DataFrame, basis: Basis) -> np.ndarray:
    fixed = df[FIXED_FEATURES].to_numpy()
    x4_basis = basis.fn(df["x4"].to_numpy())
    return np.c_[fixed, x4_basis]


def fit_and_score(train: pd.DataFrame, basis: Basis) -> dict:
    X = build_design(train, basis)
    y = train["target"].to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    cv_mae, cv_ns, cv_sn = [], [], []
    for tr, va in kf.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        p = m.predict(X[va])
        err = np.abs(p - y[va])
        cv_mae.append(err.mean())
        is_sn = train["x5_is_sent"].to_numpy()[va] > 0
        cv_ns.append(err[~is_sn].mean())
        cv_sn.append(err[is_sn].mean())
    return {
        "cv_mae": float(np.mean(cv_mae)),
        "cv_non_sent": float(np.mean(cv_ns)),
        "cv_sent": float(np.mean(cv_sn)),
    }


def anti_adversarial_bin_cv(train: pd.DataFrame, basis: Basis,
                            edges: np.ndarray) -> tuple[float, dict]:
    """Leave-one-|x4|-bin-out CV. Returns (near-gap MAE, per-bin dict)."""
    X = build_design(train, basis)
    y = train["target"].to_numpy()
    absx4 = np.abs(train["x4"].to_numpy())
    per_bin: dict[tuple[float, float], float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (absx4 >= lo) & (absx4 < hi)
        if mask.sum() < 20:
            continue
        m = LinearRegression().fit(X[~mask], y[~mask])
        per_bin[(float(lo), float(hi))] = float(np.mean(np.abs(m.predict(X[mask]) - y[mask])))

    # "Near-gap" = the two innermost bins (closest to |x4|=0.167).
    if not per_bin:
        return (np.nan, {})
    ordered = sorted(per_bin.items(), key=lambda kv: kv[0][0])
    near_gap_mae = float(np.mean([v for _, v in ordered[:2]]))
    return near_gap_mae, per_bin


def predict_on_test(train: pd.DataFrame, test: pd.DataFrame, basis: Basis) -> np.ndarray:
    Xtr = build_design(train, basis)
    Xte = build_design(test, basis)
    m = LinearRegression().fit(Xtr, train["target"].to_numpy())
    return m.predict(Xte)


# ----------------------------- reporting -----------------------------------


def bin_stats(predictions: np.ndarray, x4: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x4 >= lo) & (x4 < hi)
        if m.sum() == 0:
            continue
        p = predictions[m]
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "mean": float(p.mean()),
                     "std": float(p.std()),
                     "median": float(np.median(p))})
    return pd.DataFrame(rows)


def gap_ks(predictions: np.ndarray, x4: np.ndarray) -> float:
    """KS distance between gap-bin predictions and adjacent bin predictions.

    If f(x4) is correct and other features are independent of x4 (confirmed
    by idea #1), the predicted-target distribution should be a smooth
    continuation from adjacent bins. A step produces a distribution shift.
    """
    in_gap = (x4 >= X4_GAP[0]) & (x4 <= X4_GAP[1])
    adj = (
        ((x4 >= -0.35) & (x4 < X4_GAP[0]))
        | ((x4 > X4_GAP[1]) & (x4 <= 0.35))
    )
    gap_p = predictions[in_gap]
    adj_p = predictions[adj]
    if len(gap_p) < 20 or len(adj_p) < 20:
        return np.nan
    # Because adjacent-bin preds include the learned x4 slope, shift them
    # to the gap midpoint before comparing distributions.
    adj_centred = adj_p - (adj_p.mean() - gap_p.mean())
    return float(ks_2samp(gap_p, adj_centred).statistic)


def plot_marginal_curves(train: pd.DataFrame, results: dict[str, dict]) -> None:
    xs = np.linspace(-0.6, 0.6, 300)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Training target vs x4 as reference — aggregate residuals of all other
    # features out so the trend is the pure x4 contribution.
    ax.axvspan(*X4_GAP, color="#eee", label=f"train gap [{X4_GAP[0]}, {X4_GAP[1]}]")
    colors = plt.cm.tab20(np.linspace(0, 1, len(CANDIDATES)))
    for (basis, col) in zip(CANDIDATES, colors):
        curve = results[basis.name]["x4_curve"](xs)
        ax.plot(xs, curve, label=f"{basis.name}  CV={results[basis.name]['cv_mae']:.2f}",
                color=col, lw=1.5)
    ax.set_xlabel("x4"); ax.set_ylabel("learned x4 contribution to target")
    ax.set_title("Marginal f(x4) curves — candidates compared")
    ax.legend(fontsize=7, ncols=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "marginal_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_perbin_mae(results: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in results.items():
        pb = r["per_bin"]
        if not pb:
            continue
        xs = [(lo + hi) / 2 for (lo, hi) in pb.keys()]
        ys = list(pb.values())
        ax.plot(xs, ys, marker="o", label=f"{name} (near={r['near_gap_mae']:.2f})", lw=1)
    ax.set_xlabel("|x4| bin centre"); ax.set_ylabel("MAE on held-out bin")
    ax.set_title("Anti-adversarial CV: MAE when holding out |x4| bins")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "perbin_mae.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_gap_predictions(test: pd.DataFrame, results: dict[str, dict]) -> None:
    x4 = test["x4"].to_numpy()
    in_gap = (x4 >= X4_GAP[0]) & (x4 <= X4_GAP[1])
    adj = (
        ((x4 >= -0.35) & (x4 < X4_GAP[0]))
        | ((x4 > X4_GAP[1]) & (x4 <= 0.35))
    )
    n = len(CANDIDATES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()
    for ax, basis in zip(axes, CANDIDATES):
        p = results[basis.name]["test_pred"]
        ax.hist(p[adj], bins=30, alpha=0.45, color="#1b9e77", density=True, label="adjacent")
        ax.hist(p[in_gap], bins=30, alpha=0.45, color="#d95f02", density=True, label="gap (508)")
        ax.set_title(f"{basis.name}  KS={results[basis.name]['gap_ks']:.3f}",
                     fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Test prediction distributions: gap vs adjacent bins", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gap_predictions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def learned_x4_curve(train: pd.DataFrame, basis: Basis) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function xs -> learned x4 contribution (coef·basis(xs)).

    We refit on all of train and extract the coefficients on the x4 basis
    columns only. We also centre by subtracting the value at x4=0 so curves
    are comparable.
    """
    X = build_design(train, basis)
    y = train["target"].to_numpy()
    m = LinearRegression().fit(X, y)
    n_fixed = len(FIXED_FEATURES)
    coefs_x4 = m.coef_[n_fixed:]

    def curve(xs):
        B = basis.fn(xs)
        vals = B @ coefs_x4
        return vals - (basis.fn(np.array([0.0])) @ coefs_x4)[0]
    return curve


# ----------------------------- main ----------------------------------------


def main() -> None:
    print("=" * 72)
    print("x4 functional-form oracle")
    print("=" * 72)

    train, test = load()
    print(f"train={len(train)}  test={len(test)}  "
          f"test-in-gap={int(((test['x4'] >= X4_GAP[0]) & (test['x4'] <= X4_GAP[1])).sum())}")

    # Bin edges on |x4| for anti-adversarial CV ----------------------------
    absx4_edges = np.array([0.167, 0.22, 0.28, 0.35, 0.43, 0.52, 0.62])
    print(f"\n|x4| bin edges: {absx4_edges.tolist()}")
    print("Bin counts:", np.histogram(np.abs(train["x4"]), absx4_edges)[0].tolist())

    bin_edges_x4 = np.linspace(-0.6, 0.6, 21)

    results: dict[str, dict] = {}
    for basis in CANDIDATES:
        print(f"\n-- {basis.name}  ({basis.description})")
        scores = fit_and_score(train, basis)
        near_gap, per_bin = anti_adversarial_bin_cv(train, basis, absx4_edges)
        test_pred = predict_on_test(train, test, basis)
        curve_fn = learned_x4_curve(train, basis)
        bs = bin_stats(test_pred, test["x4"].to_numpy(), bin_edges_x4)
        ks = gap_ks(test_pred, test["x4"].to_numpy())
        results[basis.name] = {
            **scores,
            "near_gap_mae": near_gap,
            "per_bin": per_bin,
            "test_pred": test_pred,
            "x4_curve": curve_fn,
            "test_bin_stats": bs,
            "gap_ks": ks,
        }
        print(f"   CV MAE     = {scores['cv_mae']:.3f}  "
              f"(non-sent {scores['cv_non_sent']:.3f}, sent {scores['cv_sent']:.3f})")
        print(f"   near-gap MAE (extrapolation) = {near_gap:.3f}")
        print(f"   gap-KS vs adjacent bins      = {ks:.3f}")

    # Ranking CSV ----------------------------------------------------------
    rows = []
    for name, r in results.items():
        rows.append({
            "candidate": name,
            "description": next(b.description for b in CANDIDATES if b.name == name),
            "cv_mae": r["cv_mae"],
            "cv_non_sent": r["cv_non_sent"],
            "near_gap_mae": r["near_gap_mae"],
            "gap_ks": r["gap_ks"],
        })
    table = pd.DataFrame(rows)
    table["rank_sum"] = (
        table["cv_mae"].rank() + table["near_gap_mae"].rank() + table["gap_ks"].rank()
    )
    table = table.sort_values("rank_sum").reset_index(drop=True)
    table.to_csv(OUT / "candidates_cv.csv", index=False)

    print("\n" + "=" * 72)
    print("RANKING — composite of (CV MAE, near-gap CV MAE, gap KS):")
    print("=" * 72)
    print(table.to_string(index=False,
          formatters={c: "{:.3f}".format for c in
                     ("cv_mae", "cv_non_sent", "near_gap_mae", "gap_ks", "rank_sum")}))

    # Visualisations -------------------------------------------------------
    plot_marginal_curves(train, results)
    plot_perbin_mae(results)
    plot_gap_predictions(test, results)

    # Per-bin mean-prediction curves (shows whether gap predictions are smooth)
    fig, ax = plt.subplots(figsize=(10, 6))
    x4_centres = (bin_edges_x4[:-1] + bin_edges_x4[1:]) / 2
    ax.axvspan(*X4_GAP, color="#eee", label="train gap")
    colors = plt.cm.tab20(np.linspace(0, 1, len(CANDIDATES)))
    for (basis, col) in zip(CANDIDATES, colors):
        bs = results[basis.name]["test_bin_stats"]
        if bs.empty:
            continue
        xs = ((bs["lo"] + bs["hi"]) / 2).to_numpy()
        ax.plot(xs, bs["mean"], marker="o", ms=3, color=col, lw=1,
                label=f"{basis.name}  KS={results[basis.name]['gap_ks']:.2f}")
    ax.set_xlabel("x4 bin centre (test)")
    ax.set_ylabel("mean predicted target")
    ax.set_title("Mean test prediction by x4 bin — gap continuity check")
    ax.legend(fontsize=7, ncols=2, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "test_meanpred_by_x4.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nArtefacts in {OUT}")


if __name__ == "__main__":
    main()
