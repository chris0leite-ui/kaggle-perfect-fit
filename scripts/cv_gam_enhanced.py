"""Enhanced GAM — inject the hypothesis-driven learnings from
backwards-engineering and the Simpson's-paradox analysis of x9.

Baseline GAM (Round 2) used splines on every numeric feature and a linear
term on city. It hit CV MAE 4.39 and LB 6.47 inside the EBM+GAM ensemble.

Improvements tested here:

1. **Splines only on x1 and x2** (pygam's real value-add — hump / oscillating
   shapes) and pure **linear terms** everywhere else where we know the true
   slope: x4 (+31.5), x5 (-8), x8 (+14), x10/x11 (+2.6/+2.8), city (-24).

2. **x9_wc = x9 − mean(x9 | sign(x4))** — subtracts the cluster-specific
   mean of x9 so that only the within-cluster (Simpson's-paradox-true)
   signal enters the model (expected β ≈ −4, R² tiny but generalises).
   Directly removes the between-cluster pathway without dropping x9.

3. **x10·x11 interaction term** — already in Round 2's gam_interact.

4. **Parametric variant** — swap the x1 and x2 splines for A1/A2's closed
   forms (x1² and cos(5π·x2)), using an ordinary linear regression. Tests
   whether GAM's free splines add anything over the known basis.

5. **Ablations** — with vs without x9_wc; with vs without x10·x11; raw x9
   vs x9_wc. Isolates how much each piece contributes.

5-fold CV on dataset.csv (1500 rows) with KFold(shuffle=True, seed=42).
Also builds submissions for the best variants.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

DATA = REPO / "data"
SUBS = REPO / "submissions"
OUT = REPO / "plots" / "gam_enhanced"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0
SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def cluster_means_x9(train_df: pd.DataFrame) -> tuple[float, float]:
    """Return (mean x9 | x4>0, mean x9 | x4<0) from the training fold."""
    hi = train_df.loc[train_df["x4"] > 0, "x9"].mean()
    lo = train_df.loc[train_df["x4"] < 0, "x9"].mean()
    return float(hi), float(lo)


def add_x9_wc(df: pd.DataFrame, mean_hi: float, mean_lo: float) -> np.ndarray:
    """Within-cluster-centred x9: subtracts per-cluster mean based on sign(x4)."""
    return df["x9"].values - np.where(df["x4"] > 0, mean_hi, mean_lo)


def design_matrix(df: pd.DataFrame, x5_median: float,
                  mean_hi: float, mean_lo: float,
                  x1_basis: str,  # "spline", "square", "cos"
                  x2_basis: str,  # "spline", "cos"
                  include_x10x11: bool,
                  x9_mode: str,   # "wc", "raw", "none"
                  ) -> tuple[np.ndarray, list[str]]:
    """Return (X, names). x1/x2 treatment is handled outside via pygam when
    basis=='spline' — here we emit the other columns only. If basis='square'
    or 'cos' we emit the parametric basis directly as extra columns."""

    city = (df["City"] == "Zaragoza").astype(float).values
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values

    cols: list[np.ndarray] = []
    names: list[str] = []

    if x1_basis == "square":
        cols.append(df["x1"].values ** 2); names.append("x1^2")
    elif x1_basis == "cos":
        cols.append(np.cos(np.pi * df["x1"].values)); names.append("cos(pi*x1)")
    # if basis='spline', x1 is added by the pygam caller

    if x2_basis == "cos":
        cols.append(np.cos(5 * np.pi * df["x2"].values)); names.append("cos(5pi*x2)")
    # if basis='spline', x2 is added by the pygam caller

    cols.append(df["x4"].values); names.append("x4")
    cols.append(x5); names.append("x5_imp")
    cols.append(is_sent); names.append("x5_is_sent")
    cols.append(df["x8"].values); names.append("x8")
    cols.append(df["x10"].values); names.append("x10")
    cols.append(df["x11"].values); names.append("x11")
    if include_x10x11:
        cols.append(df["x10"].values * df["x11"].values); names.append("x10*x11")
    cols.append(city); names.append("city")

    if x9_mode == "wc":
        cols.append(add_x9_wc(df, mean_hi, mean_lo)); names.append("x9_wc")
    elif x9_mode == "raw":
        cols.append(df["x9"].values); names.append("x9")

    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def fit_linear(X_tr, y_tr):
    return LinearRegression().fit(X_tr, y_tr)


def fit_gam(X_tr, y_tr, x1_idx: int, x2_idx: int, lam: float = 2.0,
            n_splines: int = 20):
    from pygam import LinearGAM, l, s

    terms = None
    for i in range(X_tr.shape[1]):
        if i == x1_idx or i == x2_idx:
            term = s(i, n_splines=n_splines)
        else:
            term = l(i)
        terms = term if terms is None else terms + term
    gam = LinearGAM(terms, lam=lam)
    gam.fit(X_tr, y_tr)
    return gam


def make_X(df, x5_median, mean_hi, mean_lo, x1_basis, x2_basis,
           include_x10x11, x9_mode):
    """Return X with x1/x2 prepended as raw columns whenever basis=='spline',
    so pygam can index them. Also returns the column names and x1/x2 idx."""
    X_rest, names = design_matrix(
        df, x5_median, mean_hi, mean_lo,
        x1_basis, x2_basis, include_x10x11, x9_mode,
    )
    cols: list[np.ndarray] = []
    col_names: list[str] = []
    x1_idx = x2_idx = -1
    if x1_basis == "spline":
        cols.append(df["x1"].values); col_names.append("x1")
        x1_idx = 0
    if x2_basis == "spline":
        cols.append(df["x2"].values); col_names.append("x2")
        x2_idx = len(cols) - 1
    X = np.column_stack(cols + [X_rest]) if cols else X_rest
    col_names = col_names + names
    return X, col_names, x1_idx, x2_idx


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def run_variant(df: pd.DataFrame, x1_basis: str, x2_basis: str,
                include_x10x11: bool, x9_mode: str,
                model_kind: str, lam: float = 2.0, n_splines: int = 20):
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    learned_coef = []
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        mean_hi, mean_lo = cluster_means_x9(sub_tr)
        X_tr, names, x1_idx, x2_idx = make_X(
            sub_tr, x5_med, mean_hi, mean_lo,
            x1_basis, x2_basis, include_x10x11, x9_mode,
        )
        X_va, _, _, _ = make_X(
            sub_va, x5_med, mean_hi, mean_lo,
            x1_basis, x2_basis, include_x10x11, x9_mode,
        )
        if model_kind == "gam":
            model = fit_gam(X_tr, sub_tr["target"].values, x1_idx, x2_idx,
                            lam=lam, n_splines=n_splines)
            # GAM predict preserves input shape
            oof[va] = model.predict(X_va)
        else:
            model = fit_linear(X_tr, sub_tr["target"].values)
            oof[va] = model.predict(X_va)
            # Save x9 coef for the "raw" / "wc" ablation
            if x9_mode in ("raw", "wc"):
                idx = names.index("x9_wc" if x9_mode == "wc" else "x9")
                learned_coef.append(model.coef_[idx])
    m = mae(oof, y)
    m_ns = mae(oof[~is_sent], y[~is_sent])
    m_sn = mae(oof[is_sent], y[is_sent])
    coef_str = ""
    if learned_coef:
        coef_str = f"  β_x9_mean={np.mean(learned_coef):+.3f}"
    return m, m_ns, m_sn, oof, coef_str


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)

    variants = [
        # model        x1        x2       x10x11  x9_mode  label
        ("linear",    "square", "cos",    True,   "none",  "linear parametric  no_x9              (CV 3.70 ref)"),
        ("linear",    "square", "cos",    True,   "raw",   "linear parametric  raw x9             (contaminated β_x9)"),
        ("linear",    "square", "cos",    True,   "wc",    "linear parametric  x9_wc              (Simpson-corrected)"),
        ("gam",       "spline", "spline", False,  "none",  "GAM R2 baseline    no x10*11  no_x9"),
        ("gam",       "spline", "spline", True,   "none",  "GAM R2+interact    with x10*11  no_x9"),
        ("gam",       "spline", "spline", True,   "raw",   "GAM + raw x9                         (contaminated)"),
        ("gam",       "spline", "spline", True,   "wc",    "GAM + x9_wc                          (Simpson-corrected)"),
        ("gam",       "square", "cos",    True,   "wc",    "GAM parametric x1/x2 + x9_wc         (should match linear)"),
    ]

    print("=" * 95)
    print(f"{'variant':<60s}  {'overall':>7s}  {'non-sent':>8s}  {'sent':>7s}   notes")
    print("=" * 95)
    results = []
    for kind, x1b, x2b, x10x11, x9m, label in variants:
        t0 = time.time()
        m, m_ns, m_sn, oof, coef_str = run_variant(
            df, x1_basis=x1b, x2_basis=x2b,
            include_x10x11=x10x11, x9_mode=x9m,
            model_kind=kind,
        )
        dt = time.time() - t0
        print(f"{label:<60s}  {m:7.3f}  {m_ns:8.3f}  {m_sn:7.3f}  [{dt:4.1f}s]{coef_str}")
        results.append({
            "variant": label, "overall": m, "non_sentinel": m_ns,
            "sentinel": m_sn, "kind": kind, "x1": x1b, "x2": x2b,
            "x10x11": x10x11, "x9": x9m, "seconds": dt,
        })

    pd.DataFrame(results).to_csv(OUT / "cv_results.csv", index=False)
    print(f"\nwrote {OUT / 'cv_results.csv'}")

    # ------------------------------------------------------------------
    # Build best submissions on full dataset.
    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("Building submissions on full dataset.csv")
    print("=" * 95)

    def build(name: str, x1b: str, x2b: str, x10x11: bool, x9m: str,
              kind: str):
        x5_med = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
        mean_hi, mean_lo = cluster_means_x9(df)
        X_tr, names, x1_idx, x2_idx = make_X(
            df, x5_med, mean_hi, mean_lo, x1b, x2b, x10x11, x9m,
        )
        X_te, _, _, _ = make_X(
            test, x5_med, mean_hi, mean_lo, x1b, x2b, x10x11, x9m,
        )
        if kind == "gam":
            model = fit_gam(X_tr, df["target"].values, x1_idx, x2_idx)
        else:
            model = fit_linear(X_tr, df["target"].values)
        preds = model.predict(X_te)
        out = pd.DataFrame({"id": test["id"], "target": preds})
        out.to_csv(SUBS / f"submission_{name}.csv", index=False)
        print(f"  wrote submission_{name}.csv   "
              f"mean={preds.mean():+.3f}  range=[{preds.min():+.2f}, {preds.max():+.2f}]")

    build("gam_enhanced_x9wc",        "spline", "spline", True,  "wc",  "gam")
    build("linear_enhanced_x9wc",     "square", "cos",    True,  "wc",  "linear")


if __name__ == "__main__":
    main()
