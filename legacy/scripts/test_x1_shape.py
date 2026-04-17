"""Fit competing functional forms for x1 and see which wins.

Residualises target against every other effect in the A2 ClosedFormModel so we
isolate x1's contribution, then fits three candidates by least squares:

    A. cos(pi*x1)          (what A2 uses)
    B. x1**2               (what A1 uses, up to sign)
    C. piecewise linear:   a_neg*x1*(x1<=0) + a_pos*x1*(x1>0) + step*1(x1>0)

Reports MAE and RMSE on the residual for each. Also visualises the three fits
vs a scatter of the residuals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "formulas"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


def residualise(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (x1, residual) where residual is target minus every non-x1 effect
    from the ClosedFormModel basis (fit without x1 terms)."""
    mask = df["x5"] == SENTINEL
    x5_median = float(df.loc[~mask, "x5"].median())
    x5 = df["x5"].where(~mask, x5_median).values
    is_sent = mask.astype(float).values

    A = np.column_stack([df["x4"].values, np.ones(len(df))])
    slope_int, *_ = np.linalg.lstsq(A, df["x9"].values, rcond=None)
    x9_resid = df["x9"].values - (slope_int[0] * df["x4"].values + slope_int[1])

    city = df["City"].map({"Zaragoza": 1.0, "Albacete": 0.0}).values

    # Design matrix WITHOUT any x1 term — we want the residual to contain it.
    M = np.column_stack([
        city,
        df["x4"].values,
        df["x8"].values,
        x5,
        df["x10"].values * df["x11"].values,
        np.cos(5 * np.pi * df["x2"].values),
        x9_resid,
        is_sent,
        np.ones(len(df)),
    ])
    coef, *_ = np.linalg.lstsq(M, df["target"].values, rcond=None)
    residual = df["target"].values - M @ coef
    return df["x1"].values, residual


def fit_cos(x1: np.ndarray, r: np.ndarray) -> dict:
    """fit beta * cos(pi*x1) + c."""
    B = np.column_stack([np.cos(np.pi * x1), np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(B, r, rcond=None)
    pred = B @ coef
    return {"name": "A: beta*cos(pi*x1) + c", "coef": coef, "pred": pred}


def fit_quadratic(x1: np.ndarray, r: np.ndarray) -> dict:
    """fit a*x1^2 + b*x1 + c (include linear term for fairness)."""
    B = np.column_stack([x1 ** 2, x1, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(B, r, rcond=None)
    pred = B @ coef
    return {"name": "B: a*x1^2 + b*x1 + c", "coef": coef, "pred": pred}


def fit_piecewise(x1: np.ndarray, r: np.ndarray) -> dict:
    """fit a_neg*x1*(x1<=0) + a_pos*x1*(x1>0) + step*1(x1>0) + c."""
    pos = (x1 > 0).astype(float)
    neg = 1.0 - pos
    B = np.column_stack([x1 * neg, x1 * pos, pos, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(B, r, rcond=None)
    pred = B @ coef
    return {"name": "C: piecewise linear with step at 0", "coef": coef, "pred": pred}


def fit_piecewise_continuous(x1: np.ndarray, r: np.ndarray) -> dict:
    """V-shape: a_neg*x1*(x1<=0) + a_pos*x1*(x1>0) + c (no step; continuous at 0)."""
    pos = (x1 > 0).astype(float)
    neg = 1.0 - pos
    B = np.column_stack([x1 * neg, x1 * pos, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(B, r, rcond=None)
    pred = B @ coef
    return {"name": "D: V-shape (continuous at 0)", "coef": coef, "pred": pred}


def evaluate(name: str, y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    resid = y - pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    print(f"  {name:40s}  MAE={mae:6.3f}  RMSE={rmse:6.3f}")
    return mae, rmse


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv")
    x1, r = residualise(df)
    print(f"n={len(x1)} | residual std={r.std():.3f}")
    print(f"x1 range [{x1.min():.3f}, {x1.max():.3f}]")
    print(f"x1 > 0: {(x1 > 0).sum()},  x1 <= 0: {(x1 <= 0).sum()}")

    fits = [
        fit_cos(x1, r),
        fit_quadratic(x1, r),
        fit_piecewise(x1, r),
        fit_piecewise_continuous(x1, r),
    ]
    print("\nLeast-squares fits on x1-residual:")
    rows = []
    for f in fits:
        mae, rmse = evaluate(f["name"], r, f["pred"])
        rows.append({"name": f["name"], "coef": f["coef"], "mae": mae, "rmse": rmse})

    # Coefficients
    print("\nCoefficients:")
    for row in rows:
        print(f"  {row['name']}: {np.round(row['coef'], 3).tolist()}")

    # Visualise
    grid = np.linspace(-1.05, 1.05, 501)

    def curve_cos(g, c):
        return c[0] * np.cos(np.pi * g) + c[1]

    def curve_quad(g, c):
        return c[0] * g ** 2 + c[1] * g + c[2]

    def curve_pw(g, c):
        pos = (g > 0).astype(float); neg = 1 - pos
        return c[0] * g * neg + c[1] * g * pos + c[2] * pos + c[3]

    def curve_v(g, c):
        pos = (g > 0).astype(float); neg = 1 - pos
        return c[0] * g * neg + c[1] * g * pos + c[2]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x1, r, s=6, alpha=0.25, color="gray", label="residual after other effects")
    ax.plot(grid, curve_cos(grid, fits[0]["coef"]), lw=2, label="cos(pi*x1)")
    ax.plot(grid, curve_quad(grid, fits[1]["coef"]), lw=2, label="x1^2 + linear")
    ax.plot(grid, curve_pw(grid, fits[2]["coef"]), lw=2,
            label="piecewise linear + step")
    ax.plot(grid, curve_v(grid, fits[3]["coef"]), lw=2, ls="--",
            label="V-shape (no step)")
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("residual (target − non-x1 effects)")
    ax.set_title("Functional-form candidates for x1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "x1_shape_candidates.png", dpi=130)
    plt.close(fig)

    # Zoom near zero to see any step more clearly
    near = np.abs(x1) < 0.3
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x1[near], r[near], s=10, alpha=0.4, color="gray", label="residual (|x1|<0.3)")
    grid_z = np.linspace(-0.3, 0.3, 301)
    ax.plot(grid_z, curve_cos(grid_z, fits[0]["coef"]), lw=2, label="cos(pi*x1)")
    ax.plot(grid_z, curve_quad(grid_z, fits[1]["coef"]), lw=2, label="x1^2 + linear")
    ax.plot(grid_z, curve_pw(grid_z, fits[2]["coef"]), lw=2,
            label="piecewise linear + step")
    ax.plot(grid_z, curve_v(grid_z, fits[3]["coef"]), lw=2, ls="--",
            label="V-shape (no step)")
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("residual")
    ax.set_title("Zoom near x1=0 — any discontinuity would appear here")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "x1_shape_zoom.png", dpi=130)
    plt.close(fig)

    # Also run the same for x4 to show what a real discontinuity looks like
    print("\nDoes x1 have a distribution gap like x4?")
    print(f"  x1 range:  [{x1.min():.3f}, {x1.max():.3f}]")
    bins = np.linspace(x1.min(), x1.max(), 41)
    counts, _ = np.histogram(x1, bins=bins)
    print(f"  min bin count: {counts.min()} (out of 40 bins) — no gap if > 0")


if __name__ == "__main__":
    main()
