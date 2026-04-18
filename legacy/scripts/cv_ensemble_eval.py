"""5-fold CV evaluation of individual models and ensembles.

Question 1: How does the Round 2 EBM+GAM ensemble predict out of sample?
Question 2: Does adding the reverse-engineered formulas (A1 closed-form, A2
            ClosedFormModel) improve the ensemble?

Models:
    A1   — hard-coded closed form (no fitting)
    A2   — ClosedFormModel (linear LS on hand-picked basis)
    EBM  — Round 2 tuned ExplainableBoostingRegressor
    GAM  — pygam LinearGAM with splines on x1, x2 + linear terms

Ensembles (all use OOF predictions for honest CV):
    EBM+GAM 70/30                — current Round 2 best
    EBM+GAM+A1 stacked (Ridge)   — does the closed form add anything?
    EBM+GAM+A2 stacked (Ridge)
    EBM+GAM+A1+A2 stacked (Ridge)
    Average of all 4

Evaluation: 5-fold KFold(shuffle=True, random_state=42) on dataset.csv (1500
rows), reporting fold MAE + overall MAE. The "ensemble OOF" predictions are
built fold-by-fold so the meta-learner never sees the fold it's predicting on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "formulas"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0
SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Helpers shared across approaches
# ---------------------------------------------------------------------------

def _city(df):
    return df["City"].map({"Zaragoza": 1.0, "Albacete": 0.0}).values


def _x5(df, x5_median):
    return df["x5"].where(df["x5"] != SENTINEL, x5_median).values


# ---------------------------------------------------------------------------
# A1: hard-coded closed form (no parameters learned)
# ---------------------------------------------------------------------------

def a1_predict(df, x5_median):
    x5 = _x5(df, x5_median)
    is_zar = _city(df)
    x4_pos = (df["x4"].values > 0).astype(float)
    return (
        -100 * df["x1"].values ** 2
        + 10 * np.cos(5 * np.pi * df["x2"].values)
        + 15 * df["x4"].values
        - 8 * x5
        + 15 * df["x8"].values
        - 4 * df["x9"].values
        + df["x10"].values * df["x11"].values
        - 25 * is_zar
        + 20 * x4_pos
        + 92.5
    )


# ---------------------------------------------------------------------------
# A2: ClosedFormModel
# ---------------------------------------------------------------------------

class ClosedFormModel:
    def _design(self, df):
        mask = df["x5"] == SENTINEL
        x5 = df["x5"].where(~mask, self.x5_median_).values
        is_sent = mask.astype(float).values
        x9_resid = df["x9"].values - (self.x9_slope_ * df["x4"].values + self.x9_intercept_)
        return np.column_stack([
            _city(df),
            df["x4"].values,
            df["x8"].values,
            x5,
            df["x10"].values * df["x11"].values,
            np.cos(np.pi * df["x1"].values),
            np.cos(5 * np.pi * df["x2"].values),
            x9_resid,
            is_sent,
            np.ones(len(df)),
        ])

    def fit(self, df, y):
        mask = df["x5"] == SENTINEL
        self.x5_median_ = float(df.loc[~mask, "x5"].median())
        A = np.column_stack([df["x4"].values, np.ones(len(df))])
        slope_int, *_ = np.linalg.lstsq(A, df["x9"].values, rcond=None)
        self.x9_slope_, self.x9_intercept_ = float(slope_int[0]), float(slope_int[1])
        M = self._design(df)
        self.coef_, *_ = np.linalg.lstsq(M, y, rcond=None)
        return self

    def predict(self, df):
        return self._design(df) @ self.coef_


# ---------------------------------------------------------------------------
# A3: EBM
# ---------------------------------------------------------------------------

EBM_FEATURES = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]


def ebm_preprocess(df, x5_median):
    out = df[EBM_FEATURES].copy()
    out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
    out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
    out["city"] = _city(df)
    return out


def fit_ebm(df_train, x5_median):
    from interpret.glassbox import ExplainableBoostingRegressor
    model = ExplainableBoostingRegressor(
        interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128,
        random_state=SEED,
    )
    model.fit(ebm_preprocess(df_train, x5_median), df_train["target"].values)
    return model


# ---------------------------------------------------------------------------
# A4: GAM (Round 2 tuned)
# ---------------------------------------------------------------------------

GAM_FEATURES = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]


def gam_design(df, x5_median):
    out = df[GAM_FEATURES].copy()
    out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
    out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
    out["city"] = _city(df)
    out["x10x11"] = out["x10"] * out["x11"]
    return out


def fit_gam(df_train, x5_median):
    from pygam import LinearGAM, l, s
    X = gam_design(df_train, x5_median)
    feature_names = list(X.columns)
    spline_idx = {feature_names.index("x1"), feature_names.index("x2")}
    terms = None
    for i in range(X.shape[1]):
        term = s(i, n_splines=20) if i in spline_idx else l(i)
        terms = term if terms is None else terms + term
    gam = LinearGAM(terms, lam=2.0)
    gam.fit(X.values, df_train["target"].values)
    return gam, feature_names


# ---------------------------------------------------------------------------
# CV pipeline: build OOF predictions for every base model
# ---------------------------------------------------------------------------

def build_oof(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with OOF predictions from each base model."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    n = len(df)
    oof = pd.DataFrame({
        "y": df["target"].values,
        "fold": -np.ones(n, dtype=int),
        "a1": np.zeros(n),
        "a2": np.zeros(n),
        "ebm": np.zeros(n),
        "gam": np.zeros(n),
    })

    for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        oof.loc[va_idx, "fold"] = fold

        x5_median = float(tr.loc[tr["x5"] != SENTINEL, "x5"].median())

        # A1 — closed form, no fitting
        oof.loc[va_idx, "a1"] = a1_predict(va, x5_median)

        # A2 — ClosedFormModel
        m2 = ClosedFormModel().fit(tr, tr["target"].values)
        oof.loc[va_idx, "a2"] = m2.predict(va)

        # EBM
        ebm = fit_ebm(tr, x5_median)
        oof.loc[va_idx, "ebm"] = ebm.predict(ebm_preprocess(va, x5_median))

        # GAM
        gam, names = fit_gam(tr, x5_median)
        Xva = gam_design(va, x5_median)[names].values
        oof.loc[va_idx, "gam"] = gam.predict(Xva)

        print(f"  fold {fold}:  "
              f"A1={mae(oof.loc[va_idx, 'a1'], oof.loc[va_idx, 'y']):.3f}  "
              f"A2={mae(oof.loc[va_idx, 'a2'], oof.loc[va_idx, 'y']):.3f}  "
              f"EBM={mae(oof.loc[va_idx, 'ebm'], oof.loc[va_idx, 'y']):.3f}  "
              f"GAM={mae(oof.loc[va_idx, 'gam'], oof.loc[va_idx, 'y']):.3f}")

    return oof


def mae(y_pred, y_true):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


# ---------------------------------------------------------------------------
# Ensembles built from OOF predictions
# ---------------------------------------------------------------------------

def per_fold_mae(pred, y, fold):
    return [mae(pred[fold == k], y[fold == k]) for k in range(N_SPLITS)]


def fixed_weighted(oof, weights):
    """Weighted average with given weights dict {col: w}."""
    pred = np.zeros(len(oof))
    for col, w in weights.items():
        pred += w * oof[col].values
    return pred


def stacked_ridge(oof, cols, alpha=1.0):
    """Per-fold Ridge stacker on OOF predictions of `cols`. Returns OOF preds."""
    pred = np.zeros(len(oof))
    fold = oof["fold"].values
    y = oof["y"].values
    X = oof[cols].values
    for k in range(N_SPLITS):
        train_mask = fold != k
        test_mask = fold == k
        ridge = Ridge(alpha=alpha)
        ridge.fit(X[train_mask], y[train_mask])
        pred[test_mask] = ridge.predict(X[test_mask])
    return pred


def stacked_nnls(oof, cols):
    """Non-negative least squares stacker (no intercept)."""
    from scipy.optimize import nnls
    pred = np.zeros(len(oof))
    fold = oof["fold"].values
    y = oof["y"].values
    X = oof[cols].values
    for k in range(N_SPLITS):
        train_mask = fold != k
        test_mask = fold == k
        w, _ = nnls(X[train_mask], y[train_mask])
        pred[test_mask] = X[test_mask] @ w
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    print(f"dataset: {df.shape}")

    print("\nBuilding OOF predictions ...")
    oof = build_oof(df)

    y = oof["y"].values
    fold = oof["fold"].values

    results = []

    def record(name, pred):
        per = per_fold_mae(pred, y, fold)
        overall = mae(pred, y)
        results.append({
            "model": name,
            "cv_mae": overall,
            "fold_mean": np.mean(per),
            "fold_std": np.std(per),
            "fold_min": np.min(per),
            "fold_max": np.max(per),
        })
        return per

    print("\nIndividual models (OOF MAE per fold):")
    for col, name in [("a1", "A1 closed form"), ("a2", "A2 ClosedFormModel"),
                      ("ebm", "EBM (R2 tuned)"), ("gam", "GAM (R2 tuned)")]:
        per = record(name, oof[col].values)
        print(f"  {name:25s}  overall={mae(oof[col].values, y):.3f}  "
              f"per-fold={[round(x, 3) for x in per]}")

    print("\nEnsembles (also evaluated OOF):")
    # Fixed-weight ensembles
    pred = fixed_weighted(oof, {"ebm": 0.7, "gam": 0.3})
    record("EBM+GAM 70/30 (fixed)", pred)
    print(f"  EBM+GAM 70/30 (fixed):                  {mae(pred, y):.3f}")

    pred = fixed_weighted(oof, {"ebm": 0.5, "gam": 0.5})
    record("EBM+GAM 50/50 (fixed)", pred)

    # Stacked Ridge ensembles
    for cols, name in [
        (["ebm", "gam"], "Stacked Ridge: EBM+GAM"),
        (["ebm", "gam", "a1"], "Stacked Ridge: EBM+GAM+A1"),
        (["ebm", "gam", "a2"], "Stacked Ridge: EBM+GAM+A2"),
        (["ebm", "gam", "a1", "a2"], "Stacked Ridge: EBM+GAM+A1+A2"),
        (["ebm", "a1"], "Stacked Ridge: EBM+A1"),
        (["ebm", "a2"], "Stacked Ridge: EBM+A2"),
    ]:
        pred = stacked_ridge(oof, cols)
        record(name, pred)
        print(f"  {name:40s}  {mae(pred, y):.3f}")

    # NNLS stacking — gives interpretable non-negative weights
    print("\nNon-negative LS weights (refit on full OOF for inspection):")
    from scipy.optimize import nnls
    for cols in [["ebm", "gam"], ["ebm", "gam", "a1"], ["ebm", "gam", "a2"],
                 ["ebm", "gam", "a1", "a2"]]:
        X = oof[cols].values
        w, _ = nnls(X, y)
        pred = X @ w
        results.append({
            "model": f"NNLS: {'+'.join(c.upper() for c in cols)}",
            "cv_mae": mae(pred, y),
            "fold_mean": np.mean(per_fold_mae(pred, y, fold)),
            "fold_std": np.std(per_fold_mae(pred, y, fold)),
            "fold_min": np.min(per_fold_mae(pred, y, fold)),
            "fold_max": np.max(per_fold_mae(pred, y, fold)),
        })
        print(f"  {'+'.join(cols):20s}  weights={dict(zip(cols, np.round(w, 3).tolist()))}  fit MAE={mae(pred, y):.3f}")

    # Also report how often A1/A2 would help — correlation with EBM+GAM residuals
    base = fixed_weighted(oof, {"ebm": 0.7, "gam": 0.3})
    base_resid = y - base
    print(f"\nResidual diagnostics for current ensemble (EBM+GAM 70/30):")
    print(f"  mean |resid| = {np.mean(np.abs(base_resid)):.3f}")
    for col, name in [("a1", "A1"), ("a2", "A2")]:
        r = np.corrcoef(oof[col].values - base, base_resid)[0, 1]
        print(f"  corr( {name} - ensemble , residual ) = {r:+.3f}")

    # Save table + plot
    results_df = pd.DataFrame(results).sort_values("cv_mae").reset_index(drop=True)
    results_df.to_csv(PLOTS / "cv_results.csv", index=False)
    print("\n" + "=" * 78)
    print("CV results (sorted):")
    print("=" * 78)
    print(results_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    rd = results_df.iloc[::-1]
    ax.barh(rd["model"], rd["cv_mae"], xerr=rd["fold_std"], color="steelblue", alpha=0.85)
    for i, v in enumerate(rd["cv_mae"]):
        ax.text(v + 0.04, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("5-fold CV MAE (lower is better)")
    ax.set_title("Out-of-sample MAE by model — error bars = fold std")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cv_results.png", dpi=130)
    plt.close(fig)

    # Per-fold MAE comparison line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, name, marker in [("a1", "A1 closed form", "o"),
                              ("a2", "A2 linear", "s"),
                              ("ebm", "EBM", "D"),
                              ("gam", "GAM", "^")]:
        per = per_fold_mae(oof[col].values, y, fold)
        ax.plot(range(N_SPLITS), per, marker=marker, label=name, lw=1.5)
    pred = fixed_weighted(oof, {"ebm": 0.7, "gam": 0.3})
    ax.plot(range(N_SPLITS), per_fold_mae(pred, y, fold),
            marker="*", ms=10, lw=2, color="red", label="EBM+GAM 70/30")
    pred = stacked_ridge(oof, ["ebm", "gam", "a2"])
    ax.plot(range(N_SPLITS), per_fold_mae(pred, y, fold),
            marker="x", ms=8, lw=2, color="black", label="Stacked EBM+GAM+A2")
    ax.set_xlabel("fold")
    ax.set_ylabel("MAE")
    ax.set_title("Per-fold MAE — does adding the formulas help?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cv_per_fold.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
