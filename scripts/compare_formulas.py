"""Compare the three reverse-engineered formulas on the competition test set.

Approach 1 ("reverse-engineer-equation" branch): hard-coded closed form with a
discrete step at x4=0.

Approach 2 ("review-backwards-engineering" branch): ClosedFormModel — same
functional terms as approach 1 (cos-based x1, cos on x2, x10*x11), but x4
enters purely linearly and coefficients are fit by least squares.

Approach 3 (current Round 2 branch): trained EBM on the same feature set the
Round 2 pipeline uses — purely data-driven with no hand-picked functional form.

All three are fit / parameterised on the entire labelled dataset (dataset.csv,
1500 rows) then scored on test.csv. Predictions and plots are written to
submissions/ and plots/formulas/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT_PLOTS = REPO / "plots" / "formulas"
OUT_SUBS = REPO / "submissions"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_SUBS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


# ---------------------------------------------------------------------------
# Approach 1: hard-coded closed form from `reverse-engineer-equation`
# ---------------------------------------------------------------------------

def approach1_predict(df: pd.DataFrame, x5_median: float) -> np.ndarray:
    """target = -100*x1^2 + 10*cos(5*pi*x2) + 15*x4 - 8*x5 + 15*x8
              - 4*x9 + x10*x11 - 25*is_zaragoza + 20*1(x4>0) + 92.5.

    x5 sentinels are imputed with the training median to keep the contribution
    finite; the rest of the formula is applied exactly as published.
    """
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values
    is_zar = (df["City"] == "Zaragoza").astype(float).values
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
# Approach 2: ClosedFormModel from `review-backwards-engineering`
# ---------------------------------------------------------------------------

class ClosedFormModel:
    """Linear least squares on the hand-picked basis functions."""

    COEF_NAMES = [
        "City", "x4", "x8", "x5", "x10*x11",
        "cos(pi*x1)", "cos(5pi*x2)", "x9_resid", "x5_is_sentinel", "const",
    ]

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        mask = df["x5"] == SENTINEL
        x5 = df["x5"].where(~mask, self.x5_median_).values
        is_sent = mask.astype(float).values
        x9_resid = df["x9"].values - (self.x9_slope_ * df["x4"].values + self.x9_intercept_)
        city = df["City"].map({"Zaragoza": 1.0, "Albacete": 0.0}).values
        return np.column_stack([
            city,
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

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "ClosedFormModel":
        mask = df["x5"] == SENTINEL
        self.x5_median_ = float(df.loc[~mask, "x5"].median())
        A = np.column_stack([df["x4"].values, np.ones(len(df))])
        slope_int, *_ = np.linalg.lstsq(A, df["x9"].values, rcond=None)
        self.x9_slope_, self.x9_intercept_ = float(slope_int[0]), float(slope_int[1])
        M = self._design(df)
        self.coef_, *_ = np.linalg.lstsq(M, y, rcond=None)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._design(df) @ self.coef_


# ---------------------------------------------------------------------------
# Approach 3: EBM on the Round 2 feature set
# ---------------------------------------------------------------------------

def approach3_fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          x5_median: float) -> np.ndarray:
    from interpret.glassbox import ExplainableBoostingRegressor

    features = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]

    def preprocess(df):
        out = df[features].copy()
        out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
        out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
        out["city"] = (df["City"] == "Zaragoza").astype(float)
        return out

    X_tr = preprocess(train_df)
    X_te = preprocess(test_df)
    ebm = ExplainableBoostingRegressor(
        interactions=10,
        max_rounds=2000,
        min_samples_leaf=10,
        max_bins=128,
        random_state=42,
    )
    ebm.fit(X_tr, train_df["target"].values)
    return ebm.predict(X_te)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train: {train.shape}, test: {test.shape}")

    x5_median = float(train.loc[train["x5"] != SENTINEL, "x5"].median())
    print(f"x5 median (non-sentinel): {x5_median:.4f}")

    # Approach 1 — closed form
    y1_train = approach1_predict(train, x5_median)
    y1_test = approach1_predict(test, x5_median)
    mae1_train = np.mean(np.abs(y1_train - train["target"].values))
    print(f"Approach 1 (closed form) — train MAE: {mae1_train:.3f}")

    # Approach 2 — ClosedFormModel
    m2 = ClosedFormModel().fit(train, train["target"].values)
    y2_train = m2.predict(train)
    y2_test = m2.predict(test)
    mae2_train = np.mean(np.abs(y2_train - train["target"].values))
    print(f"Approach 2 (ClosedFormModel) — train MAE: {mae2_train:.3f}")
    for name, c in zip(m2.COEF_NAMES, m2.coef_):
        print(f"    {name:>16s}: {c:+.3f}")

    # Approach 3 — EBM
    print("Approach 3 — fitting EBM ...")
    y3_test = approach3_fit_predict(train, test, x5_median)
    # also predict on train for consistency
    from interpret.glassbox import ExplainableBoostingRegressor
    features = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]

    def preprocess(df):
        out = df[features].copy()
        out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
        out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
        out["city"] = (df["City"] == "Zaragoza").astype(float)
        return out

    ebm = ExplainableBoostingRegressor(
        interactions=10, max_rounds=2000, min_samples_leaf=10,
        max_bins=128, random_state=42,
    )
    ebm.fit(preprocess(train), train["target"].values)
    y3_train = ebm.predict(preprocess(train))
    y3_test = ebm.predict(preprocess(test))
    mae3_train = np.mean(np.abs(y3_train - train["target"].values))
    print(f"Approach 3 (EBM) — train MAE: {mae3_train:.3f}")

    # Persist predictions
    preds = pd.DataFrame({
        "id": test["id"],
        "x4": test["x4"],
        "City": test["City"],
        "approach1_closedform": y1_test,
        "approach2_clsfm_linear": y2_test,
        "approach3_ebm": y3_test,
    })
    preds.to_csv(OUT_SUBS / "formula_comparison_predictions.csv", index=False)
    print(f"wrote {OUT_SUBS / 'formula_comparison_predictions.csv'}")

    # Individual submissions (same schema as sample_submission.csv)
    for col, fname in [
        ("approach1_closedform", "submission_approach1.csv"),
        ("approach2_clsfm_linear", "submission_approach2.csv"),
        ("approach3_ebm", "submission_approach3.csv"),
    ]:
        preds[["id", col]].rename(columns={col: "target"}).to_csv(OUT_SUBS / fname, index=False)

    # Visualisations
    make_plots(preds, train, x5_median, m2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GAP_LO, GAP_HI = -0.167, 0.167


def make_plots(preds: pd.DataFrame, train: pd.DataFrame, x5_median: float,
               m2: ClosedFormModel) -> None:
    n_gap = ((preds.x4 > GAP_LO) & (preds.x4 < GAP_HI)).sum()
    print(f"Test rows in x4 gap [{GAP_LO}, {GAP_HI}]: {n_gap} ({100*n_gap/len(preds):.1f}%)")

    # Plot 1: predictions vs x4, stacked per approach, coloured by city
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    cols = ["approach1_closedform", "approach2_clsfm_linear", "approach3_ebm"]
    titles = [
        "Approach 1 — closed form with step 20·𝟙(x4>0)",
        "Approach 2 — ClosedFormModel (pure linear x4)",
        "Approach 3 — EBM (nonparametric x4)",
    ]
    for ax, col, title in zip(axes, cols, titles):
        for city, color in [("Albacete", "#1f77b4"), ("Zaragoza", "#d62728")]:
            sub = preds[preds.City == city]
            ax.scatter(sub.x4, sub[col], s=8, alpha=0.45, color=color, label=city)
        ax.axvspan(GAP_LO, GAP_HI, color="gold", alpha=0.18, label="training gap")
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("predicted target")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("x4")
    fig.suptitle("Predictions on test.csv — coloured by City; gold band = training gap", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "preds_vs_x4.png", dpi=130)
    plt.close(fig)

    # Plot 2: difference Approach1 - Approach2 as a function of x4
    diff_12 = preds["approach1_closedform"] - preds["approach2_clsfm_linear"]
    diff_13 = preds["approach1_closedform"] - preds["approach3_ebm"]
    diff_23 = preds["approach2_clsfm_linear"] - preds["approach3_ebm"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(preds.x4, diff_12, s=8, alpha=0.5, label="A1 − A2 (step vs linear)")
    ax.scatter(preds.x4, diff_13, s=8, alpha=0.5, label="A1 − A3 (step vs EBM)")
    ax.scatter(preds.x4, diff_23, s=8, alpha=0.5, label="A2 − A3 (linear vs EBM)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(GAP_LO, GAP_HI, color="gold", alpha=0.15, label="training gap")
    ax.set_xlabel("x4")
    ax.set_ylabel("prediction difference")
    ax.set_title("Pairwise prediction differences vs x4")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "pairwise_diffs.png", dpi=130)
    plt.close(fig)

    # Plot 3: isolated x4 contribution per approach (hold others at sample mean)
    x4_grid = np.linspace(-0.5, 0.5, 401)
    baseline = train.iloc[:500].copy().reset_index(drop=True)
    baseline = baseline.loc[baseline.index.repeat(1)].reset_index(drop=True)
    # build a minimal "average" row and vary x4
    means = {
        "x1": train["x1"].mean(),
        "x2": train["x2"].mean(),
        "x5": x5_median,
        "x8": train["x8"].mean(),
        "x9": train["x9"].mean(),
        "x10": train["x10"].mean(),
        "x11": train["x11"].mean(),
    }
    card = pd.DataFrame({**{k: np.full_like(x4_grid, v) for k, v in means.items()},
                         "x4": x4_grid,
                         "City": "Albacete",
                         "Country": "Spain"})
    card_zar = card.assign(City="Zaragoza")

    a1_alb = approach1_predict(card, x5_median)
    a1_zar = approach1_predict(card_zar, x5_median)
    a2_alb = m2.predict(card)
    a2_zar = m2.predict(card_zar)
    # reuse fitted EBM — refit cheaply using cached training design
    from interpret.glassbox import ExplainableBoostingRegressor

    features = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]

    def preprocess(df):
        out = df[features].copy()
        out["x5"] = out["x5"].where(out["x5"] != SENTINEL, x5_median)
        out["x5_is_sentinel"] = (df["x5"] == SENTINEL).astype(float)
        out["city"] = (df["City"] == "Zaragoza").astype(float)
        return out

    ebm = ExplainableBoostingRegressor(
        interactions=10, max_rounds=2000, min_samples_leaf=10, max_bins=128, random_state=42,
    )
    ebm.fit(preprocess(train), train["target"].values)
    a3_alb = ebm.predict(preprocess(card))
    a3_zar = ebm.predict(preprocess(card_zar))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, a1, a2, a3, title in [
        (axes[0], a1_alb, a2_alb, a3_alb, "Albacete (other vars at mean)"),
        (axes[1], a1_zar, a2_zar, a3_zar, "Zaragoza (other vars at mean)"),
    ]:
        ax.plot(x4_grid, a1, label="A1 closed form", lw=2)
        ax.plot(x4_grid, a2, label="A2 linear", lw=2)
        ax.plot(x4_grid, a3, label="A3 EBM", lw=2)
        ax.axvspan(GAP_LO, GAP_HI, color="gold", alpha=0.18, label="training gap")
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel("x4")
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("predicted target")
    fig.suptitle("Marginal effect of x4 (other features held at train mean / x5 median)", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "x4_marginal_curves.png", dpi=130)
    plt.close(fig)

    # Plot 4: violin of predictions split by gap vs non-gap
    inside = (preds.x4 > GAP_LO) & (preds.x4 < GAP_HI)
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for col, tag in zip(cols, ["A1", "A2", "A3"]):
        data.append(preds.loc[~inside, col].values)
        labels.append(f"{tag} out")
        data.append(preds.loc[inside, col].values)
        labels.append(f"{tag} IN gap")
    parts = ax.violinplot(data, showmeans=True, showmedians=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("predicted target")
    ax.set_title("Prediction distribution: outside vs inside training gap")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "gap_vs_nongap_distribution.png", dpi=130)
    plt.close(fig)

    # Summary stats
    inside_mask = inside
    summary = pd.DataFrame({
        "approach": ["A1 closed form", "A2 linear", "A3 EBM"],
        "test_mean": [preds[c].mean() for c in cols],
        "gap_mean": [preds.loc[inside_mask, c].mean() for c in cols],
        "gap_min": [preds.loc[inside_mask, c].min() for c in cols],
        "gap_max": [preds.loc[inside_mask, c].max() for c in cols],
    })
    print("\nSummary of predictions inside the training gap:")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_PLOTS / "summary.csv", index=False)

    pair_summary = pd.DataFrame({
        "x4 bin": ["x4 < -0.167", "-0.167 <= x4 <= 0", "0 < x4 <= 0.167", "x4 > 0.167"],
        "n": [
            int((preds.x4 < GAP_LO).sum()),
            int(((preds.x4 >= GAP_LO) & (preds.x4 <= 0)).sum()),
            int(((preds.x4 > 0) & (preds.x4 <= GAP_HI)).sum()),
            int((preds.x4 > GAP_HI).sum()),
        ],
        "mean |A1-A2|": [
            np.mean(np.abs(diff_12[preds.x4 < GAP_LO])),
            np.mean(np.abs(diff_12[(preds.x4 >= GAP_LO) & (preds.x4 <= 0)])),
            np.mean(np.abs(diff_12[(preds.x4 > 0) & (preds.x4 <= GAP_HI)])),
            np.mean(np.abs(diff_12[preds.x4 > GAP_HI])),
        ],
        "mean |A1-A3|": [
            np.mean(np.abs(diff_13[preds.x4 < GAP_LO])),
            np.mean(np.abs(diff_13[(preds.x4 >= GAP_LO) & (preds.x4 <= 0)])),
            np.mean(np.abs(diff_13[(preds.x4 > 0) & (preds.x4 <= GAP_HI)])),
            np.mean(np.abs(diff_13[preds.x4 > GAP_HI])),
        ],
        "mean |A2-A3|": [
            np.mean(np.abs(diff_23[preds.x4 < GAP_LO])),
            np.mean(np.abs(diff_23[(preds.x4 >= GAP_LO) & (preds.x4 <= 0)])),
            np.mean(np.abs(diff_23[(preds.x4 > 0) & (preds.x4 <= GAP_HI)])),
            np.mean(np.abs(diff_23[preds.x4 > GAP_HI])),
        ],
    })
    print("\nMean absolute pairwise differences by x4 region:")
    print(pair_summary.to_string(index=False))
    pair_summary.to_csv(OUT_PLOTS / "pairwise_summary.csv", index=False)


if __name__ == "__main__":
    main()
