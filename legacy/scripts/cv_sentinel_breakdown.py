"""Same 5-fold CV as cv_ensemble_eval.py but breaks MAE down by sentinel status.

CLAUDE.md notes that x5 sentinels (x5=999.0) are the dominant error source: in
Round 2 the ensemble had non-sentinel MAE ~1.5 vs sentinel MAE ~9.4. This
script repeats the OOF computation and reports sentinel vs non-sentinel MAE
for every individual model and ensemble — answering whether A1's lead survives
once you separate the two regimes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cv_ensemble_eval import (
    ClosedFormModel, a1_predict, build_oof, ebm_preprocess, fit_ebm,
    fit_gam, gam_design, fixed_weighted, mae, stacked_ridge,
)
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "formulas"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


def main():
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    is_sent = (df["x5"] == SENTINEL).values
    print(f"dataset: {df.shape}")
    print(f"  sentinel rows (x5=999): {is_sent.sum()} ({100*is_sent.mean():.1f}%)")
    print(f"  non-sentinel rows:      {(~is_sent).sum()} ({100*(~is_sent).mean():.1f}%)")

    print("\nBuilding OOF predictions ...")
    oof = build_oof(df)
    y = oof["y"].values

    # Build all the same predictors used in cv_ensemble_eval
    base = fixed_weighted(oof, {"ebm": 0.7, "gam": 0.3})
    stk_egA1 = stacked_ridge(oof, ["ebm", "gam", "a1"])
    stk_egA2 = stacked_ridge(oof, ["ebm", "gam", "a2"])
    stk_egA1A2 = stacked_ridge(oof, ["ebm", "gam", "a1", "a2"])
    # NNLS over full OOF
    X_eg_a1 = oof[["ebm", "gam", "a1"]].values
    w_eg_a1, _ = nnls(X_eg_a1, y)
    nnls_eg_a1 = X_eg_a1 @ w_eg_a1

    rows = []
    def evaluate(name, pred):
        overall = mae(pred, y)
        sent_mae = mae(pred[is_sent], y[is_sent])
        nons_mae = mae(pred[~is_sent], y[~is_sent])
        # Worst per-row error rank
        worst10 = np.sort(np.abs(pred - y))[-10:]
        rows.append({
            "model": name,
            "overall": overall,
            "non_sentinel": nons_mae,
            "sentinel": sent_mae,
            "worst10_mean": float(worst10.mean()),
        })

    evaluate("A1 closed form", oof["a1"].values)
    evaluate("A2 ClosedFormModel", oof["a2"].values)
    evaluate("EBM (R2 tuned)", oof["ebm"].values)
    evaluate("GAM (R2 tuned)", oof["gam"].values)
    evaluate("EBM+GAM 70/30", base)
    evaluate("Stacked EBM+GAM+A1", stk_egA1)
    evaluate("Stacked EBM+GAM+A2", stk_egA2)
    evaluate("Stacked EBM+GAM+A1+A2", stk_egA1A2)
    evaluate("NNLS EBM+GAM+A1", nnls_eg_a1)

    out = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)
    print("\nMAE breakdown:")
    print(out.to_string(index=False))
    out.to_csv(PLOTS / "cv_sentinel_breakdown.csv", index=False)

    # Plot — grouped bars
    fig, ax = plt.subplots(figsize=(11, 6))
    order = list(out["model"])
    x = np.arange(len(order))
    width = 0.27
    ax.bar(x - width, out["overall"], width, label="overall (n=1500)")
    ax.bar(x,         out["non_sentinel"], width, label=f"non-sentinel (n={int((~is_sent).sum())})")
    ax.bar(x + width, out["sentinel"], width, label=f"sentinel (n={int(is_sent.sum())})", color="firebrick")
    for xi, v in zip(x - width, out["overall"]):  ax.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x,         out["non_sentinel"]): ax.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + width, out["sentinel"]): ax.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("CV MAE")
    ax.set_title("OOF MAE — overall vs sentinel/non-sentinel")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "cv_sentinel_breakdown.png", dpi=130)
    plt.close(fig)

    # Distribution of absolute errors per group
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cols = [("a1", "A1"), ("ebm", "EBM"), ("gam", "GAM")]
    for ax, mask, title in [
        (axes[0], ~is_sent, f"non-sentinel (n={int((~is_sent).sum())})"),
        (axes[1], is_sent, f"sentinel (n={int(is_sent.sum())})"),
    ]:
        data = [np.abs(oof[c].values[mask] - y[mask]) for c, _ in cols] + [
            np.abs(base[mask] - y[mask]),
            np.abs(stk_egA1[mask] - y[mask]),
        ]
        labels = [name for _, name in cols] + ["EBM+GAM 70/30", "Stk EBM+GAM+A1"]
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title(title)
        ax.set_ylabel("|error|")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("OOF |error| distributions by sentinel status", y=1.0)
    fig.tight_layout()
    fig.savefig(PLOTS / "cv_sentinel_error_dist.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
