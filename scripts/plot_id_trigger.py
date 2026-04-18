"""Visualise the id-based clamp trigger.

Single composite figure with five panels that should make the
id<100 trigger self-evident:

(A) A1 residual vs id, coloured by (sign(x4), sign(x8)) quadrant.
    Clamp rows should form a visible cloud at id < 100 in the
    red (x4<0, x8<0) quadrant and be absent elsewhere.

(B) Zoom on id ∈ [0, 200] — shows the clean cut-off at id = 100.

(C) Histogram of id for clamp vs non-clamp (in the x4<0, x8<0
    quadrant). Two completely disjoint distributions.

(D) Residual vs x8 for clamp rows with the  -15·x8 + 1 fit line.
    Demonstrates the correction shape.

(E) Quadrant scatter (x4 × x8) with is_clamp coloured.

One extra panel:

(F) |A1 residual| per id-bucket (log-y bar chart). Bucket 0-99 is
    the only one with non-zero residuals; everything else is 0 to
    machine precision.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "clamp_search"
OUT.mkdir(parents=True, exist_ok=True)
SENTINEL = 999.0


def load() -> pd.DataFrame:
    train = pd.read_csv(DATA / "dataset.csv")
    x5m = train.loc[train["x5"] != SENTINEL, "x5"].median()
    x5 = train["x5"].where(train["x5"] != SENTINEL, x5m).values
    is_zar = (train["City"] == "Zaragoza").astype(float).values
    a1 = (
        -100 * train["x1"] ** 2
        + 10 * np.cos(5 * np.pi * train["x2"])
        + 15 * train["x4"]
        - 8 * x5
        + 15 * train["x8"]
        - 4 * train["x9"]
        + train["x10"] * train["x11"]
        - 25 * is_zar
        + 20 * (train["x4"] > 0)
        + 92.5
    )
    train["a1_resid"] = train["target"].values - a1.values
    train["is_sent"] = (train["x5"] == SENTINEL).astype(bool)
    train["quadrant"] = np.select(
        [(train["x4"] < 0) & (train["x8"] < 0),
         (train["x4"] < 0) & (train["x8"] > 0),
         (train["x4"] > 0) & (train["x8"] < 0),
         (train["x4"] > 0) & (train["x8"] > 0)],
        ["x4<0,x8<0", "x4<0,x8>0", "x4>0,x8<0", "x4>0,x8>0"],
        default="other"
    )
    return train


COLORS = {
    "x4<0,x8<0": "#d62728",   # red — the clamp quadrant
    "x4<0,x8>0": "#1f77b4",
    "x4>0,x8<0": "#2ca02c",
    "x4>0,x8>0": "#9467bd",
}


def plot(train: pd.DataFrame) -> None:
    tr = train[~train["is_sent"]].copy()
    is_clamp = tr["a1_resid"].abs() > 1.0

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    axA = fig.add_subplot(gs[0, :2])
    axB = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])
    axF = fig.add_subplot(gs[2, :])

    # ---- Panel A: residual vs id (full range) ------------------------------
    for q, c in COLORS.items():
        sub = tr[tr["quadrant"] == q]
        axA.scatter(sub["id"], sub["a1_resid"], s=8, color=c, alpha=0.6, label=q)
    axA.axvline(100, color="black", ls="--", lw=1)
    axA.axhline(0, color="gray", lw=0.5)
    axA.set_xlabel("id"); axA.set_ylabel("A1 residual")
    axA.set_title("(A) A1 residual vs id — clamp cloud sits in x4<0,x8<0 at id<100")
    axA.legend(fontsize=8, loc="upper right", ncols=2)

    # ---- Panel B: zoom id 0-200 --------------------------------------------
    m = tr["id"] < 200
    for q, c in COLORS.items():
        sub = tr[m & (tr["quadrant"] == q)]
        axB.scatter(sub["id"], sub["a1_resid"], s=14, color=c, alpha=0.75)
    axB.axvline(100, color="black", ls="--", lw=1.2, label="id = 100")
    axB.axhline(0, color="gray", lw=0.5)
    axB.set_xlabel("id"); axB.set_ylabel("A1 residual")
    axB.set_title("(B) id ∈ [0, 200] zoom")
    axB.legend(fontsize=8)

    # ---- Panel C: id histogram, clamp vs non-clamp (quadrant only) --------
    q = tr[tr["quadrant"] == "x4<0,x8<0"]
    q_clamp = q[q["a1_resid"].abs() > 1.0]
    q_noclamp = q[q["a1_resid"].abs() <= 1.0]
    bins = np.arange(0, 1500 + 50, 50)
    axC.hist(q_noclamp["id"], bins=bins, alpha=0.6, color="#1f77b4",
             label=f"non-clamp (n={len(q_noclamp)})")
    axC.hist(q_clamp["id"], bins=bins, alpha=0.9, color="#d62728",
             label=f"clamp (n={len(q_clamp)})")
    axC.axvline(100, color="black", ls="--", lw=1)
    axC.set_xlabel("id"); axC.set_ylabel("count")
    axC.set_title("(C) id histogram in x4<0,x8<0 quadrant")
    axC.legend(fontsize=8)

    # ---- Panel D: residual vs x8 with -15x8+1 fit -------------------------
    clamp = tr[(tr["quadrant"] == "x4<0,x8<0") & (tr["a1_resid"].abs() > 1.0)]
    xs = np.linspace(clamp["x8"].min(), clamp["x8"].max(), 80)
    axD.scatter(clamp["x8"], clamp["a1_resid"], s=16, color="#d62728",
                label=f"clamp rows (n={len(clamp)})")
    axD.plot(xs, -15 * xs + 1, color="black", ls="--", lw=1.5,
             label="fit: −15·x8 + 1")
    axD.set_xlabel("x8"); axD.set_ylabel("A1 residual")
    axD.set_title("(D) Correction shape on clamp rows")
    axD.legend(fontsize=8)

    # ---- Panel E: (x4, x8) scatter with is_clamp coloured ------------------
    axE.scatter(tr[~is_clamp]["x4"], tr[~is_clamp]["x8"], s=4, color="lightgray",
                alpha=0.5, label="non-clamp")
    axE.scatter(tr[is_clamp]["x4"], tr[is_clamp]["x8"], s=14, color="#d62728",
                alpha=0.9, label=f"clamp (n={is_clamp.sum()})")
    axE.axhline(0, color="gray", lw=0.5); axE.axvline(0, color="gray", lw=0.5)
    axE.set_xlabel("x4"); axE.set_ylabel("x8")
    axE.set_title("(E) Clamp rows live in x4<0,x8<0 only")
    axE.legend(fontsize=8)

    # ---- Panel F: max |residual| per id bucket -----------------------------
    buckets = np.arange(0, 1500 + 100, 100)
    centres = (buckets[:-1] + buckets[1:]) / 2
    max_abs, n_clamp = [], []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        sub = tr[(tr["id"] >= lo) & (tr["id"] < hi)]
        max_abs.append(sub["a1_resid"].abs().max() if len(sub) else 0)
        n_clamp.append(int((sub["a1_resid"].abs() > 1.0).sum()))
    colours = ["#d62728" if n > 0 else "#cccccc" for n in n_clamp]
    axF.bar(centres, max_abs, width=80, color=colours, edgecolor="black", lw=0.3)
    for x, c in zip(centres, n_clamp):
        if c:
            axF.text(x, 1.2, f"n={c}", ha="center", fontsize=8, color="#d62728")
    axF.set_xlabel("id bucket centre (width = 100)")
    axF.set_ylabel("max |A1 residual|")
    axF.set_title("(F) max |A1 residual| per id bucket — only bucket 0 has non-zero values")
    axF.axvline(100, color="black", ls="--", lw=1)

    fig.suptitle("A1 residuals: the clamp trigger is id < 100 (training-data artefact)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "id_trigger.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'id_trigger.png'}")


if __name__ == "__main__":
    train = load()
    plot(train)
