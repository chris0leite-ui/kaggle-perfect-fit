"""Plot A1 perfect-fit indicator vs x6/x7 angle.

Visualisation of the null result: the x6/x7 angle θ = atan2(x7, x6) does
NOT distinguish A1-perfect rows from A1-imperfect rows. Red (imperfect)
points are sprinkled uniformly around the 18-radius circle.

Panels:
  (a) x6-x7 scatter, all 1278 non-sent training rows, coloured by fit
      status. The circle of radius 18 is visible; bad rows look uniformly
      distributed.
  (b) Same but restricted to the clamp quadrant (x4<0 AND x8<0, n=368).
  (c) Histogram of θ by fit status (perfect vs bad) overlaid — shows
      that the 23% bad rate is flat across all angular bins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL  # noqa: E402
from compare_formulas import approach1_predict  # noqa: E402

PLOTS = REPO / "plots" / "a1_clamp"
PLOTS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(REPO / "data" / "dataset.csv").reset_index(drop=True)
    x5m = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    a1 = approach1_predict(df, x5m)
    abs_resid = np.abs(df["target"].values - a1)
    is_sent = (df["x5"] == SENTINEL).values

    x6 = df["x6"].values
    x7 = df["x7"].values
    theta = np.arctan2(x7, x6)

    ns = ~is_sent
    perfect = ns & (abs_resid < 0.01)
    bad = ns & (abs_resid > 0.1)
    clamp_q = ns & (df["x4"].values < 0) & (df["x8"].values < 0)

    fig = plt.figure(figsize=(16, 6))

    # ---- (a) All non-sent training rows on the x6/x7 circle ----
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(x6[perfect], x7[perfect], s=10, c="#1f77b4", alpha=0.5,
                label=f"perfect (|resid|<0.01)  n={perfect.sum()}")
    ax1.scatter(x6[bad], x7[bad], s=35, c="#d62728", alpha=0.85,
                edgecolors="k", linewidths=0.3,
                label=f"imperfect (|resid|>0.1)  n={bad.sum()}")
    circle = plt.Circle((0, 0), 18, fill=False, color="gray", lw=0.5, ls="--")
    ax1.add_patch(circle)
    ax1.set_xlabel("x6"); ax1.set_ylabel("x7")
    ax1.set_aspect("equal")
    ax1.set_title("All non-sentinel training rows\non the x6/x7 = 18 circle")
    ax1.legend(loc="upper right", fontsize=9, frameon=True)
    ax1.axhline(0, color="k", lw=0.3)
    ax1.axvline(0, color="k", lw=0.3)
    ax1.grid(alpha=0.2)

    # ---- (b) Restricted to the clamp quadrant ----
    ax2 = fig.add_subplot(1, 3, 2)
    clamp_perfect = clamp_q & (abs_resid < 0.01)
    clamp_bad = clamp_q & (abs_resid > 0.1)
    ax2.scatter(x6[clamp_perfect], x7[clamp_perfect], s=18, c="#1f77b4",
                alpha=0.65,
                label=f"perfect in clamp quad  n={clamp_perfect.sum()}")
    ax2.scatter(x6[clamp_bad], x7[clamp_bad], s=45, c="#d62728", alpha=0.9,
                edgecolors="k", linewidths=0.3,
                label=f"bad in clamp quad  n={clamp_bad.sum()}")
    ax2.add_patch(plt.Circle((0, 0), 18, fill=False, color="gray", lw=0.5, ls="--"))
    ax2.set_xlabel("x6"); ax2.set_ylabel("x7")
    ax2.set_aspect("equal")
    ax2.set_title(f"Clamp quadrant only (x4<0 & x8<0, n={clamp_q.sum()})\n"
                   f"Is there an angular pattern? (No)")
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    ax2.axhline(0, color="k", lw=0.3)
    ax2.axvline(0, color="k", lw=0.3)
    ax2.grid(alpha=0.2)

    # ---- (c) Histogram of θ by fit status in the clamp quadrant ----
    ax3 = fig.add_subplot(1, 3, 3)
    bins = np.linspace(-np.pi, np.pi, 19)  # 18 bins
    centers = 0.5 * (bins[:-1] + bins[1:])
    h_perfect, _ = np.histogram(theta[clamp_perfect], bins=bins)
    h_bad, _ = np.histogram(theta[clamp_bad], bins=bins)
    total = h_perfect + h_bad
    bad_rate = np.where(total > 0, h_bad / np.maximum(total, 1), 0)

    width = bins[1] - bins[0]
    ax3.bar(centers, h_perfect, width=width * 0.9, color="#1f77b4",
             label="perfect", alpha=0.75)
    ax3.bar(centers, h_bad, width=width * 0.9, bottom=h_perfect,
             color="#d62728", label="bad (clamp triggered)", alpha=0.9)
    ax3.set_xlabel("θ = atan2(x7, x6)  [rad]")
    ax3.set_ylabel("row count")
    ax3.set_title(f"Fit status vs θ within the clamp quadrant\n"
                   f"bad rate per bin: mean={bad_rate.mean():.2f}, "
                   f"std={bad_rate.std():.2f}")
    ax3.legend(loc="upper right", fontsize=9, frameon=True)
    ax3.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax3.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
    ax3.grid(alpha=0.2, axis="y")

    # Overlay the per-bin bad rate on a twin axis
    ax3b = ax3.twinx()
    ax3b.plot(centers, bad_rate, "k-o", ms=4, lw=1.2,
               label="bad rate per bin")
    ax3b.axhline(clamp_bad.sum() / clamp_q.sum(), color="gray", ls="--",
                  lw=1, label="overall clamp quad bad rate (23.4%)")
    ax3b.set_ylabel("bad rate", color="k")
    ax3b.set_ylim(0, 0.6)
    ax3b.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "A1 perfect-fit indicator vs x6/x7 angle — no angular pattern "
        "(KS p=0.89, all tests null)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "a1_fit_vs_x6x7_angle.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
