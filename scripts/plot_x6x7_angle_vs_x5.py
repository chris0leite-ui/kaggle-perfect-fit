"""Plot the x6-x7 angle (atan2) against raw x5.

x6 and x7 lie exactly on a circle of radius 18 (verified: std of sqrt(x6^2+x7^2)
is 0). The only meaningful information they carry is the angle theta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots" / "formulas"
PLOTS.mkdir(parents=True, exist_ok=True)

SENTINEL = 999.0


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv")
    theta = np.arctan2(df["x7"].values, df["x6"].values)
    sent = (df["x5"] == SENTINEL).values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: non-sentinel x5 (where x5 is observed)
    ax = axes[0]
    ax.scatter(df.loc[~sent, "x5"], theta[~sent], s=8, alpha=0.5,
               color="steelblue", label=f"non-sentinel (n={(~sent).sum()})")
    ax.set_xlabel("x5 (observed)")
    ax.set_ylabel("theta = atan2(x7, x6)  [radians]")
    ax.set_title(f"x6/x7 angle vs raw x5  (Pearson r = "
                 f"{np.corrcoef(df.loc[~sent, 'x5'], theta[~sent])[0,1]:+.4f})")
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # Right: sentinel rows (x5 = 999) — collapsed to a single column
    ax = axes[1]
    ax.scatter(np.zeros(sent.sum()), theta[sent], s=8, alpha=0.5,
               color="firebrick", label=f"sentinel x5=999 (n={sent.sum()})")
    ax.set_xticks([0])
    ax.set_xticklabels(["999"])
    ax.set_xlabel("x5 (sentinel)")
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_title("sentinel rows")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(axes[0].get_ylim())

    fig.suptitle("Angle of (x6, x7) on the radius-18 circle vs raw x5", y=1.0)
    fig.tight_layout()
    fig.savefig(PLOTS / "x6x7_angle_vs_x5.png", dpi=130)
    plt.close(fig)
    print(f"wrote {PLOTS / 'x6x7_angle_vs_x5.png'}")


if __name__ == "__main__":
    main()
