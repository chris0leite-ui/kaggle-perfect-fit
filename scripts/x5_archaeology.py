"""x5 archaeology — three investigations using the id/A1 insight.

(1) Back-solve x5 for every sentinel row: x5_true = (A1_body - target) / 8
    using the EXACT A1 formula on rows id>=100. Validate on non-sentinels.

(2) Does id predict sentinel status? id-range, id mod k, id-parity.
    If yes, sentinel selection is a hand-crafted rule, not MCAR.

(3) Does x5 have sequential / RNG structure? Plot x5 vs id, compute
    autocorrelation of x5 at lag 1..50, FFT, and test whether
    x5[i] = f(i) for a simple closed-form f (linear, trig, LCG).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, uniform, pearsonr
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "x5_archaeology"
OUT.mkdir(parents=True, exist_ok=True)
SENTINEL = 999.0


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "dataset.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["is_sent"] = (df["x5"] == SENTINEL).astype(bool)
        df["city_code"] = (df["City"] == "Zaragoza").astype(float)
        df["x1sq"] = df["x1"] ** 2
        df["cos5pi"] = np.cos(5 * np.pi * df["x2"])
        df["x10x11"] = df["x10"] * df["x11"]
    return train, test


def a1_body_no_x5(df: pd.DataFrame) -> np.ndarray:
    """A1 without the -8·x5 term. For back-solving: target = body - 8·x5."""
    return (
        -100 * df["x1sq"]
        + 10 * df["cos5pi"]
        + 15 * df["x4"]
        + 15 * df["x8"]
        - 4 * df["x9"]
        + df["x10x11"]
        - 25 * df["city_code"]
        + 20 * (df["x4"] > 0)
        + 92.5
    ).values


def back_solve_x5(df: pd.DataFrame) -> np.ndarray:
    """x5_true = (A1_body - target) / 8. Valid only where A1 holds exactly."""
    body = a1_body_no_x5(df)
    return (body - df["target"].values) / 8.0


def main() -> None:
    train, test = load()
    print("=" * 72)
    print("(1) Back-solve x5 for training sentinels")
    print("=" * 72)

    x5_solved = back_solve_x5(train)
    train["x5_solved"] = x5_solved

    # Validate on non-sentinel, id >= 100 (where A1 is exact)
    clean = train[(~train["is_sent"]) & (train["id"] >= 100)]
    err_clean = clean["x5_solved"].values - clean["x5"].values
    print(f"Non-sentinel id>=100 rows: n={len(clean)}")
    print(f"  mean(x5_solved - x5) = {err_clean.mean():+.4e}")
    print(f"  max |err|            = {np.abs(err_clean).max():.4e}")
    print(f"  std(err)             = {err_clean.std():.4e}")
    n_within = int((np.abs(err_clean) < 1e-3).sum())
    print(f"  |err| < 1e-3         = {n_within}/{len(clean)} ({100*n_within/len(clean):.1f}%)")

    # Sentinel rows: back-solved x5 values
    sent = train[train["is_sent"]].copy()
    sent_solved = sent["x5_solved"].values
    # Keep only id >= 100 sentinels (clean inversion)
    sent_clean = sent[sent["id"] >= 100]
    print(f"\nSentinel rows id>=100: n={len(sent_clean)}")
    print(f"  back-solved x5: min={sent_clean['x5_solved'].min():.3f}  "
          f"max={sent_clean['x5_solved'].max():.3f}")
    print(f"  mean={sent_clean['x5_solved'].mean():.3f}  "
          f"median={sent_clean['x5_solved'].median():.3f}  "
          f"std={sent_clean['x5_solved'].std():.3f}")
    # Is it Uniform(7,12)?
    ks_stat, ks_p = kstest((sent_clean["x5_solved"] - 7) / 5, "uniform")
    print(f"  KS test vs Uniform(7,12):  D={ks_stat:.3f}  p={ks_p:.3f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(2) Does id predict sentinel status?")
    print("=" * 72)

    # Id range buckets on train
    bins = np.arange(0, 1550, 100)
    rate = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = train[(train["id"] >= lo) & (train["id"] < hi)]
        rate.append((lo, hi, len(sub), int(sub["is_sent"].sum()), sub["is_sent"].mean()))
    print(f"{'id range':<16s} {'n':>5s} {'sentinels':>10s} {'rate':>7s}")
    for lo, hi, n, s, r in rate:
        print(f"  [{lo:4d},{hi:4d}) {n:>6d} {s:>10d} {r:>7.3f}")

    # Id modular patterns: id mod k for k = 2..10
    print("\nSentinel rate by id mod k:")
    for k in [2, 3, 5, 7, 10, 100]:
        rates = []
        for r in range(k):
            sub = train[train["id"] % k == r]
            rates.append((r, len(sub), sub["is_sent"].mean()))
        m = [f"{r}:{v:.3f}" for r, _, v in rates]
        print(f"  mod {k}: " + " ".join(m[:10]) + ("..." if k > 10 else ""))

    # Test side
    print("\nTest sentinel rates:")
    for lo, hi in zip(np.arange(1500, 3100, 200), np.arange(1700, 3100, 200)):
        sub = test[(test["id"] >= lo) & (test["id"] < hi)]
        print(f"  id in [{lo},{hi}): n={len(sub)}  sent rate={sub['is_sent'].mean():.3f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("(3) Does x5 have deterministic structure in id?")
    print("=" * 72)

    non_sent = train[(~train["is_sent"]) & (train["id"] >= 100)].sort_values("id")
    x5_arr = non_sent["x5"].values
    id_arr = non_sent["id"].values

    # Autocorrelation of x5 ordered by id (at lag 1..30)
    x5_centered = x5_arr - x5_arr.mean()
    denom = np.dot(x5_centered, x5_centered)
    acf = []
    for k in range(1, 31):
        num = np.dot(x5_centered[:-k], x5_centered[k:])
        acf.append(num / denom)
    print(f"Autocorrelation lags 1-30 on sorted-by-id x5:")
    print("  " + "  ".join(f"ρ({k})={a:+.3f}" for k, a in enumerate(acf[:10], 1)))
    print(f"  max |ρ| over lag 1-30 = {max(abs(a) for a in acf):.3f}")

    # Linear fit of x5 on id and id/1500
    m = LinearRegression().fit(non_sent[["id"]], x5_arr)
    r2_lin = 1 - np.var(x5_arr - m.predict(non_sent[["id"]])) / np.var(x5_arr)
    print(f"\nLinear x5 ~ id:  slope={m.coef_[0]:+.4e}  r²={r2_lin:.4f}")

    # Correlation with various id transforms
    print("\nPearson r(x5, f(id)) for various transforms:")
    transforms = {
        "id": id_arr,
        "id / 1500": id_arr / 1500.0,
        "sin(2π·id / 1500)": np.sin(2 * np.pi * id_arr / 1500),
        "cos(2π·id / 1500)": np.cos(2 * np.pi * id_arr / 1500),
        "sin(2π·id·φ)": np.sin(2 * np.pi * id_arr * (np.sqrt(5) - 1) / 2),
        "id mod 7": id_arr % 7,
        "id mod 5": id_arr % 5,
        "id mod 11": id_arr % 11,
        "(id * 1664525 + 1013904223) mod 2^32 / 2^32": ((id_arr.astype(np.int64) * 1664525 + 1013904223) % (2**32)) / 2**32,
    }
    for name, v in transforms.items():
        r, p = pearsonr(x5_arr, v)
        print(f"  {name:<50s}  r={r:+.4f}  p={p:.3g}")

    # FFT
    centred = x5_arr - x5_arr.mean()
    fft = np.fft.rfft(centred)
    freqs = np.fft.rfftfreq(len(centred))
    power = np.abs(fft) ** 2
    top_k = np.argsort(power[1:])[::-1][:5] + 1  # skip DC
    print(f"\nTop-5 FFT frequencies (cycles per row):")
    for k in top_k:
        print(f"  freq={freqs[k]:.4f}  power={power[k]:.2f}  period≈{1/freqs[k]:.1f}")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.scatter(train.loc[~train["is_sent"], "id"],
               train.loc[~train["is_sent"], "x5"], s=4, alpha=0.4, label="observed")
    ax.scatter(sent_clean["id"], sent_clean["x5_solved"], s=10, color="red", alpha=0.8,
               label="back-solved (sentinel, id>=100)")
    ax.axhline(7, color="gray", ls="--", lw=0.5)
    ax.axhline(12, color="gray", ls="--", lw=0.5)
    ax.axvline(100, color="black", ls="--", lw=0.5)
    ax.set_xlabel("id"); ax.set_ylabel("x5")
    ax.set_title("(a) x5 vs id (observed + back-solved sentinels)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    rates_train = [train[(train["id"] >= lo) & (train["id"] < hi)]["is_sent"].mean()
                   for lo, hi in zip(bins[:-1], bins[1:])]
    ax.bar(bins[:-1], rates_train, width=100, align="edge", color="#1f77b4",
           edgecolor="black", lw=0.3, label="train")
    test_bins = np.arange(1500, 3050, 100)
    rates_test = [test[(test["id"] >= lo) & (test["id"] < hi)]["is_sent"].mean()
                  for lo, hi in zip(test_bins[:-1], test_bins[1:])]
    ax.bar(test_bins[:-1], rates_test, width=100, align="edge", color="#ff7f0e",
           edgecolor="black", lw=0.3, alpha=0.8, label="test")
    ax.axhline(train["is_sent"].mean(), color="red", ls="--", lw=0.8,
               label=f"overall mean = {train['is_sent'].mean():.3f}")
    ax.set_xlabel("id"); ax.set_ylabel("sentinel rate")
    ax.set_title("(b) x5 sentinel rate per id-100 bucket")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(range(1, 31), acf, marker="o", ms=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1.96 / np.sqrt(len(x5_arr)), color="red", ls="--", lw=0.5,
               label=f"95% CI = ±{1.96/np.sqrt(len(x5_arr)):.3f}")
    ax.axhline(-1.96 / np.sqrt(len(x5_arr)), color="red", ls="--", lw=0.5)
    ax.set_xlabel("lag"); ax.set_ylabel("ACF of x5 (sorted by id)")
    ax.set_title("(c) Autocorrelation of x5")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(sent_clean["x5_solved"], bins=30, alpha=0.7, color="#d62728",
            density=True, label="back-solved sentinel")
    ax.hist(train.loc[~train["is_sent"] & (train["id"] >= 100), "x5"], bins=30,
            alpha=0.5, color="#1f77b4", density=True, label="observed non-sentinel")
    ax.plot([7, 12], [0.2, 0.2], color="black", lw=1.5, label="Uniform(7,12) density")
    ax.set_xlabel("x5"); ax.set_ylabel("density")
    ax.set_title("(d) Back-solved vs observed x5 distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(freqs[1:200], power[1:200])
    ax.set_xlabel("frequency (cycles per row)"); ax.set_ylabel("FFT power")
    ax.set_title("(e) FFT power spectrum of x5")
    ax.set_yscale("log")

    ax = axes[1, 2]
    # Plot x5 vs id with a clean view of possible determinism
    order = np.argsort(id_arr)
    ax.plot(id_arr[order][:200], x5_arr[order][:200], marker=".", ms=4, lw=0.5)
    ax.set_xlabel("id"); ax.set_ylabel("x5")
    ax.set_title("(f) x5 vs id, first 200 non-sentinel rows (for eyeball pattern)")

    fig.suptitle("x5 archaeology: back-solve + id-structure + sequence tests",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "x5_archaeology.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {OUT / 'x5_archaeology.png'}")

    # Save back-solved values
    pd.DataFrame({
        "id": train["id"],
        "x5_observed": train["x5"],
        "is_sent": train["is_sent"],
        "x5_solved": train["x5_solved"],
    }).to_csv(OUT / "x5_solved.csv", index=False)


if __name__ == "__main__":
    main()
