"""Clamp archaeology v2 — AND-rule search.

v1 established:
  correction on 86 clamp rows: residual ≈ -15·x8 + 1  (R²=0.994)
  x8 alone: sens 94% but spec only 44%
  no simple 1- or 2-feature threshold reaches sens/spec > 0.9

Hypothesis: the trigger is an AND of two conditions. The first is
almost certainly about x8 (marginally strongest). The second is
some secondary feature that separates the 86 clamp rows from the
190-ish non-clamp rows that also have small x8.

Strategy
--------
1. Pre-filter to rows satisfying x8 < t1  (scan t1 ∈ x8 grid).
2. Within each pre-filter, search every other feature for a second
   threshold that cleanly separates clamp from non-clamp.
3. Also try AND rules with derived features (|x8|-|x4|, x8/x4,
   integer floors, modular residues).
4. Rank final AND rules by 0.5·(sens+spec), breaking ties on
   simplicity (integer thresholds).

Additional probes
-----------------
- Triple products x_i · x_j · x_k with constants.
- Distance from (x8=-0.3): |x8 + 0.3| — since clamp x8 concentrates
  in a narrow band around -0.3.
- Integer floor / ceiling of x5, x9, x10, x11.
- `abs(x8) - abs(x4)` and similar magnitude comparisons.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "plots" / "clamp_search"
OUT.mkdir(parents=True, exist_ok=True)
SENTINEL = 999.0


def load_quadrant() -> pd.DataFrame:
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
    train["a1_resid"] = train["target"].values - a1
    train["theta"] = np.arctan2(train["x7"], train["x6"])
    q = train[(train["x4"] < 0) & (train["x8"] < 0) & (train["x5"] != SENTINEL)].copy()
    q["is_clamp"] = q["a1_resid"].abs() > 1.0
    # Derived features
    q["abs_x8_minus_abs_x4"] = q["x8"].abs() - q["x4"].abs()
    q["x8_plus_x4"] = q["x8"] + q["x4"]
    q["x8_times_x4"] = q["x8"] * q["x4"]
    q["x8_div_x4"] = q["x8"] / q["x4"]
    q["x8_centered"] = np.abs(q["x8"] + 0.3)       # band around -0.3
    q["x4_centered"] = np.abs(q["x4"] + 0.3)
    q["x8_x9"] = q["x8"] * q["x9"]
    q["x8_x10"] = q["x8"] * q["x10"]
    q["x8_x11"] = q["x8"] * q["x11"]
    q["x8_x10x11"] = q["x8"] * q["x10"] * q["x11"]
    q["x4_x10"] = q["x4"] * q["x10"]
    q["x4_x11"] = q["x4"] * q["x11"]
    q["x4_x10x11"] = q["x4"] * q["x10"] * q["x11"]
    q["x4_x8_x9"] = q["x4"] * q["x8"] * q["x9"]
    q["x4_x8_x10"] = q["x4"] * q["x8"] * q["x10"]
    q["x4_x8_x11"] = q["x4"] * q["x8"] * q["x11"]
    q["floor_x5"] = np.floor(q["x5"]).astype(int)
    q["floor_x9"] = np.floor(q["x9"]).astype(int)
    q["x5_frac"] = q["x5"] - np.floor(q["x5"])
    q["x9_frac"] = q["x9"] - np.floor(q["x9"])
    q["r_xy"] = np.sqrt(q["x6"] ** 2 + q["x7"] ** 2)
    q["x6_sign"] = np.sign(q["x6"])
    q["x7_sign"] = np.sign(q["x7"])
    q["sin_theta"] = np.sin(q["theta"])
    q["cos_theta"] = np.cos(q["theta"])
    q["tan_theta"] = np.tan(q["theta"])
    return q


def evaluate(mask: np.ndarray, y: np.ndarray) -> dict:
    tp = int((mask & y).sum()); fp = int((mask & ~y).sum())
    fn = int((~mask & y).sum()); tn = int((~mask & ~y).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sens": sens, "spec": spec,
            "score": (sens + spec) / 2}


def scan_and_rules(q: pd.DataFrame, pre: tuple[str, str, float],
                   second_feats: list[str]) -> pd.DataFrame:
    """Rules of the form (pre-condition) AND (second_feat op thr).

    `pre` = (feature, direction, threshold) — defines the pre-filter.
    """
    pf, pd_, pt = pre
    mask_pre = (q[pf] < pt).values if pd_ == "<" else (q[pf] > pt).values
    # Only search thresholds within pre-filtered rows
    is_clamp = q["is_clamp"].values
    rows = []
    for sf in second_feats:
        if sf == pf:
            continue
        vals = q[sf].values
        sub = vals[mask_pre]
        if len(sub) == 0:
            continue
        qs = np.unique(np.concatenate([
            np.quantile(sub, np.linspace(0.05, 0.95, 19)),
            np.arange(np.floor(sub.min() * 2) / 2,
                      np.ceil(sub.max() * 2) / 2 + 0.5, 0.5),
        ]))
        for t in qs:
            for direction in ("<", ">"):
                if direction == "<":
                    inner = (vals < t)
                else:
                    inner = (vals > t)
                mask = mask_pre & inner
                if not (20 <= mask.sum() <= len(q) - 20):
                    continue
                s = evaluate(mask, is_clamp)
                rows.append({
                    "rule": f"({pf} {pd_} {pt:.3f}) AND ({sf} {direction} {t:.3f})",
                    "pre": f"{pf}{pd_}{pt:.3f}", "sec": sf, "sec_thr": t,
                    "sec_dir": direction,
                    **s,
                })
    return pd.DataFrame(rows)


def scan_single(q: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    is_clamp = q["is_clamp"].values
    rows = []
    for f in features:
        vals = q[f].values
        if vals.dtype == bool:
            for direction in ("==T", "==F"):
                mask = vals if direction == "==T" else ~vals
                s = evaluate(mask, is_clamp)
                rows.append({"rule": f"{f} {direction}", **s})
            continue
        qs = np.unique(np.concatenate([
            np.quantile(vals, np.linspace(0.05, 0.95, 19)),
            np.arange(np.floor(vals.min() * 2) / 2,
                      np.ceil(vals.max() * 2) / 2 + 0.5, 0.5),
        ]))
        for t in qs:
            for direction in ("<", ">"):
                mask = (vals < t) if direction == "<" else (vals > t)
                if not (20 <= mask.sum() <= len(q) - 20):
                    continue
                s = evaluate(mask, is_clamp)
                rows.append({"rule": f"{f} {direction} {t:.3f}", **s})
    return pd.DataFrame(rows)


def main() -> None:
    q = load_quadrant()
    n = len(q); nc = int(q["is_clamp"].sum())
    print(f"Quadrant: n={n}, clamp={nc}")

    base_feats = ["x1", "x2", "x4", "x5", "x6", "x7", "x9", "x10", "x11",
                  "theta", "sin_theta", "cos_theta",
                  "abs_x8_minus_abs_x4", "x8_plus_x4", "x8_times_x4",
                  "x8_div_x4", "x8_centered", "x4_centered",
                  "x8_x9", "x8_x10", "x8_x11", "x8_x10x11",
                  "x4_x10", "x4_x11", "x4_x10x11",
                  "x4_x8_x9", "x4_x8_x10", "x4_x8_x11",
                  "floor_x5", "floor_x9", "x5_frac", "x9_frac"]

    # 1. Single rule refresh with new features
    sf = scan_single(q, base_feats + ["x8"])
    sf = sf.sort_values("score", ascending=False).reset_index(drop=True)
    print("\n=== Top 15 single rules (incl. derived) ===")
    print(sf.head(15).to_string(index=False,
          formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                      "score": "{:.3f}".format}))

    # 2. AND rules: pre-filter on x8 (various thresholds), second on everything
    all_and = []
    for pre_thr in [-0.167, -0.20, -0.22, -0.25, -0.28]:
        ar = scan_and_rules(q, ("x8", "<", pre_thr), base_feats)
        all_and.append(ar)
    and_rules = pd.concat(all_and, ignore_index=True)
    and_rules = and_rules.sort_values("score", ascending=False).reset_index(drop=True)
    print("\n=== Top 20 AND rules (pre=x8<t1 & sec=feature op t2) ===")
    print(and_rules.head(20).to_string(index=False,
          formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                      "score": "{:.3f}".format}))

    # 3. Try pre-filter on x8_centered (narrow band trigger)
    all_and2 = []
    for pre_thr in [0.05, 0.08, 0.10, 0.12, 0.15]:
        ar = scan_and_rules(q, ("x8_centered", "<", pre_thr), base_feats + ["x8"])
        all_and2.append(ar)
    band_rules = pd.concat(all_and2, ignore_index=True).sort_values("score", ascending=False)
    print("\n=== Top 20 AND rules with x8_centered pre-filter (x8 near -0.3) ===")
    print(band_rules.head(20).to_string(index=False,
          formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                      "score": "{:.3f}".format}))

    # 4. Also try: pre-filter using abs(x8)-abs(x4) (magnitude ordering)
    pre_feats = [("abs_x8_minus_abs_x4", ">", 0.0),
                 ("x8_times_x4", ">", 0.10),
                 ("floor_x5", "<", 9.5),
                 ("floor_x9", "<", 4.5)]
    all_and3 = []
    for pre in pre_feats:
        ar = scan_and_rules(q, pre, base_feats + ["x8"])
        all_and3.append(ar)
    other_rules = pd.concat(all_and3, ignore_index=True).sort_values("score", ascending=False)
    print("\n=== Top 20 AND rules with alternative pre-filters ===")
    print(other_rules.head(20).to_string(index=False,
          formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                      "score": "{:.3f}".format}))

    # 5. Try band rules: (x8 < high) AND (x8 > low)  — narrow x8 band
    is_clamp = q["is_clamp"].values
    x8v = q["x8"].values
    band_results = []
    for hi in np.arange(-0.15, -0.30, -0.01):
        for lo in np.arange(-0.30, -0.45, -0.01):
            if lo >= hi:
                continue
            mask = (x8v < hi) & (x8v > lo)
            if not (50 <= mask.sum() <= len(q) - 50):
                continue
            s = evaluate(mask, is_clamp)
            band_results.append({"rule": f"x8 in ({lo:.3f}, {hi:.3f})", **s})
    band = pd.DataFrame(band_results).sort_values("score", ascending=False)
    print("\n=== Top 15 band rules x8 in (lo, hi) ===")
    print(band.head(15).to_string(index=False,
          formatters={"sens": "{:.3f}".format, "spec": "{:.3f}".format,
                      "score": "{:.3f}".format}))

    # Save everything
    sf.to_csv(OUT / "v2_single.csv", index=False)
    and_rules.to_csv(OUT / "v2_and_x8.csv", index=False)
    band_rules.to_csv(OUT / "v2_and_centered.csv", index=False)
    other_rules.to_csv(OUT / "v2_and_other.csv", index=False)
    band.to_csv(OUT / "v2_band.csv", index=False)
    print(f"\nArtefacts in {OUT}")


if __name__ == "__main__":
    main()
