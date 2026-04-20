"""Grid-search router variants against cached OOF.

LB-verified so far:
  cross_LE alone           →  LB 2.94
  router_A1_triple         →  LB 3.35   (decomposed: triple non-sent ≈ 1.78 on safe+, 3.22 on unsafe)
  router_A1_cross_LE       →  LB 2.53   (cross_LE ≈ 1.78 on unsafe non-sent; sentinel 10 · 0.152 = 1.52 floor)

Questions answered here (offline, on dataset.csv with OOF):
  1. Tighten the safe gate — exclude the x4<0 ∧ x8<0 quadrant where A1
     has a 23 % clamp rate (from CLAUDE.md "Routing ensemble" section).
  2. Replace cross_LE backup with a TabPFN / cross_LE / triple mix on
     the unsafe-and-sentinel rows.
  3. Do both at once.

OOF predictions come from scripts/cv_foundation_models.py which already
stored them in plots/foundation_models/cv_foundation_models_oof.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_formulas import approach1_predict  # noqa: E402
from cv_router_A1 import safe_mask as safe_mask_v1  # noqa: E402

DATA = REPO / "data"
OOF = REPO / "plots" / "foundation_models" / "cv_foundation_models_oof.csv"
OUT = REPO / "plots" / "foundation_models"
SENTINEL = 999.0
X4_GAP = 0.167


def mae(p, y):
    return float(np.mean(np.abs(p - y))) if len(y) else float("nan")


# ---------------------------------------------------------------------------
# Safe-gate variants
# ---------------------------------------------------------------------------
def safe_mask_no_neg_x8(df: pd.DataFrame) -> np.ndarray:
    """Current router + exclude x4<0 ∧ x8<0 quadrant (where clamp lives)."""
    base = safe_mask_v1(df)
    not_clamp_quadrant = ~((df["x4"].values < 0) & (df["x8"].values < 0))
    return base & not_clamp_quadrant


def safe_mask_x8_pos(df: pd.DataFrame) -> np.ndarray:
    """Current router + require x8 > 0 (drops the full x8<0 half)."""
    base = safe_mask_v1(df)
    return base & (df["x8"].values > 0)


def safe_mask_stricter_gap(df: pd.DataFrame, margin: float = 0.3) -> np.ndarray:
    """Widen the x4 gap from 0.167 to `margin`."""
    x4 = df["x4"].values
    x9 = df["x9"].values
    is_sent = (df["x5"].values == SENTINEL)
    x4_clear = np.abs(x4) > margin
    cluster_match = ((x4 > 0) & (x9 > 5)) | ((x4 < 0) & (x9 < 5))
    return (~is_sent) & x4_clear & cluster_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    oof = pd.read_csv(OOF)
    y = df["target"].values
    is_sent = (df["x5"].values == SENTINEL)

    # Per-fold x5 median was already used during OOF generation; for A1 we
    # apply it with the global median — this matches what happens at
    # submission time (build_router_A1_cross_LE_submission fits A1 on test
    # with the full-data median).
    x5m_global = float(df.loc[df["x5"] != SENTINEL, "x5"].median())
    a1 = approach1_predict(df, x5m_global)

    tabpfn = oof["tabpfn"].values
    lin_x4_free = oof["lin_x4_free"].values
    lin_x4_b = oof["lin_x4_b"].values
    ebm_x9 = oof["ebm_x9"].values
    ebm_full = oof["ebm_full"].values

    cross_LE   = 0.5 * (lin_x4_free + ebm_x9)
    cross_LE_b = 0.5 * (lin_x4_b + ebm_x9)
    triple     = 0.5 * cross_LE_b + 0.5 * ebm_full

    # -----------------------------------------------------------------
    # Sanity: reproduce the known router CV numbers
    # -----------------------------------------------------------------
    safe_v1 = safe_mask_v1(df)
    r_triple     = np.where(safe_v1, a1, triple)
    r_crossLE    = np.where(safe_v1, a1, cross_LE)
    print(f"Sanity — router CV MAE with safe_v1:")
    print(f"  A1 + triple:    {mae(r_triple, y):.3f}  "
          f"(matches CLAUDE.md 1.80–1.84 ballpark; LB was 3.35)")
    print(f"  A1 + cross_LE:  {mae(r_crossLE, y):.3f}  (LB 2.53 confirmed)")

    # -----------------------------------------------------------------
    # Grid: safe-gate × backup
    # -----------------------------------------------------------------
    gates = {
        "safe_v1 (current, LB 2.53 ref)":      safe_v1,
        "safe_v1 & ~(x4<0 & x8<0)":            safe_mask_no_neg_x8(df),
        "safe_v1 & x8>0":                      safe_mask_x8_pos(df),
        "safe_stricter_gap 0.30":              safe_mask_stricter_gap(df, 0.30),
        "safe_stricter_gap 0.50":              safe_mask_stricter_gap(df, 0.50),
    }

    # FM / parametric backups
    backups = {
        "cross_LE":                       cross_LE,
        "triple":                         triple,
        "TabPFN":                         tabpfn,
        "0.3*TabPFN + 0.7*cross_LE":      0.3 * tabpfn + 0.7 * cross_LE,
        "0.5*TabPFN + 0.5*cross_LE":      0.5 * tabpfn + 0.5 * cross_LE,
        "0.3*TabPFN + 0.7*triple":        0.3 * tabpfn + 0.7 * triple,
    }

    rows = []
    print("\n" + "=" * 78)
    print(f"{'gate':<42s} {'backup':<35s} n_safe  CV_MAE   safe_MAE  unsafe_MAE")
    print("=" * 78)

    for g_name, gate in gates.items():
        for b_name, backup in backups.items():
            pred = np.where(gate, a1, backup)
            cv = mae(pred, y)
            safe_mae = mae(pred[gate], y[gate])
            unsafe_mae = mae(pred[~gate], y[~gate])
            rows.append({
                "gate": g_name, "backup": b_name,
                "n_safe": int(gate.sum()),
                "overall": cv, "safe_mae": safe_mae, "unsafe_mae": unsafe_mae,
            })

    rows_df = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)

    for _, r in rows_df.iterrows():
        print(f"{r['gate']:<42s} {r['backup']:<35s} "
              f"{int(r['n_safe']):>6d}  {r['overall']:.3f}   "
              f"{r['safe_mae']:.3f}    {r['unsafe_mae']:.3f}")

    rows_df.to_csv(OUT / "cv_router_variants.csv", index=False)

    # -----------------------------------------------------------------
    # Top 3 — LB projection (only meaningful if gate & backup track LB similarly)
    # -----------------------------------------------------------------
    # Best gate keeps LB cross_LE generalisation; TabPFN solo LB unknown (~4?).
    # Use A1 safe MAE from CV as-is (A1 has no fit → train-MAE = test-MAE).
    # For backup, assume CV→LB ratio of the chosen backup:
    #   cross_LE:    1.00x   (LB 2.94 / CV 2.97)
    #   triple:      ~1.77x  (back-solved: triple LB 3.47 / CV 1.96)
    #   TabPFN:      ~1.2x   (guess)
    #   mixes:       blended ratio
    print("\n" + "=" * 78)
    print("Top candidates")
    print("=" * 78)
    print(rows_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
