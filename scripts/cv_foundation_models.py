"""5-fold CV for tabular foundation models.

Of the 5 FMs we wanted to try (TabPFN v2, TabICL, TabDPT, CARTE, TabM),
only TabPFN v2 and TabM are reachable from this sandbox — HF and Facebook
CDNs (needed for TabICL, TabDPT, CARTE/fasttext) are blocked. TabPFN v2
ships an alternate GCS mirror that works; TabM carries no pretrained
weights (MLP ensemble trained from scratch per fold).

Reference models (from previous rounds):
  cross_LE = 0.5*(LIN_x4 + EBM_x9)      LB 2.94 (confirmed)
  triple   = 0.5*cross_LE_b + 0.5*EBM_full  CV 2.824 (top candidate)

Reports CV MAE (overall / non-sentinel / sentinel) for:

  Solo        TabPFN, TabM, cross_LE, triple
  FM ensemble 0.5*(TabPFN + TabM)
  FM + best   0.5*(FM_avg + cross_LE), 0.5*(FM_avg + triple)
              + pairwise blends with each FM

Runs in the /tmp/fm_env venv (has tabpfn, tabm, torch, interpret-core).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import FEATURES_ALL, SENTINEL, SEED, preprocess  # noqa: E402
from cv_x4_x9_swap_ensemble import ebm_features, design_matrix, fit_ebm  # noqa: E402
from cv_cross_LE_tune import LOCKED_COEFS_B, LIN_COL_ORDER  # noqa: E402

DATA = REPO / "data"
OUT = REPO / "plots" / "foundation_models"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# TabPFN v2 wrapper
# ---------------------------------------------------------------------------
def fit_predict_tabpfn(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(
        n_estimators=8,
        device=DEVICE,
        ignore_pretraining_limits=True,
        random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_va)


# ---------------------------------------------------------------------------
# TabM wrapper — train from scratch
# ---------------------------------------------------------------------------
def fit_predict_tabm(
    X_num_tr: np.ndarray, x_cat_tr: np.ndarray, y_tr: np.ndarray,
    X_num_va: np.ndarray, x_cat_va: np.ndarray,
    *, seed: int = SEED, epochs: int = 200, lr: float = 2e-3,
    weight_decay: float = 1e-4, batch_size: int = 128,
    k: int = 32, d_block: int = 256, n_blocks: int = 3, dropout: float = 0.1,
) -> np.ndarray:
    """Train TabM for one fold. Returns mean prediction over the k submodels."""
    import tabm

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_num_tr_s = scaler.fit_transform(X_num_tr).astype(np.float32)
    X_num_va_s = scaler.transform(X_num_va).astype(np.float32)

    y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
    y_tr_s = ((y_tr - y_mean) / y_std).astype(np.float32)

    # internal validation split for early stopping
    n_tr = len(X_num_tr_s)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tr)
    n_val = max(64, n_tr // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    cat_card = [int(x_cat_tr.max()) + 1]

    model = tabm.TabM.make(
        n_num_features=X_num_tr_s.shape[1],
        cat_cardinalities=cat_card,
        d_out=1, k=k, d_block=d_block, n_blocks=n_blocks, dropout=dropout,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()  # MAE is the competition metric

    def to_t(arr, dtype):
        return torch.as_tensor(arr, dtype=dtype, device=DEVICE)

    X_num_full = to_t(X_num_tr_s, torch.float32)
    X_cat_full = to_t(x_cat_tr.astype(np.int64), torch.long)
    y_full = to_t(y_tr_s, torch.float32)

    X_num_val = X_num_full[val_idx]; X_cat_val = X_cat_full[val_idx]; y_val = y_full[val_idx]
    X_num_trn = X_num_full[tr_idx]; X_cat_trn = X_cat_full[tr_idx]; y_trn = y_full[tr_idx]
    X_num_te  = to_t(X_num_va_s, torch.float32)
    X_cat_te  = to_t(x_cat_va.astype(np.int64), torch.long)

    best_val = float("inf"); best_state = None; patience = 25; waited = 0

    for epoch in range(epochs):
        model.train()
        perm_e = torch.randperm(len(X_num_trn), device=DEVICE)
        total_loss = 0.0
        for start in range(0, len(perm_e), batch_size):
            idx = perm_e[start:start + batch_size]
            opt.zero_grad()
            out = model(x_num=X_num_trn[idx], x_cat=X_cat_trn[idx])
            # out: (B, k, 1) — L1 against broadcasted target
            y_rep = y_trn[idx].view(-1, 1, 1).expand_as(out)
            loss = loss_fn(out, y_rep)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        # Validation (averaged over the k submodels)
        model.eval()
        with torch.no_grad():
            out = model(x_num=X_num_val, x_cat=X_cat_val).mean(dim=1).squeeze(-1)
            val_mae = (out - y_val).abs().mean().item()

        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(x_num=X_num_te, x_cat=X_cat_te).mean(dim=1).squeeze(-1)
        pred_s = out.cpu().numpy()
    return pred_s * y_std + y_mean


# ---------------------------------------------------------------------------
# Reference-model baselines (adapted from cv_cross_LE_tune / cv_x4_x9_swap)
# ---------------------------------------------------------------------------
def lin_x4_locked_predict(df_tr: pd.DataFrame, df_va: pd.DataFrame) -> np.ndarray:
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5_med, include_x4=True, include_x9=False)
    X_va = design_matrix(df_va, x5_med, include_x4=True, include_x9=False)
    v = np.array([LOCKED_COEFS_B[c] for c in LIN_COL_ORDER])
    return X_va @ v + (df_tr["target"].values - X_tr @ v).mean()


def lin_x4_free_predict(df_tr: pd.DataFrame, df_va: pd.DataFrame) -> np.ndarray:
    x5_med = float(df_tr.loc[df_tr["x5"] != SENTINEL, "x5"].median())
    X_tr = design_matrix(df_tr, x5_med, include_x4=True, include_x9=False)
    X_va = design_matrix(df_va, x5_med, include_x4=True, include_x9=False)
    return LinearRegression().fit(X_tr, df_tr["target"].values).predict(X_va)


# ---------------------------------------------------------------------------
# Feature prep for FMs — numeric features + binary city as cat
# ---------------------------------------------------------------------------
NUM_COLS = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]


def fm_arrays(df: pd.DataFrame, x5_median: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_num, x_cat, x5_is_sent) for TabPFN / TabM."""
    df_c = df.copy()
    df_c["x5"] = df_c["x5"].where(df_c["x5"] != SENTINEL, x5_median)
    x5_is_sent = (df["x5"] == SENTINEL).astype(np.float32).values.reshape(-1, 1)
    X_num = np.concatenate([df_c[NUM_COLS].values.astype(np.float32), x5_is_sent], axis=1)
    x_cat = (df["City"] == "Zaragoza").astype(np.int64).values.reshape(-1, 1)
    return X_num, x_cat, x5_is_sent.ravel()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def mae(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(p - y))) if len(y) else float("nan")


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    y = df["target"].values
    is_sent = (df["x5"] == SENTINEL).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = {k: np.zeros(len(df)) for k in
           ["tabpfn", "tabm", "lin_x4_free", "lin_x4_b", "ebm_x9", "ebm_full"]}

    print("=" * 78)
    print("5-fold CV — TabPFN v2 + TabM + reference baselines")
    print("=" * 78)
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5m = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())

        # --- FM-shaped inputs
        X_num_tr, x_cat_tr, _ = fm_arrays(sub_tr, x5m)
        X_num_va, x_cat_va, _ = fm_arrays(sub_va, x5m)
        # TabPFN takes a single flat array including the city col
        X_flat_tr = np.concatenate([X_num_tr, x_cat_tr.astype(np.float32)], axis=1)
        X_flat_va = np.concatenate([X_num_va, x_cat_va.astype(np.float32)], axis=1)

        t1 = time.time()
        oof["tabpfn"][va] = fit_predict_tabpfn(X_flat_tr, sub_tr["target"].values, X_flat_va)
        t_pfn = time.time() - t1

        t1 = time.time()
        oof["tabm"][va] = fit_predict_tabm(
            X_num_tr, x_cat_tr, sub_tr["target"].values,
            X_num_va, x_cat_va,
        )
        t_tabm = time.time() - t1

        # --- Reference baselines
        oof["lin_x4_free"][va] = lin_x4_free_predict(sub_tr, sub_va)
        oof["lin_x4_b"][va]    = lin_x4_locked_predict(sub_tr, sub_va)

        feats_x9 = ebm_features(with_x4=False, with_x9=True)
        X_tr = preprocess(sub_tr, feats_x9, x5m); X_va = preprocess(sub_va, feats_x9, x5m)
        oof["ebm_x9"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        feats_full = ebm_features(with_x4=True, with_x9=True)
        X_tr = preprocess(sub_tr, feats_full, x5m); X_va = preprocess(sub_va, feats_full, x5m)
        oof["ebm_full"][va] = fit_ebm(X_tr, sub_tr["target"].values).predict(X_va)

        print(f"  fold {fold+1}/{N_SPLITS}  tabpfn={t_pfn:.0f}s  tabm={t_tabm:.0f}s  "
              f"total={time.time()-t0:.0f}s")

    # -------------------------------------------------------------------
    # Derived "best" references
    # -------------------------------------------------------------------
    cross_LE = 0.5 * (oof["lin_x4_free"] + oof["ebm_x9"])                       # LB 2.94
    cross_LE_b = 0.5 * (oof["lin_x4_b"] + oof["ebm_x9"])
    triple = 0.5 * cross_LE_b + 0.5 * oof["ebm_full"]                           # CV 2.824

    fm_avg = 0.5 * (oof["tabpfn"] + oof["tabm"])

    rows = []

    def record(name: str, pred: np.ndarray) -> None:
        rows.append({
            "model": name,
            "overall": mae(pred, y),
            "non_sent": mae(pred[~is_sent], y[~is_sent]),
            "sent":     mae(pred[is_sent], y[is_sent]),
        })

    # Solo FMs and references
    record("TabPFN v2",                    oof["tabpfn"])
    record("TabM",                         oof["tabm"])
    record("cross_LE (ref, LB 2.94)",      cross_LE)
    record("triple (ref, CV 2.824)",       triple)

    # FM-only ensemble
    record("FM_avg = 0.5*(TabPFN+TabM)",   fm_avg)

    # Pairwise blends with cross_LE
    for w in [0.3, 0.5, 0.7]:
        record(f"{w:.1f}*TabPFN + {1-w:.1f}*cross_LE",
               w * oof["tabpfn"] + (1 - w) * cross_LE)
        record(f"{w:.1f}*TabM + {1-w:.1f}*cross_LE",
               w * oof["tabm"] + (1 - w) * cross_LE)
        record(f"{w:.1f}*FM_avg + {1-w:.1f}*cross_LE",
               w * fm_avg + (1 - w) * cross_LE)

    # Pairwise blends with triple
    for w in [0.3, 0.5, 0.7]:
        record(f"{w:.1f}*TabPFN + {1-w:.1f}*triple",
               w * oof["tabpfn"] + (1 - w) * triple)
        record(f"{w:.1f}*TabM + {1-w:.1f}*triple",
               w * oof["tabm"] + (1 - w) * triple)
        record(f"{w:.1f}*FM_avg + {1-w:.1f}*triple",
               w * fm_avg + (1 - w) * triple)

    # Three-way blends FM + both baselines
    for a, b, c in [(1/3, 1/3, 1/3), (0.25, 0.375, 0.375), (0.5, 0.25, 0.25)]:
        record(f"{a:.2f}*FM_avg + {b:.2f}*cross_LE + {c:.2f}*triple",
               a * fm_avg + b * cross_LE + c * triple)

    results = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)
    print("\n" + "=" * 78)
    print("Ranked by CV MAE (overall)")
    print("=" * 78)
    for _, r in results.iterrows():
        print(f"  {r['model']:<50s}  "
              f"overall={r['overall']:6.3f}  "
              f"non-sent={r['non_sent']:6.3f}  "
              f"sent={r['sent']:6.3f}")

    results.to_csv(OUT / "cv_foundation_models.csv", index=False)
    pd.DataFrame({**oof, "target": y, "is_sent": is_sent.astype(int)}).to_csv(
        OUT / "cv_foundation_models_oof.csv", index=False)
    print(f"\nWrote {OUT / 'cv_foundation_models.csv'}")


if __name__ == "__main__":
    main()
