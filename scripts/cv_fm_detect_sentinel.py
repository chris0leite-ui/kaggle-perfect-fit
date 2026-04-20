"""5-fold CV — can foundation models detect x5 sentinel rows from the other features?

Prior finding (CLAUDE.md > "Post-submission diagnostics"):
  * 222 / 1500 training rows (14.8 %) and 228 / 1500 test rows (15.2 %)
    have x5 == 999.
  * All Pearson |r(feature, is_sentinel)| < 0.06 — missingness appears MCAR.
  * A linear-logistic indicator added to the regression was insignificant
    (coef = -1.15, p = 0.51).

That analysis was linear / EBM-based. Nonlinear interactions — especially
involving x6, x7 and their derived angle — were not tested.

This script fits binary classifiers (target: x5 == 999) with all other
features including x6, x7, sin/cos(atan2(x7, x6)), on 5-fold CV.
If AUC > 0.55 we have a novel signal; MCAR is rejected.

Models:
  * TabPFN v2 (classifier)
  * TabM (trained from scratch with BCE loss)
  * HistGradientBoosting (sklearn, for reference)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import SENTINEL, SEED  # noqa: E402

DATA = REPO / "data"
OUT = REPO / "plots" / "foundation_models"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TABPFN_CLF_PATH = "/root/.cache/tabpfn/tabpfn-v2-classifier.ckpt"

NUM_COLS_NO_X5 = ["x1", "x2", "x4", "x6", "x7", "x8", "x9", "x10", "x11"]


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Features *excluding* x5. Returns (X_num, x_cat)."""
    x6 = df["x6"].values.astype(np.float32)
    x7 = df["x7"].values.astype(np.float32)
    theta = np.arctan2(x7, x6)
    sin_t = np.sin(theta).astype(np.float32).reshape(-1, 1)
    cos_t = np.cos(theta).astype(np.float32).reshape(-1, 1)
    X_num = np.concatenate([
        df[NUM_COLS_NO_X5].values.astype(np.float32),
        sin_t, cos_t,
    ], axis=1)  # 9 + 2 = 11 numeric columns
    x_cat = (df["City"] == "Zaragoza").astype(np.int64).values.reshape(-1, 1)
    return X_num, x_cat


def fit_predict_tabpfn_clf(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
    """Return P(is_sentinel=1) from TabPFN v2 classifier."""
    from tabpfn import TabPFNClassifier
    m = TabPFNClassifier(
        n_estimators=8, device=DEVICE,
        ignore_pretraining_limits=True, random_state=SEED,
        model_path=TABPFN_CLF_PATH,
    )
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_va)
    classes = list(m.classes_)
    return proba[:, classes.index(1)]


def fit_predict_tabm_clf(
    X_num_tr, x_cat_tr, y_tr, X_num_va, x_cat_va,
    *, seed: int = SEED, epochs: int = 150, lr: float = 2e-3,
    weight_decay: float = 1e-4, batch_size: int = 128,
    k: int = 32, d_block: int = 256, n_blocks: int = 3, dropout: float = 0.1,
) -> np.ndarray:
    """TabM trained for binary classification — returns P(sentinel=1)."""
    import tabm

    torch.manual_seed(seed); np.random.seed(seed)

    scaler = StandardScaler()
    X_num_tr_s = scaler.fit_transform(X_num_tr).astype(np.float32)
    X_num_va_s = scaler.transform(X_num_va).astype(np.float32)

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

    # Class balance weight (14.8 % positive)
    pos_frac = float(y_tr.mean())
    pos_weight = (1 - pos_frac) / max(pos_frac, 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def to_t(a, dt): return torch.as_tensor(a, dtype=dt, device=DEVICE)
    X_num_full = to_t(X_num_tr_s, torch.float32)
    X_cat_full = to_t(x_cat_tr.astype(np.int64), torch.long)
    y_full = to_t(y_tr.astype(np.float32), torch.float32)

    X_num_val = X_num_full[val_idx]; X_cat_val = X_cat_full[val_idx]; y_val = y_full[val_idx]
    X_num_trn = X_num_full[tr_idx]; X_cat_trn = X_cat_full[tr_idx]; y_trn = y_full[tr_idx]
    X_num_te  = to_t(X_num_va_s, torch.float32)
    X_cat_te  = to_t(x_cat_va.astype(np.int64), torch.long)

    best_val = -1.0; best_state = None; patience = 20; waited = 0
    for _ in range(epochs):
        model.train()
        pe = torch.randperm(len(X_num_trn), device=DEVICE)
        for s in range(0, len(pe), batch_size):
            idx = pe[s:s + batch_size]
            opt.zero_grad()
            logits = model(x_num=X_num_trn[idx], x_cat=X_cat_trn[idx])  # (B, k, 1)
            y_rep = y_trn[idx].view(-1, 1, 1).expand_as(logits)
            loss_fn(logits, y_rep).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_num=X_num_val, x_cat=X_cat_val).mean(dim=1).squeeze(-1)
            val_auc = roc_auc_score(
                y_val.cpu().numpy(),
                torch.sigmoid(logits).cpu().numpy(),
            ) if len(np.unique(y_val.cpu().numpy())) > 1 else 0.5
        if val_auc > best_val + 1e-4:
            best_val = val_auc
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
        logits = model(x_num=X_num_te, x_cat=X_cat_te).mean(dim=1).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def fit_predict_histgbm_clf(X_tr, y_tr, X_va):
    m = HistGradientBoostingClassifier(
        max_iter=500, max_depth=5, learning_rate=0.05,
        l2_regularization=1.0, random_state=SEED,
        class_weight="balanced",
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1]


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    y = (df["x5"] == SENTINEL).astype(np.int64).values
    print(f"Sentinel prevalence (train): {y.mean():.3%}  "
          f"({int(y.sum())} / {len(y)})")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(df)) for k in ["tabpfn", "tabm", "histgbm"]}

    print("\n" + "=" * 78)
    print("5-fold CV — binary: x5 == 999")
    print("=" * 78)
    for fold, (tr, va) in enumerate(kf.split(df)):
        t0 = time.time()
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        X_num_tr, x_cat_tr = build_features(sub_tr)
        X_num_va, x_cat_va = build_features(sub_va)
        X_flat_tr = np.concatenate([X_num_tr, x_cat_tr.astype(np.float32)], axis=1)
        X_flat_va = np.concatenate([X_num_va, x_cat_va.astype(np.float32)], axis=1)
        y_tr = y[tr]; y_va = y[va]

        t1 = time.time()
        oof["tabpfn"][va] = fit_predict_tabpfn_clf(X_flat_tr, y_tr, X_flat_va)
        t_pfn = time.time() - t1

        t1 = time.time()
        oof["tabm"][va] = fit_predict_tabm_clf(X_num_tr, x_cat_tr, y_tr,
                                               X_num_va, x_cat_va)
        t_tabm = time.time() - t1

        t1 = time.time()
        oof["histgbm"][va] = fit_predict_histgbm_clf(X_flat_tr, y_tr, X_flat_va)
        t_hgbm = time.time() - t1

        print(f"  fold {fold+1}/{N_SPLITS}  tabpfn={t_pfn:.0f}s  "
              f"tabm={t_tabm:.0f}s  hgbm={t_hgbm:.1f}s  "
              f"fold AUC: pfn={roc_auc_score(y_va, oof['tabpfn'][va]):.3f}  "
              f"tabm={roc_auc_score(y_va, oof['tabm'][va]):.3f}  "
              f"hgbm={roc_auc_score(y_va, oof['histgbm'][va]):.3f}  "
              f"[{time.time()-t0:.0f}s total]")

    # -------- summary --------
    print("\n" + "=" * 78)
    print("OOF results")
    print("=" * 78)
    rows = []
    for name, p in oof.items():
        auc = roc_auc_score(y, p)
        ap  = average_precision_score(y, p)
        print(f"  {name:<10s}  AUC={auc:.3f}  AP={ap:.3f}  (baseline AP = {y.mean():.3f})")
        rows.append({"model": name, "AUC": auc, "AP": ap})

    # Simple ensemble
    pbar = (oof["tabpfn"] + oof["tabm"] + oof["histgbm"]) / 3.0
    auc_ens = roc_auc_score(y, pbar); ap_ens = average_precision_score(y, pbar)
    print(f"  {'ensemble':<10s}  AUC={auc_ens:.3f}  AP={ap_ens:.3f}")
    rows.append({"model": "ensemble (mean)", "AUC": auc_ens, "AP": ap_ens})

    # -------- threshold-free verdict --------
    best_auc = max(r["AUC"] for r in rows)
    print("\n" + "=" * 78)
    if best_auc > 0.60:
        print(f"*** SENTINEL IS PREDICTABLE *** best AUC = {best_auc:.3f}")
        print("    prior linear analysis (Pearson |r| < 0.06) missed the signal.")
        print("    Next step: use this as an extra feature in the regressor.")
    elif best_auc > 0.55:
        print(f"Mild signal: best AUC = {best_auc:.3f}")
        print("    weak nonlinear dependence; marginal for modelling.")
    else:
        print(f"No signal detected: best AUC = {best_auc:.3f}")
        print("    sentinel-missingness is effectively MCAR  "
              "(linear analysis was correct).")
    print("=" * 78)

    pd.DataFrame(rows).to_csv(OUT / "cv_fm_detect_sentinel.csv", index=False)
    pd.DataFrame({**oof, "is_sentinel": y}).to_csv(
        OUT / "cv_fm_detect_sentinel_oof.csv", index=False)


if __name__ == "__main__":
    main()
