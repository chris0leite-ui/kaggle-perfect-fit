"""Density-ratio reweighting to break the train-only x4-x9 correlation.

Approach
--------
1. Classifier DRE — fit a classifier to distinguish the real (x4, x9)
   joint from a shuffled (x4, x9_permuted) surrogate where x4 and x9 are
   independent. The classifier's output converts to the density ratio
        w(x4, x9) = p_marginal(x4, x9) / p_joint(x4, x9)
   Reweighting the training set with w makes x4 and x9 independent in
   expectation — matching the test-set DGP.

2. Fit candidate models with sample_weight=w. Compare to unweighted
   baseline.

3. Report:
     - weighted corr(x4, x9)  (should drop from ~0.83 to ~0)
     - CV MAE unweighted      (comparable to all other scripts)
     - CV MAE weighted        (better proxy for LB since it matches p_test)
     - non-sentinel MAE

4. Build submissions for the weighted variants so they can be sent to LB.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cv_ebm_variants import FEATURES_ALL, SENTINEL, preprocess  # noqa: E402
from cv_simple_linear import design_matrix  # noqa: E402

DATA = REPO / "data"
SUBS = REPO / "submissions"
OUT = REPO / "plots" / "reweight"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
CLIP = (0.1, 10.0)


def compute_weights_classifier(df: pd.DataFrame, seed: int = SEED,
                               clip: tuple[float, float] = CLIP):
    """Classifier-based density ratio for (x4, x9) -> weights."""
    x4 = df["x4"].to_numpy()
    x9 = df["x9"].to_numpy()
    n = len(x4)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    X_pos = np.column_stack([x4, x9])
    X_neg = np.column_stack([x4, x9[perm]])
    X = np.vstack([X_pos, X_neg])
    y = np.r_[np.ones(n), np.zeros(n)]

    clf = HistGradientBoostingClassifier(
        random_state=seed, max_depth=4, max_iter=300, learning_rate=0.05
    )
    probs = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:n, 1]
    p_clip = np.clip(probs, 1e-3, 1 - 1e-3)
    w_raw = (1.0 - p_clip) / p_clip
    w = np.clip(w_raw, clip[0], clip[1])
    w = w * n / w.sum()
    return w, probs, w_raw


def compute_weights_kde(df: pd.DataFrame, bandwidth: float = 0.30,
                        clip: tuple[float, float] = CLIP):
    """KDE-based density ratio: p(x4) p(x9) / p(x4, x9).

    Gaussian KDEs on standardised (x4, x9); bandwidth defaults to 0.30 (in
    standardised units, ~Silverman for 1500 2D points).
    """
    from sklearn.neighbors import KernelDensity

    x4 = df["x4"].to_numpy()
    x9 = df["x9"].to_numpy()
    x4z = (x4 - x4.mean()) / x4.std()
    x9z = (x9 - x9.mean()) / x9.std()
    XY = np.column_stack([x4z, x9z])

    kde_joint = KernelDensity(bandwidth=bandwidth).fit(XY)
    kde_x4 = KernelDensity(bandwidth=bandwidth).fit(x4z.reshape(-1, 1))
    kde_x9 = KernelDensity(bandwidth=bandwidth).fit(x9z.reshape(-1, 1))

    log_joint = kde_joint.score_samples(XY)
    log_x4 = kde_x4.score_samples(x4z.reshape(-1, 1))
    log_x9 = kde_x9.score_samples(x9z.reshape(-1, 1))
    log_ratio = (log_x4 + log_x9) - log_joint

    w_raw = np.exp(log_ratio - log_ratio.mean())  # centre in log-space
    w = np.clip(w_raw, clip[0], clip[1])
    n = len(w)
    w = w * n / w.sum()
    return w, np.exp(log_joint), w_raw


def compute_weights_copula(df: pd.DataFrame,
                           clip: tuple[float, float] = CLIP):
    """Gaussian-copula analytical density ratio p(x4) p(x9) / p(x4, x9).

    1. Rank-transform x4 and x9 to uniform, then to standard normal (z4, z9).
    2. Under a bivariate-Gaussian assumption on the copula with correlation
       rho, the density ratio has a closed form:
         log(p_marg / p_joint) = 0.5 log(1 - rho^2)
              + 0.5 * rho * [rho*(z4^2 + z9^2) - 2*z4*z9] / (1 - rho^2)
    3. Return clipped, mean-1-normalised weights.

    Robust to non-Gaussian marginals because of the rank transform; only the
    *copula* (rank-joint) has to be approximately Gaussian.
    """
    from scipy.stats import norm, rankdata

    x4 = df["x4"].to_numpy()
    x9 = df["x9"].to_numpy()
    n = len(x4)
    u4 = (rankdata(x4) - 0.5) / n
    u9 = (rankdata(x9) - 0.5) / n
    z4 = norm.ppf(u4)
    z9 = norm.ppf(u9)
    rho = float(np.corrcoef(z4, z9)[0, 1])

    log_w = 0.5 * np.log(1 - rho ** 2) + 0.5 * rho * (
        rho * (z4 ** 2 + z9 ** 2) - 2.0 * z4 * z9
    ) / (1 - rho ** 2)
    w_raw = np.exp(log_w)
    w = np.clip(w_raw, clip[0], clip[1])
    w = w * n / w.sum()
    return w, rho, w_raw


def compute_weights(df: pd.DataFrame, method: str = "copula",
                    seed: int = SEED, clip: tuple[float, float] = CLIP):
    """Dispatcher. method ∈ {'copula', 'kde', 'classifier'}."""
    if method == "copula":
        return compute_weights_copula(df, clip=clip)
    if method == "kde":
        return compute_weights_kde(df, clip=clip)
    if method == "classifier":
        return compute_weights_classifier(df, seed=seed, clip=clip)
    raise ValueError(method)


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy))


def mae(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    err = np.abs(p - y)
    if w is None:
        return float(err.mean())
    return float(np.average(err, weights=w))


def design_matrix_with_x9(df: pd.DataFrame, x5_median: float) -> np.ndarray:
    """Simple linear design matrix including x9 — the feature reweighting targets."""
    is_sent = (df["x5"] == SENTINEL).astype(float).values
    x5 = df["x5"].where(df["x5"] != SENTINEL, x5_median).values
    city = (df["City"] == "Zaragoza").astype(float).values
    return np.column_stack([
        df["x1"].values ** 2,
        np.cos(5 * np.pi * df["x2"].values),
        df["x4"].values,
        x5,
        is_sent,
        df["x8"].values,
        df["x9"].values,
        df["x10"].values,
        df["x11"].values,
        city,
        df["x10"].values * df["x11"].values,
    ])


def cv_linear(df: pd.DataFrame, w: np.ndarray, use_weights: bool,
              with_x9: bool) -> np.ndarray:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        if with_x9:
            X_tr = design_matrix_with_x9(sub_tr, x5_med)
            X_va = design_matrix_with_x9(sub_va, x5_med)
        else:
            X_tr, _ = design_matrix(sub_tr, x5_med, include_interaction=True)
            X_va, _ = design_matrix(sub_va, x5_med, include_interaction=True)
        lr = LinearRegression()
        if use_weights:
            lr.fit(X_tr, sub_tr["target"].values, sample_weight=w[tr])
        else:
            lr.fit(X_tr, sub_tr["target"].values)
        oof[va] = lr.predict(X_va)
    return oof


def cv_ebm(df: pd.DataFrame, w: np.ndarray, use_weights: bool) -> np.ndarray:
    from interpret.glassbox import ExplainableBoostingRegressor

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    for tr, va in kf.split(df):
        sub_tr = df.iloc[tr].reset_index(drop=True)
        sub_va = df.iloc[va].reset_index(drop=True)
        x5_med = float(sub_tr.loc[sub_tr["x5"] != SENTINEL, "x5"].median())
        X_tr = preprocess(sub_tr, FEATURES_ALL, x5_med)
        X_va = preprocess(sub_va, FEATURES_ALL, x5_med)
        model = ExplainableBoostingRegressor(
            random_state=SEED, interactions=10, max_rounds=2000,
            min_samples_leaf=10, max_bins=128,
        )
        if use_weights:
            model.fit(X_tr, sub_tr["target"].values, sample_weight=w[tr])
        else:
            model.fit(X_tr, sub_tr["target"].values)
        oof[va] = model.predict(X_va)
    return oof


def build_linear_submission(train: pd.DataFrame, test: pd.DataFrame,
                            w: np.ndarray, name: str, with_x9: bool) -> None:
    x5_med = float(train.loc[train["x5"] != SENTINEL, "x5"].median())
    if with_x9:
        X_tr = design_matrix_with_x9(train, x5_med)
        X_te = design_matrix_with_x9(test, x5_med)
    else:
        X_tr, _ = design_matrix(train, x5_med, include_interaction=True)
        X_te, _ = design_matrix(test, x5_med, include_interaction=True)
    lr = LinearRegression().fit(X_tr, train["target"].values, sample_weight=w)
    preds = lr.predict(X_te)
    out = pd.DataFrame({"id": test["id"], "target": preds})
    out.to_csv(SUBS / f"submission_{name}.csv", index=False)
    print(f"  wrote submission_{name}.csv  "
          f"(mean={preds.mean():+.3f}, range=[{preds.min():+.2f}, {preds.max():+.2f}])")


def build_ebm_submission(train: pd.DataFrame, test: pd.DataFrame,
                         w: np.ndarray, name: str) -> None:
    from interpret.glassbox import ExplainableBoostingRegressor

    x5_med = float(train.loc[train["x5"] != SENTINEL, "x5"].median())
    X_tr = preprocess(train, FEATURES_ALL, x5_med)
    X_te = preprocess(test, FEATURES_ALL, x5_med)
    model = ExplainableBoostingRegressor(
        random_state=SEED, interactions=10, max_rounds=2000,
        min_samples_leaf=10, max_bins=128,
    )
    model.fit(X_tr, train["target"].values, sample_weight=w)
    preds = model.predict(X_te)
    out = pd.DataFrame({"id": test["id"], "target": preds})
    out.to_csv(SUBS / f"submission_{name}.csv", index=False)
    print(f"  wrote submission_{name}.csv  "
          f"(mean={preds.mean():+.3f}, range=[{preds.min():+.2f}, {preds.max():+.2f})]")


def plot_weights(df: pd.DataFrame, w: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sc = ax.scatter(df["x4"], df["x9"], c=w, s=12, cmap="viridis",
                    vmin=np.percentile(w, 2), vmax=np.percentile(w, 98))
    ax.set_xlabel("x4"); ax.set_ylabel("x9")
    ax.set_title(f"Training rows coloured by DRE weight\n"
                 f"(raw r={np.corrcoef(df['x4'], df['x9'])[0,1]:+.3f}, "
                 f"weighted r={weighted_corr(df['x4'], df['x9'], w):+.3f})")
    plt.colorbar(sc, ax=ax, label="sample weight")

    ax = axes[1]
    ax.hist(np.log10(w), bins=40, color="#4477aa", alpha=0.85)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("log10(weight)")
    ax.set_ylabel("count")
    ax.set_title(f"Weight distribution  clip={CLIP}  "
                 f"min={w.min():.2f} median={np.median(w):.2f} max={w.max():.2f}")

    fig.tight_layout()
    fig.savefig(OUT / "weights_overview.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT / 'weights_overview.png'}")


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv").reset_index(drop=True)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = df["target"].to_numpy()
    is_sent = (df["x5"] == SENTINEL).to_numpy()

    r_train = np.corrcoef(df["x4"], df["x9"])[0, 1]
    r_test = np.corrcoef(test["x4"], test["x9"])[0, 1]
    print(f"corr(x4, x9): train={r_train:+.3f}  test={r_test:+.3f}")

    # Joint support diagnostic — the core reason reweighting may fail.
    hi = df[df["x4"] > 0]; lo = df[df["x4"] < 0]
    print(f"joint support diagnostic:")
    print(f"  x4>0 cluster (n={len(hi)}): x9 ~ N({hi['x9'].mean():.2f}, {hi['x9'].std():.2f})")
    print(f"  x4<0 cluster (n={len(lo)}): x9 ~ N({lo['x9'].mean():.2f}, {lo['x9'].std():.2f})")
    print(f"  clusters separated by {hi['x9'].mean()-lo['x9'].mean():.2f} in x9")
    print(f"  within-cluster corr(x4, x9): "
          f"hi={np.corrcoef(hi['x4'], hi['x9'])[0,1]:+.3f}  "
          f"lo={np.corrcoef(lo['x4'], lo['x9'])[0,1]:+.3f}")
    print(f"  test corr(x4, x9)={r_test:+.3f}  =>  test has (x4>0, x9<5) and "
          f"(x4<0, x9>5) rows that DO NOT EXIST in training.")

    t0 = time.time()
    w, probs, w_raw = compute_weights(df, method="copula")
    print(f"weights computed in {time.time()-t0:.1f}s (method=copula)")
    r_w = weighted_corr(df["x4"].to_numpy(), df["x9"].to_numpy(), w)
    print(f"weighted corr(x4, x9) after DRE: {r_w:+.3f}")
    n_clipped_lo = int((w_raw < CLIP[0]).sum())
    n_clipped_hi = int((w_raw > CLIP[1]).sum())
    print(f"clipped: {n_clipped_lo} below {CLIP[0]}, {n_clipped_hi} above {CLIP[1]} "
          f"(of {len(w)} rows)")
    print(f"weight percentiles: p1={np.percentile(w,1):.3f}  "
          f"p5={np.percentile(w,5):.3f}  p50={np.median(w):.3f}  "
          f"p95={np.percentile(w,95):.3f}  p99={np.percentile(w,99):.3f}")
    plot_weights(df, w)
    pd.DataFrame({"id": df["id"], "weight": w, "prob_joint": probs}).to_csv(
        OUT / "weights.csv", index=False
    )
    print(f"  wrote {OUT / 'weights.csv'}")

    print("\n" + "=" * 78)
    print("Simple linear — three variants")
    print("=" * 78)
    for label, use_w, with_x9 in [
        ("no_x9 unweighted",   False, False),
        ("no_x9 weighted",     True,  False),
        ("with_x9 unweighted", False, True),
        ("with_x9 weighted",   True,  True),
    ]:
        t0 = time.time()
        oof = cv_linear(df, w, use_weights=use_w, with_x9=with_x9)
        m = mae(oof, y)
        m_w = mae(oof, y, w)
        m_ns = mae(oof[~is_sent], y[~is_sent])
        print(f"  {label:22s}  CV MAE={m:.3f}  CV MAE(w)={m_w:.3f}  "
              f"non-sent={m_ns:.3f}  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 78)
    print("EBM (baseline params, x9 included)")
    print("=" * 78)
    for label, use_w in [("unweighted", False), ("weighted", True)]:
        t0 = time.time()
        oof = cv_ebm(df, w, use_weights=use_w)
        m = mae(oof, y)
        m_w = mae(oof, y, w)
        m_ns = mae(oof[~is_sent], y[~is_sent])
        print(f"  {label:22s}  CV MAE={m:.3f}  CV MAE(w)={m_w:.3f}  "
              f"non-sent={m_ns:.3f}  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 78)
    print("Building weighted submissions (fit on full dataset with sample_weight=w)")
    print("=" * 78)
    build_linear_submission(df, test, w, name="linear_reweighted_with_x9", with_x9=True)
    build_linear_submission(df, test, w, name="linear_reweighted_no_x9",  with_x9=False)
    build_ebm_submission(df, test, w, name="ebm_reweighted")


if __name__ == "__main__":
    main()
