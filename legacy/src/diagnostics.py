"""Model diagnostics: SHAP, distribution shift, residual analysis, EBM explanations."""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.clusters import assign_clusters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "target"
FEATURES_FOR_SHIFT = ["x1", "x2", "x4", "x5", "x8", "x9", "x10", "x11"]
TOP_SHAP_FEATURES = ["x4", "City", "x1", "x2", "x5", "x8"]


# ---------------------------------------------------------------------------
# 5.2  Distribution shift
# ---------------------------------------------------------------------------

def compute_ks_tests(train_df, val_df, test_df, features):
    """2-sample KS test per feature: train vs val, train vs test."""
    rows = []
    for feat in features:
        tr = train_df[feat].dropna().values
        va = val_df[feat].dropna().values
        te = test_df[feat].dropna().values
        ks_val, p_val = stats.ks_2samp(tr, va)
        ks_test, p_test = stats.ks_2samp(tr, te)
        rows.append({
            "feature": feat,
            "ks_stat_val": ks_val,
            "p_val": p_val,
            "ks_stat_test": ks_test,
            "p_test": p_test,
        })
    return pd.DataFrame(rows)


def plot_distribution_shifts(train_df, val_df, test_df, features, out_dir):
    """Overlay histograms per feature across train/val/test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for feat in features:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = 40
        ax.hist(train_df[feat].dropna(), bins=bins, alpha=0.5,
                label=f"Train (n={len(train_df)})", density=True, color="steelblue")
        ax.hist(val_df[feat].dropna(), bins=bins, alpha=0.5,
                label=f"Val (n={len(val_df)})", density=True, color="coral")
        ax.hist(test_df[feat].dropna(), bins=bins, alpha=0.5,
                label=f"Test (n={len(test_df)})", density=True, color="seagreen")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {feat}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / f"dist_{feat}.png", dpi=120)
        plt.close(fig)

    # Composite summary
    n_feats = len(features)
    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(train_df[feat].dropna(), bins=30, alpha=0.5,
                label="Train", density=True, color="steelblue")
        ax.hist(val_df[feat].dropna(), bins=30, alpha=0.5,
                label="Val", density=True, color="coral")
        ax.hist(test_df[feat].dropna(), bins=30, alpha=0.5,
                label="Test", density=True, color="seagreen")
        ax.set_title(feat, fontsize=10)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distribution Shift: Train vs Val vs Test", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "dist_summary.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.3  Residual analysis
# ---------------------------------------------------------------------------

def compute_residuals(model, train_df, val_df):
    """Fit model on train, predict on val, return augmented val_df with residuals."""
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    result = val_df.copy()
    result["predicted"] = preds
    result["residual"] = result[TARGET] - preds
    result["abs_residual"] = result["residual"].abs()

    # Add cluster labels
    result["cluster"] = assign_clusters(result).values
    return result


def compute_cluster_mae(residual_df):
    """Per-cluster MAE breakdown."""
    groups = residual_df.groupby("cluster")
    rows = []
    for cluster, grp in groups:
        rows.append({
            "cluster": cluster,
            "n": len(grp),
            "mae": grp["abs_residual"].mean(),
            "std_residual": grp["residual"].std(),
        })
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def plot_residuals_vs_predicted(residual_df, model_name, out_dir):
    """Scatter: predicted vs residual."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(residual_df["predicted"], residual_df["residual"],
               alpha=0.5, s=20, c="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals vs Predicted — {model_name}")
    plt.tight_layout()
    fig.savefig(out_dir / f"resid_{model_name}_vs_predicted.png", dpi=120)
    plt.close(fig)


def plot_residuals_vs_features(residual_df, features, model_name, out_dir):
    """Scatter: residual vs each feature."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in features:
        if feat not in residual_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(residual_df[feat], residual_df["residual"],
                   alpha=0.5, s=20, c="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals vs {feat} — {model_name}")
        plt.tight_layout()
        fig.savefig(out_dir / f"resid_{model_name}_{feat}.png", dpi=120)
        plt.close(fig)


def plot_cluster_mae_comparison(all_cluster_results, out_dir):
    """Grouped bar chart: cluster x model MAE."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combine into a single DataFrame
    rows = []
    for model_name, cluster_df in all_cluster_results.items():
        for _, row in cluster_df.iterrows():
            rows.append({
                "model": model_name,
                "cluster": row["cluster"],
                "mae": row["mae"],
            })
    combined = pd.DataFrame(rows)
    if combined.empty:
        return

    clusters = sorted(combined["cluster"].unique())
    models = list(all_cluster_results.keys())
    n_models = len(models)
    x = np.arange(len(clusters))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        model_data = combined[combined["model"] == model]
        maes = [model_data[model_data["cluster"] == c]["mae"].values[0]
                if c in model_data["cluster"].values else 0
                for c in clusters]
        ax.bar(x + i * width, maes, width, label=model, alpha=0.8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Cluster MAE by Model")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(clusters, fontsize=8, rotation=15)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "cluster_mae_comparison.png", dpi=120)
    plt.close(fig)


def plot_residual_qq(residual_df, model_name, out_dir):
    """QQ plot of residuals against normal distribution."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    resid = residual_df["residual"].values
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title(f"Residual QQ Plot — {model_name}")
    plt.tight_layout()
    fig.savefig(out_dir / f"resid_{model_name}_qq.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.4  EBM native explanations
# ---------------------------------------------------------------------------

def plot_ebm_global_shapes(fitted_ebm_pipeline, out_dir):
    """Plot EBM's learned shape functions for each feature using explain_global()."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ebm = fitted_ebm_pipeline.named_steps["model"].ebm_
    exp = ebm.explain_global()

    for i, term_name in enumerate(ebm.term_names_):
        if " & " in term_name:
            continue  # skip interactions (handled separately)

        data = exp.data(i)
        names = data["names"]
        scores = np.array(data["scores"])

        if len(scores) <= 2:
            # Binary/categorical feature — bar plot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(range(len(scores)), scores, color="steelblue", alpha=0.8)
            ax.set_xticks(range(len(scores)))
            # Labels are bin edges; for binary, show 0 and 1
            if len(names) == 3:
                ax.set_xticklabels([f"{names[0]:.0f}", f"{names[2]:.0f}"])
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        else:
            # Continuous feature — line plot
            # names has one more entry than scores (bin edges)
            midpoints = [(float(names[j]) + float(names[j + 1])) / 2
                         for j in range(len(scores))]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(midpoints, scores, color="steelblue", linewidth=1.2)
            ax.fill_between(midpoints, scores, alpha=0.15, color="steelblue")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

            # Add confidence bounds if available
            if "upper_bounds" in data and "lower_bounds" in data:
                upper = np.array(data["upper_bounds"])
                lower = np.array(data["lower_bounds"])
                ax.fill_between(midpoints, lower, upper,
                                alpha=0.08, color="steelblue")

        ax.set_xlabel(term_name)
        ax.set_ylabel("Score contribution")
        ax.set_title(f"EBM Shape: {term_name}")
        plt.tight_layout()
        safe_name = term_name.replace(" ", "_")
        fig.savefig(out_dir / f"ebm_shape_{safe_name}.png", dpi=120)
        plt.close(fig)


def plot_ebm_interactions(fitted_ebm_pipeline, out_dir):
    """Plot EBM's discovered pairwise interactions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ebm = fitted_ebm_pipeline.named_steps["model"].ebm_

    interaction_terms = []
    for i, name in enumerate(ebm.term_names_):
        if " & " in name:
            importance = np.abs(ebm.term_scores_[i]).mean()
            interaction_terms.append((name, i, importance))

    if not interaction_terms:
        return

    # Sort by importance
    interaction_terms.sort(key=lambda x: x[2], reverse=True)

    # Bar chart of interaction strengths
    fig, ax = plt.subplots(figsize=(8, max(3, len(interaction_terms) * 0.4)))
    names = [t[0] for t in interaction_terms]
    importances = [t[2] for t in interaction_terms]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, importances, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean |Score|")
    ax.set_title("EBM Pairwise Interactions")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "ebm_interactions.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.1  SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap_values(model, train_df, val_df):
    """Compute SHAP values by splitting pipeline into preprocessor + estimator.

    Returns (shap_values, preprocessed_val_array, feature_names).
    """
    import shap

    # Split pipeline
    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")

    X_train = train_df.drop(columns=[TARGET])
    X_val = val_df.drop(columns=[TARGET])

    # Preprocess
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    if isinstance(X_train_proc, pd.DataFrame):
        feature_names = list(X_train_proc.columns)
        X_train_arr = X_train_proc.values
        X_val_arr = X_val_proc.values
    else:
        feature_names = [f"f{i}" for i in range(X_train_proc.shape[1])]
        X_train_arr = X_train_proc
        X_val_arr = X_val_proc

    # Choose explainer based on model type
    model_type = type(estimator).__name__
    if model_type in ("LGBMRegressor", "HistGradientBoostingRegressor"):
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_val_arr)
    else:
        # KernelExplainer for GAM, linear, etc.
        background = shap.kmeans(X_train_arr, 50)
        predict_fn = estimator.predict
        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(X_val_arr, nsamples=100)

    return sv, X_val_arr, feature_names


def plot_shap_summary(shap_values, features_array, feature_names, model_name,
                      out_dir):
    """SHAP beeswarm/summary plot."""
    import shap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, features_array, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title(f"SHAP Summary — {model_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_{model_name}_summary.png", dpi=120,
                bbox_inches="tight")
    plt.close("all")


def plot_shap_dependence(shap_values, features_array, feature_names,
                         feature_name, model_name, out_dir):
    """SHAP dependence plot for a single feature."""
    import shap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if feature_name not in feature_names:
        return
    idx = feature_names.index(feature_name)

    fig, ax = plt.subplots(figsize=(6, 5))
    shap.dependence_plot(idx, shap_values, features_array,
                         feature_names=feature_names, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence: {feature_name} — {model_name}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_{model_name}_{feature_name}.png", dpi=120)
    plt.close(fig)


def plot_shap_importance_comparison(all_shap_results, out_dir):
    """Side-by-side bar chart of mean |SHAP| per feature across models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect importances
    importance_data = {}
    all_features = set()
    for model_name, (sv, _, fnames) in all_shap_results.items():
        mean_abs = np.abs(sv).mean(axis=0)
        importance_data[model_name] = dict(zip(fnames, mean_abs))
        all_features.update(fnames)

    # Build comparison DataFrame
    all_features = sorted(all_features)
    rows = []
    for feat in all_features:
        row = {"feature": feat}
        for model_name, imp_dict in importance_data.items():
            row[model_name] = imp_dict.get(feat, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("feature")

    # Sort by mean importance across models
    df["_mean"] = df.mean(axis=1)
    df = df.sort_values("_mean", ascending=True).drop(columns=["_mean"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    df.plot.barh(ax=ax, alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance Comparison")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_importance_comparison.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.5  HTML viewer update
# ---------------------------------------------------------------------------

def _png_to_base64(path):
    """Read a PNG file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _make_card(label, img_path):
    """Generate HTML card with base64-embedded image."""
    b64 = _png_to_base64(img_path)
    return f'''<div class="card">
  <div class="card-label">{label}</div>
  <img src="data:image/png;base64,{b64}" alt="{label}">
</div>'''


def update_index_html(diag_dir, html_path=None):
    """Append diagnostic sections to plots/index.html."""
    diag_dir = Path(diag_dir)
    if html_path is None:
        html_path = diag_dir.parent / "index.html"

    # Read existing HTML
    with open(html_path, "r") as f:
        html = f.read()

    # Remove closing </body></html> to append
    html = html.replace("</body>", "").replace("</html>", "").rstrip()

    sections = []

    # --- Distribution shift section ---
    dist_summary = diag_dir / "dist_summary.png"
    if dist_summary.exists():
        cards = [_make_card("Distribution Summary", dist_summary)]
        dist_files = sorted(diag_dir.glob("dist_x*.png"))
        for f in dist_files:
            feat = f.stem.replace("dist_", "")
            cards.append(_make_card(feat, f))
        sections.append(f'''
<section>
  <h2>Distribution Shift: Train vs Val vs Test</h2>
  <div class="grid">
    {"".join(cards)}
  </div>
</section>''')

    # --- Residual analysis section ---
    cluster_cmp = diag_dir / "cluster_mae_comparison.png"
    resid_cards = []
    if cluster_cmp.exists():
        resid_cards.append(_make_card("Cluster MAE Comparison", cluster_cmp))
    for f in sorted(diag_dir.glob("resid_*_vs_predicted.png")):
        model = f.stem.replace("resid_", "").replace("_vs_predicted", "")
        resid_cards.append(_make_card(f"Residuals — {model}", f))
    for f in sorted(diag_dir.glob("resid_*_qq.png")):
        model = f.stem.replace("resid_", "").replace("_qq", "")
        resid_cards.append(_make_card(f"QQ — {model}", f))
    if resid_cards:
        sections.append(f'''
<section>
  <h2>Residual Analysis</h2>
  <div class="grid">
    {"".join(resid_cards)}
  </div>
</section>''')

    # --- SHAP section ---
    shap_cards = []
    for f in sorted(diag_dir.glob("shap_*_summary.png")):
        model = f.stem.replace("shap_", "").replace("_summary", "")
        shap_cards.append(_make_card(f"SHAP — {model}", f))
    shap_cmp = diag_dir / "shap_importance_comparison.png"
    if shap_cmp.exists():
        shap_cards.insert(0, _make_card("Importance Comparison", shap_cmp))
    if shap_cards:
        sections.append(f'''
<section>
  <h2>SHAP Explanations</h2>
  <div class="grid">
    {"".join(shap_cards)}
  </div>
</section>''')

    # --- EBM section ---
    ebm_cards = []
    for f in sorted(diag_dir.glob("ebm_shape_*.png")):
        feat = f.stem.replace("ebm_shape_", "")
        ebm_cards.append(_make_card(f"EBM Shape: {feat}", f))
    ebm_inter = diag_dir / "ebm_interactions.png"
    if ebm_inter.exists():
        ebm_cards.insert(0, _make_card("EBM Interactions", ebm_inter))
    if ebm_cards:
        sections.append(f'''
<section>
  <h2>EBM Native Explanations</h2>
  <div class="grid">
    {"".join(ebm_cards)}
  </div>
</section>''')

    # Append and close
    html += "\n".join(sections)
    html += "\n</body>\n</html>\n"

    with open(html_path, "w") as f:
        f.write(html)
