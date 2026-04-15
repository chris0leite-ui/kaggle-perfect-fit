"""Diagnostic analyses: sentinel missingness, residuals, nonlinear shapes,
R-squared ceiling, x4 bimodality."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pygam import LinearGAM, s as spline, l as linear
from scipy import stats as sp_stats

from src.clusters import replace_sentinels, find_x4_gap


# ---------------------------------------------------------------------------
# TODO 1: x5 sentinel missingness
# ---------------------------------------------------------------------------

def sentinel_indicator(x5: pd.Series, sentinel: float = 999.0) -> pd.Series:
    """Binary indicator: 1 where x5 equals sentinel, 0 elsewhere."""
    return (x5 == sentinel).astype(int)


def sentinel_cluster_crosstab(df: pd.DataFrame, cluster_col: str,
                              sentinel_col: str = "x5",
                              sentinel: float = 999.0) -> pd.DataFrame:
    """Cross-tab of sentinel vs non-sentinel rows across clusters.

    Returns DataFrame with index=cluster labels,
    columns=['sentinel', 'non_sentinel'].
    """
    indicator = sentinel_indicator(df[sentinel_col], sentinel)
    ct = pd.crosstab(df[cluster_col], indicator)
    ct.columns = ["non_sentinel", "sentinel"] if 0 in ct.columns else ["sentinel"]
    # Ensure both columns exist
    for col in ["sentinel", "non_sentinel"]:
        if col not in ct.columns:
            ct[col] = 0
    return ct[["sentinel", "non_sentinel"]]


def sentinel_chi2_test(df: pd.DataFrame, cluster_col: str,
                       sentinel_col: str = "x5",
                       sentinel: float = 999.0) -> dict:
    """Chi-squared test for independence of sentinel status across clusters."""
    ct = sentinel_cluster_crosstab(df, cluster_col, sentinel_col, sentinel)
    chi2, p, dof, expected = sp_stats.chi2_contingency(ct.values)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": pd.DataFrame(expected, index=ct.index, columns=ct.columns),
    }


def sentinel_target_regression(df: pd.DataFrame, sentinel_col: str = "x5",
                               target_col: str = "target",
                               sentinel: float = 999.0) -> dict:
    """OLS: target ~ x5_imputed + x5_is_sentinel.

    Tests whether the sentinel indicator adds predictive power beyond
    the imputed x5 value.
    """
    x5_imputed = replace_sentinels(df[sentinel_col], sentinel)
    x5_is_sentinel = sentinel_indicator(df[sentinel_col], sentinel)
    X = sm.add_constant(pd.DataFrame({
        "x5_imputed": x5_imputed.values,
        "x5_sentinel": x5_is_sentinel.values,
    }))
    model = sm.OLS(df[target_col].values, X).fit()
    return {
        "coef_x5_imputed": float(model.params["x5_imputed"]),
        "coef_x5_sentinel": float(model.params["x5_sentinel"]),
        "pvalue_x5_imputed": float(model.pvalues["x5_imputed"]),
        "pvalue_x5_sentinel": float(model.pvalues["x5_sentinel"]),
        "r_squared": float(model.rsquared),
    }


# ---------------------------------------------------------------------------
# TODO 2: Residual analysis after City + x4
# ---------------------------------------------------------------------------

def fit_city_x4_ols(df: pd.DataFrame, target_col: str = "target"):
    """Fit OLS: target ~ City_binary + x4. Returns fitted result."""
    city_bin = (df["City"] == "Zaragoza").astype(float)
    X = sm.add_constant(pd.DataFrame({
        "City": city_bin.values,
        "x4": df["x4"].values,
    }))
    return sm.OLS(df[target_col].values, X).fit()


def residual_normality_test(residuals) -> dict:
    """Shapiro-Wilk test for normality of residuals."""
    stat, p = sp_stats.shapiro(residuals)
    return {"statistic": float(stat), "p_value": float(p)}


def residual_heteroscedasticity_test(residuals, cluster_labels: pd.Series) -> dict:
    """Levene's test for equal variance of residuals across clusters."""
    groups = []
    for label in cluster_labels.unique():
        mask = cluster_labels.values == label
        groups.append(np.asarray(residuals)[mask])
    stat, p = sp_stats.levene(*groups)
    return {"statistic": float(stat), "p_value": float(p)}


def residual_stats(ols_result) -> dict:
    """Summary statistics from a fitted OLS model."""
    resid = ols_result.resid
    return {
        "r_squared": float(ols_result.rsquared),
        "adj_r_squared": float(ols_result.rsquared_adj),
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=1)),
        "residual_skew": float(sp_stats.skew(resid)),
        "residual_kurtosis": float(sp_stats.kurtosis(resid)),
    }


# ---------------------------------------------------------------------------
# TODO 3: x1/x2 nonlinear shapes within clusters
# ---------------------------------------------------------------------------

def fit_gam_per_cluster(df: pd.DataFrame, feature: str, cluster_col: str,
                        target_col: str = "target",
                        n_splines: int = 20) -> dict:
    """Fit GAM splines for feature->target within each cluster.

    Returns dict[cluster_label, dict] with x_grid, y_pred, r_squared, n.
    """
    results = {}
    for label in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == label]
        x = subset[feature].values
        y = subset[target_col].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        gam = LinearGAM(spline(0, n_splines=min(n_splines, len(x) // 3)))
        gam.fit(x.reshape(-1, 1), y)
        x_grid = np.linspace(x.min(), x.max(), 200)
        y_pred = gam.predict(x_grid.reshape(-1, 1))
        ss_res = np.sum((y - gam.predict(x.reshape(-1, 1))) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[label] = {
            "x_grid": x_grid,
            "y_pred": y_pred,
            "r_squared": float(r2),
            "n": len(x),
        }
    return results


def pooled_vs_cluster_gam_test(df: pd.DataFrame, feature: str,
                               cluster_col: str,
                               target_col: str = "target",
                               n_splines: int = 20) -> dict:
    """Compare pooled GAM vs per-cluster GAMs via residual F-test.

    Returns dict with rss_pooled, rss_cluster, f_stat, p_value.
    """
    x_all = df[feature].values
    y_all = df[target_col].values
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all = x_all[mask], y_all[mask]

    # Pooled GAM
    gam_pooled = LinearGAM(spline(0, n_splines=n_splines))
    gam_pooled.fit(x_all.reshape(-1, 1), y_all)
    rss_pooled = float(np.sum((y_all - gam_pooled.predict(x_all.reshape(-1, 1))) ** 2))
    df_pooled = len(y_all) - gam_pooled.statistics_["edof"]

    # Per-cluster GAMs
    rss_cluster = 0.0
    df_cluster_total = 0.0
    for label in sorted(df[cluster_col].unique()):
        subset = df[df[cluster_col] == label]
        x = subset[feature].values
        y = subset[target_col].values
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ns = min(n_splines, len(x) // 3)
        gam_c = LinearGAM(spline(0, n_splines=ns))
        gam_c.fit(x.reshape(-1, 1), y)
        rss_cluster += float(np.sum((y - gam_c.predict(x.reshape(-1, 1))) ** 2))
        df_cluster_total += len(y) - gam_c.statistics_["edof"]

    delta_df = df_pooled - df_cluster_total
    if delta_df > 0 and df_cluster_total > 0:
        f_stat = ((rss_pooled - rss_cluster) / delta_df) / (rss_cluster / df_cluster_total)
        p_value = float(sp_stats.f.sf(f_stat, delta_df, df_cluster_total))
    else:
        f_stat = 0.0
        p_value = 1.0

    return {
        "rss_pooled": rss_pooled,
        "rss_cluster": float(rss_cluster),
        "f_stat": float(f_stat),
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# TODO 4: Total R-squared ceiling
# ---------------------------------------------------------------------------

def _prepare_ceiling_data(df: pd.DataFrame, sentinel: float = 999.0):
    """Prepare feature matrix and target for the ceiling model.

    Returns (X_df, y) where X_df has columns:
    City, x4, x1, x2, x5, x8, x10, x11.
    """
    x5_clean = replace_sentinels(df["x5"], sentinel)
    city_bin = (df["City"] == "Zaragoza").astype(float)
    X_df = pd.DataFrame({
        "City": city_bin.values,
        "x4": df["x4"].values,
        "x1": df["x1"].values,
        "x2": df["x2"].values,
        "x5": x5_clean.values,
        "x8": df["x8"].values,
        "x10": df["x10"].values,
        "x11": df["x11"].values,
    })
    y = df["target"].values
    return X_df, y


def fit_r2_ceiling(df: pd.DataFrame, sentinel: float = 999.0,
                   target_col: str = "target") -> dict:
    """Fit combined model and report R-squared ceiling.

    Model: linear(City, x4, x5, x8, x10, x11) + spline(x1) + spline(x2).
    """
    X_df, y = _prepare_ceiling_data(df, sentinel)
    X = X_df.values

    # Column indices: City=0, x4=1, x1=2, x2=3, x5=4, x8=5, x10=6, x11=7
    terms = (linear(0) + linear(1) + spline(2, n_splines=20)
             + spline(3, n_splines=20) + linear(4) + linear(5)
             + linear(6) + linear(7))
    gam = LinearGAM(terms)
    gam.fit(X, y)

    y_pred = gam.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(y)
    p = gam.statistics_["edof"]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else r2

    return {
        "r_squared": float(r2),
        "adj_r_squared": float(adj_r2),
        "residual_std": float(np.std(y - y_pred, ddof=1)),
        "residual_mean": float(np.mean(y - y_pred)),
        "n": n,
    }


def feature_group_r2_breakdown(df: pd.DataFrame,
                               sentinel: float = 999.0) -> pd.DataFrame:
    """Incremental R-squared from feature groups added sequentially.

    Groups (in order):
      1. City + x4 (linear)
      2. + x1 (spline)
      3. + x2 (spline)
      4. + x5 + x8 + x10 + x11 (linear)
    """
    X_df, y = _prepare_ceiling_data(df, sentinel)
    X = X_df.values
    ss_tot = np.sum((y - y.mean()) ** 2)

    groups = [
        ("City + x4", linear(0) + linear(1)),
        ("+ x1 (spline)", linear(0) + linear(1) + spline(2, n_splines=20)),
        ("+ x2 (spline)", linear(0) + linear(1) + spline(2, n_splines=20)
         + spline(3, n_splines=20)),
        ("+ x5,x8,x10,x11", linear(0) + linear(1) + spline(2, n_splines=20)
         + spline(3, n_splines=20) + linear(4) + linear(5) + linear(6)
         + linear(7)),
    ]

    rows = []
    prev_r2 = 0.0
    for name, terms in groups:
        gam = LinearGAM(terms)
        gam.fit(X, y)
        ss_res = np.sum((y - gam.predict(X)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rows.append({
            "group": name,
            "cumulative_r2": float(r2),
            "marginal_r2": float(r2 - prev_r2),
        })
        prev_r2 = r2

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TODO 5: x4 bimodal origin
# ---------------------------------------------------------------------------

def x4_bimodality_test(x4_values: pd.Series) -> dict:
    """Test x4 for bimodality using gap-ratio statistic.

    Computes max_gap / median_gap among sorted consecutive differences.
    Also reports the gap location from find_x4_gap().
    """
    gap_start, gap_end, _ = find_x4_gap(x4_values)
    vals = np.sort(x4_values.dropna().values)
    diffs = np.diff(vals)
    median_gap = float(np.median(diffs))
    max_gap = float(np.max(diffs))
    ratio = max_gap / median_gap if median_gap > 0 else float("inf")

    return {
        "method": "gap_ratio",
        "statistic": ratio,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_width": gap_end - gap_start,
    }


def x4_city_distribution_test(df: pd.DataFrame) -> dict:
    """Two-sample KS test: x4 distribution in Albacete vs Zaragoza."""
    alb = df.loc[df["City"] == "Albacete", "x4"].dropna().values
    zar = df.loc[df["City"] == "Zaragoza", "x4"].dropna().values
    stat, p = sp_stats.ks_2samp(alb, zar)
    return {
        "ks_statistic": float(stat),
        "p_value": float(p),
        "mean_albacete": float(np.mean(alb)),
        "mean_zaragoza": float(np.mean(zar)),
        "std_albacete": float(np.std(alb, ddof=1)),
        "std_zaragoza": float(np.std(zar, ddof=1)),
    }


def x4_gap_analysis(x4_values: pd.Series) -> dict:
    """Characterize the x4 gap in detail."""
    gap_start, gap_end, _ = find_x4_gap(x4_values)
    vals = np.sort(x4_values.dropna().values)
    diffs = np.diff(vals)
    median_diff = float(np.median(diffs))
    gap_width = gap_end - gap_start

    n_below = int(np.sum(vals < gap_start) + np.sum(vals == gap_start))
    n_above = int(np.sum(vals > gap_end) + np.sum(vals == gap_end))
    total = len(vals)

    return {
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_width": gap_width,
        "n_below": n_below,
        "n_above": n_above,
        "frac_below": n_below / total if total > 0 else 0.0,
        "frac_above": n_above / total if total > 0 else 0.0,
        "nearest_below": float(vals[vals <= gap_start].max()) if n_below > 0 else None,
        "nearest_above": float(vals[vals >= gap_end].min()) if n_above > 0 else None,
        "exact_gap": gap_width > 2 * median_diff,
    }
