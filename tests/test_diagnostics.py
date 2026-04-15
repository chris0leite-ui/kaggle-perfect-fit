"""Tests for src/diagnostics — red-green TDD."""

import numpy as np
import pandas as pd
import pytest

from src.clusters import assign_clusters
from src.diagnostics import (
    sentinel_indicator,
    sentinel_cluster_crosstab,
    sentinel_chi2_test,
    sentinel_target_regression,
    fit_city_x4_ols,
    residual_normality_test,
    residual_heteroscedasticity_test,
    residual_stats,
    fit_gam_per_cluster,
    pooled_vs_cluster_gam_test,
    fit_r2_ceiling,
    feature_group_r2_breakdown,
    x4_bimodality_test,
    x4_city_distribution_test,
    x4_gap_analysis,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _diagnostic_df(n=200, seed=42):
    """Synthetic DataFrame mimicking competition structure with 4 clusters."""
    rng = np.random.default_rng(seed)
    n_each = n // 4

    cities = (["Albacete"] * n_each * 2) + (["Zaragoza"] * n_each * 2)
    x4_vals = np.concatenate([
        rng.uniform(-0.5, -0.2, n_each),   # Albacete low
        rng.uniform(0.2, 0.5, n_each),     # Albacete high
        rng.uniform(-0.5, -0.2, n_each),   # Zaragoza low
        rng.uniform(0.2, 0.5, n_each),     # Zaragoza high
    ])
    city_binary = np.array([1.0 if c == "Zaragoza" else 0.0 for c in cities])

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x5_clean = rng.standard_normal(n) * 3 + 10
    sentinel_mask = rng.choice(n, size=n // 8, replace=False)
    x5 = x5_clean.copy()
    x5[sentinel_mask] = 999.0

    x8 = rng.standard_normal(n)
    x10 = rng.standard_normal(n)
    x11 = rng.standard_normal(n)

    target = (23.0 * city_binary + 31.0 * x4_vals
              + 3.0 * x1**2 + 2.0 * np.sin(3 * x2)
              - 8.0 * x5_clean + 12.0 * x8 + 2.5 * x10 + 2.8 * x11
              + rng.standard_normal(n) * 5)

    return pd.DataFrame({
        "City": cities,
        "x4": x4_vals,
        "x1": x1, "x2": x2, "x5": x5,
        "x8": x8, "x10": x10, "x11": x11,
        "target": target,
    })


# ===================================================================
# TODO 1: sentinel_indicator
# ===================================================================

def test_sentinel_indicator_returns_binary():
    s = pd.Series([1.0, 999.0, 3.0, 999.0])
    result = sentinel_indicator(s)
    assert set(result.unique()).issubset({0, 1})


def test_sentinel_indicator_correct_count():
    s = pd.Series([1.0, 999.0, 3.0, 999.0, 5.0])
    result = sentinel_indicator(s)
    assert result.sum() == 2


def test_sentinel_indicator_preserves_index():
    s = pd.Series([1.0, 999.0, 3.0], index=[10, 20, 30])
    result = sentinel_indicator(s)
    assert list(result.index) == [10, 20, 30]


# ===================================================================
# TODO 1: sentinel_cluster_crosstab
# ===================================================================

def test_sentinel_crosstab_shape():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = sentinel_cluster_crosstab(df, "cluster")
    assert result.shape[0] == 4
    assert "sentinel" in result.columns
    assert "non_sentinel" in result.columns


def test_sentinel_crosstab_totals():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = sentinel_cluster_crosstab(df, "cluster")
    total = result["sentinel"].sum() + result["non_sentinel"].sum()
    assert total == len(df)


# ===================================================================
# TODO 1: sentinel_chi2_test
# ===================================================================

def test_sentinel_chi2_returns_keys():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = sentinel_chi2_test(df, "cluster")
    assert "chi2" in result
    assert "p_value" in result
    assert "dof" in result


def test_sentinel_chi2_p_value_range():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = sentinel_chi2_test(df, "cluster")
    assert 0 <= result["p_value"] <= 1


# ===================================================================
# TODO 1: sentinel_target_regression
# ===================================================================

def test_sentinel_regression_returns_keys():
    df = _diagnostic_df()
    result = sentinel_target_regression(df)
    assert "coef_x5_imputed" in result
    assert "coef_x5_sentinel" in result
    assert "pvalue_x5_sentinel" in result
    assert "r_squared" in result


def test_sentinel_regression_r2_range():
    df = _diagnostic_df()
    result = sentinel_target_regression(df)
    assert 0 <= result["r_squared"] <= 1


# ===================================================================
# TODO 2: fit_city_x4_ols
# ===================================================================

def test_fit_city_x4_ols_returns_result():
    df = _diagnostic_df()
    result = fit_city_x4_ols(df)
    assert hasattr(result, "resid")
    assert hasattr(result, "params")
    assert hasattr(result, "rsquared")


def test_fit_city_x4_ols_positive_r2():
    df = _diagnostic_df()
    result = fit_city_x4_ols(df)
    assert result.rsquared > 0


# ===================================================================
# TODO 2: residual_normality_test
# ===================================================================

def test_residual_normality_returns_keys():
    residuals = np.random.default_rng(0).standard_normal(100)
    result = residual_normality_test(residuals)
    assert "statistic" in result
    assert "p_value" in result


def test_residual_normality_normal_data():
    residuals = np.random.default_rng(0).standard_normal(100)
    result = residual_normality_test(residuals)
    assert result["p_value"] > 0.01


# ===================================================================
# TODO 2: residual_heteroscedasticity_test
# ===================================================================

def test_heteroscedasticity_returns_keys():
    residuals = np.random.default_rng(0).standard_normal(80)
    labels = pd.Series(["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20)
    result = residual_heteroscedasticity_test(residuals, labels)
    assert "statistic" in result
    assert "p_value" in result


def test_heteroscedasticity_equal_variance():
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(200)
    labels = pd.Series(["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50)
    result = residual_heteroscedasticity_test(residuals, labels)
    assert result["p_value"] > 0.01


# ===================================================================
# TODO 2: residual_stats
# ===================================================================

def test_residual_stats_keys():
    df = _diagnostic_df()
    ols_result = fit_city_x4_ols(df)
    result = residual_stats(ols_result)
    for key in ["r_squared", "adj_r_squared", "residual_mean", "residual_std",
                "residual_skew", "residual_kurtosis"]:
        assert key in result


def test_residual_stats_mean_near_zero():
    df = _diagnostic_df()
    ols_result = fit_city_x4_ols(df)
    result = residual_stats(ols_result)
    assert abs(result["residual_mean"]) < 1.0


# ===================================================================
# TODO 3: fit_gam_per_cluster
# ===================================================================

def test_gam_per_cluster_all_clusters():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = fit_gam_per_cluster(df, "x1", "cluster")
    assert len(result) == 4


def test_gam_per_cluster_keys():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = fit_gam_per_cluster(df, "x1", "cluster")
    for info in result.values():
        assert "x_grid" in info
        assert "y_pred" in info
        assert "r_squared" in info
        assert "n" in info
        assert len(info["x_grid"]) == len(info["y_pred"])


def test_gam_per_cluster_reasonable_n():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = fit_gam_per_cluster(df, "x1", "cluster")
    total_n = sum(v["n"] for v in result.values())
    assert total_n == len(df)


# ===================================================================
# TODO 3: pooled_vs_cluster_gam_test
# ===================================================================

def test_pooled_vs_cluster_returns_keys():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = pooled_vs_cluster_gam_test(df, "x1", "cluster")
    assert "rss_pooled" in result
    assert "rss_cluster" in result
    assert "f_stat" in result
    assert "p_value" in result


def test_pooled_vs_cluster_rss_relationship():
    df = _diagnostic_df()
    df["cluster"] = assign_clusters(df, x4_threshold=0.0)
    result = pooled_vs_cluster_gam_test(df, "x1", "cluster")
    assert result["rss_cluster"] <= result["rss_pooled"] + 1e-6


# ===================================================================
# TODO 4: fit_r2_ceiling
# ===================================================================

def test_r2_ceiling_returns_keys():
    df = _diagnostic_df()
    result = fit_r2_ceiling(df)
    assert "r_squared" in result
    assert "adj_r_squared" in result
    assert "residual_std" in result
    assert "n" in result


def test_r2_ceiling_high_r2():
    df = _diagnostic_df()
    result = fit_r2_ceiling(df)
    assert result["r_squared"] > 0.5


def test_r2_ceiling_n_matches():
    df = _diagnostic_df()
    result = fit_r2_ceiling(df)
    assert result["n"] == len(df)


# ===================================================================
# TODO 4: feature_group_r2_breakdown
# ===================================================================

def test_r2_breakdown_shape():
    df = _diagnostic_df()
    result = feature_group_r2_breakdown(df)
    assert isinstance(result, pd.DataFrame)
    assert "group" in result.columns
    assert "cumulative_r2" in result.columns
    assert "marginal_r2" in result.columns
    assert len(result) == 4


def test_r2_breakdown_monotonic():
    df = _diagnostic_df()
    result = feature_group_r2_breakdown(df)
    cum = result["cumulative_r2"].values
    for i in range(1, len(cum)):
        assert cum[i] >= cum[i - 1] - 1e-6


def test_r2_breakdown_marginal_positive():
    df = _diagnostic_df()
    result = feature_group_r2_breakdown(df)
    assert all(result["marginal_r2"] >= -1e-6)


# ===================================================================
# TODO 5: x4_bimodality_test
# ===================================================================

def test_bimodality_returns_keys():
    x4 = pd.Series(np.concatenate([
        np.random.default_rng(0).normal(-0.4, 0.1, 100),
        np.random.default_rng(1).normal(0.4, 0.1, 100),
    ]))
    result = x4_bimodality_test(x4)
    assert "method" in result
    assert "statistic" in result
    assert "gap_start" in result
    assert "gap_end" in result
    assert "gap_width" in result


def test_bimodality_detects_bimodal():
    x4 = pd.Series(np.concatenate([
        np.random.default_rng(0).normal(-0.4, 0.05, 200),
        np.random.default_rng(1).normal(0.4, 0.05, 200),
    ]))
    result = x4_bimodality_test(x4)
    assert result["gap_width"] > 0.3


# ===================================================================
# TODO 5: x4_city_distribution_test
# ===================================================================

def test_city_distribution_returns_keys():
    df = _diagnostic_df()
    result = x4_city_distribution_test(df)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "mean_albacete" in result
    assert "mean_zaragoza" in result


def test_city_distribution_symmetric():
    df = _diagnostic_df()
    result = x4_city_distribution_test(df)
    assert result["p_value"] > 0.01


# ===================================================================
# TODO 5: x4_gap_analysis
# ===================================================================

def test_gap_analysis_returns_keys():
    x4 = pd.Series(np.concatenate([
        np.linspace(-0.5, -0.2, 50),
        np.linspace(0.2, 0.5, 50),
    ]))
    result = x4_gap_analysis(x4)
    for key in ["gap_start", "gap_end", "gap_width", "n_below", "n_above",
                "frac_below", "frac_above", "nearest_below", "nearest_above",
                "exact_gap"]:
        assert key in result


def test_gap_analysis_correct_counts():
    x4 = pd.Series(np.concatenate([
        np.linspace(-0.5, -0.2, 50),
        np.linspace(0.2, 0.5, 60),
    ]))
    result = x4_gap_analysis(x4)
    assert result["n_below"] == 50
    assert result["n_above"] == 60
    assert result["gap_start"] == pytest.approx(-0.2)
    assert result["gap_end"] == pytest.approx(0.2)


def test_gap_analysis_detects_exact_gap():
    x4 = pd.Series(np.concatenate([
        np.linspace(-0.5, -0.2, 50),
        np.linspace(0.2, 0.5, 50),
    ]))
    result = x4_gap_analysis(x4)
    assert result["exact_gap"] is True
